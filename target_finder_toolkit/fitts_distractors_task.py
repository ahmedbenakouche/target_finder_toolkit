"""Synthetic Fitts-with-distractors target-selection task.

This task implements the synthetic comparison requested for the project:
targets and distractors are generated directly in a Qt window, then exposed to
the existing interaction techniques through the annotation-control file. In
other words, the task uses a FakeTargetFinder-style input source instead of
screen capture + YOLO detection.

The layout follows the core structure of Blanch & Ortega's CHI 2011 benchmark:
task difficulty is controlled with a Fitts-style ID = log2(1 + D / W), and
distractor density is controlled independently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from target_finder_toolkit.experimental_task import (
    DEFAULT_CHANGE_THRESH,
    DEFAULT_CAPTURE_INTERVAL,
    DEFAULT_CONFIDENCE,
    DEFAULT_DYNASPOT_LAG,
    DEFAULT_DYNASPOT_MIN_SPEED,
    DEFAULT_DYNASPOT_REDUCE_TIME,
    DEFAULT_DYNASPOT_SPOT_WIDTH,
    DEFAULT_IOU,
    DEFAULT_NINJA_CAMERA_INDEX,
    DEFAULT_NINJA_GAZE_GAIN_X,
    DEFAULT_NINJA_GAZE_GAIN_Y,
    DEFAULT_NINJA_GAZE_OFFSET_X,
    DEFAULT_NINJA_GAZE_OFFSET_Y,
    DEFAULT_NINJA_GAZE_SMOOTHING,
    DEFAULT_NINJA_SELECTION_HOLD,
    DEFAULT_NINJA_SPACING,
    EXPERIMENT_TECHNIQUES as TECHNIQUES,
    build_technique_command,
)
from target_finder_toolkit.filters import add_filter_arguments
from target_finder_toolkit.mouse_utils import restore_default_cursors
from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui
from target_finder_toolkit.windows_process_utils import (
    attach_windows_kill_on_close_job,
    close_windows_process_job,
)

try:
    import Quartz
except Exception:  # pragma: no cover - only available on macOS with PyObjC.
    Quartz = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS = 9
DEFAULT_COUNTDOWN = 0.0
DEFAULT_MAX_CLICKS = 1
DEFAULT_CURSOR_HZ = 30.0
HEADER_HEIGHT = 64
RELEASE_GUARD_MS = 120
HOME_RADIUS = 24.0
MIN_DISTRACTOR_DIAMETER = 6.0
MIN_TARGET_DIAMETER = 0.0
DISTRACTOR_SCREEN_MARGIN = 12.0

DIFFICULTY_IDS = {
    "easy": 3.0,
    "medium": 4.0,
    "hard": 5.0,
}
SYNTHETIC_ID_VALUES = (2.0, 4.0, 6.0, 8.0)
DENSITY_VALUES = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.6,
}


@dataclass(frozen=True)
class SyntheticObject:
    object_id: int
    role: str
    center: tuple[float, float]
    diameter: float
    bbox: tuple[float, float, float, float]
    distance: float
    fitts_id: float


@dataclass(frozen=True)
class SyntheticTrial:
    trial_id: int
    technique: str
    difficulty: str
    density: str
    id_value: float
    rho: float
    index_of_sparseness: float | None
    home: tuple[float, float]
    target: SyntheticObject
    distractors: tuple[SyntheticObject, ...]
    amplitude: float
    target_width: float
    nominal_target_width: float
    effective_fitts_id: float
    angle_deg: float
    layout_metadata: dict
    condition_index: int | None = None
    condition_block_index: int | None = None
    condition_repeat_index: int | None = None
    condition_metadata: dict | None = None


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._start = time.monotonic()
        self._closed = False

    def write(self, payload: dict):
        if self._closed:
            return
        event = {
            "timestamp": time.time(),
            "t": round(time.monotonic() - self._start, 6),
            **payload,
        }
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._closed:
            return
        self._fh.close()
        self._closed = True


def is_english(language: str) -> bool:
    return language == "English"


def default_log_file(participant_id: str) -> Path:
    logs_dir = PROJECT_ROOT / "synthetic_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in participant_id).strip("_")
    return logs_dir / f"{safe_id or 'participant'}_{stamp}_fitts_distractors.jsonl"


def _normalize_condition_sequence(condition_sequence: list[dict] | tuple[dict, ...] | None) -> tuple[dict, ...]:
    if not condition_sequence:
        return ()
    normalized: list[dict] = []
    for index, raw_condition in enumerate(condition_sequence, start=1):
        if not isinstance(raw_condition, dict):
            continue
        density = str(raw_condition.get("density") or "medium")
        id_value = float(raw_condition.get("id_value"))
        normalized.append(
            {
                "condition_index": int(raw_condition.get("condition_index") or index),
                "condition_block_index": index,
                "id_value": id_value,
                "difficulty": str(raw_condition.get("difficulty") or f"ID {id_value:g}"),
                "density": density,
                "rho": float(raw_condition.get("rho", DENSITY_VALUES.get(density, 0.3))),
                "technique_label": raw_condition.get("technique_label"),
                "csv_row": raw_condition.get("csv_row"),
                "csv_order": raw_condition.get("csv_order"),
                "csv_token": raw_condition.get("csv_token"),
            }
        )
    return tuple(normalized)


def point_to_list(point: QtCore.QPointF | QtCore.QPoint) -> list[float]:
    return [round(float(point.x()), 3), round(float(point.y()), 3)]


def circle_bbox(cx: float, cy: float, diameter: float) -> tuple[float, float, float, float]:
    radius = diameter / 2.0
    return (cx - radius, cy - radius, diameter, diameter)


def circle_contains(obj: SyntheticObject, x: float, y: float) -> bool:
    return math.hypot(x - obj.center[0], y - obj.center[1]) <= obj.diameter / 2.0


def object_to_log(obj: SyntheticObject) -> dict:
    return {
        "object_id": obj.object_id,
        "role": obj.role,
        "center": [round(obj.center[0], 3), round(obj.center[1], 3)],
        "diameter": round(obj.diameter, 3),
        "bbox": [round(v, 3) for v in obj.bbox],
        "distance": round(obj.distance, 3),
        "fitts_id": round(obj.fitts_id, 4),
    }


def make_object(object_id: int, role: str, center: tuple[float, float], diameter: float, home: tuple[float, float]) -> SyntheticObject:
    distance = math.hypot(center[0] - home[0], center[1] - home[1])
    return SyntheticObject(
        object_id=object_id,
        role=role,
        center=center,
        diameter=diameter,
        bbox=circle_bbox(center[0], center[1], diameter),
        distance=distance,
        fitts_id=math.log2(1.0 + distance / diameter),
    )


def _rotate_vector(distance: float, angle: float) -> tuple[float, float]:
    return (distance * math.cos(angle), distance * math.sin(angle))


def _generate_blanch_ortega_2d_distractors(
    *,
    home: tuple[float, float],
    target: SyntheticObject,
    id_value: float,
    amplitude: float,
    rho: float,
    width: int,
    height: int,
    movement_angle: float,
) -> tuple[list[SyntheticObject], dict]:
    """Generate the 2D self-similar layout from Blanch & Ortega CHI 2011.

    The appendix defines a deterministic 2D layout where density rho is split
    into tangential and radial contributions, with hexagonal packing.
    """
    if rho <= 0.0:
        return [], {
            "layout": "blanch_ortega_2011_2d",
            "rho": rho,
            "rho_t": 0.0,
            "rho_r": 0.0,
            "alpha": None,
            "beta": None,
            "scale_step": None,
            "k": None,
            "min_distractor_diameter": MIN_DISTRACTOR_DIAMETER,
        }

    id_denominator = 2.0**id_value - 1.0
    alpha = math.asin(0.5 / id_denominator)
    k = (math.pi / 4.0) * math.sin(math.pi / 3.0)
    rho_t = math.sqrt(rho / k)
    rho_r = math.sqrt(rho * k)
    beta = 2.0 * alpha / rho_t
    denominator = alpha - (math.pi / 2.0) + (rho_r / math.tan(alpha))
    if denominator <= 0.0:
        scale_step = 1.0
    else:
        scale_step = math.sqrt((math.pi / denominator) + 1.0)
    if scale_step <= 1.0:
        scale_step = 1.0001

    distractors: list[SyntheticObject] = []
    object_id = 2
    max_distance = math.hypot(width, height)
    min_distance = MIN_DISTRACTOR_DIAMETER * id_denominator
    j_min = math.ceil((-math.pi / 2.0) / beta)
    j_max = math.floor((math.pi / 2.0) / beta)

    for j in range(j_min, j_max + 1):
        ray_angle = j * beta
        radial_offset = 0.5 if j % 2 else 0.0
        i_min = math.floor(math.log(max(min_distance / amplitude, 1e-9), scale_step) - radial_offset) - 1
        i_max = math.ceil(math.log(max(max_distance / amplitude, 1e-9), scale_step) - radial_offset) + 1
        for i in range(i_min, i_max + 1):
            if i == 0 and j == 0:
                continue
            radial_distance = amplitude * (scale_step ** (i + radial_offset))
            diameter = radial_distance / id_denominator
            if diameter < MIN_DISTRACTOR_DIAMETER:
                continue
            local_x, local_y = _rotate_vector(radial_distance, ray_angle)
            world_x = home[0] + local_x * math.cos(movement_angle) - local_y * math.sin(movement_angle)
            world_y = home[1] + local_x * math.sin(movement_angle) + local_y * math.cos(movement_angle)
            obj = make_object(object_id, "distractor", (world_x, world_y), diameter, home)
            if not _inside_task_area(obj.center, obj.diameter, width, height):
                continue
            if math.hypot(obj.center[0] - target.center[0], obj.center[1] - target.center[1]) < (obj.diameter + target.diameter) * 0.50:
                continue
            distractors.append(obj)
            object_id += 1

    metadata = {
        "layout": "blanch_ortega_2011_2d",
        "rho": rho,
        "rho_t": rho_t,
        "rho_r": rho_r,
        "alpha": alpha,
        "beta": beta,
        "scale_step": scale_step,
        "k": k,
        "j_range": [j_min, j_max],
        "min_distractor_diameter": MIN_DISTRACTOR_DIAMETER,
        "max_distance": max_distance,
    }
    return distractors, metadata


def _inside_task_area(center: tuple[float, float], diameter: float, width: int, height: int) -> bool:
    radius = diameter / 2.0
    return (
        radius + DISTRACTOR_SCREEN_MARGIN <= center[0] <= width - radius - DISTRACTOR_SCREEN_MARGIN
        and HEADER_HEIGHT + radius + 24 <= center[1] <= height - radius - DISTRACTOR_SCREEN_MARGIN
    )


def generate_trial(
    *,
    trial_id: int,
    technique: str,
    difficulty: str,
    density: str,
    widget_size: tuple[int, int],
    rng: random.Random,
    id_value: float | None = None,
    condition_metadata: dict | None = None,
) -> SyntheticTrial:
    width, height = widget_size
    id_value = float(id_value) if id_value is not None else DIFFICULTY_IDS[difficulty]
    rho = DENSITY_VALUES[density]

    home = (width * 0.22, max(HEADER_HEIGHT + 130.0, height * 0.54))
    amplitude = min(width * 0.52, 760.0)
    angle_deg = 0.0
    angle = math.radians(angle_deg)

    nominal_target_width = amplitude / (2.0**id_value - 1.0)
    target_width = max(nominal_target_width, MIN_TARGET_DIAMETER)
    target_center = (
        home[0] + amplitude * math.cos(angle),
        home[1] + amplitude * math.sin(angle),
    )

    # If a small screen places the target outside the task area, reduce the
    # amplitude while keeping the requested ID by recomputing W.
    while not _inside_task_area(target_center, target_width, width, height) and amplitude > 260.0:
        amplitude *= 0.90
        nominal_target_width = amplitude / (2.0**id_value - 1.0)
        target_width = max(nominal_target_width, MIN_TARGET_DIAMETER)
        target_center = (
            home[0] + amplitude * math.cos(angle),
            home[1] + amplitude * math.sin(angle),
        )

    target = make_object(1, "target", target_center, target_width, home)
    distractors, layout_metadata = _generate_blanch_ortega_2d_distractors(
        home=home,
        target=target,
        id_value=id_value,
        amplitude=target.distance,
        rho=rho,
        width=width,
        height=height,
        movement_angle=angle,
    )

    return SyntheticTrial(
        trial_id=trial_id,
        technique=technique,
        difficulty=difficulty,
        density=density,
        id_value=id_value,
        rho=rho,
        index_of_sparseness=None if rho <= 0.0 else math.log2(1.0 / rho),
        home=home,
        target=target,
        distractors=tuple(distractors),
        amplitude=target.distance,
        target_width=target.diameter,
        nominal_target_width=nominal_target_width,
        effective_fitts_id=target.fitts_id,
        angle_deg=angle_deg,
        layout_metadata={
            **layout_metadata,
            "requested_fitts_id": id_value,
            "nominal_target_width": nominal_target_width,
            "minimum_target_diameter": MIN_TARGET_DIAMETER,
            "effective_fitts_id": target.fitts_id,
        },
        condition_index=(
            int(condition_metadata["condition_index"])
            if condition_metadata is not None and condition_metadata.get("condition_index") is not None
            else None
        ),
        condition_block_index=(
            int(condition_metadata["condition_block_index"])
            if condition_metadata is not None and condition_metadata.get("condition_block_index") is not None
            else None
        ),
        condition_repeat_index=(
            int(condition_metadata["condition_repeat_index"])
            if condition_metadata is not None and condition_metadata.get("condition_repeat_index") is not None
            else None
        ),
        condition_metadata=dict(condition_metadata or {}),
    )


class FittsDistractorsWindow(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal(int)

    def __init__(
        self,
        *,
        participant_id: str,
        technique: str,
        difficulty: str,
        density: str,
        id_value: float | None,
        trials: int,
        countdown: float,
        max_clicks: int,
        log_file: Path,
        cursor_log_hz: float,
        technique_command: list[str] | None,
        technique_log_file: Path | None,
        annotation_control_file: Path,
        language: str,
        seed: int | None,
        ninja_control_file: Path | None = None,
        external_technique_active: bool = False,
        cleanup_control_files: bool = True,
        session_metadata: dict | None = None,
        quit_application_on_complete: bool = True,
        condition_sequence: list[dict] | tuple[dict, ...] | None = None,
    ):
        super().__init__()
        self.participant_id = participant_id
        self.technique = technique
        self.difficulty = difficulty
        self.density = density
        self.id_value = float(id_value) if id_value is not None else None
        self.condition_sequence = _normalize_condition_sequence(condition_sequence)
        self.trials_per_condition = max(1, int(trials))
        self.condition_count = len(self.condition_sequence) if self.condition_sequence else 1
        self.trial_count = (
            self.condition_count * self.trials_per_condition
            if self.condition_sequence
            else self.trials_per_condition
        )
        self.countdown = max(0.0, float(countdown))
        self.max_clicks = max(1, int(max_clicks))
        self.cursor_log_hz = max(1.0, float(cursor_log_hz))
        self.technique_command = list(technique_command) if technique_command else None
        self.technique_log_file = technique_log_file
        self.annotation_control_file = Path(annotation_control_file)
        self.language = language
        self.ninja_control_file = Path(ninja_control_file) if ninja_control_file else None
        self.external_technique_active = bool(external_technique_active)
        self.cleanup_control_files = bool(cleanup_control_files)
        self.session_metadata = dict(session_metadata or {})
        self.quit_application_on_complete = bool(quit_application_on_complete)
        self.rng = random.Random(seed)

        self.logger = JsonlLogger(log_file)
        self.session_id = str(self.session_metadata.get("session_id") or log_file.stem)
        self.technique_process: subprocess.Popen | None = None
        self.current_trial: SyntheticTrial | None = None
        self.current_index = 0
        self.click_count = 0
        self.miss_count = 0
        self.trial_started_at = 0.0
        self.countdown_started_at = 0.0
        self.state = "idle"
        self.accept_clicks = False
        self.mouse_decoupled = False
        self.feedback_success: bool | None = None
        self._exit_code = 0
        self._session_ended = False
        self._aborting = False
        self._finished_emitted = False
        self._app_filter_installed = False
        self._global_keyboard_listener = None
        self._global_keyboard_listener_failed = False
        self._last_recenter_log_at = 0.0
        self._window_shown_logged = False

        if self.technique_command is not None and self._uses_ninja_cursors():
            if self.ninja_control_file is None:
                self.ninja_control_file = self.logger.path.with_suffix(".ninja_control")
            self.technique_command += ["--experiment-control-file", str(self.ninja_control_file)]
            self._set_ninja_control_state("paused")
        elif self.ninja_control_file is not None:
            self._set_ninja_control_state("paused")

        self.setMouseTracking(True)
        self.setWindowTitle("Synthetic Fitts with distractors")
        self.setMinimumSize(1100, 760)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        self.cursor_lock_timer = QtCore.QTimer(self)
        self.cursor_lock_timer.setInterval(8)
        self.cursor_lock_timer.timeout.connect(self._move_cursor_to_home)
        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(50)
        self.countdown_timer.timeout.connect(self._countdown_tick)
        self.cursor_sample_timer = QtCore.QTimer(self)
        self.cursor_sample_timer.setInterval(max(1, int(round(1000.0 / self.cursor_log_hz))))
        self.cursor_sample_timer.timeout.connect(self._write_cursor_sample)

        self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._escape_shortcut.activated.connect(lambda: self._abort_session("keyboard_escape"))
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_filter_installed = True
        self._start_global_keyboard_listener()

        self.logger.write(
            {
                "type": "session_start",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "task": "synthetic_fitts_distractors",
                "technique": self.technique,
                "difficulty": self.difficulty,
                "density": self.density,
                "trials": self.trial_count,
                "trials_per_condition": self.trials_per_condition,
                "condition_count": self.condition_count,
                "condition_sequence": list(self.condition_sequence),
                "countdown_sec": self.countdown,
                "release_guard_ms": RELEASE_GUARD_MS,
                "log_file": str(log_file),
                "technique_log_file": str(technique_log_file) if technique_log_file else None,
                "annotation_control_file": str(annotation_control_file),
                "ninja_control_file": str(self.ninja_control_file) if self.ninja_control_file else None,
                "external_technique_active": self.external_technique_active,
                "source": "synthetic_generated_targets_fake_targetfinder",
                **self.session_metadata,
            }
        )
        self._write_annotation_state("inactive")
        QtCore.QTimer.singleShot(300, self._start)

    def show_desktop_fullscreen(self):
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.show()
        self._raise_window_safely()
        self.raise_()
        self.activateWindow()
        QtWidgets.QApplication.processEvents()
        self._log_window_shown()

    def _keep_task_window_front(self):
        if not self.isVisible():
            return
        self._raise_window_safely()
        self.raise_()
        self.activateWindow()

    def _raise_window_safely(self):
        platform_name = QtGui.QGuiApplication.platformName().lower()
        if platform_name == "offscreen":
            return False
        if self.windowHandle() is None:
            return False
        try:
            return bool(raise_macos_window_above_system_ui(self, level_offset=0))
        except Exception as exc:
            self.logger.write(
                {
                    "type": "window_raise_error",
                    **self._base_event(),
                    "error": str(exc),
                }
            )
            return False

    def _log_window_shown(self):
        if self._window_shown_logged:
            return
        self._window_shown_logged = True
        self.logger.write(
            {
                "type": "window_shown",
                "fullscreen": True,
                "window_size": [int(self.width()), int(self.height())],
            }
        )

    def _start(self):
        if self._session_ended or self._aborting:
            return
        if self.technique_command is not None and self.technique_process is None:
            popen_kwargs = {
                "cwd": str(PROJECT_ROOT),
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform.startswith("win"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            self.technique_process = attach_windows_kill_on_close_job(
                subprocess.Popen(self.technique_command, **popen_kwargs)
            )
            self.logger.write(
                {
                    "type": "technique_process_start",
                    "technique": self.technique,
                    "command": self.technique_command,
                    "annotation_control_file": str(self.annotation_control_file),
                    "technique_log_file": str(self.technique_log_file) if self.technique_log_file else None,
                }
            )
            QtCore.QTimer.singleShot(1200, self._next_trial)
            return
        self._next_trial()

    def _uses_ninja_cursors(self) -> bool:
        return self.technique == "ninja_cursors" or (
            self.technique_command is not None
            and any("target_finder_toolkit.ninjacursors" in part for part in self.technique_command)
        )

    def _ninja_pretrial_state(self) -> str:
        return self._ninja_state_at_screen_center("ready")

    def _ninja_active_state(self) -> str:
        return self._ninja_state_at_screen_center("active")

    def _ninja_state_at_screen_center(self, state: str) -> str:
        if not self._uses_ninja_cursors():
            return "paused"
        center = self._screen_center_global()
        return f"{state} {int(center.x())} {int(center.y())}"

    def _screen_center_global(self) -> QtCore.QPoint:
        screen = None
        handle = self.windowHandle()
        if handle is not None:
            screen = handle.screen()
        if screen is None:
            screen = QtGui.QGuiApplication.screenAt(self.mapToGlobal(self.rect().center()))
        if screen is not None:
            return screen.geometry().center()
        return self.mapToGlobal(self.rect().center())

    def _set_ninja_control_state(self, state: str):
        if self.ninja_control_file is None:
            return
        try:
            control_state = state
            self.ninja_control_file.write_text(control_state, encoding="utf-8")
            self.logger.write({
                "type": "ninja_control_state",
                **self._base_event(),
                "state": state,
                "control_state": control_state,
            })
        except OSError as exc:
            self.logger.write({"type": "ninja_control_error", **self._base_event(), "error": str(exc)})

    def _cleanup_control_files(self):
        if not self.cleanup_control_files:
            return
        try:
            self.annotation_control_file.unlink(missing_ok=True)
        except OSError:
            pass
        if self.ninja_control_file is not None:
            try:
                self.ninja_control_file.unlink(missing_ok=True)
            except OSError:
                pass

    def _base_event(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "task": "synthetic_fitts_distractors",
            "technique": self.technique,
            "trial_id": self.current_trial.trial_id if self.current_trial else None,
        }

    def _condition_for_trial_index(self, trial_index: int) -> dict:
        if not self.condition_sequence:
            return {
                "condition_index": None,
                "condition_block_index": 1,
                "condition_repeat_index": trial_index,
                "id_value": self.id_value,
                "difficulty": self.difficulty,
                "density": self.density,
                "rho": DENSITY_VALUES.get(self.density),
            }
        sequence_index = (trial_index - 1) % len(self.condition_sequence)
        repeat_index = ((trial_index - 1) // len(self.condition_sequence)) + 1
        condition = dict(self.condition_sequence[sequence_index])
        condition["condition_block_index"] = sequence_index + 1
        condition["condition_repeat_index"] = repeat_index
        return condition

    def _next_trial(self):
        if self._session_ended or self._aborting:
            return
        self._keep_task_window_front()
        self.current_index += 1
        if self.current_index > self.trial_count:
            self._end_session("completed")
            QtCore.QTimer.singleShot(700, lambda: self._finish_window(0))
            return

        condition = self._condition_for_trial_index(self.current_index)
        self.current_trial = generate_trial(
            trial_id=self.current_index,
            technique=self.technique,
            difficulty=str(condition.get("difficulty") or self.difficulty),
            density=str(condition.get("density") or self.density),
            id_value=condition.get("id_value", self.id_value),
            widget_size=(max(900, self.width()), max(620, self.height())),
            rng=self.rng,
            condition_metadata=condition,
        )
        self.state = "countdown"
        self.accept_clicks = False
        self.feedback_success = None
        self.click_count = 0
        self.miss_count = 0
        self.countdown_timer.stop()
        self.cursor_sample_timer.stop()
        self._write_annotation_state("inactive")
        self._set_ninja_control_state(self._ninja_pretrial_state())
        self.logger.write(
            {
                "type": "trial_start",
                **self._base_event(),
                "trial": self._trial_payload(),
            }
        )
        self._lock_cursor_at_home()
        self.countdown_started_at = time.monotonic()
        self.logger.write(
            {
                "type": "countdown_start",
                **self._base_event(),
                "countdown_sec": self.countdown,
                "home_widget_position": self._home_position_log(),
            }
        )
        self._keep_task_window_front()
        if self.countdown <= 0:
            self._finish_countdown()
            return
        self.countdown_timer.start()
        self.update()

    def _home_position_log(self) -> list[float] | None:
        if self.current_trial is None:
            return None
        return [round(self.current_trial.home[0], 3), round(self.current_trial.home[1], 3)]

    def _home_global_position(self) -> QtCore.QPoint:
        if self.current_trial is None:
            return self.mapToGlobal(QtCore.QPoint(0, 0))
        home_x, home_y = self.current_trial.home
        return self.mapToGlobal(QtCore.QPoint(int(round(home_x)), int(round(home_y))))

    def _cursor_widget_position(self) -> QtCore.QPointF:
        return QtCore.QPointF(self.mapFromGlobal(QtGui.QCursor.pos()))

    def _cursor_home_error(self) -> float:
        if self.current_trial is None:
            return 0.0
        pos = self._cursor_widget_position()
        home_x, home_y = self.current_trial.home
        return math.hypot(float(pos.x()) - home_x, float(pos.y()) - home_y)

    def _set_mouse_association(self, enabled: bool) -> bool:
        # Do not call CGAssociateMouseAndMouseCursorPosition from this PyQt
        # window. On macOS this native call can crash Python after calibration
        # or when starting a Fitts block. We keep cursor recentering through
        # QCursor.setPos(), which is sufficient for the experiment start point.
        self.mouse_decoupled = False
        return False

    def _uses_semantic_pointing(self) -> bool:
        return self.technique == "semantic" or (
            self.technique_command is not None
            and any("target_finder_toolkit.semanticpointing" in part for part in self.technique_command)
        )

    def _move_cursor_to_home(self):
        if self.current_trial is None:
            return
        if self._uses_ninja_cursors():
            return
        global_pos = self._home_global_position()
        QtGui.QCursor.setPos(global_pos)
        now = time.monotonic()
        if now - self._last_recenter_log_at >= 0.25:
            self._last_recenter_log_at = now
            self.logger.write(
                {
                    "type": "cursor_recenter",
                    **self._base_event(),
                    "state": self.state,
                    "home_widget_position": self._home_position_log(),
                    "requested_screen_position": point_to_list(global_pos),
                    "actual_screen_position": point_to_list(QtGui.QCursor.pos()),
                    "actual_widget_position": point_to_list(self._cursor_widget_position()),
                    "home_error_px": round(self._cursor_home_error(), 3),
                }
            )

    def _lock_cursor_at_home(self):
        if self._uses_ninja_cursors():
            self.logger.write(
                {
                    "type": "cursor_lock_skipped",
                    **self._base_event(),
                    "reason": "ninja_cursors_screen_center_anchor",
                    "ninja_control_state": self._ninja_pretrial_state(),
                }
            )
            return
        self._move_cursor_to_home()
        if self._uses_semantic_pointing():
            self._set_mouse_association(True)
        else:
            self._set_mouse_association(False)
            self._move_cursor_to_home()
        if not self._uses_semantic_pointing() and not self.cursor_lock_timer.isActive():
            self.cursor_lock_timer.start()
        self.logger.write(
            {
                "type": "cursor_lock_start",
                **self._base_event(),
                "home_widget_position": self._home_position_log(),
                "home_error_px": round(self._cursor_home_error(), 3),
                "mouse_decoupled": self.mouse_decoupled,
            }
        )

    def _unlock_cursor_at_home(self):
        if self._uses_ninja_cursors():
            return
        if self._uses_semantic_pointing():
            self.cursor_lock_timer.stop()
            self._set_mouse_association(True)
            self.logger.write(
                {
                    "type": "cursor_lock_end",
                    **self._base_event(),
                    "home_widget_position": self._home_position_log(),
                    "home_error_px": round(self._cursor_home_error(), 3),
                    "mouse_decoupled": self.mouse_decoupled,
                }
            )
            return
        self._move_cursor_to_home()
        self._set_mouse_association(True)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self._move_cursor_to_home()
        self.cursor_lock_timer.stop()
        self._move_cursor_to_home()
        self.logger.write(
            {
                "type": "cursor_lock_end",
                **self._base_event(),
                "home_widget_position": self._home_position_log(),
                "home_error_px": round(self._cursor_home_error(), 3),
                "mouse_decoupled": self.mouse_decoupled,
            }
        )

    def _countdown_tick(self):
        if self._session_ended or self._aborting or self.current_trial is None or self.state != "countdown":
            return
        if not self._uses_ninja_cursors():
            self._move_cursor_to_home()
        if time.monotonic() - self.countdown_started_at >= self.countdown:
            self._finish_countdown()
            return
        self.update()

    def _finish_countdown(self):
        if self._session_ended or self._aborting or self.current_trial is None:
            return
        self._keep_task_window_front()
        self.state = "release_guard"
        self.countdown_timer.stop()
        if not self._uses_ninja_cursors():
            self._move_cursor_to_home()
        if not self._uses_semantic_pointing():
            self._write_annotation_state("active")
        self._set_ninja_control_state(self._ninja_active_state())
        self.logger.write(
            {
                "type": "countdown_end",
                **self._base_event(),
                "home_widget_position": self._home_position_log(),
                "home_error_px": round(self._cursor_home_error(), 3),
                "release_guard_ms": RELEASE_GUARD_MS,
            }
        )
        self.update()
        QtCore.QTimer.singleShot(RELEASE_GUARD_MS, self._begin_movement)

    def _begin_movement(self):
        if self._session_ended or self._aborting or self.current_trial is None:
            return
        self._keep_task_window_front()
        self.state = "movement"
        if self._uses_semantic_pointing():
            self._move_cursor_to_home()
        self._unlock_cursor_at_home()
        if self._uses_semantic_pointing():
            self._write_annotation_state("active")
        self.accept_clicks = True
        self.trial_started_at = time.monotonic()
        self.logger.write(
            {
                "type": "movement_start",
                **self._base_event(),
                "widget_position": point_to_list(self._cursor_widget_position()),
                "home_widget_position": self._home_position_log(),
                "home_error_px": round(self._cursor_home_error(), 3),
            }
        )
        self._write_cursor_sample()
        self.cursor_sample_timer.start()
        self.update()

    def _trial_payload(self) -> dict:
        if self.current_trial is None:
            return {}
        trial = self.current_trial
        return {
            "difficulty": trial.difficulty,
            "density": trial.density,
            "fitts_id": trial.id_value,
            "rho": trial.rho,
            "condition_index": trial.condition_index,
            "condition_block_index": trial.condition_block_index,
            "condition_repeat_index": trial.condition_repeat_index,
            "condition_count": self.condition_count,
            "trials_per_condition": self.trials_per_condition,
            "condition_metadata": dict(trial.condition_metadata or {}),
            "index_of_sparseness": None if trial.index_of_sparseness is None else round(trial.index_of_sparseness, 4),
            "home": [round(trial.home[0], 3), round(trial.home[1], 3)],
            "amplitude": round(trial.amplitude, 3),
            "target_width": round(trial.target_width, 3),
            "nominal_target_width": round(trial.nominal_target_width, 3),
            "effective_fitts_id": round(trial.effective_fitts_id, 4),
            "angle_deg": round(trial.angle_deg, 3),
            "layout_metadata": {
                key: (
                    round(value, 6)
                    if isinstance(value, float)
                    else value
                )
                for key, value in trial.layout_metadata.items()
            },
            "target": object_to_log(trial.target),
            "distractor_count": len(trial.distractors),
            "distractors": [object_to_log(obj) for obj in trial.distractors],
        }

    def _write_annotation_state(self, state: str):
        detections = []
        if state == "active" and self.current_trial is not None:
            for target_index, obj in enumerate((self.current_trial.target, *self.current_trial.distractors)):
                x, y, width, height = obj.bbox
                top_left = self.mapToGlobal(QtCore.QPoint(int(round(x)), int(round(y))))
                detections.append(
                    {
                        "id": obj.object_id,
                        "target_index": target_index,
                        "class_id": 0,
                        "class_name": "Button",
                        "x": float(top_left.x()),
                        "y": float(top_left.y()),
                        "width": width,
                        "height": height,
                        "score": 1.0,
                        "role": obj.role,
                        "synthetic_widget_bbox": [x, y, width, height],
                        "synthetic_widget_center": [obj.center[0], obj.center[1]],
                        "synthetic_distance": obj.distance,
                        "synthetic_fitts_id": obj.fitts_id,
                    }
                )
        payload = {
            "version": 1,
            "state": state,
            "source": "fake_targetfinder",
            "coordinate_space": "screen",
            "task": "synthetic_fitts_distractors",
            "trial_id": self.current_trial.trial_id if self.current_trial else None,
            "trial_key": (
                f"{self.session_id}:"
                f"{self.session_metadata.get('block_id', self.technique)}:"
                f"{self.current_trial.trial_id}"
                if self.current_trial
                else None
            ),
            "start_position_global": point_to_list(self._home_global_position()) if self.current_trial else None,
            "start_position_widget": self._home_position_log(),
            "detections": detections,
        }
        tmp_path = self.annotation_control_file.with_suffix(self.annotation_control_file.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.annotation_control_file)

    def _write_cursor_sample(self):
        if not self.accept_clicks or self.current_trial is None:
            return
        self.logger.write(
            {
                "type": "cursor_sample",
                **self._base_event(),
                "screen_position": point_to_list(QtGui.QCursor.pos()),
                "widget_position": point_to_list(self._cursor_widget_position()),
            }
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self.current_trial is None:
            return
        if not self.accept_clicks or self.state != "movement":
            return
        self.click_count += 1
        pos = event.position()
        success = circle_contains(self.current_trial.target, float(pos.x()), float(pos.y()))
        elapsed_ms = (time.monotonic() - self.trial_started_at) * 1000.0
        if not success:
            self.miss_count += 1
        self.logger.write(
            {
                "type": "click",
                **self._base_event(),
                "click_index": self.click_count,
                "widget_position": point_to_list(pos),
                "success": success,
                "movement_time_ms": round(elapsed_ms, 3),
                "miss_count": self.miss_count,
            }
        )
        if success or self.click_count >= self.max_clicks:
            self._end_trial(success, elapsed_ms)
        self.update()

    def _end_trial(self, success: bool, elapsed_ms: float):
        if self._session_ended or self._aborting:
            return
        self.state = "feedback"
        self.accept_clicks = False
        self.feedback_success = bool(success)
        self.cursor_sample_timer.stop()
        self._write_annotation_state("inactive")
        self._set_ninja_control_state("paused")
        self.logger.write(
            {
                "type": "trial_end",
                **self._base_event(),
                "success": success,
                "movement_time_ms": round(elapsed_ms, 3),
                "click_count": self.click_count,
                "miss_count": self.miss_count,
            }
        )
        self.update()
        QtCore.QTimer.singleShot(550, self._next_trial)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._abort_session("keyboard_escape")
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        if self.isVisible() and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self._abort_session("keyboard_escape")
                return True
        return super().eventFilter(watched, event)

    def _abort_session(self, reason: str):
        if self._aborting:
            return
        self._aborting = True
        self._exit_code = 130
        self._end_session(reason)
        self._stop_global_keyboard_listener()
        self._finish_window(self._exit_code)

    def _start_global_keyboard_listener(self):
        if sys.platform == "darwin":
            return
        if self._global_keyboard_listener is not None or self._global_keyboard_listener_failed:
            return
        try:
            from pynput import keyboard as pynput_keyboard
        except Exception:
            self._global_keyboard_listener_failed = True
            return

        def on_press(key):
            try:
                char = getattr(key, "char", None)
                if key == pynput_keyboard.Key.esc:
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_abort_from_global_keyboard",
                        QtCore.Qt.ConnectionType.QueuedConnection,
                    )
            except Exception:
                return

        try:
            self._global_keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self._global_keyboard_listener.start()
        except Exception:
            self._global_keyboard_listener = None
            self._global_keyboard_listener_failed = True

    def _stop_global_keyboard_listener(self):
        listener = self._global_keyboard_listener
        self._global_keyboard_listener = None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:
            pass

    @QtCore.pyqtSlot()
    def _abort_from_global_keyboard(self):
        if self.isVisible():
            self._abort_session("keyboard_escape")

    def _finish_window(self, exit_code: int):
        if self._finished_emitted:
            return
        self._finished_emitted = True
        self._exit_code = int(exit_code)
        self._stop_global_keyboard_listener()
        if self.quit_application_on_complete:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exit(self._exit_code)
            return
        self.finished.emit(self._exit_code)

    def _end_session(self, reason: str):
        if self._session_ended:
            return
        self._session_ended = True
        self.state = "ended"
        self.accept_clicks = False
        self.cursor_lock_timer.stop()
        self.countdown_timer.stop()
        self.cursor_sample_timer.stop()
        self._set_mouse_association(True)
        self._set_ninja_control_state("paused")
        self._write_annotation_state("inactive")
        if self.technique_process is not None and self.technique_process.poll() is None:
            self.technique_process.terminate()
            try:
                self.technique_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.technique_process.kill()
        close_windows_process_job(self.technique_process)
        restore_default_cursors()
        self.logger.write({"type": "session_end", "reason": reason})
        self.logger.close()
        self._cleanup_control_files()

    def closeEvent(self, event: QtGui.QCloseEvent):
        app = QtWidgets.QApplication.instance()
        if app is not None and self._app_filter_installed:
            app.removeEventFilter(self)
            self._app_filter_installed = False
        self._stop_global_keyboard_listener()
        self._end_session("window_closed" if not self._aborting else "abort_close")
        super().closeEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#f8fafc"))
        self._draw_header(painter)
        if self.current_trial is not None:
            self._draw_trial(painter)
        if self.state in {"countdown", "release_guard"}:
            self._draw_countdown(painter)
        elif self.state == "feedback":
            self._draw_feedback(painter)
        painter.end()

    def _draw_header(self, painter: QtGui.QPainter):
        painter.fillRect(QtCore.QRectF(0, 0, self.width(), HEADER_HEIGHT), QtGui.QColor("#111827"))
        painter.setPen(QtGui.QColor("white"))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        painter.setFont(font)
        if self.current_trial is None:
            text = "Synthetic Fitts with distractors" if is_english(self.language) else "Fitts synthétique avec distracteurs"
        else:
            misses_label = "misses" if is_english(self.language) else "erreurs"
            task_label = "Synthetic Fitts with distractors" if is_english(self.language) else "Fitts synthétique avec distracteurs"
            block_index = self.session_metadata.get("block_index")
            block_count = self.session_metadata.get("block_count")
            if block_index is not None and block_count is not None:
                block_label = (
                    f"Block {block_index}/{block_count}"
                    if is_english(self.language)
                    else f"Bloc {block_index}/{block_count}"
                )
            else:
                block_label = "Block" if is_english(self.language) else "Bloc"
            condition_label = (
                f"Condition {self.current_trial.condition_block_index or 1}/{self.condition_count}"
            )
            trial_label = "Trial" if is_english(self.language) else "Essai"
            repeat_index = self.current_trial.condition_repeat_index or self.current_trial.trial_id
            text = (
                f"{task_label}  "
                f"{block_label}  "
                f"{condition_label}  "
                f"{trial_label} {repeat_index}/{self.trials_per_condition}  "
                f"ID={self.current_trial.id_value:g}  "
                f"rho={self.current_trial.rho:g}  "
                f"{misses_label}={self.miss_count}"
            )
        painter.drawText(QtCore.QRectF(22, 0, self.width() - 44, HEADER_HEIGHT), QtCore.Qt.AlignmentFlag.AlignVCenter, text)

    def _draw_trial(self, painter: QtGui.QPainter):
        if self.current_trial is None:
            return
        trial = self.current_trial
        home = QtCore.QPointF(*trial.home)
        painter.setPen(QtGui.QPen(QtGui.QColor("#94a3b8"), 1.5, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(home, QtCore.QPointF(*trial.target.center))
        for obj in trial.distractors:
            self._draw_circle(painter, obj, QtGui.QColor(180, 180, 180, 190), QtGui.QColor(150, 150, 150, 210))
        self._draw_circle(painter, trial.target, QtGui.QColor(255, 64, 129, 230), QtGui.QColor(236, 32, 96, 255))
        painter.setBrush(QtGui.QColor(47, 140, 255, 70))
        painter.setPen(QtGui.QPen(QtGui.QColor("#1d4ed8"), 3))
        painter.drawEllipse(home, 13, 13)
        painter.setBrush(QtGui.QColor("#2f8cff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), 1.5))
        painter.drawEllipse(home, 5, 5)
        painter.setPen(QtGui.QPen(QtGui.QColor("#1d4ed8"), 2))
        painter.drawLine(QtCore.QPointF(home.x() - 18, home.y()), QtCore.QPointF(home.x() - 9, home.y()))
        painter.drawLine(QtCore.QPointF(home.x() + 9, home.y()), QtCore.QPointF(home.x() + 18, home.y()))
        painter.drawLine(QtCore.QPointF(home.x(), home.y() - 18), QtCore.QPointF(home.x(), home.y() - 9))
        painter.drawLine(QtCore.QPointF(home.x(), home.y() + 9), QtCore.QPointF(home.x(), home.y() + 18))

    def _draw_circle(self, painter: QtGui.QPainter, obj: SyntheticObject, fill: QtGui.QColor, stroke: QtGui.QColor):
        painter.setBrush(fill)
        painter.setPen(QtGui.QPen(stroke, 2))
        painter.drawEllipse(QtCore.QPointF(*obj.center), obj.diameter / 2.0, obj.diameter / 2.0)

    def _draw_countdown(self, painter: QtGui.QPainter):
        remaining = max(0.0, self.countdown - (time.monotonic() - self.countdown_started_at))
        overlay = QtCore.QRectF(0, HEADER_HEIGHT, self.width(), self.height() - HEADER_HEIGHT)
        painter.fillRect(overlay, QtGui.QColor(15, 23, 42, 112))
        painter.setPen(QtGui.QColor("#ffffff"))
        font = QtGui.QFont()
        font.setPointSize(34)
        font.setBold(True)
        painter.setFont(font)
        if self.state == "release_guard":
            text = "Start" if is_english(self.language) else "Départ"
        elif is_english(self.language):
            text = f"Start in {remaining:.1f} s.\nThe cursor is fixed on the blue point."
        else:
            text = f"Départ dans {remaining:.1f} s.\nLe curseur est fixé sur le point bleu."
        painter.drawText(overlay, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def _draw_feedback(self, painter: QtGui.QPainter):
        overlay = QtCore.QRectF(0, HEADER_HEIGHT, self.width(), self.height() - HEADER_HEIGHT)
        painter.fillRect(overlay, QtGui.QColor(255, 255, 255, 92))
        success = bool(self.feedback_success)
        color = QtGui.QColor("#16a34a") if success else QtGui.QColor("#dc2626")
        text = (
            "Success" if success and is_english(self.language)
            else "Failure" if is_english(self.language)
            else "Réussi" if success
            else "Échec"
        )
        painter.setPen(color)
        font = QtGui.QFont()
        font.setPointSize(54)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(overlay, QtCore.Qt.AlignmentFlag.AlignCenter, text)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a synthetic Fitts-with-distractors task.")
    parser.add_argument("--participant-id", "--participant", dest="participant_id", default="P01")
    parser.add_argument("--language", choices=["French", "English"], default="French")
    parser.add_argument("--technique", choices=TECHNIQUES, default="mouse")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--id-value", type=float, default=None, help="Explicit Fitts ID value; overrides named difficulty")
    parser.add_argument("--density", choices=tuple(DENSITY_VALUES), default="medium")
    parser.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN)
    parser.add_argument("--max-clicks", type=int, default=DEFAULT_MAX_CLICKS)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--technique-log-file", default=None)
    parser.add_argument("--cursor-log-hz", type=float, default=DEFAULT_CURSOR_HZ)
    parser.add_argument("--technique-log-cursor-hz", type=float, default=30.0)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--block-count", type=int, default=None)
    parser.add_argument("--block-id", default=None)
    parser.add_argument("--block-order", default=None)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--condition-sequence-json", default=None)
    parser.add_argument("--no-launch-technique", action="store_true")
    parser.add_argument("--annotation-control-file", default=None)
    parser.add_argument("--ninja-control-file", default=None)
    parser.add_argument("--keep-control-files", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--change-thresh", type=int, default=DEFAULT_CHANGE_THRESH)
    parser.add_argument("--capture-interval", type=float, default=DEFAULT_CAPTURE_INTERVAL)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    add_filter_arguments(parser)
    parser.add_argument("--semantic-display", action="store_true")
    parser.add_argument("--semantic-disable-accel", action="store_true")
    parser.add_argument("--dynaspot-min-speed", type=float, default=DEFAULT_DYNASPOT_MIN_SPEED)
    parser.add_argument("--dynaspot-spot-width", type=float, default=DEFAULT_DYNASPOT_SPOT_WIDTH)
    parser.add_argument("--dynaspot-lag", type=float, default=DEFAULT_DYNASPOT_LAG)
    parser.add_argument("--dynaspot-reduce-time", type=float, default=DEFAULT_DYNASPOT_REDUCE_TIME)
    parser.add_argument("--ninja-camera-index", type=int, default=DEFAULT_NINJA_CAMERA_INDEX)
    parser.add_argument("--ninja-screen-width-cm", type=float, default=None)
    parser.add_argument("--ninja-screen-height-cm", type=float, default=None)
    parser.add_argument("--ninja-spacing", type=float, default=DEFAULT_NINJA_SPACING)
    parser.add_argument("--ninja-gaze-smoothing", type=float, default=DEFAULT_NINJA_GAZE_SMOOTHING)
    parser.add_argument("--ninja-gaze-gain-x", type=float, default=DEFAULT_NINJA_GAZE_GAIN_X)
    parser.add_argument("--ninja-gaze-gain-y", type=float, default=DEFAULT_NINJA_GAZE_GAIN_Y)
    parser.add_argument("--ninja-gaze-offset-x", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_X)
    parser.add_argument("--ninja-gaze-offset-y", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_Y)
    parser.add_argument("--ninja-selection-hold", type=float, default=DEFAULT_NINJA_SELECTION_HOLD)
    parser.add_argument("--ninja-lock-on-dwell", action="store_true")
    parser.add_argument("--ninja-hide-gaze-point", action="store_true")
    parser.add_argument("--ninja-hide-debug-status", action="store_true")
    parser.add_argument("--ninja-snap-system-cursor-to-active", action="store_true")
    parser.add_argument("--ninja-calib-points", type=int, choices=[5, 9, 13], default=5)
    parser.add_argument("--ninja-auto-calibrate", action="store_true")
    parser.add_argument("--ninja-with-targetfinder", dest="ninja_without_targetfinder", action="store_false")
    parser.set_defaults(ninja_without_targetfinder=True)
    parser.add_argument("--no-technique-log", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Run without preserving synthetic task JSONL logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    condition_sequence = None
    if args.condition_sequence_json:
        try:
            parsed_conditions = json.loads(args.condition_sequence_json)
            if isinstance(parsed_conditions, list):
                condition_sequence = parsed_conditions
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --condition-sequence-json: {exc}") from exc
    if args.summary_only:
        summary_condition = (condition_sequence or [None])[0]
        trial = generate_trial(
            trial_id=1,
            technique=args.technique,
            difficulty=str(summary_condition.get("difficulty") if summary_condition else args.difficulty),
            density=str(summary_condition.get("density") if summary_condition else args.density),
            id_value=(summary_condition.get("id_value") if summary_condition else args.id_value),
            widget_size=(1200, 800),
            rng=random.Random(args.seed),
            condition_metadata=summary_condition,
        )
        print(
            json.dumps(
                {
                    "trial": {
                        "target": object_to_log(trial.target),
                        "distractor_count": len(trial.distractors),
                        "fitts_id": trial.id_value,
                        "rho": trial.rho,
                    }
                },
                indent=2,
            )
        )
        return 0

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    temp_dir_obj = None
    if args.no_log:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="target_finder_synthetic_task_")
        log_file = Path(temp_dir_obj.name) / "synthetic_task.jsonl"
        args.no_technique_log = True
    else:
        log_file = Path(args.log_file).expanduser().resolve() if args.log_file else default_log_file(args.participant_id)
    annotation_control_file = (
        Path(args.annotation_control_file).expanduser().resolve()
        if args.annotation_control_file
        else log_file.with_suffix(".annotations.json")
    )
    technique_log_file = None if args.no_technique_log else (
        Path(args.technique_log_file).expanduser().resolve()
        if args.technique_log_file
        else log_file.with_name(f"{log_file.stem}_{args.technique}_runtime.jsonl")
    )
    technique_command = None if args.no_launch_technique else build_technique_command(args, technique_log_file, annotation_control_file)
    session_metadata = {
        key: value
        for key, value in {
            "session_id": args.session_id,
            "block_index": args.block_index,
            "block_count": args.block_count,
            "block_id": args.block_id,
            "block_order": args.block_order,
            "trial_offset": args.trial_offset,
        }.items()
        if value is not None
    }

    window = FittsDistractorsWindow(
        participant_id=args.participant_id,
        technique=args.technique,
        difficulty=args.difficulty,
        density=args.density,
        id_value=args.id_value,
        trials=args.trials,
        countdown=args.countdown,
        max_clicks=args.max_clicks,
        log_file=log_file,
        cursor_log_hz=args.cursor_log_hz,
        technique_command=technique_command,
        technique_log_file=technique_log_file,
        annotation_control_file=annotation_control_file,
        language=args.language,
        seed=args.seed,
        ninja_control_file=Path(args.ninja_control_file).expanduser().resolve() if args.ninja_control_file else None,
        external_technique_active=bool(args.no_launch_technique and args.technique != "mouse"),
        cleanup_control_files=not args.keep_control_files,
        session_metadata=session_metadata,
        condition_sequence=condition_sequence,
    )
    if args.windowed:
        window.show()
    else:
        window.show_desktop_fullscreen()
    try:
        exit_code = app.exec()
        return int(window._exit_code or exit_code or 0)
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
