"""
Controlled target-selection experimental task prototype.

This module turns the annotated TargetFinder dataset into simple Fitts-style
click trials:

- load screenshot images and YOLO-format target annotations,
- convert normalized annotations to pixel-space bounding boxes,
- sample targets by index of difficulty: log2(1 + D / W),
- display the screenshot and highlighted target,
- reset the cursor, run a countdown, collect clicks, and log trial results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from target_finder_toolkit.filters import add_filter_arguments
from target_finder_toolkit.logging_utils import make_default_log_path
from target_finder_toolkit.mouse_utils import restore_default_cursors

try:
    from target_finder_toolkit.targetfinder import CLASS_NAMES
except Exception:  # pragma: no cover - targetfinder deps may be unavailable.
    CLASS_NAMES = {}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/Users/tangxinqi/Desktop/stage/data/web")
DIFFICULTY_BINS = {
    "easy": (1.5, 2.5),
    "medium": (2.5, 3.5),
    "hard": (3.5, 5.0),
}
TECHNIQUES = [
    "mouse",
    "targetfinder",
    "bubble",
    "semantic",
    "dynaspot",
    "ninja_cursors",
]
TECHNIQUE_MODULES = {
    "targetfinder": "target_finder_toolkit.targetfinder",
    "bubble": "target_finder_toolkit.bubblecursor",
    "semantic": "target_finder_toolkit.semanticpointing",
    "dynaspot": "target_finder_toolkit.dynaspot",
    "ninja_cursors": "target_finder_toolkit.ninjacursors",
}
DEFAULT_CHANGE_THRESH = 100
DEFAULT_CAPTURE_INTERVAL = 1 / 30
DEFAULT_CONFIDENCE = 0.28
DEFAULT_IOU = 0.3
DEFAULT_DYNASPOT_MIN_SPEED = 100.0
DEFAULT_DYNASPOT_SPOT_WIDTH = 32.0
DEFAULT_DYNASPOT_LAG = 0.12
DEFAULT_DYNASPOT_REDUCE_TIME = 0.18
DEFAULT_NINJA_CAMERA_INDEX = 0
DEFAULT_NINJA_SPACING = 320.0
DEFAULT_NINJA_GAZE_SMOOTHING = 0.35
DEFAULT_NINJA_GAZE_GAIN_X = 1.0
DEFAULT_NINJA_GAZE_GAIN_Y = 1.0
DEFAULT_NINJA_GAZE_OFFSET_X = 0.0
DEFAULT_NINJA_GAZE_OFFSET_Y = -200.0
DEFAULT_NINJA_SELECTION_HOLD = 2.0


@dataclass(frozen=True)
class TargetAnnotation:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    distance: float
    width_metric: float
    fitts_id: float
    source_line: str


@dataclass(frozen=True)
class DatasetImage:
    image_path: Path
    label_path: Path
    width: int
    height: int
    targets: tuple[TargetAnnotation, ...]


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    technique: str
    difficulty: str
    image_path: str
    label_path: str
    image_size: tuple[int, int]
    target_index: int
    target_class_id: int
    target_class_name: str
    target_bbox: tuple[float, float, float, float]
    target_center: tuple[float, float]
    start_position: tuple[float, float]
    distance: float
    width_metric: float
    fitts_id: float


def class_name_for_id(class_id: int) -> str:
    if isinstance(CLASS_NAMES, dict):
        return str(CLASS_NAMES.get(class_id, class_id))
    try:
        return str(CLASS_NAMES[class_id])
    except Exception:
        return str(class_id)


def yolo_to_bbox(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    width = width_norm * image_width
    height = height_norm * image_height
    x = (x_center_norm - width_norm / 2.0) * image_width
    y = (y_center_norm - height_norm / 2.0) * image_height
    x = max(0.0, min(float(image_width), x))
    y = max(0.0, min(float(image_height), y))
    width = max(0.0, min(float(image_width) - x, width))
    height = max(0.0, min(float(image_height) - y, height))
    return x, y, width, height


def compute_fitts_id(
    start: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], float, float, float]:
    x, y, width, height = bbox
    center = (x + width / 2.0, y + height / 2.0)
    distance = math.hypot(center[0] - start[0], center[1] - start[1])
    width_metric = max(1.0, min(width, height))
    fitts_id = math.log2(1.0 + distance / width_metric)
    return center, distance, width_metric, fitts_id


def load_dataset(data_dir: Path, *, min_target_size: float = 4.0) -> list[DatasetImage]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths = sorted(
        path
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for path in data_dir.glob(ext)
        if path.with_suffix(".txt").is_file()
    )
    dataset: list[DatasetImage] = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        image = QtGui.QImage(str(image_path))
        if image.isNull():
            continue
        image_width = image.width()
        image_height = image.height()
        start = (image_width / 2.0, image_height / 2.0)
        targets: list[TargetAnnotation] = []

        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                continue
            bbox = yolo_to_bbox(x_center, y_center, width, height, image_width, image_height)
            if bbox[2] < min_target_size or bbox[3] < min_target_size:
                continue
            center, distance, width_metric, fitts_id = compute_fitts_id(start, bbox)
            targets.append(
                TargetAnnotation(
                    class_id=class_id,
                    class_name=class_name_for_id(class_id),
                    bbox=bbox,
                    center=center,
                    distance=distance,
                    width_metric=width_metric,
                    fitts_id=fitts_id,
                    source_line=line,
                )
            )

        if targets:
            dataset.append(
                DatasetImage(
                    image_path=image_path,
                    label_path=label_path,
                    width=image_width,
                    height=image_height,
                    targets=tuple(targets),
                )
            )

    if not dataset:
        raise RuntimeError(f"No annotated images found in {data_dir}")
    return dataset


def sample_trials(
    dataset: list[DatasetImage],
    *,
    technique: str,
    count: int,
    difficulty: str,
    seed: int | None = None,
) -> list[TrialSpec]:
    rng = random.Random(seed)
    candidates: list[tuple[str, DatasetImage, int, TargetAnnotation]] = []
    difficulty_names = list(DIFFICULTY_BINS) if difficulty == "mixed" else [difficulty]

    for difficulty_name in difficulty_names:
        low, high = DIFFICULTY_BINS[difficulty_name]
        bucket = [
            (difficulty_name, item, idx, target)
            for item in dataset
            for idx, target in enumerate(item.targets)
            if low <= target.fitts_id < high
        ]
        candidates.extend(bucket)

    if not candidates:
        raise RuntimeError(f"No targets found for difficulty={difficulty!r}")

    trials: list[TrialSpec] = []
    for trial_id in range(1, count + 1):
        difficulty_name, item, target_index, target = rng.choice(candidates)
        trials.append(
            TrialSpec(
                trial_id=trial_id,
                technique=technique,
                difficulty=difficulty_name,
                image_path=str(item.image_path),
                label_path=str(item.label_path),
                image_size=(item.width, item.height),
                target_index=target_index,
                target_class_id=target.class_id,
                target_class_name=target.class_name,
                target_bbox=target.bbox,
                target_center=target.center,
                start_position=(item.width / 2.0, item.height / 2.0),
                distance=target.distance,
                width_metric=target.width_metric,
                fitts_id=target.fitts_id,
            )
        )
    return trials


def safe_log_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def default_log_path(*, technique: str, difficulty: str) -> Path:
    logs_dir = PROJECT_ROOT / "task_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    technique_name = safe_log_name(technique)
    difficulty_name = safe_log_name(difficulty)
    return logs_dir / f"{stamp}_task_trials_{technique_name}_{difficulty_name}.jsonl"


def default_technique_log_path(technique: str) -> Path:
    return make_default_log_path(PROJECT_ROOT, f"{technique}_during_task")


def build_technique_command(args, technique_log_file: Path | None) -> list[str] | None:
    if args.technique == "mouse":
        return None
    module_name = TECHNIQUE_MODULES.get(args.technique)
    if module_name is None:
        raise ValueError(f"Unsupported launch technique: {args.technique}")

    cmd = [sys.executable, "-m", module_name]
    cmd += [
        "--change-thresh",
        str(args.change_thresh),
        "--capture-interval",
        str(args.capture_interval),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
        "--filter",
        args.filter,
        "--filter-freq",
        str(args.filter_freq),
        "--filter-min-cutoff",
        str(args.filter_min_cutoff),
        "--filter-beta",
        str(args.filter_beta),
        "--filter-d-cutoff",
        str(args.filter_d_cutoff),
    ]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if technique_log_file is not None:
        cmd += ["--log-file", str(technique_log_file), "--log-cursor-hz", str(args.technique_log_cursor_hz)]

    if args.technique == "semantic":
        if args.semantic_display:
            cmd.append("--display")
        if args.semantic_disable_accel:
            cmd.append("--disable-accel")
    elif args.technique == "dynaspot":
        cmd += [
            "--min-speed",
            str(args.dynaspot_min_speed),
            "--spot-width",
            str(args.dynaspot_spot_width),
            "--lag",
            str(args.dynaspot_lag),
            "--reduce-time",
            str(args.dynaspot_reduce_time),
        ]
    elif args.technique == "ninja_cursors":
        cmd += [
            "--camera-index",
            str(args.ninja_camera_index),
            "--ninja-spacing",
            str(args.ninja_spacing),
            "--gaze-smoothing",
            str(args.ninja_gaze_smoothing),
            "--gaze-gain-x",
            str(args.ninja_gaze_gain_x),
            "--gaze-gain-y",
            str(args.ninja_gaze_gain_y),
            "--gaze-offset-x",
            str(args.ninja_gaze_offset_x),
            "--gaze-offset-y",
            str(args.ninja_gaze_offset_y),
            "--selection-hold",
            str(args.ninja_selection_hold),
            "--calib-points",
            str(args.ninja_calib_points),
        ]
        if args.ninja_screen_width_cm is not None:
            cmd += ["--screen-width-cm", str(args.ninja_screen_width_cm)]
        if args.ninja_screen_height_cm is not None:
            cmd += ["--screen-height-cm", str(args.ninja_screen_height_cm)]
        if args.ninja_lock_on_dwell:
            cmd.append("--lock-on-dwell")
        if args.ninja_hide_gaze_point:
            cmd.append("--hide-gaze-point")
        if args.ninja_auto_calibrate:
            cmd.append("--auto-calibrate")
        if args.ninja_without_targetfinder:
            cmd.append("--without-targetfinder")

    return cmd


class TrialCanvas(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._image = QtGui.QImage()
        self._target_bbox: tuple[float, float, float, float] | None = None
        self._show_all_targets = False
        self._all_targets: tuple[TargetAnnotation, ...] = ()
        self._image_rect = QtCore.QRectF()
        self._status_text = ""
        self._message_text = ""
        self.setMouseTracking(True)
        self.setMinimumSize(640, 420)

    def set_trial(
        self,
        image_path: str,
        target_bbox: tuple[float, float, float, float],
        *,
        all_targets: tuple[TargetAnnotation, ...] = (),
        show_all_targets: bool = False,
    ):
        self._image = QtGui.QImage(image_path)
        self._target_bbox = target_bbox
        self._all_targets = all_targets
        self._show_all_targets = show_all_targets
        self.update()

    def set_status_text(self, text: str):
        self._status_text = text
        self.update()

    def set_message_text(self, text: str):
        self._message_text = text
        self.update()

    def image_to_widget_rect(self, bbox: tuple[float, float, float, float]) -> QtCore.QRectF:
        if self._image.isNull() or self._image_rect.isNull():
            return QtCore.QRectF()
        scale_x = self._image_rect.width() / self._image.width()
        scale_y = self._image_rect.height() / self._image.height()
        x, y, width, height = bbox
        return QtCore.QRectF(
            self._image_rect.left() + x * scale_x,
            self._image_rect.top() + y * scale_y,
            width * scale_x,
            height * scale_y,
        )

    def widget_to_image_point(self, point: QtCore.QPointF) -> tuple[float, float] | None:
        if self._image.isNull() or not self._image_rect.contains(point):
            return None
        scale_x = self._image.width() / self._image_rect.width()
        scale_y = self._image.height() / self._image_rect.height()
        return (
            (point.x() - self._image_rect.left()) * scale_x,
            (point.y() - self._image_rect.top()) * scale_y,
        )

    def image_center_global(self) -> QtCore.QPoint:
        if self._image_rect.isNull():
            return self.mapToGlobal(self.rect().center())
        center = self._image_rect.center()
        return self.mapToGlobal(QtCore.QPoint(int(center.x()), int(center.y())))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(18, 18, 18))

        if self._image.isNull():
            painter.setPen(QtGui.QColor(80, 80, 80))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Aucune image")
            painter.end()
            return

        target_size = self._image.size().scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        left = (self.width() - target_size.width()) / 2.0
        top = (self.height() - target_size.height()) / 2.0
        self._image_rect = QtCore.QRectF(left, top, target_size.width(), target_size.height())
        painter.drawImage(self._image_rect, self._image)

        if self._show_all_targets:
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 180, 80, 130), 1.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for target in self._all_targets:
                painter.drawRect(self.image_to_widget_rect(target.bbox))

        if self._target_bbox is not None:
            rect = self.image_to_widget_rect(self._target_bbox)
            painter.setPen(QtGui.QPen(QtGui.QColor(230, 40, 40), 4))
            painter.setBrush(QtGui.QColor(255, 30, 30, 38))
            painter.drawRoundedRect(rect, 8, 8)

        self._draw_overlay_text(painter)
        painter.end()

    def _draw_overlay_text(self, painter: QtGui.QPainter):
        margin = 24
        if self._status_text:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 150))
            status_rect = QtCore.QRectF(margin, margin, self.width() - margin * 2, 54)
            painter.drawRoundedRect(status_rect, 14, 14)
            painter.setPen(QtGui.QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(16)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                status_rect.adjusted(18, 0, -18, 0),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                self._status_text,
            )

        if self._message_text:
            font = painter.font()
            font.setPointSize(34)
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_width = min(metrics.horizontalAdvance(self._message_text) + 64, self.width() - margin * 2)
            message_rect = QtCore.QRectF(
                (self.width() - text_width) / 2,
                self.height() - 110,
                text_width,
                70,
            )
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 160))
            painter.drawRoundedRect(message_rect, 18, 18)
            painter.setPen(QtGui.QColor(255, 80, 80))
            painter.drawText(message_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._message_text)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        point = self.widget_to_image_point(event.position())
        if point is not None:
            self.clicked.emit(point[0], point[1])


class ExperimentalTaskWindow(QtWidgets.QWidget):
    def __init__(
        self,
        trials: list[TrialSpec],
        dataset_by_image: dict[str, DatasetImage],
        *,
        log_file: Path,
        countdown_sec: int = 3,
        max_clicks: int = 1,
        show_all_targets: bool = False,
        technique_command: list[str] | None = None,
        technique_log_file: Path | None = None,
        technique_start_delay_sec: float = 3.0,
        fullscreen: bool = True,
    ):
        super().__init__()
        self.trials = trials
        self.dataset_by_image = dataset_by_image
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.countdown_sec = max(0, int(countdown_sec))
        self.max_clicks = max(1, int(max_clicks))
        self.show_all_targets = show_all_targets
        self.technique_command = list(technique_command) if technique_command else None
        self.technique_log_file = Path(technique_log_file) if technique_log_file else None
        self.technique_start_delay_sec = max(0.0, float(technique_start_delay_sec))
        self.fullscreen = bool(fullscreen)
        self.technique_process: subprocess.Popen | None = None
        self.current_index = -1
        self.current_trial: TrialSpec | None = None
        self.trial_started_at = 0.0
        self.click_count = 0
        self.miss_count = 0
        self._countdown_remaining = 0
        self._accept_clicks = False
        self._session_ended = False

        self.setWindowTitle("Tache experimentale")
        self.resize(1200, 820)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = TrialCanvas()
        self.canvas.clicked.connect(self._handle_click)
        layout.addWidget(self.canvas, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick_countdown)

        self.cursor_lock_timer = QtCore.QTimer(self)
        self.cursor_lock_timer.setInterval(16)
        self.cursor_lock_timer.timeout.connect(self._set_cursor_to_start)

        self.technique_watch_timer = QtCore.QTimer(self)
        self.technique_watch_timer.setInterval(1000)
        self.technique_watch_timer.timeout.connect(self._check_technique_process)

        self._write_event(
            {
                "type": "session_start",
                "log_kind": "experimental_task_trials",
                "task": "controlled_target_selection",
                "trial_count": len(self.trials),
                "countdown_sec": self.countdown_sec,
                "max_clicks": self.max_clicks,
                "technique_process_enabled": self.technique_command is not None,
                "technique_command": self.technique_command,
                "technique_log_file": str(self.technique_log_file) if self.technique_log_file else None,
            }
        )
        if self.technique_command is not None:
            QtCore.QTimer.singleShot(300, self._start_technique_process)
            first_trial_delay = int(max(0.5, self.technique_start_delay_sec) * 1000)
        else:
            first_trial_delay = 200
        QtCore.QTimer.singleShot(first_trial_delay, self.next_trial)

    def closeEvent(self, event):
        self.timer.stop()
        self.cursor_lock_timer.stop()
        self._stop_technique_process()
        self._end_session("window_close")
        super().closeEvent(event)

    def _write_event(self, payload: dict):
        payload = {"timestamp": time.time(), **payload}
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _end_session(self, reason: str):
        if self._session_ended:
            return
        self._session_ended = True
        self._write_event({"type": "session_end", "reason": reason})

    def _start_technique_process(self):
        if self.technique_command is None or self.technique_process is not None:
            return
        popen_kwargs = {"cwd": str(PROJECT_ROOT)}
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        try:
            self.technique_process = subprocess.Popen(self.technique_command, **popen_kwargs)
        except Exception as exc:
            self._write_event(
                {
                    "type": "technique_process_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": self.technique_command,
                }
            )
            self._set_status_text(f"Echec du lancement de la technique : {exc}")
            return
        self._write_event(
            {
                "type": "technique_process_start",
                "pid": self.technique_process.pid,
                "command": self.technique_command,
                "technique_log_file": str(self.technique_log_file) if self.technique_log_file else None,
            }
        )
        self.technique_watch_timer.start()

    def _check_technique_process(self):
        if self.technique_process is None:
            self.technique_watch_timer.stop()
            return
        exit_code = self.technique_process.poll()
        if exit_code is None:
            return
        self._write_event({"type": "technique_process_exit", "exit_code": exit_code})
        self.technique_process = None
        self.technique_watch_timer.stop()
        self._set_status_text(f"{self._status_text()} | Technique arretee ({exit_code})")

    def _stop_technique_process(self):
        if self.technique_process is None:
            self.technique_watch_timer.stop()
            restore_default_cursors()
            return
        proc = self.technique_process
        self.technique_process = None
        self.technique_watch_timer.stop()
        try:
            if proc.poll() is None:
                if sys.platform.startswith("win"):
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        finally:
            restore_default_cursors()
            self._write_event(
                {
                    "type": "technique_process_stop",
                    "exit_code": proc.poll(),
                }
            )

    def _set_cursor_to_start(self):
        QtGui.QCursor.setPos(self.canvas.image_center_global())

    def _status_text(self) -> str:
        return self.canvas._status_text

    def _set_status_text(self, text: str):
        self.canvas.set_status_text(text)

    def _set_message_text(self, text: str):
        self.canvas.set_message_text(text)

    def next_trial(self):
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self._set_status_text(f"Termine. Journal : {self.log_file}")
            self._set_message_text("Termine")
            self._stop_technique_process()
            self._end_session("completed")
            QtCore.QTimer.singleShot(1200, QtWidgets.QApplication.instance().quit)
            return

        self.current_trial = self.trials[self.current_index]
        item = self.dataset_by_image[self.current_trial.image_path]
        self.click_count = 0
        self.miss_count = 0
        self._accept_clicks = False
        self.canvas.set_trial(
            self.current_trial.image_path,
            self.current_trial.target_bbox,
            all_targets=item.targets,
            show_all_targets=self.show_all_targets,
        )
        self._set_status_text(
            f"Essai {self.current_trial.trial_id}/{len(self.trials)} | "
            f"Technique: {self.current_trial.technique} | "
            f"Difficulte: {self.current_trial.difficulty} | "
            f"ID={self.current_trial.fitts_id:.2f} | "
            f"Cible: {self.current_trial.target_class_name}"
        )
        self._write_event({"type": "trial_start", **asdict(self.current_trial)})
        QtCore.QTimer.singleShot(100, self._start_countdown)

    def _start_countdown(self):
        self._accept_clicks = False
        self._set_cursor_to_start()
        self.cursor_lock_timer.start()
        self._countdown_remaining = self.countdown_sec
        if self._countdown_remaining <= 0:
            self._begin_click_phase()
            return
        self._set_message_text(str(self._countdown_remaining))
        self.timer.start()

    def _tick_countdown(self):
        self._set_cursor_to_start()
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            self.timer.stop()
            self._begin_click_phase()
        else:
            self._set_message_text(str(self._countdown_remaining))

    def _begin_click_phase(self):
        self.cursor_lock_timer.stop()
        self._set_cursor_to_start()
        self._set_message_text("Cliquez sur la cible rouge")
        self._accept_clicks = True
        self.trial_started_at = time.monotonic()

    def _target_contains(self, x: float, y: float) -> bool:
        if self.current_trial is None:
            return False
        tx, ty, tw, th = self.current_trial.target_bbox
        return tx <= x <= tx + tw and ty <= y <= ty + th

    def _handle_click(self, x: float, y: float):
        if not self._accept_clicks or self.current_trial is None:
            return
        self.click_count += 1
        success = self._target_contains(x, y)
        elapsed_ms = (time.monotonic() - self.trial_started_at) * 1000.0
        if not success:
            self.miss_count += 1

        self._write_event(
            {
                "type": "trial_click",
                "trial_id": self.current_trial.trial_id,
                "technique": self.current_trial.technique,
                "click_index": self.click_count,
                "click_position": [round(x, 3), round(y, 3)],
                "success": success,
                "movement_time_ms": round(elapsed_ms, 3),
                "miss_count": self.miss_count,
                "target_bbox": list(self.current_trial.target_bbox),
                "target_class_id": self.current_trial.target_class_id,
                "target_class_name": self.current_trial.target_class_name,
                "fitts_id": round(self.current_trial.fitts_id, 4),
                "distance": round(self.current_trial.distance, 3),
                "width_metric": round(self.current_trial.width_metric, 3),
            }
        )

        if success or self.click_count >= self.max_clicks:
            self._accept_clicks = False
            self._write_event(
                {
                    "type": "trial_end",
                    "trial_id": self.current_trial.trial_id,
                    "technique": self.current_trial.technique,
                    "success": success,
                    "movement_time_ms": round(elapsed_ms, 3),
                    "click_count": self.click_count,
                    "miss_count": self.miss_count,
                }
            )
            self._set_message_text("Reussi" if success else "Echec")
            QtCore.QTimer.singleShot(700, self.next_trial)
        else:
            self._set_message_text("Echec - recommencez")

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)


def print_dataset_summary(dataset: list[DatasetImage]):
    all_targets = [target for item in dataset for target in item.targets]
    ids = [target.fitts_id for target in all_targets]
    print(f"images={len(dataset)} targets={len(all_targets)}")
    print(f"fitts_id min={min(ids):.2f} median={sorted(ids)[len(ids)//2]:.2f} max={max(ids):.2f}")
    for name, (low, high) in DIFFICULTY_BINS.items():
        count = sum(1 for value in ids if low <= value < high)
        print(f"{name}: {count} targets in [{low}, {high})")


def main():
    parser = argparse.ArgumentParser(description="Run a controlled target-selection experimental task prototype")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing .png/.txt annotated pairs")
    parser.add_argument("--technique", choices=TECHNIQUES, default="mouse", help="Technique label to store in task logs")
    parser.add_argument("--trials", type=int, default=12, help="Number of trials to generate")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed", help="Target difficulty bin")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible trial sampling")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before each trial starts")
    parser.add_argument("--max-clicks", type=int, default=1, help="Maximum clicks allowed per trial")
    parser.add_argument("--log-file", default=None, help="JSONL log path")
    parser.add_argument("--show-all-targets", action="store_true", help="Draw all annotated targets in green for debugging")
    parser.add_argument("--windowed", action="store_true", help="Run in a normal window instead of fullscreen")
    parser.add_argument("--summary-only", action="store_true", help="Only print dataset and trial summary; do not open GUI")
    parser.add_argument("--no-launch-technique", action="store_true", help="Do not start the selected overlay process")
    parser.add_argument("--technique-start-delay", type=float, default=3.0, help="Seconds to wait before the first trial after launching a technique")
    parser.add_argument("--technique-log-file", default=None, help="Optional JSONL log path for the launched technique process")
    parser.add_argument("--no-technique-log", action="store_true", help="Do not write a separate technique runtime log")
    parser.add_argument("--technique-log-cursor-hz", type=float, default=30.0, help="Cursor sampling rate for the technique runtime log")
    parser.add_argument("--model-path", default=None, help="Optional YOLO model path for launched TargetFinder-based techniques")
    parser.add_argument("--change-thresh", type=int, default=DEFAULT_CHANGE_THRESH, help="Screen-change threshold for launched techniques")
    parser.add_argument("--capture-interval", type=float, default=DEFAULT_CAPTURE_INTERVAL, help="Screen capture interval for launched techniques")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="YOLO confidence for launched techniques")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="YOLO IoU for launched techniques")
    add_filter_arguments(parser)
    parser.add_argument("--semantic-display", action="store_true", help="Show Semantic Pointing visual guides")
    parser.add_argument("--semantic-disable-accel", action="store_true", help="Disable system mouse acceleration for Semantic Pointing")
    parser.add_argument("--dynaspot-min-speed", type=float, default=DEFAULT_DYNASPOT_MIN_SPEED, help="DynaSpot minimum speed threshold")
    parser.add_argument("--dynaspot-spot-width", type=float, default=DEFAULT_DYNASPOT_SPOT_WIDTH, help="DynaSpot maximum spot diameter")
    parser.add_argument("--dynaspot-lag", type=float, default=DEFAULT_DYNASPOT_LAG, help="DynaSpot shrink lag")
    parser.add_argument("--dynaspot-reduce-time", type=float, default=DEFAULT_DYNASPOT_REDUCE_TIME, help="DynaSpot reduction time")
    parser.add_argument("--ninja-camera-index", type=int, default=DEFAULT_NINJA_CAMERA_INDEX, help="Ninja webcam index")
    parser.add_argument("--ninja-screen-width-cm", type=float, default=None, help="Ninja physical screen width in cm")
    parser.add_argument("--ninja-screen-height-cm", type=float, default=None, help="Ninja physical screen height in cm")
    parser.add_argument("--ninja-spacing", type=float, default=DEFAULT_NINJA_SPACING, help="Ninja cursor spacing in pixels")
    parser.add_argument("--ninja-gaze-smoothing", type=float, default=DEFAULT_NINJA_GAZE_SMOOTHING, help="Ninja gaze smoothing")
    parser.add_argument("--ninja-gaze-gain-x", type=float, default=DEFAULT_NINJA_GAZE_GAIN_X, help="Ninja gaze horizontal gain")
    parser.add_argument("--ninja-gaze-gain-y", type=float, default=DEFAULT_NINJA_GAZE_GAIN_Y, help="Ninja gaze vertical gain")
    parser.add_argument("--ninja-gaze-offset-x", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_X, help="Ninja gaze horizontal offset")
    parser.add_argument("--ninja-gaze-offset-y", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_Y, help="Ninja gaze vertical offset")
    parser.add_argument("--ninja-selection-hold", type=float, default=DEFAULT_NINJA_SELECTION_HOLD, help="Ninja dwell duration before lock")
    parser.add_argument("--ninja-lock-on-dwell", action="store_true", help="Require Ninja dwell lock before click")
    parser.add_argument("--ninja-hide-gaze-point", action="store_true", help="Hide Ninja red gaze point")
    parser.add_argument("--ninja-calib-points", type=int, choices=[5, 9, 13], default=5, help="Ninja calibration point count")
    parser.add_argument("--ninja-auto-calibrate", action="store_true", help="Start Ninja calibration automatically")
    parser.add_argument("--ninja-with-targetfinder", dest="ninja_without_targetfinder", action="store_false", help="Enable TargetFinder detections inside Ninja Cursors")
    parser.set_defaults(ninja_without_targetfinder=True)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.data_dir))
    trials = sample_trials(
        dataset,
        technique=args.technique,
        count=args.trials,
        difficulty=args.difficulty,
        seed=args.seed,
    )

    print_dataset_summary(dataset)
    print("sample_trials:")
    for trial in trials[: min(5, len(trials))]:
        print(
            f"  trial={trial.trial_id} difficulty={trial.difficulty} "
            f"ID={trial.fitts_id:.2f} image={Path(trial.image_path).name} "
            f"bbox={[round(v, 1) for v in trial.target_bbox]}"
        )

    if args.summary_only:
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dataset_by_image = {str(item.image_path): item for item in dataset}
    log_file = (
        Path(args.log_file).expanduser()
        if args.log_file
        else default_log_path(technique=args.technique, difficulty=args.difficulty)
    )
    launch_technique = args.technique != "mouse" and not args.no_launch_technique
    technique_log_file = None
    if launch_technique and not args.no_technique_log:
        technique_log_file = (
            Path(args.technique_log_file).expanduser()
            if args.technique_log_file
            else default_technique_log_path(args.technique)
        )
    technique_command = build_technique_command(args, technique_log_file) if launch_technique else None
    window = ExperimentalTaskWindow(
        trials,
        dataset_by_image,
        log_file=log_file,
        countdown_sec=args.countdown,
        max_clicks=args.max_clicks,
        show_all_targets=args.show_all_targets,
        technique_command=technique_command,
        technique_log_file=technique_log_file,
        technique_start_delay_sec=args.technique_start_delay,
        fullscreen=not args.windowed,
    )
    if args.windowed:
        window.show()
    else:
        window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
