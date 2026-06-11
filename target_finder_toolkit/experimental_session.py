"""Run a full counterbalanced controlled experiment session."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from target_finder_toolkit.experimental_task import (
    DEFAULT_CAPTURE_INTERVAL,
    DEFAULT_CHANGE_THRESH,
    DEFAULT_CONFIDENCE,
    DEFAULT_DATA_DIR,
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
    PROJECT_ROOT,
    ExperimentalTaskWindow,
    build_technique_command,
    load_dataset,
    sample_trials,
)
from target_finder_toolkit.filters import add_filter_arguments


TECHNIQUES = ("bubble", "dynaspot", "semantic", "ninja_cursors")
DIFFICULTIES = ("easy", "medium", "hard")
TECHNIQUE_LABELS = {
    "bubble": "Bubble Cursor",
    "dynaspot": "DynaSpot",
    "semantic": "Pointage sémantique",
    "ninja_cursors": "Ninja Cursors",
}


@dataclass(frozen=True)
class ExperimentBlock:
    block_id: str
    technique: str
    difficulty: str
    trials: int


@dataclass
class PreloadedTechnique:
    technique: str
    command: list[str]
    process: subprocess.Popen
    annotation_control_file: Path
    technique_log_file: Path | None = None
    output_buffer: str = ""
    exit_logged: bool = False
    ready: bool = False


def make_blocks(trials_per_block: int) -> list[ExperimentBlock]:
    blocks: list[ExperimentBlock] = []
    for technique in TECHNIQUES:
        for difficulty in DIFFICULTIES:
            blocks.append(
                ExperimentBlock(
                    block_id=f"{technique}_{difficulty}",
                    technique=technique,
                    difficulty=difficulty,
                    trials=int(trials_per_block),
                )
            )
    return blocks


def counterbalanced_order(
    blocks: list[ExperimentBlock],
    *,
    participant_id: str,
    seed: int | None = None,
) -> list[ExperimentBlock]:
    """Return a deterministic participant-specific block order."""
    rng_seed = f"{participant_id}:{seed}" if seed is not None else participant_id
    rng = random.Random(rng_seed)
    ordered = list(blocks)
    rng.shuffle(ordered)
    return ordered


def write_event(log_file: Path, payload: dict):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": time.time(), **payload}, ensure_ascii=False) + "\n")


def write_annotation_control_file(path: Path, *, state: str):
    payload = {
        "version": 1,
        "state": state,
        "detections": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def add_session_technique_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model-path", default=None, help="Optional YOLO model path for launched techniques")
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
    parser.add_argument("--ninja-calib-points", type=int, choices=[5, 9, 13], default=5)
    parser.add_argument("--ninja-auto-calibrate", action="store_true")
    parser.add_argument("--ninja-with-targetfinder", dest="ninja_without_targetfinder", action="store_false")
    parser.set_defaults(ninja_without_targetfinder=True)
    parser.add_argument("--technique-log-cursor-hz", type=float, default=30.0)


def task_runtime_args(args) -> list[str]:
    values = [
        "--change-thresh", str(args.change_thresh),
        "--capture-interval", str(args.capture_interval),
        "--confidence", str(args.confidence),
        "--iou", str(args.iou),
        "--filter", args.filter,
        "--filter-freq", str(args.filter_freq),
        "--filter-min-cutoff", str(args.filter_min_cutoff),
        "--filter-beta", str(args.filter_beta),
        "--filter-d-cutoff", str(args.filter_d_cutoff),
        "--technique-log-cursor-hz", str(args.technique_log_cursor_hz),
        "--dynaspot-min-speed", str(args.dynaspot_min_speed),
        "--dynaspot-spot-width", str(args.dynaspot_spot_width),
        "--dynaspot-lag", str(args.dynaspot_lag),
        "--dynaspot-reduce-time", str(args.dynaspot_reduce_time),
        "--ninja-camera-index", str(args.ninja_camera_index),
        "--ninja-spacing", str(args.ninja_spacing),
        "--ninja-gaze-smoothing", str(args.ninja_gaze_smoothing),
        "--ninja-gaze-gain-x", str(args.ninja_gaze_gain_x),
        "--ninja-gaze-gain-y", str(args.ninja_gaze_gain_y),
        "--ninja-gaze-offset-x", str(args.ninja_gaze_offset_x),
        "--ninja-gaze-offset-y", str(args.ninja_gaze_offset_y),
        "--ninja-selection-hold", str(args.ninja_selection_hold),
        "--ninja-calib-points", str(args.ninja_calib_points),
    ]
    if args.model_path:
        values += ["--model-path", args.model_path]
    if args.semantic_display:
        values.append("--semantic-display")
    if args.semantic_disable_accel:
        values.append("--semantic-disable-accel")
    if args.ninja_screen_width_cm is not None:
        values += ["--ninja-screen-width-cm", str(args.ninja_screen_width_cm)]
    if args.ninja_screen_height_cm is not None:
        values += ["--ninja-screen-height-cm", str(args.ninja_screen_height_cm)]
    if args.ninja_lock_on_dwell:
        values.append("--ninja-lock-on-dwell")
    if args.ninja_hide_gaze_point:
        values.append("--ninja-hide-gaze-point")
    if args.ninja_auto_calibrate:
        values.append("--ninja-auto-calibrate")
    if not args.ninja_without_targetfinder:
        values.append("--ninja-with-targetfinder")
    return values


def _format_block_label(block: ExperimentBlock) -> str:
    technique = TECHNIQUE_LABELS.get(block.technique, block.technique)
    return f"{technique} · difficulté {block.difficulty}"


def _technique_instruction(block: ExperimentBlock) -> str:
    if block.technique == "ninja_cursors":
        return (
            "Rappel Ninja Cursors : huit curseurs sont affichés. "
            "Regardez le curseur que vous voulez utiliser, puis cliquez avec ce curseur pour sélectionner la cible."
        )
    return ""


def _ensure_qapplication():
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def create_session_screen(*, windowed: bool):
    """Create the fullscreen black screen used between experimental phases."""

    from PyQt6 import QtCore, QtGui, QtWidgets

    try:
        from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui
    except Exception:
        raise_macos_window_above_system_ui = None

    class SessionScreen(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self._windowed = bool(windowed)
            self.aborted = False
            self._keyboard_grabbed = False
            self.setWindowTitle("Expérience")
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            if not self._windowed:
                self.setWindowFlags(
                    QtCore.Qt.WindowType.Window
                    | QtCore.Qt.WindowType.FramelessWindowHint
                    | QtCore.Qt.WindowType.WindowStaysOnTopHint
                )
            self.setStyleSheet(
                """
                QWidget {
                    background: #050505;
                    color: #f2f2f2;
                    font-family: Helvetica, Arial, sans-serif;
                }
                QLabel#Title {
                    font-size: 50px;
                    font-weight: 800;
                }
                QLabel#Body {
                    font-size: 30px;
                    font-weight: 500;
                    color: #d7d7d7;
                }
                QLabel#Hint {
                    font-size: 22px;
                    color: #9a9a9a;
                }
                QPushButton#ContinueButton {
                    background: #f2f2f2;
                    color: #111111;
                    border: 0;
                    border-radius: 18px;
                    font-size: 28px;
                    font-weight: 800;
                    min-width: 260px;
                    min-height: 76px;
                    padding: 10px 42px;
                }
                QPushButton#ContinueButton:hover {
                    background: #ffffff;
                }
                """
            )
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(64, 48, 64, 48)
            layout.setSpacing(18)
            layout.addStretch(1)

            self.title_label = QtWidgets.QLabel("")
            self.title_label.setObjectName("Title")
            self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.title_label.setWordWrap(True)
            layout.addWidget(self.title_label)

            self.body_label = QtWidgets.QLabel("")
            self.body_label.setObjectName("Body")
            self.body_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.body_label.setWordWrap(True)
            layout.addWidget(self.body_label)

            self.hint_label = QtWidgets.QLabel("")
            self.hint_label.setObjectName("Hint")
            self.hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.hint_label.setWordWrap(True)
            layout.addWidget(self.hint_label)

            self.continue_button = QtWidgets.QPushButton("Continuer")
            self.continue_button.setObjectName("ContinueButton")
            self.continue_button.clicked.connect(self._continue)
            button_row = QtWidgets.QHBoxLayout()
            button_row.addStretch(1)
            button_row.addWidget(self.continue_button)
            button_row.addStretch(1)
            layout.addLayout(button_row)
            layout.addStretch(1)

            self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
            self._escape_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            self._escape_shortcut.activated.connect(self._abort)
            self._q_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
            self._q_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            self._q_shortcut.activated.connect(self._abort)
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.installEventFilter(self)

        def eventFilter(self, watched, event):
            if self.isVisible() and event.type() == QtCore.QEvent.Type.KeyPress:
                key = event.key()
                if key in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
                    self._abort()
                    return True
                if self.continue_button.isVisible() and key in (
                    QtCore.Qt.Key.Key_Return,
                    QtCore.Qt.Key.Key_Enter,
                    QtCore.Qt.Key.Key_Space,
                ):
                    self._continue()
                    return True
            return super().eventFilter(watched, event)

        def show_content(
            self,
            *,
            title: str,
            body: str = "",
            hint: str = "",
            button_text: str | None = None,
            level_offset: int = 2,
        ):
            self.aborted = False
            self.title_label.setText(title)
            self.body_label.setText(body)
            self.hint_label.setText(hint)
            if button_text:
                self.continue_button.setText(button_text)
                self.continue_button.show()
                self.continue_button.setFocus()
            else:
                self.continue_button.hide()
            if self._windowed:
                self.resize(980, 640)
                self.show()
            else:
                screen = QtWidgets.QApplication.primaryScreen()
                if screen is not None:
                    self.setGeometry(screen.geometry())
                self.show()
                if raise_macos_window_above_system_ui is not None:
                    raise_macos_window_above_system_ui(self, level_offset=level_offset)
            self.raise_()
            self.activateWindow()
            self._grab_session_keyboard()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()

        def wait_for_continue(self) -> bool:
            app = QtWidgets.QApplication.instance()
            if app is None:
                return False
            result = app.exec()
            return result == 130 or bool(self.aborted)

        def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
                self._abort()
                return
            if self.continue_button.isVisible() and event.key() in (
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Enter,
                QtCore.Qt.Key.Key_Space,
            ):
                self._continue()
                return
            super().keyPressEvent(event)

        def hide(self):
            self._release_session_keyboard()
            super().hide()

        def closeEvent(self, event):
            self._release_session_keyboard()
            super().closeEvent(event)

        def _grab_session_keyboard(self):
            if sys.platform == "darwin":
                # Qt keyboard grabs can crash with the macOS input-method
                # service. The event filter and shortcuts are enough here.
                return
            if self._keyboard_grabbed:
                return
            try:
                self.grabKeyboard()
                self._keyboard_grabbed = True
            except Exception:
                self._keyboard_grabbed = False

        def _release_session_keyboard(self):
            if not self._keyboard_grabbed:
                return
            try:
                self.releaseKeyboard()
            except Exception:
                pass
            self._keyboard_grabbed = False

        @QtCore.pyqtSlot()
        def _continue(self):
            self._release_session_keyboard()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exit(0)

        @QtCore.pyqtSlot()
        def _abort(self):
            self.aborted = True
            self._release_session_keyboard()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exit(130)

    _ensure_qapplication()
    return SessionScreen()


def show_break_screen(
    *,
    current_block: ExperimentBlock,
    next_block: ExperimentBlock,
    windowed: bool,
    screen=None,
):
    """Keep a rest page visible until the participant chooses to continue."""

    owned_screen = screen is None
    screen = screen or create_session_screen(windowed=windowed)
    next_instruction = _technique_instruction(next_block)
    body = (
        f"Bloc terminé : {_format_block_label(current_block)}\n"
        f"Prochain bloc : {_format_block_label(next_block)}"
    )
    if next_instruction:
        body += f"\n\n{next_instruction}"
    screen.show_content(
        title="Pause",
        body=body,
        hint=(
            "Vous pouvez faire une pause aussi longtemps que nécessaire.\n"
            "Cliquez sur Continuer, ou appuyez sur Entrée / Espace, pour reprendre."
        ),
        button_text="Continuer",
    )
    aborted = screen.wait_for_continue()
    if owned_screen:
        screen.hide()
    return aborted


def build_block_command(
    args,
    block: ExperimentBlock,
    *,
    session_id: str,
    block_index: int,
    block_count: int,
    block_order: list[ExperimentBlock],
    trial_offset: int,
    block_log_file: Path,
    technique_log_file: Path,
    annotation_control_file: Path | None,
    ninja_control_file: Path | None,
    no_launch_technique: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "target_finder_toolkit.experimental_task",
        "--data-dir",
        str(args.data_dir),
        "--technique",
        block.technique,
        "--difficulty",
        block.difficulty,
        "--trials",
        str(block.trials),
        "--countdown",
        str(args.countdown),
        "--max-clicks",
        str(args.max_clicks),
        "--participant-id",
        args.participant,
        "--session-id",
        session_id,
        "--block-index",
        str(block_index),
        "--block-count",
        str(block_count),
        "--block-id",
        block.block_id,
        "--block-order",
        ",".join(item.block_id for item in block_order),
        "--trial-offset",
        str(trial_offset),
        "--log-file",
        str(block_log_file),
        "--technique-log-file",
        str(technique_log_file),
        "--cursor-log-hz",
        str(args.cursor_log_hz),
        "--no-task-session-events",
    ]
    if args.show_all_targets:
        cmd.append("--show-all-targets")
    if args.windowed:
        cmd.append("--windowed")
    if args.no_technique_log:
        cmd.append("--no-technique-log")
    if args.technique_start_delay is not None:
        cmd += ["--technique-start-delay", str(args.technique_start_delay)]
    if no_launch_technique:
        cmd.append("--no-launch-technique")
        cmd.append("--keep-control-files")
    if annotation_control_file is not None:
        cmd += ["--annotation-control-file", str(annotation_control_file)]
    if ninja_control_file is not None and block.technique == "ninja_cursors":
        cmd += ["--ninja-control-file", str(ninja_control_file)]
    return cmd + task_runtime_args(args) + extra_args


def _popen_technique(command: list[str]) -> subprocess.Popen:
    popen_kwargs = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if sys.platform.startswith("win"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, **popen_kwargs)
    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass
    return proc


def _drain_preloaded_outputs(
    processes: dict[str, PreloadedTechnique],
    session_log: Path,
) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for technique, item in processes.items():
        proc = item.process
        if proc.stdout is None:
            continue
        chunks = []
        while True:
            try:
                data = os.read(proc.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        if not chunks:
            continue
        item.output_buffer += "".join(chunks)
        while True:
            newline_idx = item.output_buffer.find("\n")
            if newline_idx < 0:
                break
            line = item.output_buffer[:newline_idx].rstrip("\r")
            item.output_buffer = item.output_buffer[newline_idx + 1:]
            if not line:
                continue
            write_event(
                session_log,
                {
                    "type": "technique_process_output",
                    "technique": technique,
                    "line": line,
                },
            )
            calib_prefix = "__NINJA_CALIB__ "
            runtime_prefix = "__NINJA_EVENT__ "
            if technique == "ninja_cursors" and line.startswith(calib_prefix):
                try:
                    payload = json.loads(line[len(calib_prefix):])
                except Exception:
                    payload = {}
                if payload:
                    write_event(session_log, {"type": "ninja_calibration_event", **payload})
                    events.append((technique, payload))
            elif technique == "ninja_cursors" and line.startswith(runtime_prefix):
                try:
                    payload = json.loads(line[len(runtime_prefix):])
                except Exception:
                    payload = {}
                if payload:
                    if payload.get("event") == "ready":
                        item.ready = True
                    write_event(session_log, {"type": "ninja_runtime_event", **payload})
                    events.append((technique, payload))
        exit_code = proc.poll()
        if exit_code is not None and not item.exit_logged:
            item.exit_logged = True
            write_event(
                session_log,
                {
                    "type": "technique_process_exit",
                    "technique": technique,
                    "exit_code": exit_code,
                },
            )
    return events


def _build_preload_command(
    args,
    *,
    technique: str,
    annotation_control_file: Path,
    technique_log_file: Path | None,
    ninja_control_file: Path | None,
) -> list[str]:
    technique_args = argparse.Namespace(**vars(args))
    technique_args.technique = technique
    if technique == "ninja_cursors":
        # In full sessions, calibration is started explicitly by the session
        # screen after the participant has read the instructions.
        technique_args.ninja_auto_calibrate = False
    command = build_technique_command(
        technique_args,
        technique_log_file,
        annotation_control_file,
    )
    if command is None:
        raise RuntimeError(f"No command generated for technique {technique}")
    if technique == "ninja_cursors" and ninja_control_file is not None:
        command += ["--experiment-control-file", str(ninja_control_file)]
    return command


def start_preloaded_techniques(
    args,
    *,
    session_id: str,
    output_dir: Path,
    session_log: Path,
    ninja_control_file: Path,
) -> dict[str, PreloadedTechnique]:
    processes: dict[str, PreloadedTechnique] = {}
    for technique in TECHNIQUES:
        annotation_control_file = output_dir / f"{technique}.annotations.json"
        write_annotation_control_file(annotation_control_file, state="inactive")
        technique_log_file = None if args.no_technique_log else output_dir / f"{session_id}_{technique}_runtime.jsonl"
        command = _build_preload_command(
            args,
            technique=technique,
            annotation_control_file=annotation_control_file,
            technique_log_file=technique_log_file,
            ninja_control_file=ninja_control_file,
        )
        proc = _popen_technique(command)
        processes[technique] = PreloadedTechnique(
            technique=technique,
            command=command,
            process=proc,
            annotation_control_file=annotation_control_file,
            technique_log_file=technique_log_file,
        )
        write_event(
            session_log,
            {
                "type": "technique_process_start",
                "technique": technique,
                "pid": proc.pid,
                "command": command,
                "annotation_control_file": str(annotation_control_file),
                "technique_log_file": str(technique_log_file) if technique_log_file else None,
            },
        )
    return processes


def wait_for_initial_ninja_calibration(
    processes: dict[str, PreloadedTechnique],
    session_log: Path,
    *,
    app=None,
    abort_check=None,
) -> str:
    if "ninja_cursors" not in processes:
        return "missing"
    print("waiting for Ninja Cursors calibration...")
    while True:
        if app is not None:
            app.processEvents()
        if abort_check is not None and abort_check():
            return "aborted"
        for technique, payload in _drain_preloaded_outputs(processes, session_log):
            if technique == "ninja_cursors" and payload.get("event") in {"calibrated", "failed", "cancelled"}:
                print(f"Ninja Cursors calibration: {payload.get('event')}")
                return str(payload.get("event"))
        proc = processes["ninja_cursors"].process
        exit_code = proc.poll()
        if exit_code is not None:
            write_event(
                session_log,
                {
                    "type": "technique_process_exit",
                    "technique": "ninja_cursors",
                    "exit_code": exit_code,
                    "during": "initial_calibration",
                },
            )
            return "exited"
        time.sleep(0.1)


def wait_for_ninja_ready(
    processes: dict[str, PreloadedTechnique],
    session_log: Path,
    *,
    app=None,
    abort_check=None,
) -> str:
    item = processes.get("ninja_cursors")
    if item is None:
        return "missing"
    if item.ready:
        return "ready"
    print("waiting for Ninja Cursors to become ready...")
    while True:
        if app is not None:
            app.processEvents()
        if abort_check is not None and abort_check():
            return "aborted"
        if item.ready:
            return "ready"
        for technique, payload in _drain_preloaded_outputs(processes, session_log):
            if technique == "ninja_cursors" and payload.get("event") == "ready":
                item.ready = True
                print("Ninja Cursors ready.")
                return "ready"
        exit_code = item.process.poll()
        if exit_code is not None:
            write_event(
                session_log,
                {
                    "type": "technique_process_exit",
                    "technique": "ninja_cursors",
                    "exit_code": exit_code,
                    "during": "wait_ready",
                },
            )
            return "exited"
        time.sleep(0.1)


def wait_for_preloaded_startup(
    processes: dict[str, PreloadedTechnique],
    session_log: Path,
    *,
    seconds: float,
    app=None,
    abort_check=None,
) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if abort_check is not None and abort_check():
            return True
        _drain_preloaded_outputs(processes, session_log)
        time.sleep(0.1)
    _drain_preloaded_outputs(processes, session_log)
    if app is not None:
        app.processEvents()
    return bool(abort_check is not None and abort_check())


def cleanup_session_resources(
    *,
    preloaded_processes: dict[str, PreloadedTechnique],
    session_log: Path,
    ninja_control_file: Path | None = None,
    session_screen=None,
    app=None,
):
    try:
        if session_screen is not None:
            session_screen.hide()
        if app is not None:
            app.processEvents()
    except Exception:
        pass
    if preloaded_processes:
        stop_preloaded_techniques(preloaded_processes, session_log)
    if ninja_control_file is not None:
        try:
            ninja_control_file.unlink(missing_ok=True)
        except OSError:
            pass


def stop_preloaded_techniques(
    processes: dict[str, PreloadedTechnique],
    session_log: Path,
):
    for item in processes.values():
        write_annotation_control_file(item.annotation_control_file, state="inactive")
    for technique, item in processes.items():
        proc = item.process
        try:
            if proc.poll() is None:
                if sys.platform.startswith("win"):
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if not sys.platform.startswith("win"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        finally:
            write_event(
                session_log,
                {
                    "type": "technique_process_stop",
                    "technique": technique,
                    "exit_code": proc.poll(),
                },
            )


def run_block_in_process(
    args,
    block: ExperimentBlock,
    *,
    session_id: str,
    block_index: int,
    block_count: int,
    block_order: list[ExperimentBlock],
    trial_offset: int,
    dataset,
    dataset_by_image: dict[str, object],
    block_log_file: Path,
    technique_log_file: Path,
    annotation_control_file: Path | None,
    ninja_control_file: Path | None,
    preloaded: PreloadedTechnique | None,
    transition_screen=None,
) -> int:
    from PyQt6 import QtCore

    app = _ensure_qapplication()
    trials = sample_trials(
        dataset,
        technique=block.technique,
        count=block.trials,
        difficulty=block.difficulty,
        seed=None,
    )

    launch_inside_task = preloaded is None and block.technique != "mouse"
    task_annotation_control_file = annotation_control_file
    if task_annotation_control_file is None and launch_inside_task:
        task_annotation_control_file = block_log_file.with_name(f"block_{block_index:02d}_{block.block_id}.annotations.json")

    technique_command = None
    if launch_inside_task:
        technique_args = argparse.Namespace(**vars(args))
        technique_args.technique = block.technique
        technique_command = build_technique_command(
            technique_args,
            None if args.no_technique_log else technique_log_file,
            task_annotation_control_file,
        )

    window = ExperimentalTaskWindow(
        trials,
        dataset_by_image,
        log_file=block_log_file,
        countdown_sec=args.countdown,
        max_clicks=args.max_clicks,
        show_all_targets=args.show_all_targets,
        technique_command=technique_command,
        technique_log_file=None if args.no_technique_log else technique_log_file,
        annotation_control_file=task_annotation_control_file,
        ninja_control_file=ninja_control_file if block.technique == "ninja_cursors" else None,
        external_technique_active=bool(preloaded),
        cleanup_control_files=not bool(preloaded),
        technique_start_delay_sec=args.technique_start_delay if args.technique_start_delay is not None else 3.0,
        cursor_log_hz=args.cursor_log_hz,
        fullscreen=not args.windowed,
        emit_session_events=False,
        session_metadata={
            "participant_id": args.participant,
            "session_id": session_id,
            "block_index": block_index,
            "block_count": block_count,
            "block_id": block.block_id,
            "block_order": ",".join(item.block_id for item in block_order),
            "trial_offset": trial_offset,
        },
    )
    if args.windowed:
        window.show()
    else:
        window.show_desktop_fullscreen()
    app.processEvents()
    if transition_screen is not None:
        # Keep the black transition screen visible briefly while the next task
        # window finishes becoming fullscreen. This avoids exposing the real
        # desktop between two experimental blocks on macOS.
        QtCore.QTimer.singleShot(150, transition_screen.hide)
        app.processEvents()

    qt_exit_code = app.exec()
    exit_code = int(window._exit_code or qt_exit_code or 0)

    if transition_screen is not None and exit_code == 0:
        transition_screen.show_content(
            title="",
            body="",
            hint="",
            button_text=None,
        )
        app.processEvents()

    window.close()
    window.deleteLater()
    app.processEvents()
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Run a counterbalanced experimental session made of technique/difficulty blocks."
    )
    parser.add_argument("--participant", required=True, help="Participant id, e.g. P01")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Annotated dataset directory")
    parser.add_argument("--trials-per-block", type=int, default=6, help="Trials per technique/difficulty block")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds passed to each block")
    parser.add_argument("--max-clicks", type=int, default=1, help="Maximum clicks per trial")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed combined with participant id")
    parser.add_argument("--output-dir", default=None, help="Directory for session and block logs")
    # Kept for compatibility with older panel/CLI invocations. Breaks are now
    # always manual: the participant continues when ready.
    parser.add_argument("--break-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pause-between-blocks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-all-targets", action="store_true", help="Debug: show all annotated targets")
    parser.add_argument("--windowed", action="store_true", help="Run each block in a window")
    parser.add_argument("--no-technique-log", action="store_true", help="Disable separate runtime technique logs")
    parser.add_argument("--cursor-log-hz", type=float, default=30.0, help="Experiment-level cursor sampling rate")
    parser.add_argument("--technique-start-delay", type=float, default=None, help="Override technique startup delay")
    parser.add_argument("--no-preload-techniques", action="store_true", help="Fallback: launch each technique inside each block")
    parser.add_argument("--dry-run", action="store_true", help="Print generated order without running blocks")
    add_session_technique_arguments(parser)
    args, extra_args = parser.parse_known_args()

    if args.trials_per_block <= 0:
        raise SystemExit("--trials-per-block must be positive")

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.participant}_{session_stamp}"
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROJECT_ROOT / "experience_logs" / session_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = make_blocks(args.trials_per_block)
    ordered_blocks = counterbalanced_order(
        blocks,
        participant_id=args.participant,
        seed=args.seed,
    )
    block_count = len(ordered_blocks)
    total_trials = sum(block.trials for block in ordered_blocks)
    session_log = output_dir / f"{session_id}_session.jsonl"

    print("experimental session")
    print(f"  participant={args.participant}")
    print(f"  session_id={session_id}")
    print(f"  blocks={block_count}")
    print(f"  total_trials={total_trials}")
    print("  order:")
    for index, block in enumerate(ordered_blocks, start=1):
        print(f"    {index:02d}. {block.block_id} ({block.trials} trials)")

    session_started_at = time.time()

    def write_session_end(reason: str, **fields):
        write_event(
            session_log,
            {
                "type": "session_end",
                "reason": reason,
                "total_duration_sec": round(time.time() - session_started_at, 3),
                **fields,
            },
        )

    write_event(
        session_log,
        {
            "type": "session_start",
            "participant_id": args.participant,
            "session_id": session_id,
            "data_dir": str(args.data_dir),
            "trials_per_block": args.trials_per_block,
            "block_count": block_count,
            "total_trials": total_trials,
            "block_order": [asdict(block) for block in ordered_blocks],
            "extra_args": extra_args,
        },
    )

    if args.dry_run:
        write_session_end("dry_run")
        return

    app = _ensure_qapplication()
    dataset = load_dataset(Path(args.data_dir))
    dataset_by_image = {str(item.image_path): item for item in dataset}
    session_screen = create_session_screen(windowed=args.windowed)
    session_screen.show_content(
        title="Initialisation de l'expérience",
        body=(
            "Préparation de la session contrôlée.\n"
            "Chargement des techniques et des paramètres expérimentaux..."
        ),
        hint="Veuillez patienter.",
        button_text=None,
    )
    write_event(session_log, {"type": "initialization_start"})

    ninja_control_file = output_dir / "ninja_cursors.control"
    ninja_control_file.write_text("paused", encoding="utf-8")
    preloaded_processes: dict[str, PreloadedTechnique] = {}
    if not args.no_preload_techniques:
        print("preloading technique processes...")
        preloaded_processes = start_preloaded_techniques(
            args,
            session_id=session_id,
            output_dir=output_dir,
            session_log=session_log,
            ninja_control_file=ninja_control_file,
        )
        wait_for_preloaded_startup(
            preloaded_processes,
            session_log,
            seconds=args.technique_start_delay if args.technique_start_delay is not None else 3.0,
            app=app,
            abort_check=lambda: bool(session_screen.aborted),
        )
        if session_screen.aborted:
            write_session_end("escape_during_initialization")
            cleanup_session_resources(
                preloaded_processes=preloaded_processes,
                session_log=session_log,
                ninja_control_file=ninja_control_file,
                session_screen=session_screen,
                app=app,
            )
            raise SystemExit(130)
        if args.ninja_auto_calibrate:
            session_screen.show_content(
                title="Initialisation de l'expérience",
                body=(
                    "Préparation du suivi du regard.\n"
                    "Veuillez patienter pendant l'initialisation de Ninja Cursors avant la calibration..."
                ),
                hint="Cette étape peut prendre quelques secondes selon la machine.",
                button_text=None,
            )
            ninja_ready = wait_for_ninja_ready(
                preloaded_processes,
                session_log,
                app=app,
                abort_check=lambda: bool(session_screen.aborted),
            )
            if ninja_ready == "aborted":
                write_session_end("escape_during_initialization")
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
            if ninja_ready == "exited":
                write_session_end("ninja_exited_during_initialization")
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
            session_screen.show_content(
                title="Calibration du regard",
                body=(
                    "Avant de commencer l'expérience, une calibration du regard va être effectuée.\n"
                    "Des points rouges apparaîtront successivement à l'écran.\n"
                    "Regardez chaque point rouge sans bouger la tête jusqu'au point suivant.\n"
                    "Après la calibration, un écran vous indiquera que l'expérience peut commencer."
                ),
                hint="Cliquez sur Commencer quand vous êtes prêt(e).",
                button_text="Commencer la calibration",
            )
            write_event(session_log, {"type": "calibration_instructions"})
            if session_screen.wait_for_continue():
                write_session_end("escape_before_calibration")
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
            session_screen.show_content(
                title="Calibration en cours",
                body="Regardez le point rouge affiché à l'écran jusqu'à la fin de la calibration.",
                hint="Ne cliquez pas et évitez de bouger la tête pendant cette étape.",
                button_text=None,
                level_offset=0,
            )
            ninja_control_file.write_text("calibrate", encoding="utf-8")
            write_event(session_log, {"type": "calibration_start_requested"})
            calibration_result = wait_for_initial_ninja_calibration(
                preloaded_processes,
                session_log,
                app=app,
                abort_check=lambda: bool(session_screen.aborted),
            )
            ninja_control_file.write_text("paused", encoding="utf-8")
            if calibration_result in {"aborted", "cancelled"}:
                write_session_end(
                    "escape_during_calibration"
                    if calibration_result == "aborted"
                    else "calibration_cancelled"
                )
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
            if calibration_result == "exited":
                write_session_end("ninja_exited_during_calibration")
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
            session_screen.show_content(
                title="Calibration terminée",
                body=(
                    "L'expérience va maintenant commencer.\n"
                    "Vous allez sélectionner les cibles indiquées à l'écran avec les différentes techniques."
                ),
                hint="Cliquez sur Commencer quand vous êtes prêt(e).",
                button_text="Commencer l'expérience",
            )
            write_event(session_log, {"type": "experiment_start_instructions"})
            if session_screen.wait_for_continue():
                write_session_end("escape_before_first_block")
                cleanup_session_resources(
                    preloaded_processes=preloaded_processes,
                    session_log=session_log,
                    ninja_control_file=ninja_control_file,
                    session_screen=session_screen,
                    app=app,
                )
                raise SystemExit(130)
    write_event(session_log, {"type": "initialization_end"})

    try:
        trial_offset = 0
        for block_index, block in enumerate(ordered_blocks, start=1):
            _drain_preloaded_outputs(preloaded_processes, session_log)
            block_prefix = f"block_{block_index:02d}_{block.block_id}"
            block_log_file = session_log
            technique_log_file = output_dir / f"{block_prefix}_technique.jsonl"
            preloaded = preloaded_processes.get(block.technique)
            if preloaded is not None and preloaded.process.poll() is not None:
                write_session_end(
                    "preloaded_technique_exited",
                    failed_block_index=block_index,
                    failed_block_id=block.block_id,
                    technique=block.technique,
                    exit_code=preloaded.process.poll(),
                )
                raise SystemExit(preloaded.process.poll() or 1)
            cmd = build_block_command(
                args,
                block,
                session_id=session_id,
                block_index=block_index,
                block_count=block_count,
                block_order=ordered_blocks,
                trial_offset=trial_offset,
                block_log_file=block_log_file,
                technique_log_file=technique_log_file,
                annotation_control_file=preloaded.annotation_control_file if preloaded else None,
                ninja_control_file=ninja_control_file if preloaded_processes else None,
                no_launch_technique=bool(preloaded),
                extra_args=extra_args,
            )

            write_event(
                session_log,
                {
                    "type": "block_start",
                    "block_index": block_index,
                    "block_count": block_count,
                    "trial_offset": trial_offset,
                    **asdict(block),
                    "block_log_file": str(block_log_file),
                    "technique_log_file": str(technique_log_file),
                    "preloaded_technique": bool(preloaded),
                    "annotation_control_file": str(preloaded.annotation_control_file) if preloaded else None,
                    "command": cmd,
                    "block_runtime": "in_process",
                },
            )
            print(f"\nstarting block {block_index}/{block_count}: {block.block_id}")
            started = time.time()
            result_code = run_block_in_process(
                args,
                block,
                session_id=session_id,
                block_index=block_index,
                block_count=block_count,
                block_order=ordered_blocks,
                trial_offset=trial_offset,
                dataset=dataset,
                dataset_by_image=dataset_by_image,
                block_log_file=block_log_file,
                technique_log_file=technique_log_file,
                annotation_control_file=preloaded.annotation_control_file if preloaded else None,
                ninja_control_file=ninja_control_file if preloaded_processes else None,
                preloaded=preloaded,
                transition_screen=session_screen,
            )
            elapsed = time.time() - started
            _drain_preloaded_outputs(preloaded_processes, session_log)
            if preloaded:
                write_annotation_control_file(preloaded.annotation_control_file, state="inactive")
            ninja_control_file.write_text("paused", encoding="utf-8")
            write_event(
                session_log,
                {
                    "type": "block_end",
                    "block_index": block_index,
                    "block_count": block_count,
                    "trial_offset": trial_offset,
                    **asdict(block),
                    "returncode": result_code,
                    "elapsed_sec": round(elapsed, 3),
                },
            )
            if result_code != 0:
                reason = "escape_in_block" if result_code == 130 else "block_failed"
                write_session_end(reason, failed_block_index=block_index, returncode=result_code)
                raise SystemExit(result_code)

            trial_offset += block.trials
            if block_index < block_count:
                next_block = ordered_blocks[block_index]
                print(f"pause avant le prochain bloc: {next_block.block_id}")
                pause_started = time.time()
                write_event(
                    session_log,
                    {
                        "type": "pause_start",
                        "after_block_index": block_index,
                        "current_block_id": block.block_id,
                        "next_block_id": next_block.block_id,
                        "current_block": asdict(block),
                        "next_block": asdict(next_block),
                    },
                )
                break_aborted = show_break_screen(
                    current_block=block,
                    next_block=next_block,
                    windowed=args.windowed,
                    screen=session_screen,
                )
                pause_duration = time.time() - pause_started
                _drain_preloaded_outputs(preloaded_processes, session_log)
                write_event(
                    session_log,
                    {
                        "type": "pause_end",
                        "after_block_index": block_index,
                        "current_block_id": block.block_id,
                        "next_block_id": next_block.block_id,
                        "duration_sec": round(pause_duration, 3),
                        "aborted": bool(break_aborted),
                    },
                )
                if break_aborted:
                    write_session_end("escape_on_break", after_block_index=block_index)
                    raise SystemExit(130)

        write_session_end("completed")
        print(f"\nsession completed: {session_log}")
    finally:
        try:
            session_screen.hide()
            app.processEvents()
        except Exception:
            pass
        if preloaded_processes:
            stop_preloaded_techniques(preloaded_processes, session_log)
        try:
            ninja_control_file.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
