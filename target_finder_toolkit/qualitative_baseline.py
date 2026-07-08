"""Qualitative task sequence for Phase I observation.

This module is intentionally separate from the controlled Phase II experiment.
It provides ecological baseline tasks for Phase I observation and writes logs to
qualitative_logs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS_PER_TASK = 1
DEFAULT_CURSOR_HZ = 30.0
TASK_ORDER = ("cursor_stability", "long_distance", "dense_interface")
TASK_LABELS_EN = {
    "cursor_stability": "Cursor stability",
    "long_distance": "Long-distance movement",
    "dense_interface": "Dense interface",
}
TASK_LABELS_FR = {
    "cursor_stability": "Stabilite du curseur",
    "long_distance": "Mouvement longue distance",
    "dense_interface": "Interface dense",
}
TASK_INSTRUCTIONS_EN = {
    "cursor_stability": "Move through Menu 2 into the submenu, then click Submenu 1.",
    "long_distance": "Drag the blue square into the green drop zone and release it inside the target area.",
    "dense_interface": "Click the red highlighted item in the dense grid.",
}
TASK_INSTRUCTIONS_FR = {
    "cursor_stability": "Passez par Menu 2 vers le sous-menu, puis cliquez Submenu 1.",
    "long_distance": "Faites glisser le carré bleu jusque dans la zone verte, puis relâchez-le dedans.",
    "dense_interface": "Cliquez l'élément rouge surligné dans la grille dense.",
}


@dataclass(frozen=True)
class Phase:
    key: str
    technique: str
    filter_name: str
    task_order: tuple[str, ...]
    label_en: str
    label_fr: str


QUALITATIVE_PHASES = (
    Phase(
        "standard_sans_filter",
        "standard_mouse",
        "none",
        TASK_ORDER,
        "Standard mouse without filter",
        "Souris standard sans filtre",
    ),
    Phase(
        "standard_avec_filter",
        "standard_mouse",
        "one_euro",
        TASK_ORDER,
        "Standard mouse with 1€ filter",
        "Souris standard avec filtre 1€",
    ),
    Phase(
        "bubble",
        "bubble",
        "none",
        ("cursor_stability", "dense_interface"),
        "Bubble Cursor",
        "Bubble Cursor",
    ),
    Phase(
        "semantic",
        "semantic",
        "none",
        ("cursor_stability", "dense_interface"),
        "Semantic Pointing",
        "Pointage semantique",
    ),
    Phase(
        "ninja",
        "ninja_cursors",
        "none",
        TASK_ORDER,
        "Ninja Cursors",
        "Ninja Cursors",
    ),
)


def is_english(language: str | None) -> bool:
    return str(language or "").strip().lower().startswith("en")


def default_log_path(participant_id: str) -> Path:
    logs_dir = PROJECT_ROOT / "qualitative_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in participant_id).strip("_") or "participant"
    return logs_dir / f"{safe_id}_{stamp}_qualitative_sequence.jsonl"


def rect_to_list(rect: QtCore.QRectF) -> list[float]:
    return [round(rect.x(), 3), round(rect.y(), 3), round(rect.width(), 3), round(rect.height(), 3)]


def point_to_list(point: QtCore.QPointF | QtCore.QPoint) -> list[float]:
    return [round(float(point.x()), 3), round(float(point.y()), 3)]


@dataclass(frozen=True)
class Trial:
    global_trial_id: int
    task: str
    task_trial_id: int
    variant: int


class JsonlLogger:
    def __init__(self, log_file: Path | None):
        self.path = Path(log_file) if log_file is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        else:
            self._fh = None
        self._start = time.monotonic()
        self._closed = False

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def write(self, payload: dict):
        if self._closed or self._fh is None:
            return
        payload = {"timestamp": time.time(), "t": round(self.elapsed(), 6), **payload}
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._closed:
            return
        if self._fh is not None:
            self._fh.close()
        self._closed = True


class QualitativeBaselineWindow(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        participant_id: str,
        trials_per_task: int,
        log_file: Path | None,
        cursor_hz: float,
        language: str,
        seed: int | None,
        fullscreen: bool = True,
    ):
        super().__init__()
        self.participant_id = participant_id
        self.trials_per_task = int(trials_per_task)
        self.language = language
        self.fullscreen_requested = bool(fullscreen)
        self.logger = JsonlLogger(log_file)
        self.session_id = log_file.stem if log_file is not None else f"{participant_id}_qualitative_no_log"
        self.rng = random.Random(seed)
        self.phases = list(QUALITATIVE_PHASES)
        self.phase_index = -1
        self.phase: Phase | None = None
        self._next_global_trial_id = 1
        self.trials: list[Trial] = []
        self.current_index = -1
        self.current_trial: Trial | None = None
        self.trial_started_at = 0.0
        self.in_trial = False
        self.completed = False
        self.pause_active = False
        self.finished = False
        self.phase_preparing = False
        self.phase_preparation_title = ""
        self.phase_preparation_body = ""
        self._phase_process_started = False
        self._phase_process_ready = False
        self._phase_prepare_started_at = 0.0
        self._ninja_waiting_for_calibration = False
        self._desktop_fullscreen_applied = False
        self.error_count = 0
        self.click_count = 0
        self.last_region: str | None = None
        self.feedback_text = ""
        self.feedback_success = True
        self.feedback_until = 0.0

        self.dragging = False
        self.drag_offset = QtCore.QPointF()
        self.drag_item_rect = QtCore.QRectF()
        self.hover_menu_open = False
        self.entered_path = False
        self.dense_target_index = 0
        self.pause_button_rect = QtCore.QRectF()
        self.finish_button_rect = QtCore.QRectF()
        self.next_phase_label = ""
        self.previous_phase_label = ""
        self.technique_process: subprocess.Popen | None = None
        self._phase_process_stdout_thread: threading.Thread | None = None
        self._phase_process_output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._phase_process_output_timer = QtCore.QTimer(self)
        self._phase_process_output_timer.setInterval(50)
        self._phase_process_output_timer.timeout.connect(self._drain_phase_process_output)
        self.keep_front_timer = QtCore.QTimer(self)
        self.keep_front_timer.setInterval(300)
        self.keep_front_timer.timeout.connect(self._ensure_task_window_visible)
        self.preparation_cursor_lock_timer = QtCore.QTimer(self)
        self.preparation_cursor_lock_timer.setInterval(16)
        self.preparation_cursor_lock_timer.timeout.connect(self._lock_preparation_cursor_to_center)
        control_parent = log_file.parent if log_file is not None else PROJECT_ROOT / "qualitative_logs"
        control_parent.mkdir(parents=True, exist_ok=True)
        self.annotation_control_file = control_parent / f"{self.session_id}.annotations.json"
        self.ninja_control_file = control_parent / f"{self.session_id}.ninja_control"
        self.standard_control_file = control_parent / f"{self.session_id}.standard_control"

        self.cursor_timer = QtCore.QTimer(self)
        self.cursor_timer.setInterval(max(1, int(round(1000.0 / max(float(cursor_hz), 1.0)))))
        self.cursor_timer.timeout.connect(self._write_cursor_sample)

        self.setMouseTracking(True)
        self.setWindowTitle("Qualitative Tasks")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(900, 620)
        self.logger.write(
            {
                "type": "session_start",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "phase_order": [
                    {
                        "phase": phase.key,
                        "technique": phase.technique,
                        "filter": phase.filter_name,
                        "tasks": list(phase.task_order),
                    }
                    for phase in self.phases
                ],
                "trials_per_task": self.trials_per_task,
                "total_trials": sum(len(phase.task_order) * self.trials_per_task for phase in self.phases),
                "log_file": str(log_file) if log_file is not None else None,
                "logging_enabled": log_file is not None,
            }
        )
        self._write_annotation_control_state("paused")
        self._write_ninja_control_state("paused")
        self._write_standard_control_state("paused")
        QtCore.QTimer.singleShot(250, self._start_next_phase)

    def _labels(self) -> dict[str, str]:
        return TASK_LABELS_EN if is_english(self.language) else TASK_LABELS_FR

    def _instructions(self) -> dict[str, str]:
        return TASK_INSTRUCTIONS_EN if is_english(self.language) else TASK_INSTRUCTIONS_FR

    def _phase_label(self, phase: Phase | None = None) -> str:
        phase = phase or self.phase
        if phase is None:
            return ""
        return phase.label_en if is_english(self.language) else phase.label_fr

    def _make_trials(self, task_order: tuple[str, ...]) -> list[Trial]:
        trials: list[Trial] = []
        global_id = self._next_global_trial_id
        for task in task_order:
            for task_trial_id in range(1, self.trials_per_task + 1):
                trials.append(Trial(global_id, task, task_trial_id, self.rng.randint(0, 10_000)))
                global_id += 1
        self._next_global_trial_id = global_id
        return trials

    def _base_event(self) -> dict:
        task = self.current_trial.task if self.current_trial else None
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "phase_index": self.phase_index + 1 if self.phase is not None else None,
            "phase_count": len(self.phases),
            "phase": self.phase.key if self.phase is not None else None,
            "phase_label": self._phase_label(),
            "technique": self.phase.technique if self.phase is not None else None,
            "filter": self.phase.filter_name if self.phase is not None else None,
            "global_trial_id": self.current_trial.global_trial_id if self.current_trial else None,
            "task": task,
            "task_label": self._labels().get(task, task) if task else None,
            "task_trial_id": self.current_trial.task_trial_id if self.current_trial else None,
        }

    def _start_next_phase(self):
        self.pause_active = False
        self.finished = False
        self.phase_index += 1
        if self.phase_index >= len(self.phases):
            self._finish_session()
            return

        self.phase = self.phases[self.phase_index]
        self.current_index = -1
        self.current_trial = None
        self.trials = self._make_trials(self.phase.task_order)
        self._phase_process_started = False
        self._ninja_waiting_for_calibration = False
        self.phase_preparing = False
        self.phase_preparation_title = ""
        self.phase_preparation_body = ""
        self.logger.write(
            {
                "type": "phase_start",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "phase_index": self.phase_index + 1,
                "phase_count": len(self.phases),
                "phase": self.phase.key,
                "phase_label": self._phase_label(),
                "technique": self.phase.technique,
                "filter": self.phase.filter_name,
                "tasks": list(self.phase.task_order),
                "trial_count": len(self.trials),
            }
        )
        self._ensure_task_window_visible()
        self.next_trial()

    @QtCore.pyqtSlot()
    def _ensure_task_window_visible(self):
        if self.fullscreen_requested:
            self.show_desktop_fullscreen()
            return
        if not self.isVisible():
            self.show()
        raise_macos_window_above_system_ui(self, level_offset=0)
        self.raise_()
        self.activateWindow()

    def show_desktop_fullscreen(self):
        """Cover the screen without entering the macOS fullscreen Space."""
        if not self._desktop_fullscreen_applied:
            self.setWindowFlags(
                QtCore.Qt.WindowType.Window
                | QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                self.setGeometry(screen.geometry())
            self._desktop_fullscreen_applied = True
        if not self.isVisible():
            self.show()
        raise_macos_window_above_system_ui(self, level_offset=0)
        self.raise_()
        self.activateWindow()

    def _complete_phase(self):
        if self.phase is None:
            self._finish_session()
            return
        completed_phase = self.phase
        self.phase_preparing = False
        self.preparation_cursor_lock_timer.stop()
        self._ninja_waiting_for_calibration = False
        self._write_annotation_control_state("paused")
        self._write_ninja_control_state("paused")
        self._write_standard_control_state("paused")
        self._stop_phase_process()
        self.logger.write(
            {
                "type": "phase_end",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "phase_index": self.phase_index + 1,
                "phase_count": len(self.phases),
                "phase": completed_phase.key,
                "phase_label": self._phase_label(completed_phase),
                "technique": completed_phase.technique,
                "filter": completed_phase.filter_name,
                "reason": "completed",
            }
        )
        if self.phase_index + 1 >= len(self.phases):
            self._finish_session()
            return
        self.previous_phase_label = self._phase_label(completed_phase)
        self.next_phase_label = self._phase_label(self.phases[self.phase_index + 1])
        self.current_trial = None
        self.in_trial = False
        self.pause_active = True
        self.update()

    def _phase_process_log_file(self) -> Path | None:
        if self.logger.path is None or self.phase is None:
            return None
        return self.logger.path.with_name(f"{self.session_id}_{self.phase.key}_runtime.jsonl")

    def _start_phase_process(self) -> bool:
        if self.phase is None or self.phase.key == "standard_sans_filter":
            return False
        if self._phase_process_started and self.technique_process is not None:
            return True
        cmd = self._build_phase_process_command(self.phase)
        if not cmd:
            return False
        self.logger.write(
            {
                "type": "phase_process_start",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "phase": self.phase.key,
                "command": cmd,
            }
        )
        try:
            self.technique_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
            self._phase_process_started = True
            self._start_phase_output_reader(self.phase.key, self.technique_process)
            self._phase_process_output_timer.start()
            if self.phase.key != "ninja":
                self.keep_front_timer.start()
                for delay_ms in (50, 150, 350):
                    QtCore.QTimer.singleShot(delay_ms, self._ensure_task_window_visible)
                if self.phase.key in {"bubble", "semantic"}:
                    stop_delay_ms = max(400, self._phase_start_delay_ms() - 100)
                    QtCore.QTimer.singleShot(stop_delay_ms, self.keep_front_timer.stop)
            return True
        except Exception as exc:
            self.technique_process = None
            self.logger.write(
                {
                    "type": "phase_process_error",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key,
                    "error": str(exc),
                }
            )
            return False

    def _start_phase_output_reader(self, phase_key: str, proc: subprocess.Popen):
        stream = proc.stdout
        if stream is None:
            return

        def _reader():
            try:
                for raw_line in stream:
                    line = raw_line.rstrip("\r\n")
                    if line:
                        self._phase_process_output_queue.put((phase_key, line))
            except Exception as exc:
                self._phase_process_output_queue.put((phase_key, f"__reader_error__ {type(exc).__name__}: {exc}"))

        self._phase_process_stdout_thread = threading.Thread(target=_reader, daemon=True)
        self._phase_process_stdout_thread.start()

    def _drain_phase_process_output(self):
        drained = False
        while True:
            try:
                phase_key, line = self._phase_process_output_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            self.logger.write(
                {
                    "type": "phase_process_output",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": phase_key,
                    "line": line,
                }
            )
            if phase_key == "ninja":
                self._handle_ninja_process_output(line)
            elif phase_key == "bubble":
                self._handle_bubble_process_output(line)
        proc = self.technique_process
        if proc is not None and proc.poll() is not None:
            if self._ninja_waiting_for_calibration:
                self._ninja_waiting_for_calibration = False
                self._activate_current_trial()
            if not drained:
                self._phase_process_output_timer.stop()

    def _handle_bubble_process_output(self, line: str):
        if line.strip() == "__BUBBLE_EVENT__ ready":
            self._phase_process_ready = True
            self.logger.write(
                {
                    "type": "bubble_ready",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": "bubble",
                }
            )

    def _handle_ninja_process_output(self, line: str):
        calib_prefix = "__NINJA_CALIB__ "
        runtime_prefix = "__NINJA_EVENT__ "
        if line.startswith(runtime_prefix):
            try:
                payload = json.loads(line[len(runtime_prefix):])
            except Exception:
                payload = {}
            if payload:
                self.logger.write(
                    {
                        "type": "ninja_runtime_event",
                        "participant_id": self.participant_id,
                        "session_id": self.session_id,
                        "phase": "ninja",
                        **payload,
                    }
                )
            return
        if not line.startswith(calib_prefix):
            return
        try:
            payload = json.loads(line[len(calib_prefix):])
        except Exception:
            payload = {}
        if payload:
            self.logger.write(
                {
                    "type": "ninja_calibration_event",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": "ninja",
                    **payload,
                }
            )
        event = payload.get("event")
        if event in {"calibrated", "failed", "cancelled"} and self._ninja_waiting_for_calibration:
            self._ninja_waiting_for_calibration = False
            self._write_ninja_control_state(self._center_control_state("ready"))
            self.keep_front_timer.start()
            self._ensure_task_window_visible()
            QtCore.QTimer.singleShot(350, self._activate_current_trial)

    def _build_phase_process_command(self, phase: Phase) -> list[str]:
        log_file = self._phase_process_log_file()
        common = ["--log-file", str(log_file), "--log-cursor-hz", "30"] if log_file is not None else []
        if phase.key == "standard_avec_filter":
            return [
                sys.executable,
                "-m",
                "target_finder_toolkit.standard_mouse",
                "--filter",
                "one_euro",
                "--without-targetfinder",
                "--control-file",
                str(self.standard_control_file),
                *common,
            ]
        if phase.key == "bubble":
            return [
                sys.executable,
                "-m",
                "target_finder_toolkit.bubblecursor",
                "--annotation-control-file",
                str(self.annotation_control_file),
                "--include-text-targets",
                "--disable-keyboard-quit",
                *common,
            ]
        if phase.key == "semantic":
            return [
                sys.executable,
                "-m",
                "target_finder_toolkit.semanticpointing",
                "--annotation-control-file",
                str(self.annotation_control_file),
                "--disable-keyboard-quit",
                *common,
            ]
        if phase.key == "ninja":
            return [
                sys.executable,
                "-m",
                "target_finder_toolkit.ninjacursors",
                "--annotation-control-file",
                str(self.annotation_control_file),
                "--experiment-control-file",
                str(self.ninja_control_file),
                "--disable-keyboard-quit",
                "--hide-debug-status",
                "--hide-gaze-point",
                "--snap-system-cursor-to-active",
                "--auto-calibrate",
                "--auto-calibrate-delay",
                "0.3",
                *common,
            ]
        return []

    def _stop_phase_process(self):
        proc = self.technique_process
        self.technique_process = None
        self.keep_front_timer.stop()
        self._phase_process_output_timer.stop()
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.logger.write(
            {
                "type": "phase_process_stop",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "phase": self.phase.key if self.phase is not None else None,
                "exit_code": proc.poll(),
            }
        )

    def next_trial(self):
        if self.current_trial is not None:
            self._end_trial(success=self.completed, reason="completed" if self.completed else "advanced")
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self._complete_phase()
            return

        self.current_trial = self.trials[self.current_index]
        self.in_trial = False
        self.phase_preparing = False
        self.completed = False
        self.error_count = 0
        self.click_count = 0
        self.last_region = None
        self.feedback_text = ""
        self.feedback_success = True
        self.feedback_until = 0.0
        self.dragging = False
        self.hover_menu_open = False
        self.entered_path = False
        self._setup_trial_geometry()
        self._write_annotation_control_state("active")
        if self.phase is not None and self.phase.key == "standard_avec_filter":
            self._write_standard_control_state(self._center_control_state("ready"))
        if self._needs_phase_process_start():
            self.phase_preparing = True
            self._phase_process_ready = False
            self._phase_prepare_started_at = time.monotonic()
            self._set_phase_preparation_message()
            if self.phase is not None and self.phase.key == "bubble":
                self._reset_system_cursor_to_trial_center()
                self.preparation_cursor_lock_timer.start()
            if self.phase is not None and self.phase.key == "ninja":
                self._write_ninja_control_state("calibrate")
            if self._start_phase_process():
                if self.phase is not None and self.phase.key == "ninja":
                    self._ninja_waiting_for_calibration = True
                    self.logger.write(
                        {
                            "type": "ninja_calibration_wait",
                            **self._base_event(),
                        }
                    )
                    self.update()
                    return
                self.update()
                QtCore.QTimer.singleShot(100, self._wait_for_phase_ready_then_activate)
                return
        self._activate_current_trial()

    def _needs_phase_process_start(self) -> bool:
        return (
            self.phase is not None
            and self.phase.key != "standard_sans_filter"
            and not self._phase_process_started
        )

    def _phase_start_delay_ms(self) -> int:
        if self.phase is None:
            return 0
        if self.phase.key in {"bubble", "semantic"}:
            return 1200
        if self.phase.key == "standard_avec_filter":
            return 900
        return 0

    def _wait_for_phase_ready_then_activate(self):
        if not self.phase_preparing or self.current_trial is None or self.pause_active or self.finished:
            self.preparation_cursor_lock_timer.stop()
            return
        if self.phase is None:
            self._activate_current_trial()
            return

        if self.phase.key == "bubble":
            self._reset_system_cursor_to_trial_center()

        proc = self.technique_process
        if proc is not None and proc.poll() is not None:
            if is_english(self.language):
                self.phase_preparation_title = "Bubble Cursor did not start"
                self.phase_preparation_body = "The interaction technique stopped before the trial could begin. Press Esc to quit this session."
            else:
                self.phase_preparation_title = "Bubble Cursor n'a pas démarré"
                self.phase_preparation_body = "La technique d'interaction s'est arrêtée avant le début de l'essai. Appuyez sur Échap pour quitter cette session."
            self.logger.write(
                {
                    "type": "phase_process_failed_before_ready",
                    **self._base_event(),
                    "phase": self.phase.key,
                    "returncode": proc.returncode,
                }
            )
            self.update()
            return

        elapsed_ms = (time.monotonic() - self._phase_prepare_started_at) * 1000.0
        ready = elapsed_ms >= self._phase_start_delay_ms()
        if self.phase.key == "bubble":
            ready = ready and self._phase_process_ready

        if ready:
            self._activate_current_trial()
            return
        QtCore.QTimer.singleShot(100, self._wait_for_phase_ready_then_activate)

    def _set_phase_preparation_message(self):
        label = self._phase_label()
        if self.phase is not None and self.phase.key == "ninja":
            if is_english(self.language):
                self.phase_preparation_title = "Ninja Cursors calibration"
                self.phase_preparation_body = (
                    "Look at each red calibration point until the next one appears. "
                    "The first Ninja trial starts automatically after calibration."
                )
            else:
                self.phase_preparation_title = "Calibration Ninja Cursors"
                self.phase_preparation_body = (
                    "Regardez chaque point rouge de calibration jusqu'au point suivant. "
                    "Le premier essai Ninja commence automatiquement après la calibration."
                )
            return
        if is_english(self.language):
            self.phase_preparation_title = f"Preparing {label}"
            if self.phase is not None and self.phase.key == "bubble":
                self.phase_preparation_body = "Bubble Cursor is starting. Keep the cursor at the center point; the task will begin when Bubble is ready."
            else:
                self.phase_preparation_body = "The interaction technique is starting. The task will begin automatically in a moment."
        else:
            self.phase_preparation_title = f"Préparation de {label}"
            if self.phase is not None and self.phase.key == "bubble":
                self.phase_preparation_body = "Bubble Cursor démarre. Gardez le curseur au centre ; la tâche commencera lorsque Bubble sera prêt."
            else:
                self.phase_preparation_body = "La technique d'interaction démarre. La tâche commencera automatiquement dans un instant."

    def _activate_current_trial(self):
        if self.current_trial is None or self.pause_active or self.finished:
            return
        self.preparation_cursor_lock_timer.stop()
        self._reset_system_cursor_to_trial_center()
        self._write_annotation_control_state("active")
        if self.phase is not None and self.phase.key == "standard_avec_filter":
            self._write_standard_control_state(self._center_control_state("active"))
        if self.phase is not None and self.phase.key == "ninja":
            self._write_ninja_control_state(self._center_control_state("active"))
        self.phase_preparing = False
        self.in_trial = True
        self.trial_started_at = time.monotonic()
        self.logger.write(
            {
                "type": "trial_start",
                **self._base_event(),
                "trial_index": self.current_index + 1,
                "trial_count": len(self.trials),
                "variant": self.current_trial.variant,
                "geometry": self._geometry_payload(),
            }
        )
        self.cursor_timer.start()
        if self.phase is None or self.phase.key not in {"bubble", "semantic", "ninja"}:
            self._ensure_task_window_visible()
        self.update()

    def _trial_center_global(self) -> QtCore.QPoint:
        return self.mapToGlobal(self.rect().center())

    def _reset_system_cursor_to_trial_center(self):
        center = self._trial_center_global()
        try:
            QtGui.QCursor.setPos(center)
        except Exception as exc:
            self.logger.write(
                {
                    "type": "cursor_reset_error",
                    **self._base_event(),
                    "error": str(exc),
                }
            )

    def _lock_preparation_cursor_to_center(self):
        if (
            self.phase_preparing
            and not self.in_trial
            and self.current_trial is not None
            and self.phase is not None
            and self.phase.key == "bubble"
        ):
            self._reset_system_cursor_to_trial_center()
            return
        self.preparation_cursor_lock_timer.stop()

    def _center_control_state(self, state: str) -> str:
        center = self._trial_center_global()
        token = self.current_trial.global_trial_id if self.current_trial is not None else "none"
        return f"{state} {int(center.x())} {int(center.y())} {token}"

    def _setup_trial_geometry(self):
        rect = self.rect()
        w = max(900, rect.width())
        h = max(620, rect.height())
        if self.current_trial is None:
            return

        if self.current_trial.task == "long_distance":
            size = 54
            y = 150 + (self.current_trial.variant % 5) * 70
            self.drag_item_rect = QtCore.QRectF(90, y, size, size)
        elif self.current_trial.task == "dense_interface":
            self.dense_target_index = self.current_trial.variant % 40
        _ = w, h

    def _geometry_payload(self) -> dict:
        if self.current_trial is None:
            return {}
        task = self.current_trial.task
        if task == "cursor_stability":
            return {
                "main_menu": rect_to_list(self._main_menu_rect()),
                "corridor": rect_to_list(self._corridor_rect()),
                "submenu": rect_to_list(self._submenu_rect()),
                "target": rect_to_list(self._stability_target_rect()),
            }
        if task == "long_distance":
            return {
                "start_item": rect_to_list(self.drag_item_rect),
                "drop_zone": rect_to_list(self._drop_zone_rect()),
            }
        if task == "dense_interface":
            return {
                "grid": rect_to_list(self._dense_grid_rect()),
                "target_index": self.dense_target_index,
                "target": rect_to_list(self._dense_cell_rect(self.dense_target_index)),
            }
        return {}

    def _detection_from_rect(
        self,
        rect: QtCore.QRectF,
        *,
        det_id: int,
        class_id: int = 0,
        class_name: str = "Button",
        role: str,
    ) -> dict:
        origin = self.mapToGlobal(QtCore.QPoint(0, 0))
        return {
            "id": det_id,
            "target_index": det_id - 1,
            "class_id": class_id,
            "class_name": class_name,
            "x": float(origin.x()) + float(rect.x()),
            "y": float(origin.y()) + float(rect.y()),
            "width": float(rect.width()),
            "height": float(rect.height()),
            "score": 1.0,
            "role": role,
            "source_line_number": det_id,
            "source_line": f"{role} {rect_to_list(rect)}",
        }

    def _current_detections(self) -> list[dict]:
        if self.current_trial is None:
            return []
        detections: list[dict] = []
        task = self.current_trial.task
        det_id = 1
        if task == "cursor_stability":
            for index in range(5):
                detections.append(
                    self._detection_from_rect(
                        self._menu_item_rect(index),
                        det_id=det_id,
                        role=f"menu_{index + 1}",
                    )
                )
                det_id += 1
            if self.hover_menu_open:
                submenu = self._submenu_rect()
                for index in range(5):
                    item = QtCore.QRectF(submenu.x(), submenu.y() + index * 58, submenu.width(), 58)
                    detections.append(
                        self._detection_from_rect(
                            item,
                            det_id=det_id,
                            role=f"submenu_{index + 1}",
                        )
                    )
                    det_id += 1
        elif task == "long_distance":
            detections.append(
                self._detection_from_rect(
                    self.drag_item_rect,
                    det_id=det_id,
                    role="drag_item",
                )
            )
            det_id += 1
            detections.append(
                self._detection_from_rect(
                    self._drop_zone_rect(),
                    det_id=det_id,
                    role="drop_zone",
                )
            )
        elif task == "dense_interface":
            for index in range(40):
                detections.append(
                    self._detection_from_rect(
                        self._dense_cell_rect(index),
                        det_id=det_id,
                        role=f"dense_cell_{index + 1}",
                    )
                )
                det_id += 1
        return detections

    def _write_annotation_control_state(self, state: str):
        if self.annotation_control_file is None:
            return
        detections = self._current_detections() if state == "active" else []
        start_global = self._trial_center_global()
        payload = {
            "version": 1,
            "state": state,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "phase": self.phase.key if self.phase is not None else None,
            "technique": self.phase.technique if self.phase is not None else None,
            "filter": self.phase.filter_name if self.phase is not None else None,
            "trial_id": self.current_trial.task_trial_id if self.current_trial is not None else None,
            "trial_key": (
                f"{self.session_id}:{self.phase.key if self.phase else 'none'}:"
                f"{self.current_trial.global_trial_id if self.current_trial else 'none'}"
            ),
            "start_position_global": [float(start_global.x()), float(start_global.y())],
            "detections": detections,
        }
        tmp_path = self.annotation_control_file.with_suffix(self.annotation_control_file.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.annotation_control_file)
        except Exception as exc:
            self.logger.write(
                {
                    "type": "annotation_control_error",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key if self.phase is not None else None,
                    "error": str(exc),
                }
            )

    def _write_ninja_control_state(self, state: str):
        try:
            self.ninja_control_file.write_text(state, encoding="utf-8")
            self.logger.write(
                {
                    "type": "ninja_control_state",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key if self.phase is not None else None,
                    "state": state,
                }
            )
        except Exception as exc:
            self.logger.write(
                {
                    "type": "ninja_control_error",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key if self.phase is not None else None,
                    "error": str(exc),
                }
            )

    def _write_standard_control_state(self, state: str):
        try:
            self.standard_control_file.write_text(state, encoding="utf-8")
            self.logger.write(
                {
                    "type": "standard_control_state",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key if self.phase is not None else None,
                    "state": state,
                }
            )
        except Exception as exc:
            self.logger.write(
                {
                    "type": "standard_control_error",
                    "participant_id": self.participant_id,
                    "session_id": self.session_id,
                    "phase": self.phase.key if self.phase is not None else None,
                    "error": str(exc),
                }
            )

    def _top_bar_height(self) -> float:
        return 118.0

    def _main_menu_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(80, 135, 300, 390)

    def _menu_item_rect(self, index: int) -> QtCore.QRectF:
        item_h = self._main_menu_rect().height() / 5
        return QtCore.QRectF(
            self._main_menu_rect().x(),
            self._main_menu_rect().y() + index * item_h,
            self._main_menu_rect().width(),
            item_h,
        )

    def _target_menu_rect(self) -> QtCore.QRectF:
        return self._menu_item_rect(1)

    def _corridor_rect(self) -> QtCore.QRectF:
        target = self._target_menu_rect()
        return QtCore.QRectF(target.right(), target.y(), 2, target.height())

    def _submenu_rect(self) -> QtCore.QRectF:
        target = self._target_menu_rect()
        return QtCore.QRectF(
            target.right() + 2,
            target.y() + 8,
            340,
            292,
        )

    def _stability_target_rect(self) -> QtCore.QRectF:
        submenu = self._submenu_rect()
        return QtCore.QRectF(submenu.x(), submenu.y(), submenu.width(), 58)

    def _drop_zone_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(max(620, self.width() - 270), max(180, self.height() / 2 - 95), 170, 170)

    def _dense_grid_rect(self) -> QtCore.QRectF:
        width = 8 * 58
        height = 5 * 48
        return QtCore.QRectF((self.width() - width) / 2, 170, width, height)

    def _dense_cell_rect(self, index: int) -> QtCore.QRectF:
        grid = self._dense_grid_rect()
        col = index % 8
        row = index // 8
        return QtCore.QRectF(grid.x() + col * 58 + 8, grid.y() + row * 48 + 7, 42, 34)

    def _task_region(self, pos: QtCore.QPointF) -> str:
        if self.current_trial is None:
            return "none"
        task = self.current_trial.task
        if task == "cursor_stability":
            if self._stability_target_rect().contains(pos):
                return "stability_target"
            if self._corridor_rect().contains(pos):
                return "corridor"
            if self._submenu_rect().contains(pos):
                return "submenu"
            if self._target_menu_rect().contains(pos):
                return "target_menu"
            if self._main_menu_rect().contains(pos):
                return "main_menu"
        elif task == "long_distance":
            if self._drop_zone_rect().contains(pos):
                return "drop_zone"
            if self.drag_item_rect.contains(pos):
                return "drag_item"
        elif task == "dense_interface":
            for index in range(40):
                if self._dense_cell_rect(index).contains(pos):
                    return f"dense_cell_{index}"
        return "background"

    def _allowed_stability_region(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addRect(self._target_menu_rect())
        if self.hover_menu_open:
            path.addRect(self._corridor_rect())
            path.addRect(self._submenu_rect())
        return path

    @staticmethod
    def _distance_to_rect(pos: QtCore.QPointF, rect: QtCore.QRectF) -> float:
        dx = max(float(rect.left()) - float(pos.x()), 0.0, float(pos.x()) - float(rect.right()))
        dy = max(float(rect.top()) - float(pos.y()), 0.0, float(pos.y()) - float(rect.bottom()))
        return math.hypot(dx, dy)

    def _bubble_stability_hover_position(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        candidates: list[tuple[float, str, QtCore.QRectF]] = []
        for index in range(5):
            rect = self._menu_item_rect(index)
            role = "target_menu" if index == 1 else f"menu_{index + 1}"
            candidates.append((self._distance_to_rect(pos, rect), role, rect))
        if self.hover_menu_open:
            submenu = self._submenu_rect()
            for index in range(5):
                rect = QtCore.QRectF(submenu.x(), submenu.y() + index * 58, submenu.width(), 58)
                role = "stability_target" if index == 0 else f"submenu_{index + 1}"
                candidates.append((self._distance_to_rect(pos, rect), role, rect))
        if not candidates:
            return pos
        _, role, rect = min(candidates, key=lambda item: item[0])
        if role in {"target_menu", "stability_target"}:
            return rect.center()
        return pos

    def _stability_hover_position(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        if self.phase is not None and self.phase.key == "bubble":
            return self._bubble_stability_hover_position(pos)
        return pos

    def _write_cursor_sample(self):
        if not self.in_trial or self.current_trial is None:
            return
        global_pos = QtGui.QCursor.pos()
        widget_pos = self.mapFromGlobal(global_pos)
        pos_f = QtCore.QPointF(widget_pos)
        if self.current_trial.task == "cursor_stability":
            self._handle_stability_move(pos_f)
        self.logger.write(
            {
                "type": "cursor_sample",
                **self._base_event(),
                "screen_position": point_to_list(global_pos),
                "widget_position": point_to_list(pos_f),
                "region": self._task_region(pos_f),
            }
        )

    def _log_region_change(self, pos: QtCore.QPointF):
        region = self._task_region(pos)
        if region == self.last_region:
            return
        previous = self.last_region
        self.last_region = region
        self.logger.write(
            {
                "type": "region_change",
                **self._base_event(),
                "from_region": previous,
                "to_region": region,
                "widget_position": point_to_list(pos),
            }
        )

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if not self.in_trial or self.current_trial is None:
            return
        pos = event.position()
        self._log_region_change(pos)
        if self.current_trial.task == "cursor_stability":
            self._handle_stability_move(pos)
        elif self.current_trial.task == "long_distance" and self.dragging:
            self.drag_item_rect.moveTopLeft(pos - self.drag_offset)
            self.update()
        self._write_annotation_control_state("active")

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.pause_active or self.finished:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.in_trial or self.current_trial is None:
            return
        pos = event.position()
        self.click_count += 1
        self.logger.write(
            {
                "type": "mouse_down",
                **self._base_event(),
                "button": "left",
                "widget_position": point_to_list(pos),
                "region": self._task_region(pos),
            }
        )
        if self.current_trial.task == "long_distance" and self.drag_item_rect.contains(pos):
            self.dragging = True
            self.drag_offset = pos - self.drag_item_rect.topLeft()
            self.logger.write({"type": "drag_start", **self._base_event(), "widget_position": point_to_list(pos)})

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.pause_active:
            if self.pause_button_rect.contains(event.position()):
                self._start_next_phase()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.finished:
            if self.finish_button_rect.contains(event.position()):
                QtWidgets.QApplication.instance().quit()
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.in_trial or self.current_trial is None:
            return
        pos = event.position()
        region = self._task_region(pos)
        self.logger.write(
            {
                "type": "mouse_up",
                **self._base_event(),
                "button": "left",
                "widget_position": point_to_list(pos),
                "region": region,
            }
        )

        if self.current_trial.task == "cursor_stability":
            success = self._stability_target_rect().contains(pos)
            self._log_click(pos, success=success, target="stability_target")
            if success:
                self._complete_current_trial("target_clicked")
            else:
                self._log_error("miss_click", pos)
        elif self.current_trial.task == "long_distance":
            if self.dragging:
                drop_zone = self._drop_zone_rect()
                overlap = drop_zone.intersected(self.drag_item_rect)
                item_area = max(1.0, self.drag_item_rect.width() * self.drag_item_rect.height())
                overlap_ratio = (overlap.width() * overlap.height()) / item_area
                center_inside = drop_zone.contains(self.drag_item_rect.center())
                release_inside = drop_zone.contains(pos)
                success = center_inside or release_inside or overlap_ratio >= 0.5
                self.logger.write(
                    {
                        "type": "drag_end",
                        **self._base_event(),
                        "widget_position": point_to_list(pos),
                        "item_rect": rect_to_list(self.drag_item_rect),
                        "item_center": point_to_list(self.drag_item_rect.center()),
                        "drop_zone": rect_to_list(drop_zone),
                        "release_inside_drop_zone": release_inside,
                        "item_center_inside_drop_zone": center_inside,
                        "overlap_ratio": round(float(overlap_ratio), 4),
                        "success": success,
                    }
                )
                self.dragging = False
                if success:
                    self._complete_current_trial("dropped_in_zone")
                else:
                    self._log_error("drop_outside_target", pos)
            else:
                self._log_click(pos, success=False, target="drag_item")
                self._log_error("click_without_drag", pos)
        elif self.current_trial.task == "dense_interface":
            hit_index = self._dense_hit_index(pos)
            success = hit_index == self.dense_target_index
            self._log_click(pos, success=success, target=f"dense_cell_{self.dense_target_index}", hit=f"dense_cell_{hit_index}" if hit_index is not None else None)
            if success:
                self._complete_current_trial("target_clicked")
            else:
                self._log_error("dense_miss", pos, hit_index=hit_index)

    def _handle_stability_move(self, pos: QtCore.QPointF):
        pos = self._stability_hover_position(pos)
        in_target_menu = self._target_menu_rect().contains(pos)
        in_main_menu = self._main_menu_rect().contains(pos)
        in_corridor = self._corridor_rect().contains(pos)
        in_submenu = self._submenu_rect().contains(pos)

        if in_target_menu:
            self.hover_menu_open = True
        elif in_main_menu:
            self.hover_menu_open = False
            self.entered_path = False
        elif not in_corridor and not in_submenu:
            if self.hover_menu_open and self.entered_path:
                self._log_error("left_narrow_path", pos)
            self.hover_menu_open = False
            self.entered_path = False

        if in_corridor:
            self.entered_path = True
        if self.hover_menu_open and self.entered_path and not self._allowed_stability_region().contains(pos):
            self._log_error("left_narrow_path", pos)
            self.entered_path = False
        self._write_annotation_control_state("active")
        self.update()

    def _dense_hit_index(self, pos: QtCore.QPointF) -> int | None:
        for index in range(40):
            if self._dense_cell_rect(index).contains(pos):
                return index
        return None

    def _log_click(self, pos: QtCore.QPointF, *, success: bool, target: str, hit: str | None = None):
        self.logger.write(
            {
                "type": "click",
                **self._base_event(),
                "click_index": self.click_count,
                "widget_position": point_to_list(pos),
                "target": target,
                "hit": hit,
                "success": success,
            }
        )

    def _log_error(self, reason: str, pos: QtCore.QPointF, **fields):
        self.error_count += 1
        self._show_feedback(
            "Failure - try again" if is_english(self.language) else "Echec - reessayez",
            success=False,
            duration_ms=900,
        )
        self.logger.write(
            {
                "type": "error",
                **self._base_event(),
                "reason": reason,
                "error_count": self.error_count,
                "widget_position": point_to_list(pos),
                **fields,
            }
        )

    def _complete_current_trial(self, reason: str):
        self.completed = True
        self._end_trial(success=True, reason=reason)
        self._show_feedback(
            "Success" if is_english(self.language) else "Reussi",
            success=True,
            duration_ms=900,
        )
        self.update()
        QtCore.QTimer.singleShot(950, self.next_trial)

    def _show_feedback(self, text: str, *, success: bool, duration_ms: int):
        self.feedback_text = text
        self.feedback_success = bool(success)
        self.feedback_until = time.monotonic() + max(0.1, duration_ms / 1000.0)
        self.update()
        QtCore.QTimer.singleShot(duration_ms, self.update)

    def _end_trial(self, *, success: bool, reason: str):
        if not self.in_trial or self.current_trial is None:
            return
        self.in_trial = False
        self.cursor_timer.stop()
        self.logger.write(
            {
                "type": "trial_end",
                **self._base_event(),
                "success": success,
                "reason": reason,
                "duration_sec": round(time.monotonic() - self.trial_started_at, 6),
                "click_count": self.click_count,
                "error_count": self.error_count,
            }
        )

    def _end_session(self, reason: str):
        self.cursor_timer.stop()
        self.keep_front_timer.stop()
        self.preparation_cursor_lock_timer.stop()
        self._write_annotation_control_state("paused")
        self._write_ninja_control_state("paused")
        self._write_standard_control_state("paused")
        self._stop_phase_process()
        self._cleanup_annotation_control_file()
        self.logger.write(
            {
                "type": "session_end",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "reason": reason,
                "total_duration_sec": round(self.logger.elapsed(), 6),
            }
        )
        self.logger.close()

    def _cleanup_annotation_control_file(self):
        try:
            self.annotation_control_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self.ninja_control_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self.standard_control_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _finish_session(self):
        self.finished = True
        self.pause_active = False
        self.phase_preparing = False
        self.current_trial = None
        self.in_trial = False
        self._end_session("completed")
        self.update()
        QtCore.QTimer.singleShot(1800, QtWidgets.QApplication.instance().quit)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            if self.in_trial:
                self._end_trial(success=False, reason="user_abort")
            self._end_session("user_abort")
            QtWidgets.QApplication.instance().quit()
            return
        if self.phase_preparing:
            return
        if event.key() in {QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}:
            if self.pause_active:
                self._start_next_phase()
                return
            if self.finished:
                QtWidgets.QApplication.instance().quit()
                return
            if self.in_trial:
                self._end_trial(success=False, reason="skipped")
                self.completed = False
                self._show_feedback(
                    "Failure" if is_english(self.language) else "Echec",
                    success=False,
                    duration_ms=900,
                )
                QtCore.QTimer.singleShot(950, self.next_trial)
            else:
                self.next_trial()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.pause_active:
            self._draw_pause_screen(painter)
            painter.end()
            return
        if self.finished:
            self._draw_finish_screen(painter)
            painter.end()
            return
        painter.fillRect(self.rect(), QtGui.QColor("#f7f7f4"))
        self._draw_header(painter)
        if self.current_trial is None:
            painter.end()
            return
        if self.current_trial.task == "cursor_stability":
            self._draw_stability_task(painter)
        elif self.current_trial.task == "long_distance":
            self._draw_distance_task(painter)
        elif self.current_trial.task == "dense_interface":
            self._draw_dense_task(painter)
        if self.phase_preparing:
            self._draw_preparation_overlay(painter)
        self._draw_feedback(painter)
        painter.end()

    def _draw_pause_screen(self, painter: QtGui.QPainter):
        painter.fillRect(self.rect(), QtGui.QColor("#050505"))
        title_font = QtGui.QFont()
        title_font.setPointSize(44)
        title_font.setBold(True)
        body_font = QtGui.QFont()
        body_font.setPointSize(24)
        hint_font = QtGui.QFont()
        hint_font.setPointSize(18)
        button_font = QtGui.QFont()
        button_font.setPointSize(25)
        button_font.setBold(True)

        if is_english(self.language):
            title = "Pause"
            previous = f"Previous phase completed: {self.previous_phase_label}."
            next_text = f"Next phase: {self.next_phase_label}."
            hint = "When the pause is over, click the button to continue."
            button = "Continue"
            ninja_extra = (
                "Ninja Cursors starts with eye-tracking calibration: look at each red point until it changes. "
                "Then 8 cursors appear around the screen center at the start of every trial. "
                "Look at the cursor you want to use; when it turns orange, click normally. "
                "For drag tasks, keep looking at the same active cursor while holding the mouse button."
            )
        else:
            title = "Pause"
            previous = f"Phase précédente terminée : {self.previous_phase_label}."
            next_text = f"Phase suivante : {self.next_phase_label}."
            hint = "Quand la pause est terminée, cliquez sur le bouton pour continuer."
            button = "Continuer"
            ninja_extra = (
                "Ninja Cursors commence par une calibration du regard : regardez chaque point rouge jusqu'au changement. "
                "Ensuite, 8 curseurs apparaissent autour du centre de l'écran au début de chaque essai. "
                "Regardez le curseur que vous voulez utiliser ; quand il devient orange, cliquez normalement. "
                "Pour glisser-déposer, gardez le regard sur le même curseur actif pendant que le bouton est maintenu."
            )
        extra = ""
        if self.phase_index + 1 < len(self.phases) and self.phases[self.phase_index + 1].key == "ninja":
            extra = "\n\n" + ninja_extra
        body_height = 310 if extra else 150
        hint_y = self.height() * (0.64 if extra else 0.51)
        button_y = self.height() * (0.73 if extra else 0.62)

        painter.setPen(QtGui.QColor("white"))
        painter.setFont(title_font)
        painter.drawText(QtCore.QRectF(60, self.height() * 0.20, self.width() - 120, 80), QtCore.Qt.AlignmentFlag.AlignCenter, title)
        painter.setFont(body_font)
        painter.drawText(
            QtCore.QRectF(80, self.height() * 0.34, self.width() - 160, body_height),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            previous + "\n" + next_text + extra,
        )
        painter.setPen(QtGui.QColor("#b8b8b8"))
        painter.setFont(hint_font)
        painter.drawText(
            QtCore.QRectF(80, hint_y, self.width() - 160, 60),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            hint,
        )
        width = min(620.0, self.width() * 0.55)
        self.pause_button_rect = QtCore.QRectF((self.width() - width) / 2, button_y, width, 92)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("white"))
        painter.drawRoundedRect(self.pause_button_rect, 24, 24)
        painter.setPen(QtGui.QColor("#111111"))
        painter.setFont(button_font)
        painter.drawText(self.pause_button_rect, QtCore.Qt.AlignmentFlag.AlignCenter, button)

    def _draw_preparation_overlay(self, painter: QtGui.QPainter):
        painter.save()
        painter.fillRect(self.rect(), QtGui.QColor(5, 5, 5, 170))
        title_font = QtGui.QFont()
        title_font.setPointSize(34)
        title_font.setBold(True)
        body_font = QtGui.QFont()
        body_font.setPointSize(22)
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(title_font)
        painter.drawText(
            QtCore.QRectF(70, self.height() * 0.34, self.width() - 140, 70),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self.phase_preparation_title,
        )
        painter.setPen(QtGui.QColor("#e5e7eb"))
        painter.setFont(body_font)
        painter.drawText(
            QtCore.QRectF(110, self.height() * 0.46, self.width() - 220, 150),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            self.phase_preparation_body,
        )
        painter.restore()

    def _draw_finish_screen(self, painter: QtGui.QPainter):
        painter.fillRect(self.rect(), QtGui.QColor("#050505"))
        title_font = QtGui.QFont()
        title_font.setPointSize(42)
        title_font.setBold(True)
        body_font = QtGui.QFont()
        body_font.setPointSize(24)
        if is_english(self.language):
            title = "Qualitative tasks completed"
            body = "The full qualitative sequence is finished. The window will close automatically."
        else:
            title = "Tâches qualitatives terminées"
            body = "La séquence qualitative complète est terminée. La fenêtre va se fermer automatiquement."
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(title_font)
        painter.drawText(QtCore.QRectF(60, self.height() * 0.34, self.width() - 120, 80), QtCore.Qt.AlignmentFlag.AlignCenter, title)
        painter.setPen(QtGui.QColor("#d7dde8"))
        painter.setFont(body_font)
        painter.drawText(
            QtCore.QRectF(80, self.height() * 0.48, self.width() - 160, 100),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            body,
        )

    def _draw_feedback(self, painter: QtGui.QPainter):
        if not self.feedback_text or time.monotonic() >= self.feedback_until:
            return
        painter.save()
        box_width = min(520.0, max(300.0, self.width() * 0.42))
        box = QtCore.QRectF(
            (self.width() - box_width) / 2.0,
            self.height() / 2.0 - 48.0,
            box_width,
            96.0,
        )
        fill = QtGui.QColor(17, 122, 72, 220) if self.feedback_success else QtGui.QColor(180, 35, 24, 220)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 22, 22)
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(box, QtCore.Qt.AlignmentFlag.AlignCenter, self.feedback_text)
        painter.restore()

    def _draw_header(self, painter: QtGui.QPainter):
        painter.fillRect(QtCore.QRectF(0, 0, self.width(), self._top_bar_height()), QtGui.QColor("#202124"))
        title_font = QtGui.QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        instruction_font = QtGui.QFont()
        instruction_font.setPointSize(17)
        instruction_font.setBold(True)
        if self.current_trial is None:
            text = "Qualitative tasks"
            instruction = ""
        else:
            label = self._labels().get(self.current_trial.task, self.current_trial.task)
            errors_label = "Errors" if is_english(self.language) else "Erreurs"
            text = (
                f"{self._phase_label()} | {label}  "
                f"{self.current_trial.task_trial_id}/{self.trials_per_task}    "
                f"{errors_label}: {self.error_count}"
            )
            instruction = self._instructions().get(self.current_trial.task, "")
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(title_font)
        painter.drawText(QtCore.QRectF(24, 12, self.width() - 48, 34), QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        painter.setPen(QtGui.QColor("#d7dde8"))
        painter.setFont(instruction_font)
        painter.drawText(
            QtCore.QRectF(24, 54, self.width() - 48, 50),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.TextFlag.TextWordWrap,
            instruction,
        )

    def _draw_stability_task(self, painter: QtGui.QPainter):
        painter.setPen(QtGui.QPen(QtGui.QColor("#333333"), 2))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawRoundedRect(self._main_menu_rect(), 4, 4)
        for i in range(5):
            item = self._menu_item_rect(i)
            if i == 1 and self.hover_menu_open:
                painter.fillRect(item, QtGui.QColor("#eeeeee"))
            painter.drawLine(item.bottomLeft(), item.bottomRight())
            painter.drawText(item.adjusted(22, 0, -18, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, f"Menu {i + 1}")
            if i == 1:
                arrow = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(item.right() - 26, item.center().y() - 7),
                        QtCore.QPointF(item.right() - 26, item.center().y() + 7),
                        QtCore.QPointF(item.right() - 14, item.center().y()),
                    ]
                )
                painter.setBrush(QtGui.QColor("#333333"))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(arrow)
                painter.setPen(QtGui.QPen(QtGui.QColor("#333333"), 2))

        if self.hover_menu_open:
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawRoundedRect(self._submenu_rect(), 4, 4)
            for i in range(5):
                item = QtCore.QRectF(self._submenu_rect().x(), self._submenu_rect().y() + i * 58, self._submenu_rect().width(), 58)
                if i == 0:
                    painter.fillRect(item, QtGui.QColor("#eeeeee"))
                painter.drawLine(item.bottomLeft(), item.bottomRight())
                painter.drawText(item.adjusted(24, 0, -18, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, f"Submenu {i + 1}")

    def _draw_distance_task(self, painter: QtGui.QPainter):
        drop_zone = self._drop_zone_rect()
        item_in_zone = drop_zone.intersects(self.drag_item_rect)

        painter.setBrush(QtGui.QColor("#dcfce7" if not item_in_zone else "#bbf7d0"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#15803d"), 3, QtCore.Qt.PenStyle.DashLine))
        painter.drawRoundedRect(drop_zone, 10, 10)

        painter.save()
        if item_in_zone:
            painter.setOpacity(0.58)
        painter.setPen(QtGui.QPen(QtGui.QColor("#1f2937"), 2))
        painter.setBrush(QtGui.QColor("#dbeafe"))
        painter.drawRoundedRect(self.drag_item_rect, 8, 8)
        painter.setOpacity(1.0)
        painter.setPen(QtGui.QColor("#111827"))
        painter.drawText(self.drag_item_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "A")
        painter.restore()

    def _draw_dense_task(self, painter: QtGui.QPainter):
        for index in range(40):
            rect = self._dense_cell_rect(index)
            is_target = index == self.dense_target_index
            painter.setBrush(QtGui.QColor("#f8d7da" if is_target else "#ffffff"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#b42318" if is_target else "#6b7280"), 3 if is_target else 1))
            painter.drawRoundedRect(rect, 4, 4)
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(index + 1))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the qualitative task sequence.")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--trials-per-task", type=int, default=DEFAULT_TRIALS_PER_TASK)
    parser.add_argument("--cursor-log-hz", type=float, default=DEFAULT_CURSOR_HZ)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--no-log", action="store_true", help="Run without writing qualitative JSONL logs")
    parser.add_argument("--language", default="French")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--windowed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    log_file = None if args.no_log else (
        Path(args.log_file).expanduser().resolve() if args.log_file else default_log_path(args.participant)
    )
    window = QualitativeBaselineWindow(
        participant_id=args.participant,
        trials_per_task=max(1, args.trials_per_task),
        log_file=log_file,
        cursor_hz=args.cursor_log_hz,
        language=args.language,
        seed=args.seed,
        fullscreen=not args.windowed,
    )
    if args.windowed:
        window.resize(1100, 740)
        window.show()
    else:
        window.show_desktop_fullscreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
