"""Qualitative baseline tasks using the standard mouse only.

This module is intentionally separate from the controlled Phase II experiment.
It provides ecological baseline tasks for Phase I observation and writes logs to
qualitative_logs/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets


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
    "long_distance": "Faites glisser le carre bleu jusque dans la zone verte, puis relachez-le dedans.",
    "dense_interface": "Cliquez l'element rouge surligne dans la grille dense.",
}


def is_english(language: str | None) -> bool:
    return str(language or "").strip().lower().startswith("en")


def default_log_path(participant_id: str) -> Path:
    logs_dir = PROJECT_ROOT / "qualitative_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in participant_id).strip("_") or "participant"
    return logs_dir / f"{safe_id}_{stamp}_normal_mouse_baseline.jsonl"


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
    ):
        super().__init__()
        self.participant_id = participant_id
        self.trials_per_task = int(trials_per_task)
        self.language = language
        self.logger = JsonlLogger(log_file)
        self.session_id = log_file.stem if log_file is not None else f"{participant_id}_qualitative_no_log"
        self.rng = random.Random(seed)
        self.trials = self._make_trials()
        self.current_index = -1
        self.current_trial: Trial | None = None
        self.trial_started_at = 0.0
        self.in_trial = False
        self.completed = False
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

        self.cursor_timer = QtCore.QTimer(self)
        self.cursor_timer.setInterval(max(1, int(round(1000.0 / max(float(cursor_hz), 1.0)))))
        self.cursor_timer.timeout.connect(self._write_cursor_sample)

        self.setMouseTracking(True)
        self.setWindowTitle("Normal Mouse Baseline")
        self.setMinimumSize(900, 620)
        self.logger.write(
            {
                "type": "session_start",
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "task_order": list(TASK_ORDER),
                "trials_per_task": self.trials_per_task,
                "total_trials": len(self.trials),
                "technique": "normal_mouse",
                "log_file": str(log_file) if log_file is not None else None,
                "logging_enabled": log_file is not None,
            }
        )
        QtCore.QTimer.singleShot(250, self.next_trial)

    def _labels(self) -> dict[str, str]:
        return TASK_LABELS_EN if is_english(self.language) else TASK_LABELS_FR

    def _instructions(self) -> dict[str, str]:
        return TASK_INSTRUCTIONS_EN if is_english(self.language) else TASK_INSTRUCTIONS_FR

    def _make_trials(self) -> list[Trial]:
        trials: list[Trial] = []
        global_id = 1
        for task in TASK_ORDER:
            for task_trial_id in range(1, self.trials_per_task + 1):
                trials.append(Trial(global_id, task, task_trial_id, self.rng.randint(0, 10_000)))
                global_id += 1
        return trials

    def _base_event(self) -> dict:
        task = self.current_trial.task if self.current_trial else None
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "technique": "normal_mouse",
            "global_trial_id": self.current_trial.global_trial_id if self.current_trial else None,
            "task": task,
            "task_label": self._labels().get(task, task) if task else None,
            "task_trial_id": self.current_trial.task_trial_id if self.current_trial else None,
        }

    def next_trial(self):
        if self.current_trial is not None:
            self._end_trial(success=self.completed, reason="completed" if self.completed else "advanced")
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self._end_session("completed")
            QtCore.QTimer.singleShot(800, QtWidgets.QApplication.instance().quit)
            return

        self.current_trial = self.trials[self.current_index]
        self.in_trial = True
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
        self.trial_started_at = time.monotonic()
        self._setup_trial_geometry()
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
        self.update()

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

    def _write_cursor_sample(self):
        if not self.in_trial or self.current_trial is None:
            return
        global_pos = QtGui.QCursor.pos()
        widget_pos = self.mapFromGlobal(global_pos)
        pos_f = QtCore.QPointF(widget_pos)
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

    def mousePressEvent(self, event: QtGui.QMouseEvent):
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
                success = self._drop_zone_rect().intersects(self.drag_item_rect) and self._drop_zone_rect().contains(self.drag_item_rect.center())
                self.logger.write(
                    {
                        "type": "drag_end",
                        **self._base_event(),
                        "widget_position": point_to_list(pos),
                        "item_rect": rect_to_list(self.drag_item_rect),
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

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            if self.in_trial:
                self._end_trial(success=False, reason="user_abort")
            self._end_session("user_abort")
            QtWidgets.QApplication.instance().quit()
            return
        if event.key() in {QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}:
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
        self._draw_feedback(painter)
        painter.end()

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
            text = "Normal Mouse Baseline"
            instruction = ""
        else:
            label = self._labels().get(self.current_trial.task, self.current_trial.task)
            text = f"{label}  {self.current_trial.task_trial_id}/{self.trials_per_task}    Errors: {self.error_count}"
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
    parser = argparse.ArgumentParser(description="Run qualitative normal-mouse baseline tasks.")
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
    )
    if args.windowed:
        window.resize(1100, 740)
        window.show()
    else:
        window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
