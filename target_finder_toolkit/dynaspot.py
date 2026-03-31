"""
DynaSpot Demo
=============

This module demonstrates the **DynaSpot** interaction technique
using the TargetFinder toolkit.

Notes
-----
- This script is for demonstration purposes only.
- It is **not part of the core TargetFinder API**, but shows how the
  toolkit can support another pointing technique.
"""

import os
import sys
import time
import signal
import math
import pyautogui
from PyQt6 import QtWidgets, QtGui, QtCore
from pynput import keyboard, mouse
import argparse

from target_finder_toolkit.targetfinder import TargetFinder
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors

__all__ = ["dynaspot", "main"]


class DynaSpot(QtWidgets.QWidget):
    """
    PyQt overlay widget implementing a DynaSpot-like speed-dependent area cursor.

    The cursor behaves like a point cursor at low speed and gradually grows
    a circular activation area when the pointer moves faster.
    """

    MIN_SPEED = 120.0
    MAX_SPEED = 1600.0
    MIN_RADIUS = 0.5
    MAX_RADIUS = 28.0
    LAG = 0.08
    REDUCE_TIME = 0.20
    GROWTH_SMOOTHING = 0.35
    SHRINK_SMOOTHING = 0.18
    TEXT_MARGIN = 20

    def __init__(self, detector: TargetFinder):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        if sys.platform == "darwin":
            self.detector.hide_overlay_during_capture = False

        self._mouse_listener = None
        self._keyboard_listener = None
        self._cursor_refresh_timer = None
        self._last_rehide_at = 0.0

        self.enabled = True
        self._simulating_click = False
        self._last_target = None
        self._spot_radius = self.MIN_RADIUS
        self._last_cursor_pos = QtCore.QPointF(QtGui.QCursor.pos())
        now = time.monotonic()
        self._last_sample_at = now
        self._last_motion_at = now
        self._shrink_start_at = None
        self._shrink_start_radius = 0.5

        self._start_mouse_listener()
        self._start_keyboard_listener()

        geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(geom)

        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
        if sys.platform != "darwin":
            flags |= QtCore.Qt.WindowType.Tool
        if sys.platform.startswith("linux"):
            flags |= QtCore.Qt.WindowType.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.detector.start()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10)

        if sys.platform == "darwin":
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(self._rehide_cursor)
            self._cursor_refresh_timer.start(30)

    def _screen_for_point(self, x: int, y: int):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return QtWidgets.QApplication.primaryScreen()
        return app.screenAt(QtCore.QPoint(int(x), int(y))) or QtWidgets.QApplication.primaryScreen()

    def _in_system_reserved_area(self, x: int, y: int) -> bool:
        if sys.platform != "darwin":
            return False
        screen = self._screen_for_point(x, y)
        if screen is None:
            return False
        point = QtCore.QPoint(int(x), int(y))
        return not screen.availableGeometry().contains(point)

    def _update_spot_radius(self, cursor_pos: QtCore.QPointF):
        now = time.monotonic()
        dt = max(now - self._last_sample_at, 1e-3)
        dx = cursor_pos.x() - self._last_cursor_pos.x()
        dy = cursor_pos.y() - self._last_cursor_pos.y()
        dist = math.hypot(dx, dy)
        speed = dist / dt

        moved = dist > 0.5
        if moved:
            self._last_motion_at = now
            self._shrink_start_at = None
            if speed <= self.MIN_SPEED:
                target_radius = self.MIN_RADIUS
            else:
                speed_ratio = min(1.0, (speed - self.MIN_SPEED) / max(1.0, self.MAX_SPEED - self.MIN_SPEED))
                eased_ratio = 1.0 - (1.0 - speed_ratio) ** 2
                target_radius = self.MIN_RADIUS + eased_ratio * (self.MAX_RADIUS - self.MIN_RADIUS)
            self._spot_radius += (target_radius - self._spot_radius) * self.GROWTH_SMOOTHING
        else:
            idle_for = now - self._last_motion_at
            if idle_for >= self.LAG:
                if self._shrink_start_at is None:
                    self._shrink_start_at = now
                    self._shrink_start_radius = self._spot_radius
                progress = min(1.0, (now - self._shrink_start_at) / self.REDUCE_TIME)
                target_radius = self.MIN_RADIUS + (self._shrink_start_radius - self.MIN_RADIUS) * (1.0 - progress)
                self._spot_radius += (target_radius - self._spot_radius) * self.SHRINK_SMOOTHING

        self._last_sample_at = now
        self._last_cursor_pos = QtCore.QPointF(cursor_pos)

    def _select_target(self, cx: float, cy: float, detections):
        cursor_on_non_text = False
        for x, y, w, h, score, cls_id in detections:
            if cls_id != 3 and x <= cx <= x + w and y <= cy <= y + h:
                cursor_on_non_text = True
                break
        if not cursor_on_non_text:
            for x, y, w, h, score, cls_id in detections:
                if cls_id == 3:
                    if ((x - self.TEXT_MARGIN) <= cx <= (x + w + self.TEXT_MARGIN)
                            and (y - self.TEXT_MARGIN) <= cy <= (y + h + self.TEXT_MARGIN)):
                        return None

        candidates = []
        for x, y, w, h, score, cls_id in detections:
            if cls_id == 3:
                continue
            cx_box = x + w / 2
            cy_box = y + h / 2
            dx = max(0.0, abs(cx - cx_box) - w / 2)
            dy = max(0.0, abs(cy - cy_box) - h / 2)
            int_d = math.hypot(dx, dy)
            if int_d <= self._spot_radius:
                center_d = math.hypot(cx - cx_box, cy - cy_box)
                candidates.append((center_d, int_d, cx_box, cy_box, w, h, cls_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, tx, ty, w, h, cls_id = candidates[0]
        return (tx, ty, w, h, cls_id)

    def paintEvent(self, event):
        detections = self.detector.get_detections()
        if not self.enabled:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pos = QtGui.QCursor.pos()
        cx, cy = pos.x(), pos.y()

        self._update_spot_radius(QtCore.QPointF(pos))
        target = self._select_target(cx, cy, detections) if detections else None

        if target is None:
            self._last_target = None
        else:
            tx, ty, w, h, _cls_id = target
            self._last_target = (
                tx / self.detector.sx,
                ty / self.detector.sy,
                w / self.detector.sx,
                h / self.detector.sy,
            )
        if self._spot_radius > self.MIN_RADIUS + 0.5:
            radius_ratio = min(1.0, (self._spot_radius - self.MIN_RADIUS) / max(1.0, self.MAX_RADIUS - self.MIN_RADIUS))
            ring_alpha = int(135 + 75 * radius_ratio)
            fill_alpha = int(10 + 26 * radius_ratio)
            pen = QtGui.QPen(QtGui.QColor(70, 255, 120, ring_alpha), 3)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(70, 255, 120, fill_alpha))
            painter.drawEllipse(QtCore.QPointF(cx, cy), self._spot_radius, self._spot_radius)

            inner_radius = max(3.0, self._spot_radius * 0.42)
            inner_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, int(45 + 55 * radius_ratio)), 1)
            painter.setPen(inner_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), inner_radius, inner_radius)

        self._draw_fake_cursor(painter, cx, cy)
        painter.end()

    def _draw_fake_cursor(self, painter, cx, cy):
        active = self._spot_radius > self.MIN_RADIUS + 0.5
        pen = QtGui.QPen(
            QtGui.QColor(255, 255, 255, 240 if active else 220),
            2,
        )
        painter.setPen(pen)
        fill = QtGui.QColor(255, 255, 255, 22 if active else 0)
        painter.setBrush(fill if active else QtCore.Qt.BrushStyle.NoBrush)
        radius_cursor = 6
        center = QtCore.QPointF(cx, cy)
        painter.drawEllipse(center, radius_cursor, radius_cursor)
        line_len = 4
        gap = radius_cursor + 1
        painter.drawLine(cx, cy - gap - line_len, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + gap + line_len)
        painter.drawLine(cx + gap, cy, cx + gap + line_len, cy)
        painter.drawLine(cx - gap - line_len, cy, cx - gap, cy)

    @QtCore.pyqtSlot()
    def toggle_dynaspot(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.detector.start()
            hide_cursor_everywhere()
        else:
            self.detector.stop()
            restore_default_cursors()
            self._last_target = None
            self._spot_radius = self.MIN_RADIUS
        self.update()

    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        restore_default_cursors()
        os._exit(0)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'd':
                    QtCore.QMetaObject.invokeMethod(self, "toggle_dynaspot", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key.char == 'q':
                    QtCore.QMetaObject.invokeMethod(self, "stop_and_quit", QtCore.Qt.ConnectionType.QueuedConnection)
            except AttributeError:
                pass

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    @QtCore.pyqtSlot()
    def _rehide_cursor(self):
        if not self.enabled:
            return
        hide_cursor_everywhere()
        if sys.platform == "darwin":
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(75, hide_cursor_everywhere)

    def _start_mouse_listener(self):
        def queue_rehide(force=False):
            if sys.platform != "darwin":
                return
            now = time.monotonic()
            if not force and now - self._last_rehide_at < 0.008:
                return
            self._last_rehide_at = now
            QtCore.QMetaObject.invokeMethod(
                self,
                "_rehide_cursor",
                QtCore.Qt.ConnectionType.QueuedConnection,
            )

        def on_move(x, y):
            queue_rehide()

        def on_click(x, y, button, pressed):
            if button == button.left:
                queue_rehide(force=True)
            if pressed and button == button.left:
                if self._in_system_reserved_area(x, y):
                    return
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_simulate_click",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(int, x),
                    QtCore.Q_ARG(int, y),
                )

        self._mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self._mouse_listener.start()

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        if self._in_system_reserved_area(orig_x, orig_y):
            return
        if not (self._last_target and self.enabled):
            return

        tx, ty, w, h = self._last_target
        if tx - w / 2 <= orig_x <= tx + w / 2 and ty - h / 2 <= orig_y <= ty + h / 2:
            return

        self._mouse_listener.stop()
        try:
            pyautogui.mouseUp(button='left')
            pyautogui.moveTo(tx, ty)
            pyautogui.click()
            pyautogui.moveTo(orig_x, orig_y)
        except pyautogui.FailSafeException:
            pass
        finally:
            self._rehide_cursor()
            self._start_mouse_listener()


def dynaspot(detector: TargetFinder):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hide_cursor_everywhere()
    ov = DynaSpot(detector)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description="Launch the DynaSpot overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    dynaspot(det)


if __name__ == "__main__":
    main()
