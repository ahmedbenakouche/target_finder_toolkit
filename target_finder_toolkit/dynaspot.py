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
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D
from target_finder_toolkit.logging_utils import SessionLogger

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

    def __init__(
        self,
        detector: TargetFinder,
        cursor_filter=None,
        logger=None,
        *,
        min_speed: float | None = None,
        max_speed: float | None = None,
        min_radius: float | None = None,
        max_radius: float | None = None,
        lag: float | None = None,
        reduce_time: float | None = None,
        growth_smoothing: float | None = None,
        shrink_smoothing: float | None = None,
    ):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        if sys.platform == "darwin":
            self.detector.hide_overlay_during_capture = False
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.min_speed = float(self.MIN_SPEED if min_speed is None else min_speed)
        self.max_speed = float(self.MAX_SPEED if max_speed is None else max_speed)
        self.min_radius = float(self.MIN_RADIUS if min_radius is None else min_radius)
        self.max_radius = float(self.MAX_RADIUS if max_radius is None else max_radius)
        self.lag = float(self.LAG if lag is None else lag)
        self.reduce_time = float(self.REDUCE_TIME if reduce_time is None else reduce_time)
        self.growth_smoothing = float(self.GROWTH_SMOOTHING if growth_smoothing is None else growth_smoothing)
        self.shrink_smoothing = float(self.SHRINK_SMOOTHING if shrink_smoothing is None else shrink_smoothing)

        self._mouse_listener = None
        self._keyboard_listener = None
        self._cursor_refresh_timer = None
        self._last_rehide_at = 0.0

        self.enabled = True
        self._simulating_click = False
        self._last_target = None
        self._last_target_info = None
        self._spot_radius = self.min_radius
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
            if speed <= self.min_speed:
                target_radius = self.min_radius
            else:
                speed_ratio = min(1.0, (speed - self.min_speed) / max(1.0, self.max_speed - self.min_speed))
                eased_ratio = 1.0 - (1.0 - speed_ratio) ** 2
                target_radius = self.min_radius + eased_ratio * (self.max_radius - self.min_radius)
            self._spot_radius += (target_radius - self._spot_radius) * self.growth_smoothing
        else:
            idle_for = now - self._last_motion_at
            if idle_for >= self.lag:
                if self._shrink_start_at is None:
                    self._shrink_start_at = now
                    self._shrink_start_radius = self._spot_radius
                progress = min(1.0, (now - self._shrink_start_at) / max(self.reduce_time, 1e-6))
                target_radius = self.min_radius + (self._shrink_start_radius - self.min_radius) * (1.0 - progress)
                self._spot_radius += (target_radius - self._spot_radius) * self.shrink_smoothing

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
        raw_x, raw_y = float(pos.x()), float(pos.y())
        cx, cy = raw_x, raw_y
        if self.cursor_filter is not None:
            cx, cy = self.cursor_filter.filter(raw_x, raw_y)

        self._update_spot_radius(QtCore.QPointF(cx, cy))
        target = self._select_target(cx, cy, detections) if detections else None

        if target is None:
            self._last_target = None
            self._last_target_info = None
        else:
            tx, ty, w, h, _cls_id = target
            self._last_target = (
                tx / self.detector.sx,
                ty / self.detector.sy,
                w / self.detector.sx,
                h / self.detector.sy,
            )
            self._last_target_info = self.detector.find_detection_by_geometry(
                tx, ty, w, h, class_id=_cls_id
            )
        if self._spot_radius > self.min_radius + 0.5:
            radius_ratio = min(1.0, (self._spot_radius - self.min_radius) / max(1.0, self.max_radius - self.min_radius))
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
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=cx,
                filtered_y=cy,
                technique="dynaspot",
                filter_name=self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                dynaspot_radius=round(float(self._spot_radius), 3),
                has_target=self._last_target is not None,
            )
        painter.end()

    def _draw_fake_cursor(self, painter, cx, cy):
        active = self._spot_radius > self.min_radius + 0.5
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
        painter.drawLine(QtCore.QLineF(cx, cy - gap - line_len, cx, cy - gap))
        painter.drawLine(QtCore.QLineF(cx, cy + gap, cx, cy + gap + line_len))
        painter.drawLine(QtCore.QLineF(cx + gap, cy, cx + gap + line_len, cy))
        painter.drawLine(QtCore.QLineF(cx - gap - line_len, cy, cx - gap, cy))

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
            self._spot_radius = self.min_radius
        self.update()

    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        self.enabled = False
        self.detector.stop()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        restore_default_cursors()
        if self.logger is not None:
            self.logger.log_session_end(reason="quit")
            self.logger.close()
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event):
        self.enabled = False
        self.detector.stop()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        restore_default_cursors()
        super().closeEvent(event)

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
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._last_target_info,
                )
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
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[round(tx, 3), round(ty, 3)],
                    redirected=True,
                    target=self._last_target_info,
                )
            self._rehide_cursor()
            self._start_mouse_listener()


def dynaspot(
    detector: TargetFinder,
    cursor_filter=None,
    logger=None,
    **dynaspot_kwargs,
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hide_cursor_everywhere()
    ov = DynaSpot(detector, cursor_filter=cursor_filter, logger=logger, **dynaspot_kwargs)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if logger is not None:
        logger.log_session_end(reason="app_exit")
        logger.close()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description="Launch the DynaSpot overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    parser.add_argument('--filter', choices=sorted(FILTER_OPTIONS.keys()), default="none", help="Optional pointer filter")
    parser.add_argument('--log-file', default=None, help="Optional JSONL log file path")
    parser.add_argument('--log-cursor-hz', type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument('--min-speed', type=float, default=DynaSpot.MIN_SPEED, help="Minimum speed before the DynaSpot radius starts growing")
    parser.add_argument('--max-speed', type=float, default=DynaSpot.MAX_SPEED, help="Speed at which the DynaSpot radius reaches its maximum")
    parser.add_argument('--min-radius', type=float, default=DynaSpot.MIN_RADIUS, help="Minimum DynaSpot radius")
    parser.add_argument('--max-radius', type=float, default=DynaSpot.MAX_RADIUS, help="Maximum DynaSpot radius")
    parser.add_argument('--lag', type=float, default=DynaSpot.LAG, help="Delay before the radius begins shrinking after motion stops")
    parser.add_argument('--reduce-time', type=float, default=DynaSpot.REDUCE_TIME, help="Time used to shrink the radius back toward its minimum")
    parser.add_argument('--growth-smoothing', type=float, default=DynaSpot.GROWTH_SMOOTHING, help="Smoothing factor applied while the spot grows")
    parser.add_argument('--shrink-smoothing', type=float, default=DynaSpot.SHRINK_SMOOTHING, help="Smoothing factor applied while the spot shrinks")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    cursor_filter = PointFilter2D(args.filter)
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="dynaspot",
            filter_name=args.filter,
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
            min_speed=args.min_speed,
            max_speed=args.max_speed,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            lag=args.lag,
            reduce_time=args.reduce_time,
            growth_smoothing=args.growth_smoothing,
            shrink_smoothing=args.shrink_smoothing,
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    dynaspot(
        det,
        cursor_filter=cursor_filter,
        logger=logger,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        lag=args.lag,
        reduce_time=args.reduce_time,
        growth_smoothing=args.growth_smoothing,
        shrink_smoothing=args.shrink_smoothing,
    )


if __name__ == "__main__":
    main()
