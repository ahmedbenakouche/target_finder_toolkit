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
from target_finder_toolkit.annotation_detector import AnnotationDetector
from target_finder_toolkit.mouse_utils import restore_default_cursors
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D, add_filter_arguments, filter_kwargs_from_args
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui

__all__ = ["dynaspot", "main"]

_SESSION_STOP_REASON = None

pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0


class DynaSpot(QtWidgets.QWidget):
    """
    PyQt overlay widget implementing the paper-style DynaSpot speed-dependent area cursor.

    The system cursor remains visible and acts as the cursor center. A translucent
    circular activation area grows when pointer speed exceeds a threshold. When the
    area overlaps multiple targets, the closest one to the cursor center is selected.
    """

    POINT_WIDTH = 1.0
    MIN_SPEED = 100.0
    SPOT_WIDTH = 128.0
    LAG = 0.300
    REDUCE_TIME = 0.500
    GROWTH_FACTOR = 1.2
    CO_EXPONENTIAL_POWER = 3.0
    CLICK_EPSILON = 3.0

    def __init__(
        self,
        detector: TargetFinder,
        cursor_filter=None,
        logger=None,
        *,
        min_speed: float | None = None,
        spot_width: float | None = None,
        lag: float | None = None,
        reduce_time: float | None = None,
        include_text_targets: bool = False,
    ):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        if sys.platform == "darwin":
            self.detector.hide_overlay_during_capture = False
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.min_speed = float(self.MIN_SPEED if min_speed is None else min_speed)
        self.spot_width = float(self.SPOT_WIDTH if spot_width is None else spot_width)
        self.lag = float(self.LAG if lag is None else lag)
        self.reduce_time = float(self.REDUCE_TIME if reduce_time is None else reduce_time)
        self.include_text_targets = bool(include_text_targets)

        self._mouse_listener = None
        self._keyboard_listener = None

        self.enabled = True
        self._last_target = None
        self._last_target_click = None
        self._last_target_info = None
        self._current_target = None
        self._current_non_text_target = None
        self._pending_click_target = None
        self._pending_click_point = None
        self._selected_detection = None
        self._spot_current_width = self.POINT_WIDTH
        self._last_speed_point = QtCore.QPointF(QtGui.QCursor.pos())
        now = time.monotonic()
        self._last_input_at = now
        self._last_motion_at = now
        self._shrink_start_at = None
        self._shrink_start_width = self.POINT_WIDTH

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

    def _spot_radius(self) -> float:
        return self._spot_current_width / 2.0

    def _update_spot_width(self):
        now = time.monotonic()
        idle_for = now - self._last_motion_at
        if idle_for < self.lag:
            return

        if self._shrink_start_at is None:
            self._shrink_start_at = now
            self._shrink_start_width = self._spot_current_width
        progress = min(1.0, (now - self._shrink_start_at) / max(self.reduce_time, 1e-6))
        reduction_fraction = progress ** self.CO_EXPONENTIAL_POWER
        self._spot_current_width = self.POINT_WIDTH + (
            self._shrink_start_width - self.POINT_WIDTH
        ) * (1.0 - reduction_fraction)

    @QtCore.pyqtSlot(int, int)
    def _handle_pointer_move_event(self, x: int, y: int):
        now = time.monotonic()
        speed_x, speed_y = float(x), float(y)
        if self.cursor_filter is not None:
            speed_x, speed_y = self.cursor_filter.filter(speed_x, speed_y)
        speed_point = QtCore.QPointF(speed_x, speed_y)
        dt = max(now - self._last_input_at, 1e-3)
        dx = speed_point.x() - self._last_speed_point.x()
        dy = speed_point.y() - self._last_speed_point.y()
        dist = math.hypot(dx, dy)
        speed = dist / dt

        moved = dist > 0.5  # Ignore tiny jitter between input events.
        if moved:
            self._last_motion_at = now
            self._shrink_start_at = None
            if speed >= self.min_speed:
                self._spot_current_width = min(
                    self.spot_width,
                    max(self.POINT_WIDTH, self._spot_current_width * self.GROWTH_FACTOR),
                )

        self._last_input_at = now
        self._last_speed_point = QtCore.QPointF(speed_point)

    def _select_target(self, cx: float, cy: float):
        radius = self._spot_radius()
        candidates = []
        for det in self.detector.get_detection_dicts():
            if det.get("class_id") == 3 and not self.include_text_targets:
                continue
            x = float(det["x"])
            y = float(det["y"])
            w = float(det["width"])
            h = float(det["height"])
            cx_box = x + w / 2.0
            cy_box = y + h / 2.0
            dx = max(0.0, abs(cx - cx_box) - w / 2.0)
            dy = max(0.0, abs(cy - cy_box) - h / 2.0)
            int_d = math.hypot(dx, dy)
            if int_d <= radius:
                center_d = math.hypot(cx - cx_box, cy - cy_box)
                area = w * h
                candidates.append((center_d, area, -float(det.get("score", 0.0)), det))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def _set_selected_target(self, selected_det):
        self._selected_detection = selected_det
        if selected_det is None:
            self._last_target = None
            self._last_target_click = None
            self._last_target_info = None
            return

        tx = float(selected_det["x"]) + float(selected_det["width"]) / 2.0
        ty = float(selected_det["y"]) + float(selected_det["height"]) / 2.0
        w = float(selected_det["width"])
        h = float(selected_det["height"])
        self._last_target = (tx, ty, w, h)
        scale_x = max(float(getattr(self.detector, "sx", 1.0) or 1.0), 1e-6)
        scale_y = max(float(getattr(self.detector, "sy", 1.0) or 1.0), 1e-6)
        if sys.platform == "darwin":
            self._last_target_click = (tx, ty)
        else:
            self._last_target_click = (tx / scale_x, ty / scale_y)
        self._last_target_info = TargetFinder._compact_detection(selected_det)

    def _draw_selected_target(self, painter: QtGui.QPainter):
        if self._selected_detection is None:
            return
        x = float(self._selected_detection["x"])
        y = float(self._selected_detection["y"])
        w = float(self._selected_detection["width"])
        h = float(self._selected_detection["height"])
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80, 220), 3))
        painter.setBrush(QtGui.QColor(255, 80, 80, 44))
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), 8, 8)

    def paintEvent(self, event):
        if not self.enabled:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pos = QtGui.QCursor.pos()
        raw_x, raw_y = float(pos.x()), float(pos.y())
        cx, cy = raw_x, raw_y
        speed_x, speed_y = raw_x, raw_y
        if self.cursor_filter is not None:
            speed_x, speed_y = self.cursor_filter.filter(raw_x, raw_y)

        self._update_spot_width()
        self._current_non_text_target = self.detector.find_detection_for_point(
            raw_x,
            raw_y,
            include_text=False,
            fallback_nearest=False,
        )
        self._current_target = self.detector.find_detection_for_point(
            raw_x,
            raw_y,
            include_text=True,
            fallback_nearest=False,
        )
        self._set_selected_target(self._select_target(cx, cy))

        if self._spot_current_width > self.POINT_WIDTH + 0.25:
            radius = self._spot_radius()
            painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60, 170), 2))
            painter.setBrush(QtGui.QColor(145, 145, 145, 68))
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
        self._draw_selected_target(painter)
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=speed_x,
                filtered_y=speed_y,
                technique="dynaspot",
                filter_name=self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                **(self.cursor_filter.params if self.cursor_filter is not None else {}),
                dynaspot_width=round(float(self._spot_current_width), 3),
                dynaspot_radius=round(float(self._spot_radius()), 3),
                has_target=self._last_target is not None,
            )
        painter.end()

    @QtCore.pyqtSlot()
    def toggle_dynaspot(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.detector.start()
        else:
            self.detector.stop()
            self._last_target = None
            self._last_target_click = None
            self._last_target_info = None
            self._selected_detection = None
            self._spot_current_width = self.POINT_WIDTH
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
        restore_default_cursors()
        self._last_target = None
        self._last_target_click = None
        self._last_target_info = None
        if self.logger is not None:
            self.logger.log_session_end(reason="quit")
            self.logger.close()
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event):
        reason = _SESSION_STOP_REASON or "window_close"
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
        restore_default_cursors()
        self._last_target = None
        self._last_target_click = None
        self._last_target_info = None
        if self.logger is not None:
            self.logger.log_session_end(reason=reason)
            self.logger.close()
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

    def _start_mouse_listener(self):
        def on_move(x, y):
            QtCore.QMetaObject.invokeMethod(
                self,
                "_handle_pointer_move_event",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(int, int(round(x))),
                QtCore.Q_ARG(int, int(round(y))),
            )

        def on_click(x, y, button, pressed):
            if sys.platform == "darwin":
                return
            if pressed and button == button.left:
                if self._in_system_reserved_area(x, y):
                    return
                self._snapshot_click_target()
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_simulate_click",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(int, x),
                    QtCore.Q_ARG(int, y),
                )

        kwargs = {"on_move": on_move, "on_click": on_click}
        if sys.platform == "darwin":
            kwargs["darwin_intercept"] = self._intercept_mouse_event
        self._mouse_listener = mouse.Listener(**kwargs)
        self._mouse_listener.start()

    def _intercept_mouse_event(self, event_type, event):
        try:
            import Quartz
        except Exception:
            return event

        if event_type not in (
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventLeftMouseUp,
        ):
            return event

        try:
            px, py = Quartz.CGEventGetLocation(event)
        except Exception:
            return event

        if self._in_system_reserved_area(int(px), int(py)):
            return event

        if event_type == Quartz.kCGEventLeftMouseDown:
            self._pending_click_point = self._last_target
            self._pending_click_target = self._last_target_info

        target = self._pending_click_point or self._last_target
        redirected = False
        effective_x = float(px)
        effective_y = float(py)
        if target is not None:
            tx, ty, w, h = target
            redirected = not (tx - w / 2 <= px <= tx + w / 2 and ty - h / 2 <= py <= ty + h / 2)
            if redirected:
                Quartz.CGEventSetLocation(event, Quartz.CGPointMake(float(tx), float(ty)))
                effective_x = float(tx)
                effective_y = float(ty)

        if event_type == Quartz.kCGEventLeftMouseUp:
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[round(float(px), 3), round(float(py), 3)],
                    effective=[round(effective_x, 3), round(effective_y, 3)],
                    redirected=redirected,
                    target=self._resolve_click_target_for_log(
                        effective_x,
                        effective_y,
                        self._pending_click_target,
                        fallback_nearest=False,
                    ),
                )
            self._pending_click_point = None
            self._pending_click_target = None
        return event

    def _snapshot_click_target(self):
        self._pending_click_point = self._last_target
        self._pending_click_target = (
            dict(self._last_target_info) if self._last_target_info is not None else None
        )

    def _take_click_target_snapshot(self):
        point = self._pending_click_point
        target = self._pending_click_target
        self._pending_click_point = None
        self._pending_click_target = None
        return point, target

    def _resolve_click_target_for_log(self, x, y, fallback_info=None, *, fallback_nearest=False):
        if fallback_info is not None:
            return dict(fallback_info)
        if self._current_non_text_target is not None:
            return dict(self._current_non_text_target)
        if self._current_target is not None:
            return dict(self._current_target)
        target = self.detector.find_detection_for_point(
            float(x),
            float(y),
            include_text=False,
            fallback_nearest=False,
        )
        if target is not None:
            return target
        target = self.detector.find_detection_for_point(
            float(x),
            float(y),
            include_text=True,
            fallback_nearest=False,
        )
        if target is not None:
            return target
        if not fallback_nearest:
            return None
        return self.detector.find_detection_for_point(
            float(x),
            float(y),
            include_text=False,
            fallback_nearest=True,
        )

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        click_target, click_target_info = self._take_click_target_snapshot()
        if click_target is None:
            click_target = self._last_target
        if click_target_info is None:
            click_target_info = self._last_target_info

        if self._in_system_reserved_area(orig_x, orig_y):
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._resolve_click_target_for_log(
                        orig_x,
                        orig_y,
                        click_target_info,
                        fallback_nearest=False,
                    ),
                )
            return
        if not (click_target and self.enabled):
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._resolve_click_target_for_log(
                        orig_x,
                        orig_y,
                        click_target_info,
                        fallback_nearest=True,
                    ),
                )
            return

        tx, ty, w, h = click_target
        if tx - w / 2 <= orig_x <= tx + w / 2 and ty - h / 2 <= orig_y <= ty + h / 2:
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._resolve_click_target_for_log(
                        orig_x,
                        orig_y,
                        click_target_info,
                        fallback_nearest=True,
                    ),
                )
            return

        click_tx, click_ty = (
            (tx, ty) if click_target is not self._last_target else self._last_target_click
        )
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        try:
            self._send_click(orig_x, orig_y, click_tx, click_ty)
        except pyautogui.FailSafeException:
            pass
        finally:
            if self.logger is not None:
                self.logger.log_click(
                    technique="dynaspot",
                    raw=[orig_x, orig_y],
                    effective=[round(click_tx, 3), round(click_ty, 3)],
                    redirected=True,
                    target=self._resolve_click_target_for_log(
                        click_tx,
                        click_ty,
                        click_target_info,
                        fallback_nearest=True,
                    ),
                )
            self._start_mouse_listener()

    def _send_click(self, orig_x: float, orig_y: float, tx: float, ty: float):
        if sys.platform == "darwin":
            self._send_macos_click(float(orig_x), float(orig_y), float(tx), float(ty))
            return

        pyautogui.mouseUp(button='left')
        pyautogui.moveTo(tx, ty)
        pyautogui.click()
        pyautogui.moveTo(orig_x, orig_y)

    def _send_macos_click(self, orig_x: float, orig_y: float, tx: float, ty: float):
        import Quartz

        def post_mouse_event(event_type, x, y):
            event = Quartz.CGEventCreateMouseEvent(
                None,
                event_type,
                (x, y),
                Quartz.kCGMouseButtonLeft,
            )
            if event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
                Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 1)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

        post_mouse_event(Quartz.kCGEventMouseMoved, tx, ty)
        time.sleep(0.015)
        post_mouse_event(Quartz.kCGEventLeftMouseDown, tx, ty)
        time.sleep(0.025)
        post_mouse_event(Quartz.kCGEventLeftMouseUp, tx, ty)
        time.sleep(0.03)
        post_mouse_event(Quartz.kCGEventMouseMoved, orig_x, orig_y)


def dynaspot(
    detector: TargetFinder,
    cursor_filter=None,
    logger=None,
    **dynaspot_kwargs,
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ov = DynaSpot(detector, cursor_filter=cursor_filter, logger=logger, **dynaspot_kwargs)
    ov.show()
    raise_macos_window_above_system_ui(ov, level_offset=1)
    def _handle_signal(sig, frame):
        global _SESSION_STOP_REASON
        _SESSION_STOP_REASON = "stop_button" if sig in {
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGBREAK", None),
        } else "signal_interrupt"
        QtWidgets.QApplication.quit()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)
    exit_code = app.exec()
    restore_default_cursors()
    if logger is not None:
        logger.log_session_end(reason=_SESSION_STOP_REASON or "app_exit")
        logger.close()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description="Launch the DynaSpot overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    add_filter_arguments(parser)
    parser.add_argument('--log-file', default=None, help="Optional JSONL log file path")
    parser.add_argument('--log-cursor-hz', type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument('--min-speed', type=float, default=DynaSpot.MIN_SPEED, help="Minimum speed threshold (px/s) before the DynaSpot area starts growing")
    parser.add_argument('--spot-width', type=float, default=DynaSpot.SPOT_WIDTH, help="Maximum DynaSpot spot width (diameter in pixels)")
    parser.add_argument('--lag', type=float, default=DynaSpot.LAG, help="Delay before the spot starts shrinking after motion stops")
    parser.add_argument('--reduce-time', type=float, default=DynaSpot.REDUCE_TIME, help="Time used to shrink the spot back to a point")
    parser.add_argument('--annotation-control-file', default=None, help="Use controlled-task annotations instead of live YOLO detection")
    parser.add_argument('--include-text-targets', action='store_true', help="Allow Text annotations to be selected by DynaSpot")
    args = parser.parse_args()

    if args.annotation_control_file:
        det = AnnotationDetector(args.annotation_control_file)
    else:
        if args.model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            args.model_path = os.path.join(here, "yolo26s_1280.pt")
        det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    cursor_filter = PointFilter2D(args.filter, **filter_kwargs_from_args(args))
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="dynaspot",
            filter_name=args.filter,
            **filter_kwargs_from_args(args),
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
            min_speed=args.min_speed,
            spot_width=args.spot_width,
            lag=args.lag,
            reduce_time=args.reduce_time,
            detection_source="annotations" if args.annotation_control_file else "yolo",
            annotation_control_file=args.annotation_control_file,
            include_text_targets=bool(args.include_text_targets or args.annotation_control_file),
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    dynaspot(
        det,
        cursor_filter=cursor_filter,
        logger=logger,
        min_speed=args.min_speed,
        spot_width=args.spot_width,
        lag=args.lag,
        reduce_time=args.reduce_time,
        include_text_targets=bool(args.include_text_targets or args.annotation_control_file),
    )


if __name__ == "__main__":
    main()
