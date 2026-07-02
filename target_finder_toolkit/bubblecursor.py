"""
Bubble Cursor Demo
==================

This module demonstrates the **Bubble Cursor** interaction technique
using the TargetFinder toolkit.

Notes
-----
- This script is for demonstration purposes only.
- It is **not part of the core TargetFinder API**, but shows how the
  toolkit can be used for novel interaction techniques.
"""


import os
import sys
import time
import signal
import atexit
import threading
import numpy as np
import cv2
import mss
from ultralytics import YOLO
from PyQt6 import QtWidgets, QtGui, QtCore
import math
import pyautogui
from pynput import keyboard, mouse
import argparse
from target_finder_toolkit.targetfinder import CLASS_NAMES, TargetFinder
from target_finder_toolkit.annotation_detector import FakeTargetFinder
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D, add_filter_arguments, filter_kwargs_from_args
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui

__all__ = ["bubble_cursor", "main"]


_CURSOR_RESTORE_REGISTERED = False
_SESSION_STOP_REASON = None

class BubbleCursor(QtWidgets.QWidget):
    """
    PyQt overlay widget implementing the Bubble Cursor technique.

    Principle
    ---------
    At each frame, the system computes the intersecting distance (IntD)
    from the pointer to all detected widgets. The bubble radius is set
    to encompass the closest widget but not overlap the second closest.
    A visual "bubble" is drawn around the pointer and the target.

    Parameters
    ----------
    detector : TargetFinder
        Detector providing widget bounding boxes (logical coordinates).

    Notes
    -----
    - The overlay is transparent and always on top.
    - Clicking outside a widget inside the bubble is redirected
      to the nearest detected target.
    - The real cursor is hidden and replaced by a drawn "fake" cursor.
    """
    def __init__(
        self,
        detector: TargetFinder,
        cursor_filter=None,
        logger=None,
        *,
        include_text_targets=False,
        disable_keyboard_quit=False,
    ):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        if sys.platform == "darwin":
            self.detector.hide_overlay_during_capture = False
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.include_text_targets = bool(include_text_targets)
        self.disable_keyboard_quit = bool(disable_keyboard_quit)
        self._last_target = None  # to store the active target
        self._last_target_click = None
        self._last_target_info = None
        self._hover_text_info = None
        self._pending_click_target = None
        self._pending_click_target_logical = None
        self._pending_click_target_info = None
        self._pending_click_enabled = False
        self._mouse_listener = None
        self._last_rehide_at = 0.0
        self.bubble_enabled = True
        self._start_mouse_listener()
        self._start_keyboard_listener()

        # Full-screen geometry (Qt DPI-aware)
        geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(geom)

        # Window flags for frameless, always-on-top, click-through & transparent
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

        # Start detection thread
        self.detector.start()

        # refresh to update the overlay
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10) # 10 ms
        self._cursor_refresh_timer = None
        if sys.platform == "darwin":
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(self._rehide_cursor)
            self._cursor_refresh_timer.start(30)

    def _detector_active(self) -> bool:
        is_active = getattr(self.detector, "is_active", None)
        return not callable(is_active) or bool(is_active())

    def _resolve_target_info(self, tx, ty, w, h, cls_id, score):
        target_info = self.detector.find_detection_by_geometry(
            tx, ty, w, h, class_id=cls_id
        )
        if target_info is not None:
            return target_info

        target_info = self.detector.find_detection_for_point(
            float(tx),
            float(ty),
            include_text=(cls_id == 3),
            fallback_nearest=True,
        )
        if target_info is not None:
            return target_info

        return {
            "id": None,
            "x": int(round(tx - w / 2)),
            "y": int(round(ty - h / 2)),
            "w": int(round(w)),
            "h": int(round(h)),
            "score": round(float(score), 4),
            "class": CLASS_NAMES.get(int(cls_id), str(cls_id)),
            "class_id": int(cls_id),
        }

    def _screen_for_point(self, x: int, y: int):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return QtWidgets.QApplication.primaryScreen()
        return app.screenAt(QtCore.QPoint(int(x), int(y))) or QtWidgets.QApplication.primaryScreen()

    def _snapshot_click_target(self):
        target_info = None
        click_target = tuple(self._last_target_click) if self._last_target_click else None
        logical_target = tuple(self._last_target) if self._last_target else None
        if self._last_target_info is not None:
            target_info = self._last_target_info
        if target_info is None and self._hover_text_info is not None:
            target_info = self._hover_text_info
        self._pending_click_target = click_target
        self._pending_click_target_logical = logical_target
        self._pending_click_target_info = dict(target_info) if target_info is not None else None
        self._pending_click_enabled = bool(self.bubble_enabled)

    def _take_click_target_snapshot(self):
        target = self._pending_click_target
        logical_target = self._pending_click_target_logical
        target_info = self._pending_click_target_info
        enabled = self._pending_click_enabled
        self._pending_click_target = None
        self._pending_click_target_logical = None
        self._pending_click_target_info = None
        self._pending_click_enabled = False
        return target, logical_target, target_info, enabled

    def _resolve_click_target_for_log(self, x, y, fallback_info=None, *, fallback_nearest=False):
        if fallback_info is not None:
            return dict(fallback_info)
        target_info = self.detector.find_detection_for_point(
            float(x),
            float(y),
            include_text=True,
            fallback_nearest=fallback_nearest,
        )
        return target_info

    def _in_system_reserved_area(self, x: int, y: int) -> bool:
        if sys.platform != "darwin":
            return False
        screen = self._screen_for_point(x, y)
        if screen is None:
            return False
        point = QtCore.QPoint(int(x), int(y))
        return not screen.availableGeometry().contains(point)

    # === Paint ===
    def paintEvent(self, event):
        detections = self.detector.get_detections()
        if not self.bubble_enabled or not detections:
            return
        bubble_candidates = [
            det for det in detections
            if self.include_text_targets or int(det[5]) != 3
        ]

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Pointer position (logical coordinates)
        pos = QtGui.QCursor.pos()
        cx, cy = pos.x(), pos.y()
        raw_x, raw_y = float(cx), float(cy)
        filtered_x, filtered_y = raw_x, raw_y
        if self.cursor_filter is not None:
            filtered_x, filtered_y = self.cursor_filter.filter(raw_x, raw_y)

        text_margin = 20
        cursor_on_non_text = False
        for x, y, w, h, score, cls_id in bubble_candidates:
            if cls_id != 3 and x <= cx <= x + w and y <= cy <= y + h:
                cursor_on_non_text = True
                break
        if not self.include_text_targets and not cursor_on_non_text:
            for x, y, w, h, score, cls_id in detections:
                if cls_id == 3:
                    if ((x - text_margin) <= cx <= (x + w + text_margin) and
                            (y - text_margin) <= cy <= (y + h + text_margin)):
                        self._last_target = None
                        self._last_target_click = None
                        self._last_target_info = None
                        self._hover_text_info = self.detector.find_detection_for_point(
                            float(cx),
                            float(cy),
                            include_text=True,
                            fallback_nearest=True,
                        )
                        self._draw_fake_cursor(painter, cx, cy)
                        painter.end()
                        return

        # Compute distances from pointer to each box edge
        distances = []
        for x, y, w, h, score, cls_id in bubble_candidates:
            cx_box = x + w/2
            cy_box = y + h/2

            # Intersecting Distance (IntD)
            dx = max(0.0, abs(cx - cx_box) - w/2)
            dy = max(0.0, abs(cy - cy_box) - h/2)
            IntD = math.hypot(dx, dy)

            distances.append((IntD, cx_box, cy_box, w, h, score, cls_id))

        # Draw bubble only when not hovering over text
        if not distances:
            self._last_target = None
            self._last_target_click = None
            self._last_target_info = None
            self._hover_text_info = None
            self._draw_fake_cursor(painter,cx,cy)
            painter.end()
            return

        distances.sort(key=lambda t: t[0])
        IntD1, tx, ty, w, h, nearest_score, nearest_cls_id = distances[0]

        if nearest_cls_id == 3 and not self.include_text_targets:
            self._last_target = None
            self._last_target_click = None
            self._last_target_info = None
            self._hover_text_info = self._resolve_target_info(
                tx, ty, w, h, nearest_cls_id, nearest_score
            )
            self._draw_fake_cursor(painter,cx,cy)
            painter.end()
            return
        self._hover_text_info = None
        self._last_target = (tx, ty, w, h)
        scale_x = max(float(getattr(self.detector, "sx", 1.0) or 1.0), 1e-6)
        scale_y = max(float(getattr(self.detector, "sy", 1.0) or 1.0), 1e-6)
        if sys.platform == "darwin":
            self._last_target_click = (tx, ty)
        else:
            self._last_target_click = (tx / scale_x, ty / scale_y)
        self._last_target_info = self._resolve_target_info(
            tx, ty, w, h, nearest_cls_id, nearest_score
        )

        # Containment Distance (ConD1)
        x = tx - w / 2
        y = ty - h / 2
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        ConD1 = max([math.hypot(cx - px, cy - py) for (px, py) in corners])
        if len(distances) > 1:
            IntD2 = distances[1][0]
            radius = min(ConD1, IntD2)
            if radius == IntD2:
                gap = 0.01 * radius
                radius = int(radius - gap)  # to avoid touching the second target
        else:
            radius = ConD1

        # build main and env bubble with different centers
        main_path = QtGui.QPainterPath()
        main_path.addEllipse(cx - radius, cy - radius, 2 * radius, 2 * radius)

        # env bubble centered on widget center
        env_path = QtGui.QPainterPath()
        t = 1  # t interpolates between 0 = sharp‑cornered rectangle / 1 = fully rounded (ellipse‑like)
        r = min(w, h) / 2 * t  # corner radius
        d = math.hypot(r, r) - r
        env_path.addRoundedRect(tx - w/2 -d, ty - h/2 - d, w + 2*d, h + 2*d, r + d, r + d)

        # Or we can also use a circle around it or a ellipse
        # reinforce radius: half max dimension of widget = max(w, h) / 2.0
        # ew, eh = w * math.sqrt(2), h * math.sqrt(2)
        # env_path.addEllipse(tx - ew/2, ty - eh/2, ew, eh)

        union_path = main_path.united(env_path)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 3)
        painter.setPen(pen)
        painter.drawPath(union_path)
        self._draw_fake_cursor(painter,cx,cy)
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=filtered_x,
                filtered_y=filtered_y,
                technique="bubble",
                filter_name=self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                **(self.cursor_filter.params if self.cursor_filter is not None else {}),
                bubble_radius=round(float(radius), 3),
                has_target=self._last_target is not None,
            )
        painter.end()
        return


    # FAKE CURSOR
    def _draw_fake_cursor(self,painter,cx,cy):
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        radius_cursor = 6
        center = QtCore.QPointF(cx, cy)
        painter.drawEllipse(center, radius_cursor, radius_cursor)
        line_len = 4
        gap = radius_cursor + 1
        painter.drawLine(QtCore.QLineF(cx, cy - gap - line_len, cx, cy - gap))
        painter.drawLine(QtCore.QLineF(cx, cy + gap, cx, cy + gap + line_len))
        painter.drawLine(QtCore.QLineF(cx + gap, cy, cx + gap + line_len, cy))
        painter.drawLine(QtCore.QLineF(cx - gap - line_len, cy, cx - gap, cy))
    

    # === Toggle bubble on/off ===
    @QtCore.pyqtSlot()
    def toggle_bubble(self):
        self.bubble_enabled = not self.bubble_enabled
        if self.bubble_enabled:
            self.detector.start()
            hide_cursor_everywhere()
        else:
            self.detector.stop()
            restore_default_cursors()
        self.update()  # repaint immediately

    # === Quit ===
    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        self.bubble_enabled = False
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
        reason = _SESSION_STOP_REASON or "window_close"
        self.bubble_enabled = False
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
            self.logger.log_session_end(reason=reason)
            self.logger.close()
        super().closeEvent(event)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'b':
                    QtCore.QMetaObject.invokeMethod(self, "toggle_bubble", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key.char == 'q' and not self.disable_keyboard_quit:
                    QtCore.QMetaObject.invokeMethod(self, "stop_and_quit", QtCore.Qt.ConnectionType.QueuedConnection)
            except AttributeError:
                pass
        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    @QtCore.pyqtSlot()
    def _rehide_cursor(self):
        if not self.bubble_enabled or not self._detector_active():
            return
        hide_cursor_everywhere()
        if sys.platform == "darwin":
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(75, hide_cursor_everywhere)

    # === Global mouse listener + click simulation ===
    def _start_mouse_listener(self):
        def queue_rehide(force=False):
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
            if not self._detector_active():
                return
            queue_rehide()

        def on_click(x, y, button, pressed):
            if not self._detector_active():
                return
            if sys.platform == "darwin":
                return
            if button == button.left:
                queue_rehide(force=True)
            if pressed and button == button.left:
                self._snapshot_click_target()
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(int, x), QtCore.Q_ARG(int, y))

        kwargs = {"on_move": on_move, "on_click": on_click}
        if sys.platform == "darwin":
            kwargs["darwin_intercept"] = self._intercept_mouse_event
        self._mouse_listener = mouse.Listener(**kwargs)
        self._mouse_listener.start()

    def _intercept_mouse_event(self, event_type, event):
        if not self._detector_active():
            return event
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
            self._snapshot_click_target()

        logical_target = self._pending_click_target_logical or self._last_target
        target_info = self._pending_click_target_info
        if target_info is None:
            if self._last_target_info is not None:
                target_info = dict(self._last_target_info)
            elif self._hover_text_info is not None:
                target_info = dict(self._hover_text_info)

        if logical_target is None:
            if event_type == Quartz.kCGEventLeftMouseUp and self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
                    raw=[round(float(px), 3), round(float(py), 3)],
                    effective=[round(float(px), 3), round(float(py), 3)],
                    redirected=False,
                    target=self._resolve_click_target_for_log(
                        float(px),
                        float(py),
                        target_info,
                        fallback_nearest=True,
                    ),
                )
                self._pending_click_target = None
                self._pending_click_target_logical = None
                self._pending_click_target_info = None
                self._pending_click_enabled = False
            return event

        tx, ty, w, h = logical_target
        if self._pending_click_target is not None:
            click_x, click_y = self._pending_click_target
        elif self._last_target_click is not None:
            click_x, click_y = self._last_target_click
        else:
            click_x = float(tx)
            click_y = float(ty)
        redirected = not (tx - w / 2 <= px <= tx + w / 2 and ty - h / 2 <= py <= ty + h / 2)
        if redirected:
            Quartz.CGEventSetLocation(event, Quartz.CGPointMake(float(click_x), float(click_y)))

        if event_type == Quartz.kCGEventLeftMouseUp:
            if self.logger is not None:
                effective_x = float(click_x) if redirected else float(px)
                effective_y = float(click_y) if redirected else float(py)
                self.logger.log_click(
                    technique="bubble",
                    raw=[round(float(px), 3), round(float(py), 3)],
                    effective=[round(effective_x, 3), round(effective_y, 3)],
                    redirected=redirected,
                    target=self._resolve_click_target_for_log(
                        effective_x,
                        effective_y,
                        target_info,
                        fallback_nearest=True,
                    ),
                )
            self._pending_click_target = None
            self._pending_click_target_logical = None
            self._pending_click_target_info = None
            self._pending_click_enabled = False

        return event

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        if not self._detector_active():
            return
        click_target, logical_target, click_target_info, click_enabled = self._take_click_target_snapshot()
        if click_target is None:
            click_target = self._last_target_click
        if logical_target is None:
            logical_target = self._last_target
        if click_target_info is None:
            click_target_info = self._last_target_info
        if not click_enabled:
            click_enabled = self.bubble_enabled

        if self._in_system_reserved_area(orig_x, orig_y):
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
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
        if not (click_target and click_enabled):
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
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

        if logical_target is None or click_target is None:
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
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

        tx, ty, w, h = logical_target

        # If click inside detected rectangle let it pass
        if tx-w/2 <= orig_x <= tx+w/2 and ty-h/2 <= orig_y <= ty+h/2:
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
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

        # Otherwise simulate target click
        click_tx, click_ty = click_target
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        try:
            self._send_click(orig_x, orig_y, click_tx, click_ty)
        except pyautogui.FailSafeException:
            # to prevent the script from crashing when the mouse hits a corner
            pass
        finally:
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
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
            self._rehide_cursor()
            self._start_mouse_listener() # restart the listener

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


def bubble_cursor(
    detector: TargetFinder,
    cursor_filter=None,
    logger=None,
    *,
    include_text_targets=False,
    disable_keyboard_quit=False,
):
    """Launch the Bubble Cursor overlay.

    This replaces the system cursor with a dynamic "bubble" that
    expands to always contain the closest widget detected by
    :class:`TargetFinder`.

    Args:
        detector (TargetFinder): Initialized detector (YOLO model loaded).

    Returns:
        None: Blocks until the Qt application is closed.
    """
    global _CURSOR_RESTORE_REGISTERED
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if not _CURSOR_RESTORE_REGISTERED:
        atexit.register(restore_default_cursors)
        _CURSOR_RESTORE_REGISTERED = True
    ov = BubbleCursor(
        detector,
        cursor_filter=cursor_filter,
        logger=logger,
        include_text_targets=include_text_targets,
        disable_keyboard_quit=disable_keyboard_quit,
    )
    ov.show()
    raise_macos_window_above_system_ui(ov, level_offset=1)
    is_active = getattr(detector, "is_active", None)
    if not callable(is_active) or bool(is_active()):
        if sys.platform == "darwin":
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
        else:
            hide_cursor_everywhere()
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


# CLI usage
def main():
    """Command-line entry point for the Bubble Cursor demo.

    CLI arguments:
        - -model-path (str, optional): Path to YOLO .pt weights.
            Defaults to ``yolo26s_1280.pt`` in the package.
        - -change-thresh (int, optional): Threshold for screen change detection.
            Higher = less sensitive. Default = ``100``.
        - -capture-interval (float, optional): Delay in seconds between captures.
            Lower = higher refresh rate, more CPU/GPU usage. Default = ``1/30``.
        - -confidence (float, optional): YOLO confidence threshold in ``[0, 1]``.
            Default = ``0.28``.
        - -iou (float, optional): IoU threshold for YOLO NMS in ``[0, 1]``.
            Controls overlap merging. Default = ``0.3``.

    Keyboard shortcuts:
        - **b**: Toggle between the Bubble Cursor and the normal system cursor.
        - **q**: Quit the program.

    **Example:** ``bubblecursor --change-thresh 200 --confidence 0.3 --iou 0.4``

    Returns:
        Starts the Qt event loop until exit.
    """
    parser = argparse.ArgumentParser(description="Launch the BubbleCursor overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    add_filter_arguments(parser)
    parser.add_argument('--log-file', default=None, help="Optional JSONL log file path")
    parser.add_argument('--log-cursor-hz', type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument('--annotation-control-file', default=None, help="Use controlled-task annotations instead of live YOLO detection")
    parser.add_argument('--include-text-targets', action='store_true', help="Allow Text annotations to be acquired by the bubble")
    parser.add_argument('--disable-keyboard-quit', action='store_true', help="Disable the overlay-level q quit shortcut; controlled experiments handle quitting")
    args = parser.parse_args()

    if args.annotation_control_file:
        det = FakeTargetFinder(args.annotation_control_file)
    else:
        if args.model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            args.model_path = os.path.join(here, "yolo26s_1280.pt")
        det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)

    if args.model_path is None and not args.annotation_control_file:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "yolo26s_1280.pt")
    cursor_filter = PointFilter2D(args.filter, **filter_kwargs_from_args(args)) if args.filter != "none" else None
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="bubble",
            filter_name=args.filter,
            **filter_kwargs_from_args(args),
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
            detection_source="annotations" if args.annotation_control_file else "yolo",
            annotation_control_file=args.annotation_control_file,
            include_text_targets=bool(args.include_text_targets or args.annotation_control_file),
            keyboard_quit_enabled=not args.disable_keyboard_quit,
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    bubble_cursor(
        det,
        cursor_filter=cursor_filter,
        logger=logger,
        include_text_targets=bool(args.include_text_targets or args.annotation_control_file),
        disable_keyboard_quit=args.disable_keyboard_quit,
    )

if __name__ == "__main__":
    main()
