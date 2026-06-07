"""
Semantic Pointing Demo
======================

This module demonstrates the **Semantic Pointing** interaction technique
using the TargetFinder toolkit.

Notes
-----
- This script is for demonstration purposes only.
- It is **not part of the core TargetFinder API**, but illustrates how
  the toolkit can support advanced interaction techniques.
"""



import os
import sys
import time
import signal
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
from target_finder_toolkit.targetfinder import TargetFinder
from target_finder_toolkit.annotation_detector import AnnotationDetector
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors, disable_mouse_acceleration, restore_mouse_acceleration
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D, add_filter_arguments, filter_kwargs_from_args
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui



__all__ = ["semantic_pointing", "main"]

_SESSION_STOP_REASON = None

# def Omega_rc(u, b):
#     """
#     Raised‐cosine normalisée telle que ∫{-1/2}^{+1/2} Omega_rc(u,b) du = 1.
#     - u : distance normalisée = max(|dx|, |dy|)
#     - b : roll‐off ∈ [0,1]
#     """
#     # Facteur de normalisation A(b) = 1 - b/2 + b/π
#     A = 1.0 - (b / 2.0) + (b / math.pi)
#
#     # Cas plateau (valeur = 1/A) si |u| ≤ (1 - b)/2
#     if u <= (1.0 - b) / 2.0:
#         return 1.0 / A
#
#     # Cas de la transition cosinus si (1 - b)/2 < |u| ≤ (1 + b)/2
#     if u <= (1.0 + b) / 2.0:
#         arg = (u - (1.0 - b) / 2.0) * (math.pi / b)
#         return (0.5 * (1.0 + math.cos(arg))) / A
#
#     # Hors zone d’influence : 0
#     return 0.0

class ControlPanel(QtWidgets.QWidget):
    """Floating parameter panel drawn manually, controlled by keyboard."""
    def __init__(self,detector: TargetFinder):
        super().__init__()
        self.detector = detector

        self.setWindowTitle("Semantic Pointing Controls")
        self.setFixedSize(360,140)

        flages = (
            QtCore.Qt.WindowType.Tool
            |   QtCore.Qt.WindowType.WindowStaysOnTopHint
            |   QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setWindowFlags(flages)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.selected_index = 0

        # paramater definitions
        self.params = [
            {
                "key": "confidence",
                "label": "Confidence",
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "key": "change_thresh",
                "label": "Change threshold",
                "min": 0,
                "max": 300,
                "step": 5,
            },
        ]
    def toggle_visible(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.update()
    
    def move_selection_up(self):
        self.selected_index = max(0,self.selected_index - 1)
        self.update()

    def move_selection_down(self):
        self.selected_index = min(len(self.params)- 1,self.selected_index + 1)
        self.update()
    
    def adjust_current(self,direction):
        params = self.params[self.selected_index]
        key = params["key"]
        step = params["step"]
        min_val = params["min"]
        max_val = params["max"]

        if key == "confidence":
            current = float(self.detector.conf)
            new_value = current + direction * step
            new_value = max(min_val, min(max_val,new_value))
            self.detector.conf = round(new_value,2)

        elif key == "change_thresh":
            current = int(self.detector.change_thresh)
            new_value = current + direction * step
            new_value = max(min_val,min(max_val,new_value))
            self.detector.change_thresh = int(new_value)

        self.update()
    
    def _get_value(self,key):
        if key == "confidence":
            return float(self.detector.conf)
        elif key == "change_thresh":
            return int(self.detector.change_thresh)
        return 0
    
    def _format_value(self, key, value):
        if key == "confidence":
            return f"{value:.2f}"
        return str(int(value))
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # background
        bg_rect = self.rect()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(30,30,30,220))
        painter.drawRoundedRect(bg_rect,12,12)

        # title
        painter.setPen(QtGui.QColor(255,255,255))
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(16,24,"Parameters")

        # content
        label_font = QtGui.QFont()
        label_font.setPointSize(13)
        painter.setFont(label_font)

        start_y = 50
        block_h = 55
        bar_x = 110
        bar_w = 170
        bar_h = 10

        for i,param in enumerate(self.params):
            y = start_y + i * block_h
            value = self._get_value(param["key"])
            min_val = param["min"]
            max_val = param["max"]
            
            if max_val > min_val:
                ratio = (value - min_val) / (max_val - min_val)
            else:
                ratio = 0.0
            ratio = max(0.0, min(1.0,ratio))

            # selected row highlight
            if i == self.selected_index:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(255, 255, 255, 25))
                painter.drawRoundedRect(10, y - 20, self.width() - 20, 40, 8, 8)

            # label
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(20, y, param["label"])

            # current value
            current_text = self._format_value(param["key"], value)
            text_rect = QtCore.QRect(bar_x, y - 20, bar_w, 20)
            painter.setPen(QtGui.QColor(255,255,255))
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, current_text)

            # min / max
            painter.setPen(QtGui.QColor(180, 180, 180))
            painter.drawText(bar_x - 28, y + 18, self._format_value(param["key"], min_val))
            painter.drawText(bar_x + bar_w + 8, y + 18, self._format_value(param["key"], max_val))

            # bar background
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(90, 90, 90))
            painter.drawRoundedRect(bar_x, y + 8, bar_w, bar_h, 5, 5)

            # filled bar
            painter.setBrush(QtGui.QColor(80, 170, 255))
            painter.drawRoundedRect(bar_x, y + 8, int(bar_w * ratio), bar_h, 5, 5)

        painter.end()
        
class SemanticPointing(QtWidgets.QWidget):
    """
    PyQt overlay widget implementing the Semantic Pointing technique.

    Principle
    ---------
    For each frame, bell-shaped weights are aggregated around each
    detected widget. The speed factor `s` varies from 1 to `self.s`,
    slowing down the fake cursor near targets and facilitating selection.

    Parameters
    ----------
    detector : TargetFinder
        Detector providing widget bounding boxes (logical coordinates).
    display : bool, optional
        If True, highlights the target box and its physical motor-space area.
    disable_accel : bool, optional
        If True, disables system mouse acceleration during the session.

    Notes
    -----
    - The overlay is a transparent, always-on-top window.
    - Real clicks are redirected to the fake cursor position during
      click simulation.
    - The Qt timer is set to 10 ms for smooth cursor rendering;
      this does not increase the detector’s inference frequency.
    """
    def __init__(self, detector: TargetFinder, display = False, disable_accel = False, cursor_filter=None, logger=None):
        super().__init__()
        self._is_macos = sys.platform == "darwin"
        self.display = display
        self.disable_accel = disable_accel
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.detector = detector
        detector.overlay_window = self
        if self._is_macos and self.display:
            # Avoid visible flashing when the semantic visual guides are enabled.
            # This changes only the capture/display strategy, not the pointing algorithm.
            self.detector.hide_overlay_during_capture = False
        self._mouse_listener = None
        self._cursor_refresh_timer = None
        self._last_rehide_at = 0.0
        self._current_target = None
        self._current_non_text_target = None
        self._pending_click_target = None
        self._pending_click_raw = None
        self._pending_click_effective = None
        self.enabled = True
        self._simulating_click = False
        self._start_mouse_listener()
        self._start_keyboard_listener()

        # Initial cursor state
        init = QtCore.QPointF(QtGui.QCursor.pos())
        self.prev_real = init
        self.fake_pos = QtCore.QPointF(init)
        self.s = 2  # semantic index reflecting desired speed

        # Full-screen geometry (Qt DPI-aware)
        self.geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(self.geom)

        # Window flags for frameless, always-on-top, click-through & transparent
        flags = (
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        if not self._is_macos:
            flags |= QtCore.Qt.WindowType.Tool
        if sys.platform.startswith("linux"):
            flags |= QtCore.Qt.WindowType.X11BypassWindowManagerHint

        self._base_flags = flags
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Start detection thread
        self.detector.start()

        # refresh to update drawing and especially manage the speed of fake cursor
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10)
        if self._is_macos:
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(self._rehide_cursor)
            self._cursor_refresh_timer.start(16)
        #Separate floating control window
        self.control_panel = ControlPanel(self.detector)
        self.control_panel.move(20,20)
        self.control_panel.hide()
        if self._is_macos:
            self._set_overlay_clickthrough(True)

    def _detector_active(self) -> bool:
        is_active = getattr(self.detector, "is_active", None)
        return not callable(is_active) or bool(is_active())

    # === Paint ===
    def paintEvent(self, event):
        if not self.enabled or not self._detector_active():
            return
        detections = self.detector.get_detections()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        raw_real = QtCore.QPointF(QtGui.QCursor.pos())
        raw_x, raw_y = raw_real.x(), raw_real.y()
        filt_x, filt_y = raw_x, raw_y
        real = QtCore.QPointF(raw_real)

        edge_hit = (
            sys.platform != "darwin"
            and (
                raw_real.x() <= 0
                or raw_real.x() >= self.geom.width() - 1
                or raw_real.y() <= 0
                or raw_real.y() >= self.geom.height() - 1
            )
        )

        # Keep the original Windows/Linux edge-reset behavior so the real cursor
        # does not get trapped at the physical screen border. When a filter is
        # enabled, reset its state to the fake cursor position as well to avoid
        # a feedback loop on the next frame.
        if edge_hit:
            reset_x = float(self.fake_pos.x())
            reset_y = float(self.fake_pos.y())
            QtGui.QCursor.setPos(int(reset_x), int(reset_y))
            raw_real = QtCore.QPointF(QtGui.QCursor.pos())
            raw_x, raw_y = raw_real.x(), raw_real.y()
            real = QtCore.QPointF(raw_real)
            self.prev_real = raw_real
            if self.cursor_filter is not None:
                self.cursor_filter.reset(reset_x, reset_y)

        # update delta
        # during click simulation the real cursor is at a different position implying a movement of fake cursor
        # that should be ignored
        if not self._simulating_click:
            raw_dx = raw_real.x() - self.prev_real.x()
            raw_dy = raw_real.y() - self.prev_real.y()
            self.prev_real = raw_real
            dx, dy = raw_dx, raw_dy
        else:
            dx = dy = 0

        # Sum bell-shaped weights for all detected widgets
        total_bell = 0.0
        for (x, y, w, h, score, cls_id) in detections:
            cx = x + w / 2
            cy = y + h / 2
            ux = abs(self.fake_pos.x() - cx) / w
            uy = abs(self.fake_pos.y() - cy) / h
            u_i = max(ux, uy)
            bell_i = math.log(3) / (math.cosh(math.log(3) * u_i) ** 2)
            total_bell += bell_i

            # # Raised‐cosine
            # beta = 1
            # u_i = max(ux, uy)
            # omega_i = Omega_rc(u_i, beta)  # nouveau « poids » raised‐cosine
            # total_bell += omega_i

        # speed factor that changes from normal speed 1 to 1/s (speed divided by s),
        # so if s=2 the speed is halved and the motor-space width = 2 * screen-space width
        s = 1 + (self.s - 1) * total_bell

        # Move fake cursor applying speed factor
        new_x = self.fake_pos.x() + dx / s
        new_y = self.fake_pos.y() + dy / s

        if self.cursor_filter is not None and not self._is_macos:
            filt_x, filt_y = self.cursor_filter.filter(new_x, new_y)
            new_x, new_y = filt_x, filt_y
        else:
            filt_x, filt_y = new_x, new_y

        # Clamp within screen bounds
        clamped_x = max(0, min(new_x, self.geom.width() - 1))
        clamped_y = max(0, min(new_y, self.geom.height() - 1))
        self.fake_pos.setX(clamped_x)
        self.fake_pos.setY(clamped_y)
        self._current_non_text_target = self.detector.find_detection_for_point(
            float(self.fake_pos.x()),
            float(self.fake_pos.y()),
            include_text=False,
            fallback_nearest=False,
        )
        self._current_target = self.detector.find_detection_for_point(
            float(self.fake_pos.x()),
            float(self.fake_pos.y()),
            include_text=True,
            fallback_nearest=False,
        )
        if self._is_macos and not self._simulating_click:
            sync_point = QtCore.QPoint(int(self.fake_pos.x()), int(self.fake_pos.y()))
            QtGui.QCursor.setPos(sync_point)
            real = QtCore.QPointF(sync_point)
        self.prev_real = real
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=filt_x,
                filtered_y=filt_y,
                technique="semantic",
                filter_name=self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                **(self.cursor_filter.params if self.cursor_filter is not None else {}),
                fake_x=round(self.fake_pos.x(), 3),
                fake_y=round(self.fake_pos.y(), 3),
                scale=round(float(s), 4),
            )

        # If display flag is set highlight the target widget and its semantic area
        if self.display:
            for (x, y, w, h, score, cls_id) in detections:
                rect = QtCore.QRectF(x, y, w, h)
                if rect.contains(self.fake_pos):
                    cx = x + w / 2
                    cy = y + h / 2

                    # widget bounding box
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 2)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawRect(rect)

                    # the size of the widget in the physical space S*W
                    rect2 = QtCore.QRectF(cx - w * self.s / 2, cy - h * self.s / 2, w * self.s, h * self.s)
                    pen2 = QtGui.QPen(QtGui.QColor(255, 0, 0, 200),2, QtCore.Qt.PenStyle.DashLine)
                    painter.setPen(pen2)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawRect(rect2)

                    # movement in the physical space s*w
                    # coordinates within rect2
                    ix = rect2.left() + (self.fake_pos.x() - rect.left()) * self.s
                    iy = rect2.top() + (self.fake_pos.y() - rect.top()) * self.s

                    # FAKE CURSOR in the physical space
                    pen3 = QtGui.QPen(QtGui.QColor(255, 0, 0, 200), 2)
                    painter.setPen(pen3)
                    painter.setBrush(QtGui.QColor(255, 0, 0, 100))
                    painter.drawEllipse(QtCore.QPointF(ix, iy), 6, 6)


        # FAKE CURSOR
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        radius_cursor = 6
        painter.drawEllipse(self.fake_pos.toPoint(), radius_cursor, radius_cursor)
        line_len = 4
        gap = radius_cursor + 1
        cx, cy = int(self.fake_pos.x()), int(self.fake_pos.y())
        painter.drawLine(cx, cy - gap - line_len, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + gap + line_len)
        painter.drawLine(cx + gap, cy, cx + gap + line_len, cy)
        painter.drawLine(cx - gap - line_len, cy, cx - gap, cy)

        # semi-transparent rectangle to block real clicks
        # (because the real cursor position ≠ the fake cursor position)
        if not self._simulating_click and not self._is_macos:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 1))
            painter.drawRect(QtWidgets.QApplication.primaryScreen().geometry())

        painter.end()


    # === Toggle bubble on/off ===
    @QtCore.pyqtSlot()
    def toggle(self):
        self.enabled = not self.enabled

        if self.enabled:
            self.detector.start()
            hide_cursor_everywhere()
            if self.disable_accel:
                disable_mouse_acceleration()
            if sys.platform.startswith("linux"):
                self._set_overlay_clickthrough(False)
            elif self._is_macos:
                self._set_overlay_clickthrough(True)
        else:
            self.detector.stop()
            restore_default_cursors()
            if self.disable_accel:
                restore_mouse_acceleration()
            if sys.platform.startswith("linux") or self._is_macos:
                self._set_overlay_clickthrough(True)
        self.update() # repaint immediately
    
    def _set_overlay_clickthrough(self,enabled):
        if not (sys.platform.startswith("linux") or self._is_macos):
            return
        flags = self._base_flags
        if enabled:
            flags |= QtCore.Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.show()
        if not self._is_macos:
            self.raise_()
        if hasattr(self, "control_panel") and self.control_panel and self.control_panel.isVisible():
            self.control_panel.raise_()
        QtWidgets.QApplication.processEvents()

    @QtCore.pyqtSlot()
    def _rehide_cursor(self):
        if not self.enabled or not self._detector_active():
            return
        hide_cursor_everywhere()
        if self._is_macos:
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(75, hide_cursor_everywhere)

    @QtCore.pyqtSlot()
    def toggle_panel(self):
        self.control_panel.toggle_visible()
    @QtCore.pyqtSlot()
    def panel_up(self):
        self.control_panel.move_selection_up()
    @QtCore.pyqtSlot()
    def panel_down(self):
        self.control_panel.move_selection_down()
    @QtCore.pyqtSlot()
    def panel_left(self):
        self.control_panel.adjust_current(-1)
    @QtCore.pyqtSlot()
    def panel_right(self):
        self.control_panel.adjust_current(1)


    # === Quit ===
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
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.stop()
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        if hasattr(self, "control_panel") and self.control_panel:
            self.control_panel.close()
        restore_default_cursors()
        if self.disable_accel:
            restore_mouse_acceleration()
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
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.stop()
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        if hasattr(self, "control_panel") and self.control_panel:
            self.control_panel.close()
        restore_default_cursors()
        if self.disable_accel:
            restore_mouse_acceleration()
        if self.logger is not None:
            self.logger.log_session_end(reason=reason)
            self.logger.close()
        super().closeEvent(event)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if hasattr(key, "char") and key.char == 'b':
                    QtCore.QMetaObject.invokeMethod(self, "toggle", QtCore.Qt.ConnectionType.QueuedConnection)
                    return
                if hasattr(key, "char") and key.char == 'q':
                    QtCore.QMetaObject.invokeMethod(self, "stop_and_quit", QtCore.Qt.ConnectionType.QueuedConnection)
                    return
                if hasattr(key, "char") and key.char == 's':
                    QtCore.QMetaObject.invokeMethod(self, "toggle_panel", QtCore.Qt.ConnectionType.QueuedConnection)
                    return
            except AttributeError:
                pass
            if self.control_panel.isVisible():
                if key == keyboard.Key.up:
                    QtCore.QMetaObject.invokeMethod(self, "panel_up", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key == keyboard.Key.down:
                    QtCore.QMetaObject.invokeMethod(self, "panel_down", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key == keyboard.Key.left:
                    QtCore.QMetaObject.invokeMethod(self, "panel_left", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key == keyboard.Key.right:
                    QtCore.QMetaObject.invokeMethod(self, "panel_right", QtCore.Qt.ConnectionType.QueuedConnection)
                
        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    def _resolve_click_target(self, raw_x, raw_y):
        """Resolve the logged click target without inventing a nearest fallback.

        Priority:
        1. Exact hit on a non-text control under the semantic cursor.
        2. Exact hit on any target under the semantic cursor.
        3. Exact hit on a non-text control at the raw click point.
        4. Exact hit on any target at the raw click point.

        This keeps empty-space clicks at null while still allowing text / text
        labels to be logged when they are the only exact match.
        """
        if self._current_non_text_target is not None:
            return self._current_non_text_target
        if self._current_target is not None:
            return self._current_target

        points = (
            (float(self.fake_pos.x()), float(self.fake_pos.y())),
            (float(raw_x), float(raw_y)),
        )
        for px, py in points:
            target = self.detector.find_detection_for_point(
                px,
                py,
                include_text=False,
                fallback_nearest=False,
            )
            if target is not None:
                return target
            target = self.detector.find_detection_for_point(
                px,
                py,
                include_text=True,
                fallback_nearest=False,
            )
            if target is not None:
                return target
        return None

    def _snapshot_click_target(self, raw_x, raw_y):
        self._pending_click_raw = [round(float(raw_x), 3), round(float(raw_y), 3)]
        self._pending_click_effective = [
            round(float(self.fake_pos.x()), 3),
            round(float(self.fake_pos.y()), 3),
        ]
        self._pending_click_target = self._resolve_click_target(raw_x, raw_y)

    def _consume_click_snapshot(self):
        target = self._pending_click_target
        raw = self._pending_click_raw
        effective = self._pending_click_effective
        self._pending_click_target = None
        self._pending_click_raw = None
        self._pending_click_effective = None
        return raw, effective, target


    # === Global mouse listener + click simulation ===
    def _start_mouse_listener(self):
        def on_move(x, y):
            if not self._detector_active() or not self._is_macos or not self.enabled:
                return
            now = time.monotonic()
            if now - self._last_rehide_at < 0.016:
                return
            self._last_rehide_at = now
            QtCore.QMetaObject.invokeMethod(self, "_rehide_cursor", QtCore.Qt.ConnectionType.QueuedConnection)

        def on_click(x, y, button, pressed):
            if not self._detector_active():
                return
            if button == button.left and self._is_macos and pressed:
                self._snapshot_click_target(x, y)
                raw, effective, click_target = self._consume_click_snapshot()
                if self.logger is not None:
                    self.logger.log_click(
                        technique="semantic",
                        raw=raw,
                        effective=effective,
                        redirected=False,
                        target=click_target,
                    )
                QtCore.QMetaObject.invokeMethod(self, "_rehide_cursor", QtCore.Qt.ConnectionType.QueuedConnection)
                return
            if pressed and button == button.left:
                self._snapshot_click_target(x, y)
                # simulate in the Qt thread
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection)
        self._mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        self._mouse_listener.start()

    @QtCore.pyqtSlot()
    def _simulate_click(self):
        if not self._detector_active():
            return
        if self._is_macos:
            return
        if self.enabled:
            # simulate the click by removing the semi-transparent rectangle that blocks clicks
            pyautogui.mouseUp(button='left') # simulate button release
            self._mouse_listener.stop() # stop listener
            self._simulating_click = True # activate the flag "windows"
            self.update()  # request immediate Qt repaint
            QtWidgets.QApplication.processEvents()  # force immediate event processing
            if sys.platform.startswith("linux") or self._is_macos:
                self._set_overlay_clickthrough(True)
            QtGui.QCursor.setPos(int(self.fake_pos.x()), int(self.fake_pos.y())) # move real cursor
                                                                                 # note: this doesn’t work on Windows
                                                                                 # if the semi-transparent rectangle isn’t drawn
            # pyautogui.moveTo(self.fake_pos.x() / self.detector.sx, self.fake_pos.y() / self.detector.sy)
            # pyautogui.moveTo is slower when the real cursor is stuck at the edge
            try:
                pyautogui.click()
            except pyautogui.FailSafeException:
                # to prevent the script from crashing when the mouse hits a corner
                pass
            finally:
                if sys.platform.startswith("linux") or self._is_macos:
                    self._set_overlay_clickthrough(False)
                self._simulating_click = False # deactivate the flag and resynchronize
                self.prev_real = QtCore.QPointF(QtGui.QCursor.pos())
                raw, effective, click_target = self._consume_click_snapshot()
                if raw is None or effective is None:
                    raw = [round(self.prev_real.x(), 3), round(self.prev_real.y(), 3)]
                    effective = [round(self.fake_pos.x(), 3), round(self.fake_pos.y(), 3)]
                    click_target = self._resolve_click_target(self.prev_real.x(), self.prev_real.y())
                if self.logger is not None:
                    self.logger.log_click(
                        technique="semantic",
                        raw=raw,
                        effective=effective,
                        redirected=True,
                        target=click_target,
                    )
                if self._is_macos:
                    self._rehide_cursor()
                    self.raise_()
                    if hasattr(self, "control_panel") and self.control_panel and self.control_panel.isVisible():
                        self.control_panel.raise_()
                self._start_mouse_listener() # restart the listener

def semantic_pointing(detector: TargetFinder, display = False, disable_accel=False, cursor_filter=None, logger=None):
    """Launch the Semantic Pointing overlay with a given detector.

    The overlay draws a **fake cursor** whose speed dynamically changes
    depending on proximity to detected widgets. This alters their
    motor-space representation, making them easier to acquire.

    Args:
        detector (TargetFinder): Initialized TargetFinder (YOLO model loaded).
        display (bool, optional): If True, highlight the target box and its
            motor-space area (S×W). Defaults to False.
        disable_accel (bool, optional): If True, disable OS mouse
            acceleration. Defaults to False.

    Keyboard shortcuts:
        - **b**: Toggle between Semantic Pointing and the normal system cursor.
        - **q**: Quit the program.

    Returns:
        None: Blocks until the Qt application is closed.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if disable_accel:
        disable_mouse_acceleration()
    ov = SemanticPointing(detector, display, disable_accel, cursor_filter=cursor_filter, logger=logger)
    ov.show()
    raise_macos_window_above_system_ui(ov, level_offset=1)
    is_active = getattr(detector, "is_active", None)
    if not callable(is_active) or bool(is_active()):
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
    if disable_accel:
        restore_mouse_acceleration()
    if logger is not None:
        logger.log_session_end(reason=_SESSION_STOP_REASON or "app_exit")
        logger.close()
    sys.exit(exit_code)


# CLI usage
def main():
    """Command-line entry point for the Semantic Pointing demo.

    CLI arguments:
        --model-path (str, optional): Path to YOLO .pt weights.
            Defaults to ``yolo26s_1280.pt`` in the package.
        --change-thresh (int, optional): Screen-change L2 threshold on a
            down-scaled frame. Higher = less sensitive. Default: ``100``.
        --capture-interval (float, optional): Delay in seconds between captures.
            Lower = higher refresh rate, more CPU/GPU. Default: ``1/30``.
        --confidence (float, optional): YOLO confidence threshold in ``[0, 1]``.
            Default: ``0.28``.
        --iou (float, optional): IoU threshold for YOLO NMS in ``[0, 1]``.
            Controls overlap merging. Default: ``0.3``.
        --disable-accel (flag): Disable system mouse acceleration.  

        --display (flag): Show the target bounding box and its motor-space area.

    Keyboard shortcuts:
        - **b**: Toggle between Semantic Pointing and the normal system cursor.
        - **q**: Quit the program.

    **Example:** ``semanticpointing --disable-accel --display --confidence 0.4``

    Returns:
        Starts the Qt event loop until exit.
    """
    parser = argparse.ArgumentParser(description="Launch the Semantic Pointing overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    parser.add_argument('--disable-accel', action='store_true', help="Disable system mouse acceleration")
    parser.add_argument('--display', action='store_true', help="Enable on-screen display of target boxe and physical area")
    add_filter_arguments(parser)
    parser.add_argument('--log-file', default=None, help="Optional JSONL log file path")
    parser.add_argument('--log-cursor-hz', type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument('--annotation-control-file', default=None, help="Use controlled-task annotations instead of live YOLO detection")
    args = parser.parse_args()

    if args.annotation_control_file:
        det = AnnotationDetector(args.annotation_control_file)
    else:
        if args.model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            args.model_path = os.path.join(here, "yolo26s_1280.pt")
        det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    cursor_filter = PointFilter2D(args.filter, **filter_kwargs_from_args(args)) if args.filter != "none" else None
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="semantic",
            filter_name=args.filter,
            **filter_kwargs_from_args(args),
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
            display=args.display,
            disable_accel=args.disable_accel,
            detection_source="annotations" if args.annotation_control_file else "yolo",
            annotation_control_file=args.annotation_control_file,
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    semantic_pointing(det, args.display, args.disable_accel, cursor_filter=cursor_filter, logger=logger)

if __name__ == "__main__":
    main()
