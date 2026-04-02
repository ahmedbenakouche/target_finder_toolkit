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
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors, disable_mouse_acceleration, restore_mouse_acceleration
import math

__all__ = ["semantic_pointing", "main"]

# def Omega_rc(u, b):
#     """
#     Raised-cosine normalisée telle que ∫{-1/2}^{+1/2} Omega_rc(u,b) du = 1.
#     - u : distance normalisée = max(|dx|, |dy|)
#     - b : roll-off ∈ [0,1]
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
#     # Hors zone d'influence : 0
#     return 0.0

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
      this does not increase the detector's inference frequency.
    """
    def __init__(self, detector: TargetFinder, display = False, disable_accel = False):
        super().__init__()
        self._is_macos = sys.platform == "darwin"
        self.display = display
        self.disable_accel = disable_accel
        self.detector = detector
        detector.overlay_window = self
        if self._is_macos and self.display:
            self.detector.hide_overlay_during_capture = False
        self._mouse_listener = None
        self._cursor_refresh_timer = None
        self._last_rehide_at = 0.0
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
        self._timer.start(10) # 10 ms
        if self._is_macos:
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(self._rehide_cursor)
            self._cursor_refresh_timer.start(16)
            self._set_overlay_clickthrough(True)

    def _screen_for_point(self, x: int, y: int):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return QtWidgets.QApplication.primaryScreen()
        return app.screenAt(QtCore.QPoint(int(x), int(y))) or QtWidgets.QApplication.primaryScreen()

    def _in_system_reserved_area(self, x: int, y: int) -> bool:
        if not self._is_macos:
            return False
        screen = self._screen_for_point(x, y)
        if screen is None:
            return False
        point = QtCore.QPoint(int(x), int(y))
        return not screen.availableGeometry().contains(point)

    # === Paint ===
    def paintEvent(self, event):
        if not self.enabled:
            return
        detections = self.detector.get_detections()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        raw_real = QtCore.QPointF(QtGui.QCursor.pos())
        real = QtCore.QPointF(raw_real)

        # If the real cursor is stuck at an edge
        if (
            not self._is_macos
            and (
                real.x() <= 0 or real.x() >= self.geom.width() - 1
                or real.y() <= 0 or real.y() >= self.geom.height() - 1
            )
        ):
            QtGui.QCursor.setPos(int(self.fake_pos.x()), int(self.fake_pos.y()))
            real = QtCore.QPointF(QtGui.QCursor.pos())
            self.prev_real = real

        # update delta
        # during click simulation the real cursor is at a different position implying a movement of fake cursor
        # that should be ignored
        if not self._simulating_click:
            dx = real.x() - self.prev_real.x()
            dy = real.y() - self.prev_real.y()
            self.prev_real = real
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

            # # Raised-cosine
            # beta = 1
            # u_i = max(ux, uy)
            # omega_i = Omega_rc(u_i, beta)  # nouveau « poids » raised-cosine
            # total_bell += omega_i

        # speed factor that changes from normal speed 1 to 1/s (speed divided by s),
        # so if s=2 the speed is halved and the motor-space width = 2 * screen-space width
        s = 1 + (self.s - 1) * total_bell

        # Move fake cursor applying speed factor
        new_x = self.fake_pos.x() + dx / s
        new_y = self.fake_pos.y() + dy / s

        # Clamp within screen bounds
        clamped_x = max(0, min(new_x, self.geom.width() - 1))
        clamped_y = max(0, min(new_y, self.geom.height() - 1))
        self.fake_pos.setX(clamped_x)
        self.fake_pos.setY(clamped_y)
        if self._is_macos and not self._simulating_click:
            sync_point = QtCore.QPoint(int(self.fake_pos.x()), int(self.fake_pos.y()))
            QtGui.QCursor.setPos(sync_point)
            real = QtCore.QPointF(sync_point)
        self.prev_real = real

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

    def _set_overlay_clickthrough(self, enabled):
        if not (sys.platform.startswith("linux") or self._is_macos):
            return
        flags = self._base_flags
        if enabled:
            flags |= QtCore.Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.show()
        if not self._is_macos:
            self.raise_()
        QtWidgets.QApplication.processEvents()

    @QtCore.pyqtSlot()
    def _rehide_cursor(self):
        if not self.enabled:
            return
        hide_cursor_everywhere()
        if self._is_macos:
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(75, hide_cursor_everywhere)

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
        restore_default_cursors()
        if self.disable_accel:
            restore_mouse_acceleration()
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
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.stop()
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        restore_default_cursors()
        if self.disable_accel:
            restore_mouse_acceleration()
        super().closeEvent(event)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'b':
                    QtCore.QMetaObject.invokeMethod(self, "toggle", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key.char == 'q':
                    QtCore.QMetaObject.invokeMethod(self, "stop_and_quit", QtCore.Qt.ConnectionType.QueuedConnection)
            except AttributeError:
                pass
        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()


    # === Global mouse listener + click simulation ===
    def _start_mouse_listener(self):
        def on_move(x, y):
            if not self._is_macos or not self.enabled:
                return
            now = time.monotonic()
            if now - self._last_rehide_at < 0.016:
                return
            self._last_rehide_at = now
            QtCore.QMetaObject.invokeMethod(self, "_rehide_cursor", QtCore.Qt.ConnectionType.QueuedConnection)

        def on_click(x, y, button, pressed):
            if button == button.left and self._is_macos and pressed:
                QtCore.QMetaObject.invokeMethod(self, "_rehide_cursor", QtCore.Qt.ConnectionType.QueuedConnection)
            if pressed and button == button.left:
                if self._in_system_reserved_area(x, y):
                    return
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection)
        self._mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        self._mouse_listener.start()

    @QtCore.pyqtSlot()
    def _simulate_click(self):
        if self._is_macos:
            return
        if self.enabled:
            pos = QtGui.QCursor.pos()
            if self._in_system_reserved_area(pos.x(), pos.y()):
                return
            pyautogui.mouseUp(button='left')
            self._mouse_listener.stop()
            self._simulating_click = True
            self.update()
            QtWidgets.QApplication.processEvents()
            if sys.platform.startswith("linux") or self._is_macos:
                self._set_overlay_clickthrough(True)
            QtGui.QCursor.setPos(int(self.fake_pos.x()), int(self.fake_pos.y()))
            try:
                pyautogui.click()
            except pyautogui.FailSafeException:
                pass
            finally:
                if sys.platform.startswith("linux") or self._is_macos:
                    self._set_overlay_clickthrough(False)
                self._simulating_click = False
                self.prev_real = QtCore.QPointF(QtGui.QCursor.pos())
                self._start_mouse_listener()

def semantic_pointing(detector: TargetFinder, display = False, disable_accel=False):
    """Launch the Semantic Pointing overlay with a given detector.

    The overlay draws a **fake cursor** whose speed dynamically changes
    depending on proximity to detected widgets. This alters their
    motor-space representation, making them easier to acquire.

    Args:
        detector (TargetFinder): Initialized TargetFinder (YOLO model loaded).
        display (bool, optional): If True, highlight the target box and its
            motor-space area (S*W). Defaults to False.
        disable_accel (bool, optional): If True, disable OS mouse
            acceleration. Defaults to False.

    Keyboard shortcuts:
        - **b**: Toggle between Semantic Pointing and the normal system cursor.
        - **q**: Quit the program.

    Returns:
        None: Blocks until the Qt application is closed.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hide_cursor_everywhere()
    if disable_accel:
        disable_mouse_acceleration()
    ov  = SemanticPointing(detector, display, disable_accel)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    signal.signal(signal.SIGTERM, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if disable_accel:
        restore_mouse_acceleration()
    sys.exit(exit_code)


# CLI usage
def main():
    """Command-line entry point for the Semantic Pointing demo.

    CLI arguments:
        --model-path (str, optional): Path to YOLO .pt weights.
            Defaults to ``best.pt`` in the package.
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
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0-1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0-1.0)")
    parser.add_argument('--disable-accel', action='store_true', help="Disable system mouse acceleration")
    parser.add_argument('--display', action='store_true', help="Enable on-screen display of target boxe and physical area")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    semantic_pointing(det, args.display, args.disable_accel)

if __name__ == "__main__":
    main()
