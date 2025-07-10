# semanticpointing.py

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

class SemanticPointing(QtWidgets.QWidget):
    def __init__(self, detector: TargetFinder, display = False, disable_accel = False):
        super().__init__()
        self.display = display
        self.disable_accel = disable_accel
        self.detector = detector
        detector.overlay_window = self
        self._mouse_listener = None
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
                | QtCore.Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Start detection thread
        self.detector.start()

        # refresh to update drawing and especially manage the speed of fake cursor
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10) # 10 ms

    # === Paint ===
    def paintEvent(self, event):
        if not self.enabled or not self.detector.get_detections():
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        real = QtCore.QPointF(QtGui.QCursor.pos())

        # If the real cursor is stuck at an edge
        if (real.x() <= 0 or real.x() >= self.geom.width() - 1
                or real.y() <= 0 or real.y() >= self.geom.height() - 1):
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
        for (x, y, w, h, score, cls_id) in self.detector.get_detections():
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

        # Clamp within screen bounds
        clamped_x = max(0, min(new_x, self.geom.width() - 1))
        clamped_y = max(0, min(new_y, self.geom.height() - 1))
        self.fake_pos.setX(clamped_x)
        self.fake_pos.setY(clamped_y)
        self.prev_real = real

        # If display flag is set highlight the target widget and its semantic area
        if self.display:
            for (x, y, w, h, score, cls_id) in self.detector.get_detections():
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
                    pen3 = QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 2)
                    painter.setPen(pen3)
                    painter.setBrush(QtGui.QColor(255, 255, 0, 100))
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
        if not self._simulating_click:
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
        else:
            self.detector.stop()
            restore_default_cursors()
            if self.disable_accel:
                restore_mouse_acceleration()
        self.update() # repaint immediately

    # === Quit ===
    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        restore_default_cursors()
        if self.disable_accel:
            restore_mouse_acceleration()
        os._exit(0)

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
        def on_click(x, y, button, pressed):
            if pressed and button == button.left:
                # simulate in the Qt thread
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection)
        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._mouse_listener.start()

    @QtCore.pyqtSlot()
    def _simulate_click(self):
        if self.enabled:
            # simulate the click by removing the semi-transparent rectangle that blocks clicks
            pyautogui.mouseUp(button='left') # simulate button release
            self._mouse_listener.stop() # stop listener
            self._simulating_click = True # activate the flag
            self.update()  # request immediate Qt repaint
            QtWidgets.QApplication.processEvents()  # force immediate event processing
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
                self._simulating_click = False # deactivate the flag and resynchronize
                self.prev_real = QtCore.QPointF(QtGui.QCursor.pos())
                self._start_mouse_listener() # restart the listener

def semantic_pointing(detector: TargetFinder, display = False, disable_accel=False):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hide_cursor_everywhere()
    if disable_accel:
        disable_mouse_acceleration()
    ov  = SemanticPointing(detector, display, disable_accel)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if disable_accel:
        restore_mouse_acceleration()
    sys.exit(exit_code)


# CLI usage
def main():
    parser = argparse.ArgumentParser(description="Launch the Semantic Pointing overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30,
                        help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--display', action='store_true',
                        help="Enable on-screen display of target boxe and physical area")
    parser.add_argument('--disable-accel', action='store_true', help="Disable system mouse acceleration")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(__file__)
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence)
    semantic_pointing(det, args.display, args.disable_accel)

if __name__ == "__main__":
    main()