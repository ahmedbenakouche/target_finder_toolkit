# bubblecursor.py

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
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors
from importlib import resources


class BubbleCursor(QtWidgets.QWidget):
    def __init__(self, detector: TargetFinder):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        self._last_target = None  # to store the active target
        self._mouse_listener = None
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
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
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

    # === Paint ===
    def paintEvent(self, event):
        if not self.bubble_enabled or not self.detector.get_detections():
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Pointer position (logical coordinates)
        pos = QtGui.QCursor.pos()
        cx, cy = pos.x(), pos.y()
                
        # Compute distances from pointer to each box edge
        distances = []
        for x, y, w, h, *_ in self.detector.get_detections():
            cx_box = x + w/2
            cy_box = y + h/2

            # Intersecting Distance (IntD)
            dx = max(0.0, abs(cx - cx_box) - w/2)
            dy = max(0.0, abs(cy - cy_box) - h/2)
            IntD = math.hypot(dx, dy)

            distances.append((IntD, cx_box, cy_box, w, h))

        if distances:

            distances.sort(key=lambda t: t[0])
            IntD1, tx, ty, w, h = distances[0]
            self._last_target = (
            tx / self.detector.sx, ty / self.detector.sy, w / self.detector.sx, h / self.detector.sy)

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

            # FAKE CURSOR
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            radius_cursor = 6
            center = QtCore.QPointF(cx, cy)
            painter.drawEllipse(center, radius_cursor, radius_cursor)
            line_len = 4
            gap = radius_cursor + 1
            painter.drawLine(cx, cy - gap - line_len, cx, cy - gap)
            painter.drawLine(cx, cy + gap, cx, cy + gap + line_len)
            painter.drawLine(cx + gap, cy, cx + gap + line_len, cy)
            painter.drawLine(cx - gap - line_len, cy, cx - gap, cy)

            painter.end()
        else:
            return

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
        restore_default_cursors()
        os._exit(0)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'b':
                    QtCore.QMetaObject.invokeMethod(self, "toggle_bubble", QtCore.Qt.ConnectionType.QueuedConnection)
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
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(int, x), QtCore.Q_ARG(int, y))
        self._mouse_listener = mouse.Listener(on_click=on_click)
        self._mouse_listener.start()

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        if self._last_target and self.bubble_enabled:
            tx, ty, w, h = self._last_target

            # If click inside detected rectangle let it pass
            if tx-w/2 <= orig_x <= tx+w/2 and ty-h/2 <= orig_y <= ty+h/2:
                return

            # Otherwise simulate target click
            self._mouse_listener.stop()  # stop listener
            try:
                pyautogui.mouseUp(button='left')  # simulate button release
                pyautogui.moveTo(tx, ty) # move to and click the targeted widget
                pyautogui.click()
                pyautogui.moveTo(orig_x, orig_y) # move the mouse back to its original position
            except pyautogui.FailSafeException:
                # to prevent the script from crashing when the mouse hits a corner
                pass
            finally:
                self._start_mouse_listener() # restart the listener


def bubble_cursor(detector: TargetFinder):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hide_cursor_everywhere()
    ov = BubbleCursor(detector)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    sys.exit(exit_code)


# CLI usage
def main():
    parser = argparse.ArgumentParser(description="Launch the BubbleCursor overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(__file__)
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    bubble_cursor(det)

if __name__ == "__main__":
    main()
