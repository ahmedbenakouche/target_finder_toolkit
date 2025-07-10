# targetfinder.py

import os
import sys
import time
import threading
import numpy as np
import cv2
import mss
from ultralytics import YOLO
from PyQt6 import QtWidgets, QtGui, QtCore
import argparse

# Mapping of class IDs to names
CLASS_NAMES = {
    0: "Button",
    1: "ToggleButton",
    2: "Hyperlink",
    3: "Text",
    4: "TextInput",
    5: "Slider"
}

class TargetFinder:
    """Continuous widget detection and storage of bounding boxes"""
    def __init__(self, model_path=None, change_thresh=100, capture_interval=1/30, confidence=0.28):
        # Ensure a QApplication exists for primaryScreen()
        if QtWidgets.QApplication.instance() is None:
            self._app = QtWidgets.QApplication(sys.argv)
        if model_path is None:
            here = os.path.dirname(__file__)
            model_path = os.path.join(here, "best.pt")
        self.model = YOLO(model_path)
        self.change_thresh = change_thresh
        self.interval = capture_interval
        self.conf = confidence
        self.detections = []   # (x, y, w, h, score, cls_id)
        self.sx = 1.0
        self.sy = 1.0
        self._stop = False
        self.overlay_window = None

    def start(self):
        """Start the capture+inference loop in a separate thread"""
        self._stop = False
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        """Stop the detection loop"""
        self._stop = True

    def get_detections(self):
        """Return the list [(x,y,w,h,score,cls_id), ...]"""
        return self.detections

    def _capture_loop(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        prev_small = None

        while not self._stop:
            # Low-res screenshot for change detection
            frame = np.array(sct.grab(monitor))[..., :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

            if prev_small is None or cv2.norm(small, prev_small, cv2.NORM_L2) > self.change_thresh:
                prev_small = small.copy()

                # Hide the overlay before full-resolution capture
                if self.overlay_window:
                    QtCore.QMetaObject.invokeMethod(self.overlay_window, "hide",
                        QtCore.Qt.ConnectionType.QueuedConnection)
                time.sleep(0.01)  # allow time for the window to hide

                # Full-resolution capture without overlay
                full = np.array(sct.grab(monitor))[..., :3]

                # Re-show the overlay
                if self.overlay_window:
                    QtCore.QMetaObject.invokeMethod(self.overlay_window, "show",
                        QtCore.Qt.ConnectionType.QueuedConnection)

                # YOLO inference
                results = self.model(full, conf=self.conf, verbose = False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()

                # DPI-aware scaling
                h_phy, w_phy = full.shape[:2]
                screen_geom = QtWidgets.QApplication.primaryScreen().geometry()
                self.sx = screen_geom.width() / w_phy
                self.sy = screen_geom.height() / h_phy

                self.detections = [(int(x1 * self.sx), int(y1 * self.sy), int((x2 - x1) * self.sx), int((y2 - y1) * self.sy), float(score),
                                    int(cls_id)) for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids)]

            # Pause to limit loop frequency
            # Prevents saturating RAM and CPU with too many captures
            # On very powerful machines with lots of RAM we can set self.interval = 0
            time.sleep(self.interval)


class OverlayWindow(QtWidgets.QWidget):
    """Transparent window for drawing bounding boxes"""
    PALETTE = [
        QtGui.QColor(0, 255, 0, 200),    # Button → green
        QtGui.QColor(255, 0, 0, 200),    # ToggleButton → red
        QtGui.QColor(0, 0, 255, 200),    # Hyperlink → blue
        QtGui.QColor(255, 255, 0, 200),  # Text → yellow
        QtGui.QColor(255, 0, 255, 200),  # TextInput → magenta
        QtGui.QColor(0, 255, 255, 200),  # Slider → cyan
    ]

    def __init__(self, detector: TargetFinder):
        super().__init__()
        self.detector = detector
        # Link overlay to the detector so it can hide/show it
        detector.overlay_window = self

        # Full-screen geometry (Qt DPI-aware)
        geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(geom)

        # Window flags for frameless, always-on-top, click-through & transparent
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Start detection thread
        self.detector.start()

        # Timer to force update() at the same frequency as detection
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(int(self.detector.interval * 1000))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        for x, y, w, h, score, cls_id in self.detector.get_detections():
            color = self.PALETTE[cls_id % len(self.PALETTE)]
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

            # Label
            label = f"{score:.2f}"
            fm = painter.fontMetrics()
            tw, th = fm.horizontalAdvance(label), fm.height()
            painter.fillRect(x, y - th, tw + 4, th, QtGui.QColor(0, 0, 0, 120))
            painter.drawText(x + 2, y - 2, label)

        painter.end()


def show_detections(detector: TargetFinder):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ov  = OverlayWindow(detector)
    ov.show()
    sys.exit(app.exec())


# CLI usage
def main():
    parser = argparse.ArgumentParser(description="Launch the TargetFinder overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30,
                        help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(__file__)
        args.model_path = os.path.join(here, "best.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence)
    show_detections(det)

if __name__ == "__main__":
    main()


