"""
targetfinder.py
===============

Continuous widget detection from screen captures or static images.

Core features
-------------
- Real-time screen capture and YOLO-based detection of GUI widgets.
- Callback mechanism to provide the current detections, the captured frame
  (optional), and the lists of added/removed detections.
- Optional PyQt overlay to visualize bounding boxes in real time.
- One-shot detection on static images with optional annotated export.

Notes
-----
This is the **main entry point** of the toolkit. Other modules
(`postprocess.py`, `mouse_utils.py`) provide utilities and helpers,
while `bubblecursor.py` and `semanticpointing.py` are demonstration examples.
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


def _iou(a, b):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    a : tuple (x, y, w, h)
    b : tuple (x, y, w, h)

    Returns
    -------
    float
        IoU value in [0, 1].
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a, area_b = aw * ah, bw * bh
    return inter / float(area_a + area_b - inter + 1e-9)



def _require_number(name, val):
    """Convert value to float or raise a TypeError with context."""
    try:
        return float(val)
    except Exception:
        raise TypeError(f"{name} must be a number, got {type(val).__name__}")

def _require_between(name, val, lo, hi, inclusive=True):
    """
    Validate that a number is within a range.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    val : any
        Value to check.
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    inclusive : bool, optional
        If True, interval is closed [lo, hi].
        If False, interval is open (lo, hi).

    Returns
    -------
    float
        The validated numeric value.
    """
    v = _require_number(name, val)
    if inclusive:
        if not (lo <= v <= hi):
            raise ValueError(f"{name} ({v}) must be in [{lo}, {hi}]")
    else:
        if not (lo < v < hi):
            raise ValueError(f"{name} ({v}) must be in ({lo}, {hi})")
    return v


class TargetFinder:
    """
    Continuous screen capture + widget detection using YOLO.

    Provides:
    - real-time loop over screen captures with change detection
    - persistent IDs for tracked detections
    - one-shot image detection with optional export
    """
    
    def __init__(self, model_path=None, change_thresh=100, capture_interval=1/30, confidence=0.28, iou = 0.3):
        """
        Initialize the TargetFinder detector.

        Parameters
        ----------
        model_path : str or None
            Path to YOLO .pt file. If None, defaults to 'best.pt' in this folder.
        change_thresh : int
            Threshold for triggering re-detection on screen changes.
        capture_interval : float
            Interval between screen captures (seconds).
        confidence : float
            YOLO confidence threshold (0.0–1.0).
        iou : float
            YOLO IoU threshold (0.0–1.0).
        """
        
        # Ensure a QApplication exists for primaryScreen()
        if QtWidgets.QApplication.instance() is None:
            self._app = QtWidgets.QApplication(sys.argv)

        # Resolve model path (fallback to local best.pt)
        if model_path is None:
            here = os.path.dirname(__file__)
            model_path = os.path.join(here, "best.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model_path not found: {model_path}")

        # Load YOLO model
        self.model = YOLO(model_path)

        # Validate and store core thresholds/intervals
        self.change_thresh = int(_require_between("change_thresh", change_thresh, 0, 1e12))
        self.interval      = float(_require_between("capture_interval", capture_interval, 0, 1e6))
        self.conf          = float(_require_between("confidence", confidence, 0.0, 1.0))
        self.iou           = float(_require_between("iou", iou, 0.0, 1.0))

        # Public snapshot used by the overlay: list of tuples (x, y, w, h, score, cls_id)
        self.detections = []

        # Screen scaling (physical capture → Qt screen geometry)
        self.sx = 1.0
        self.sy = 1.0

        # Control flags for the background loop
        self._stop = False
        self.overlay_window = None  # will be set by OverlayWindow

        # Tracking state to keep stable IDs between frames
        self._prev_det_dicts = []   # previous frame detection dicts (with "id")
        self._next_id = 1           # next stable ID to assign

        # Callback configuration
        self._on_change = None      # callable or None
        self._with_frame = False    # whether to forward the frame to callback
        self._diff_iou = 0.5        # IoU threshold for added/removed + ID conservation

    def set_callback(self, fn, with_frame=False, diff_iou=0.5):
        """
        Register a callback to be invoked after each detection update.

        Parameters
        ----------
        fn : callable or None
            Signature: fn(detections, added, removed, frame)
            - detections: current list[dict]
            - added: newly appeared list[dict]
            - removed: disappeared list[dict]
            - frame: RGB np.ndarray or None (depending on with_frame)
        with_frame : bool, optional
            If True, pass the full-resolution RGB frame to the callback.
        diff_iou : float, optional
            IoU threshold used to determine added/removed and to keep stable IDs.
        """
        if fn is not None and not callable(fn):
            raise TypeError("on_change must be a callable or None")
        self._on_change = fn
        self._with_frame = bool(with_frame)
        self._diff_iou = float(_require_between("diff_iou", diff_iou, 0.0, 1.0))


    def start(self):
        """Start the capture+inference loop in a separate thread"""
        self._stop = False
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        """Stop the detection loop"""
        self._stop = True

    def get_detections(self):
        """
        Get current detections as a list of tuples.

        Notes
        -----
        The coordinates are in **logical screen space**, i.e. they already
        include DPI-aware scaling to match the primary screen geometry.

        Returns
        -------
        list[tuple]
            Each entry is (x, y, w, h, score, cls_id).
        """
        return self.detections

    def _build_dicts(self, boxes_xyxy, scores, class_ids, scale_x, scale_y):
        """
        Build detection dicts from YOLO raw outputs.

        Notes
        -----
        The (x, y, width, height) values are converted from YOLO pixel
        coordinates and multiplied by `scale_x`/`scale_y` so they are
        expressed in **logical screen space** (DPI-aware).

        Returns
        -------
        list[dict]
            Each dict has: id, x, y, width, height, score, class_id, class_name.
        """
        dets = []
        for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, class_ids):
            dets.append({
                "id": None,
                "x": int(x1 * scale_x),
                "y": int(y1 * scale_y),
                "width": int((x2 - x1) * scale_x),
                "height": int((y2 - y1) * scale_y),
                "score": float(score),
                "class_id": int(cls_id),
                "class_name": CLASS_NAMES.get(int(cls_id), str(int(cls_id))),
            })
        return dets

    def _assign_ids_and_diff(self, prev, curr):
        """
        Assign stable IDs across frames by class-aware IoU matching, and compute
        the sets of added/removed detections.

        ID lifecycle / policy
        ---------------------
        - Monotonic IDs: `self._next_id` increases over time; we never decrement or
          reuse past IDs in the current session.
        - No re-identification of disappeared widgets: if a widget is absent at
          this frame (no IoU match ≥ self._diff_iou with same class), it is marked
          as *removed*. Its ID is considered *expired* and will NOT be re-assigned to
          any future detection — even if a visually similar widget reappears later.
          (We do not track through long occlusions or disappearance/reappearance.)
        - Class-aware matching: candidates are matched only within the same class_id.
        - Greedy matching by best IoU: for each current detection, we pick the
          unmatched previous detection with the highest IoU (if ≥ self._diff_iou).
        - New IDs for unmatched current detections: anything not matched to a
          previous detection receives a fresh, never-before-used ID.

        Parameters
        ----------
        prev : list[dict]
            Previous-frame detections with assigned "id".
        curr : list[dict]
            Current-frame detections (their "id" will be set/kept here).

        Returns
        -------
        (added, removed) : tuple(list[dict], list[dict])
            - added: detections in `curr` that received *new* IDs (unmatched).
            - removed: detections in `prev` that were *not matched* this frame.
        """
        used_prev = set()
        used_curr = set()

        # Greedy IoU match per class (class-aware ID conservation)
        for ci, cd in enumerate(curr):
            c_rect = (cd["x"], cd["y"], cd["width"], cd["height"])
            c_cls = cd["class_id"]
            best_j, best_iou = -1, 0.0
            for pj, pd in enumerate(prev):
                if pj in used_prev or pd["class_id"] != c_cls:
                    continue
                p_rect = (pd["x"], pd["y"], pd["width"], pd["height"])
                iou = _iou(c_rect, p_rect)
                if iou > best_iou:
                    best_iou, best_j = iou, pj

            # If match is good enough, keep the previous ID
            if best_j >= 0 and best_iou >= self._diff_iou:
                used_prev.add(best_j)
                used_curr.add(ci)
                curr[ci]["id"] = prev[best_j]["id"]  # conserve ID

        # Assign new IDs to unmatched current detections
        for i, cd in enumerate(curr):
            if i not in used_curr:
                cd["id"] = self._next_id
                self._next_id += 1

        # Compute sets for added/removed (by unmatched indices)
        added = [cd for i, cd in enumerate(curr) if i not in used_curr]
        removed = [pd for j, pd in enumerate(prev) if j not in used_prev]
        return added, removed

    def detect_image(self, image_path, save_annotated=False, save_json=False):
        """
        Run detection on a single image and optionally save results.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        save_annotated : bool
            If True, saves an annotated version of the image at the same location
            with suffix '_annotated.png'.
        save_json : bool
            If True, saves detections as a JSON file at the same location
            with suffix '_detections.json'.

        Returns
        -------
        detections : list of dict
            List of detections in the format:
            {id, x, y, width, height, score, class_id, class_name}
        """
        # Read image (BGR as expected by OpenCV/Ultralytics)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Single-shot YOLO inference
        results = self.model(img_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        # Build detection dicts in image coordinates (no scaling)
        detections = self._build_dicts(
            boxes_xyxy=boxes,
            scores=scores,
            class_ids=class_ids,
            scale_x=1.0,
            scale_y=1.0
        )
        # Assign unique IDs (no temporal tracking in one-shot mode)
        for d in detections:
            d["id"] = int(self._next_id)
            self._next_id += 1

        # Optionally save annotated image (PNG)
        if save_annotated:
            annotated = img_bgr.copy()
            for d in detections:
                x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{d['class_name']}:{d['score']:.2f}"
                cv2.putText(annotated, label, (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = image_path.rsplit(".", 1)[0] + "_annotated.png"
            cv2.imwrite(out_path, annotated)

        # Optionally save JSON detections
        if save_json:
            import json
            out_json = image_path.rsplit(".", 1)[0] + "_detections.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(detections, f, ensure_ascii=False, indent=2)

        return detections


    def _capture_loop(self):
        """Internal loop: capture screen, run inference, invoke callback if needed."""
        # Create an MSS grabber and select primary monitor
        sct = mss.mss()
        monitor = sct.monitors[0]
        prev_small = None     # last low-res grayscale for change detection

        while not self._stop:
            # Low-res screenshot for change detection
            frame = np.array(sct.grab(monitor))[..., :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

            # Trigger detection only if significant screen change is detected
            if prev_small is None or cv2.norm(small, prev_small, cv2.NORM_L2) > self.change_thresh:
                prev_small = small.copy()

                # Hide the overlay before full-resolution capture
                if self.overlay_window:
                    QtCore.QMetaObject.invokeMethod(self.overlay_window, "hide",
                        QtCore.Qt.ConnectionType.QueuedConnection)
                time.sleep(0.03)  # allow time for the overlay to disappear

                # Full-resolution capture without overlay
                full = np.array(sct.grab(monitor))[..., :3]

                # Re-show the overlay
                if self.overlay_window:
                    QtCore.QMetaObject.invokeMethod(self.overlay_window, "show",
                        QtCore.Qt.ConnectionType.QueuedConnection)

                # YOLO inference
                results = self.model(full, conf=self.conf, iou=self.iou, verbose = False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()

                # DPI-aware scaling
                h_phy, w_phy = full.shape[:2]
                screen_geom = QtWidgets.QApplication.primaryScreen().geometry()
                self.sx = screen_geom.width() / w_phy
                self.sy = screen_geom.height() / h_phy

                # Store detections as list of tuples for the overlay painter
                self.detections = [(int(x1 * self.sx), int(y1 * self.sy), int((x2 - x1) * self.sx), int((y2 - y1) * self.sy), float(score),
                                    int(cls_id)) for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids)]

                # Notify callback with dicts and (optionally) the frame
                if self._on_change:
                    det_dicts = self._build_dicts(boxes, scores, class_ids, self.sx, self.sy)
                    added, removed = self._assign_ids_and_diff(self._prev_det_dicts, det_dicts)
                    frame_to_send = None
                    if self._with_frame:
                        frame_to_send = cv2.resize(full, (screen_geom.width(), screen_geom.height()), interpolation=cv2.INTER_AREA)
                    try:
                        self._on_change(det_dicts, added, removed, frame_to_send)
                    except Exception:
                        pass
                    # Keep current detections for next-frame tracking
                    self._prev_det_dicts = det_dicts


            # Pause to limit loop frequency
            # Prevents saturating RAM and CPU with too many captures
            # On very powerful machines with lots of RAM we can set self.interval = 0
            time.sleep(self.interval)


class OverlayWindow(QtWidgets.QWidget):
    """
    Transparent full-screen overlay window for drawing bounding boxes.

    Colors are assigned per class ID.
    """
    PALETTE = [
        QtGui.QColor(0, 255, 0, 200),    # Button → green
        QtGui.QColor(255, 0, 0, 200),    # ToggleButton → red
        QtGui.QColor(0, 0, 255, 200),    # Hyperlink → blue
        QtGui.QColor(255, 255, 0, 200),  # Text → yellow
        QtGui.QColor(255, 0, 255, 200),  # TextInput → magenta
        QtGui.QColor(0, 255, 255, 200),  # Slider → cyan
    ]

    def __init__(self, detector: TargetFinder):
        """
        Construct an always-on-top, click-through, transparent overlay.

        Parameters
        ----------
        detector : TargetFinder
            The detector instance from which to read detections.
        """
        super().__init__()
        self.detector = detector

        # Link overlay to the detector so it can hide/show the window during capture
        detector.overlay_window = self

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
            # Avoid window manager interference on some X11 setups
            flags |= QtCore.Qt.WindowType.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Start the detection loop (background thread)
        self.detector.start()

        # refresh to update the overlay
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10)  # 10 ms

    def paintEvent(self, event):
        """
        Paint bounding boxes and scores on the transparent overlay.
        """
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
    """
    Launch a PyQt application showing detections in a transparent overlay window.

    Parameters
    ----------
    detector : TargetFinder
        The detector instance to visualize.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ov  = OverlayWindow(detector)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    sys.exit(app.exec())


# CLI usage
def main():
    """
    CLI entry point for launching TargetFinder overlay.

    Example
    -------
    python -m targetfinder --model-path best.pt --confidence 0.3 --iou 0.4
    """
    parser = argparse.ArgumentParser(description="Launch the TargetFinder overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "best.pt")

    # Instantiate detector and start overlay
    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    show_detections(det)

if __name__ == "__main__":
    main()

