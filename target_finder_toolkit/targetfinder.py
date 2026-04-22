"""
Detection of GUI widgets from live screen captures or static images.

Core features:
    - Real-time screen capture and YOLO-based detection of GUI widgets.
    - Callback mechanism to provide the current detections, the captured frame (optional), and the lists of added/removed detections.
    - Optional PyQt overlay to visualize bounding boxes in real time.
    - One-shot detection on static images with optional annotated export.

Notes:
    This is the **main entry point** of the toolkit. Utility helpers live in
    ``postprocess.py`` and demonstration apps are in ``bubblecursor.py`` and
    ``semanticpointing.py``.
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
import pyautogui

__all__ = ["TargetFinder", "show_detections", "main"]


class ScreenCaptureError(Exception):
    pass

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
    """Compute IoU between two (x, y, w, h) boxes.

    Args:
        a (tuple[int, int, int, int]): First box (x, y, w, h).
        b (tuple[int, int, int, int]): Second box (x, y, w, h).

    Returns:
        float: IoU in ``[0, 1]``.
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
    try:
        return float(val)
    except Exception:
        raise TypeError(f"{name} must be a number, got {type(val).__name__}")

def _require_between(name, val, lo, hi, inclusive=True):
    v = _require_number(name, val)
    if inclusive:
        if not (lo <= v <= hi):
            raise ValueError(f"{name} ({v}) must be in [{lo}, {hi}]")
    else:
        if not (lo < v < hi):
            raise ValueError(f"{name} ({v}) must be in ({lo}, {hi})")
    return v


class TargetFinder:
    def __init__(self, model_name=None, change_thresh=100, capture_interval=1/30, confidence=0.4, iou = 0.3, imgsz=640):
        """Initialize the detector.

        Args:
            model_path (str | None, optional):
                Path to a YOLO ``.pt`` weights file.  
                If ``None``, the detector loads the default model (``best.pt``)
                packaged with the toolkit. You can also supply your own trained
                YOLOv8 model.

            change_thresh (int, optional):
                Screen change detection threshold (L2 distance on a down-scaled
                screen).  
                A **higher value** makes the detector **less sensitive**
                to small variations.  
                Default = ``100``.

            capture_interval (float, optional):
                Delay in **seconds** between consecutive screen captures.  
                Lower values = higher refresh rate but also higher CPU/GPU usage.  
                Default = ``1/30 ≈ 0.033s (30 FPS)``.

            confidence (float, optional):
                Minimum YOLO confidence score required to keep a detection.  
                Must be in the range ``[0.0 – 1.0]``.  
                Default = ``0.28``.

            iou (float, optional):
                Intersection-over-Union (IoU) threshold used for YOLO’s
                Non-Maximum Suppression (NMS).  
                Determines when two overlapping bounding boxes should be considered
                the same object (higher = stricter merging, lower = more boxes kept).  
                Must be in the range ``[0.0 – 1.0]``.  
                Default = ``0.3``.

        Raises:
            FileNotFoundError:
                If the specified YOLO model file cannot be found.
            ValueError:
                If numeric arguments are outside their valid ranges.
            TypeError:
                If numeric arguments have invalid types.
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
        self.imgsz         = int(imgsz)

        # Public snapshot used by the overlay: list of tuples (x, y, w, h, score, cls_id)
        self.detections = []

        # Screen scaling (physical capture → Qt screen geometry)
        self.sx = 1.0
        self.sy = 1.0

        # Control flags for the background loop
        self._stop = False
        self.hide_overlay_during_capture = True
        self.overlay_window = {}  # screen_key -> OverlayWindow, set by OverlayWindow or caller

        # Tracking state to keep stable IDs between frames
        self._prev_det_dicts = []   # previous frame detection dicts (with "id")
        self._next_id = 1           # next stable ID to assign

        # Callback configuration
        self._on_change = None      # callable or None
        self._with_frame = False    # whether to forward the frame to callback
        self._diff_iou = 0.5        # IoU threshold for added/removed + ID conservation

    def set_callback(self, fn, with_frame=False, diff_iou=0.5):
        """Register a callback invoked after each detection update.

        The callback is invoked as:
            ``fn(detections, added, removed, frame)``

        Args:
            fn (Callable | None):
                User callback function, or ``None`` to disable.  
                The function receives:
                    - ``detections (list[dict])``: All current detections (widgets).
                    - ``added (list[dict])``: Detections that appeared since the last frame.
                    - ``removed (list[dict])``: Detections that disappeared since the last frame.
                    - ``frame (np.ndarray | None)``: RGB screen frame
                      (only provided if ``with_frame=True``).

                Each detection dictionary has the following keys:
                    - ``id (int)``: Unique identifier for a detection,
                      preserved across frames as long as the widget remains visible.
                    - ``x (int)``: Top-left X coordinate.
                    - ``y (int)``: Top-left Y coordinate.
                    - ``width (int)``: Bounding box width.
                    - ``height (int)``: Bounding box height.
                    - ``score (float)``: Confidence score in ``[0, 1]``.
                    - ``class_id (int)``: YOLO class index.
                    - ``class_name (str)``: class label (e.g., "Button").

            with_frame (bool, optional):
                If ``True``, include the RGB frame in the callback.
                Default = ``False``.

            diff_iou (float, optional):
                IoU threshold used to decide whether a detection is considered
                the same widget across frames (for ID conservation and added/removed lists).
                Must be in ``[0, 1]``. Default = ``0.5``.

        Raises:
            TypeError: If ``fn`` is not callable.
            ValueError: If ``diff_iou`` is outside ``[0, 1]``.
        """
        if fn is not None and not callable(fn):
            raise TypeError("on_change must be a callable or None")
        self._on_change = fn
        self._with_frame = bool(with_frame)
        self._diff_iou = float(_require_between("diff_iou", diff_iou, 0.0, 1.0))

    @staticmethod
    def get_monitor_from_mouse(sct):
        """Return the mss monitor dict and its 1-based index for the screen
        that currently contains the mouse pointer."""
        x, y = pyautogui.position()
        for i, monitor in enumerate(sct.monitors[1:], start=1):
            if (
                monitor["left"] <= x < monitor["left"] + monitor["width"]
                and monitor["top"] <= y < monitor["top"] + monitor["height"]
            ):
                return monitor, i
        raise ScreenCaptureError("Mouse is not on any known monitor")

    def match_monitor_to_overlay(self, monitor):
        """Match an mss monitor dict to the corresponding OverlayWindow.

        Returns:
            tuple: (active_overlay, list_of_other_overlays)
        """
        other_overlays = []
        active_overlay = None
        for _, overlay in self.overlay_window.items():
            if (
                hasattr(overlay, "screen_geometry")
                and overlay.screen_geometry.x() == monitor["left"]
                and overlay.screen_geometry.y() == monitor["top"]
                and overlay.screen_geometry.width() == monitor["width"]
                and overlay.screen_geometry.height() == monitor["height"]
            ):
                active_overlay = overlay
            else:
                other_overlays.append(overlay)
        if active_overlay is None:
            raise ScreenCaptureError("No overlay matches the active monitor")
        return active_overlay, other_overlays

    def start(self):
        """Start the capture+inference loop in a separate thread."""
        self._stop = False
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        """Stop the detection loop."""
        self._stop = True

    def get_detections(self):
        """Return current detections as tuples. Coordinates are in **logical screen space**, i.e. they already include DPI-aware scaling to match the primary screen geometry.

        Returns:
            ``[(x, y, w, h, score, class_id) ...]``.
        """
        return self.detections

    def _build_dicts(self, boxes_xyxy, scores, class_ids, scale_x, scale_y):
        """Convert raw YOLO outputs to detection dicts.

        Returns:
            list[dict]: Dicts with keys
            ``id, x, y, width, height, score, class_id, class_name``.
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
        """Assign stable IDs via class-aware IoU matching; compute added/removed.

        Args:
            prev (list[dict]): Previous detections (with assigned ``id``).
            curr (list[dict]): Current detections (``id`` will be set).

        Returns:
            tuple[list[dict], list[dict]]: ``(added, removed)``.
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

    def detect_array(self, img_bgr):
        """Run detection on a BGR numpy array.

        Args:
            img_bgr (np.ndarray): BGR image array (H, W, 3).

        Returns:
            Detections with keys ``id, x, y, width, height, score, class_id, class_name``.
        """
        results = self.model(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        detections = self._build_dicts(
            boxes_xyxy=boxes, scores=scores, class_ids=class_ids,
            scale_x=1.0, scale_y=1.0,
        )
        for d in detections:
            d["id"] = int(self._next_id)
            self._next_id += 1
        return detections

    def detect_image(self, image_path, save_annotated=False, save_json=False):
        """Run detection on a single image and optionally save results.

        Args:
            image_path (str): Path to input image.
            save_annotated (bool, optional): If ``True``, write
                ``*_annotated.png`` next to the image. Defaults to ``False``.
            save_json (bool, optional): If ``True``, write
                ``*_detections.json`` next to the image. Defaults to ``False``.

        Returns:
            Detections with keys ``id, x, y, width, height, score, class_id, class_name``.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        # Read image (BGR as expected by OpenCV/Ultralytics)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Single-shot YOLO inference
        results = self.model(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, end2end=False, verbose=False)[0]
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
        # Colorblind-safe palette (Wong 2011) — BGR order for OpenCV
        _ANNOTATE_PALETTE = {
            0: (0, 159, 230),    # Button → Orange
            1: (233, 180, 86),   # ToggleButton → Sky Blue
            2: (115, 158, 0),    # Hyperlink → Bluish Green
            3: (66, 228, 240),   # Text → Yellow
            4: (178, 114, 0),    # TextInput → Blue
            5: (0, 94, 213),     # Slider → Vermillion
        }
        if save_annotated:
            annotated = img_bgr.copy()
            for d in detections:
                x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
                color = _ANNOTATE_PALETTE.get(d["class_id"], (0, 255, 0))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = f"{d['class_name']}:{d['score']:.2f}"
                # 라벨 배경 (가독성)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 4, max(0, y - 1)), color, -1)
                cv2.putText(annotated, label, (x + 2, max(0, y - 4)),
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
        sct = mss.mss()
        prev_small = None

        while not self._stop:
            # Determine which monitor to capture
            try:
                monitor, _ = self.get_monitor_from_mouse(sct)
            except ScreenCaptureError:
                monitor = sct.monitors[0]

            # Low-res screenshot for change detection
            frame = np.array(sct.grab(monitor))[..., :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

            if prev_small is None or cv2.norm(small, prev_small, cv2.NORM_L2) > self.change_thresh:
                # Activate/deactivate per-screen overlays if available
                active_overlay = None
                has_per_screen = (
                    len(self.overlay_window) > 1
                    and any(hasattr(ov, "screen_geometry") for ov in self.overlay_window.values())
                )
                if has_per_screen:
                    try:
                        active_overlay, other_overlays = self.match_monitor_to_overlay(monitor)
                        active_overlay.activate()
                        for ov in other_overlays:
                            ov.reset()
                    except ScreenCaptureError:
                        pass

                prev_small = small.copy()

                # Hide overlay(s) before full-resolution capture when needed
                target_ov = active_overlay if active_overlay else None
                if self.overlay_window and self.hide_overlay_during_capture:
                    if target_ov:
                        QtCore.QMetaObject.invokeMethod(
                            target_ov, "hide",
                            QtCore.Qt.ConnectionType.QueuedConnection,
                        )
                    else:
                        for ov in self.overlay_window.values():
                            QtCore.QMetaObject.invokeMethod(
                                ov, "hide",
                                QtCore.Qt.ConnectionType.QueuedConnection,
                            )
                    time.sleep(0.03)

                # Full-resolution capture
                full = np.array(sct.grab(monitor))[..., :3]

                # Re-show overlay(s)
                if self.overlay_window and self.hide_overlay_during_capture:
                    if target_ov:
                        QtCore.QMetaObject.invokeMethod(
                            target_ov, "show",
                            QtCore.Qt.ConnectionType.QueuedConnection,
                        )
                    else:
                        for ov in self.overlay_window.values():
                            QtCore.QMetaObject.invokeMethod(
                                ov, "show",
                                QtCore.Qt.ConnectionType.QueuedConnection,
                            )

                # Reset non-active overlays (clear leftover detections)
                if has_per_screen and active_overlay:
                    for ov in other_overlays:
                        ov.reset()

                # YOLO inference
                results = self.model(full, conf=self.conf, iou=self.iou, imgsz=self.imgsz, end2end=False, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()

                # DPI-aware scaling
                h_phy, w_phy = full.shape[:2]
                if active_overlay and hasattr(active_overlay, "screen_geometry"):
                    screen_geom = active_overlay.screen_geometry
                else:
                    screen_geom = QtWidgets.QApplication.primaryScreen().geometry()
                self.sx = screen_geom.width() / w_phy
                self.sy = screen_geom.height() / h_phy

                # Store detections as tuples for the overlay painter
                self.detections = [
                    (int(x1 * self.sx), int(y1 * self.sy),
                     int((x2 - x1) * self.sx), int((y2 - y1) * self.sy),
                     float(score), int(cls_id))
                    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids)
                ]

                # Notify callback
                if self._on_change:
                    det_dicts = self._build_dicts(boxes, scores, class_ids, self.sx, self.sy)
                    added, removed = self._assign_ids_and_diff(self._prev_det_dicts, det_dicts)
                    frame_to_send = None
                    if self._with_frame:
                        frame_to_send = cv2.resize(
                            full, (screen_geom.width(), screen_geom.height()),
                            interpolation=cv2.INTER_AREA,
                        )
                    try:
                        self._on_change(det_dicts, added, removed, frame_to_send)
                    except Exception:
                        pass
                    self._prev_det_dicts = det_dicts

            time.sleep(self.interval)


class OverlayWindow(QtWidgets.QWidget):
    PALETTE = [
        QtGui.QColor(0, 255, 0, 200),    # Button
        QtGui.QColor(255, 0, 0, 200),    # ToggleButton
        QtGui.QColor(0, 0, 255, 200),    # Hyperlink
        QtGui.QColor(255, 255, 0, 200),  # Text
        QtGui.QColor(255, 0, 255, 200),  # TextInput
        QtGui.QColor(0, 255, 255, 200),  # Slider
    ]

    def __init__(self, detector: TargetFinder, screen: QtGui.QScreen = None):
        super().__init__()
        self.detector = detector
        self._is_macos = sys.platform == "darwin"
        self.active = True

        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()

        # Register in detector's overlay dict
        self.detector.overlay_window[str(screen.name())] = self

        # Per-screen geometry
        self.setScreen(screen)
        self.screen_geometry = screen.geometry()
        self.resize(self.screen_geometry.size())

        # Window flags
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
        if not self._is_macos:
            flags |= QtCore.Qt.WindowType.Tool
        if sys.platform.startswith("linux"):
            flags |= QtCore.Qt.WindowType.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10)

    def paintEvent(self, event):
        if not self.active:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        for x, y, w, h, score, cls_id in self.detector.get_detections():
            color = self.PALETTE[cls_id % len(self.PALETTE)]
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

            label = f"{score:.2f}"
            fm = painter.fontMetrics()
            tw, th = fm.horizontalAdvance(label), fm.height()
            painter.fillRect(x, y - th, tw + 4, th, QtGui.QColor(0, 0, 0, 120))
            painter.drawText(x + 2, y - 2, label)

        painter.end()

    def activate(self):
        self.active = True

    def reset(self):
        self.active = False
        self.update()


def show_detections(detector: TargetFinder):
    """Launch a PyQt application showing detections in a transparent overlay window.

    Args:
        detector (TargetFinder): The detector instance providing the detections.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ov  = OverlayWindow(detector)
    ov.show()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    sys.exit(app.exec())


# CLI usage
def main():
    """CLI entry point for launching the TargetFinder overlay.

    **Example:**  ``python -m target_finder_toolkit.targetfinder --confidence 0.3 --iou 0.4``

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

