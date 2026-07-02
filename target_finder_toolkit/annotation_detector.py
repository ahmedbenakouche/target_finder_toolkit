"""TargetFinder-compatible detector backed by controlled-task annotations."""

from __future__ import annotations

import json
from pathlib import Path

from target_finder_toolkit.targetfinder import CLASS_NAMES


class AnnotationDetector:
    """Expose static experimental annotations through the TargetFinder API subset.

    The controlled experimental task displays screenshots from the annotated
    dataset. Interaction techniques should therefore use those annotations, not
    a live YOLO pass over the current desktop.
    """

    def __init__(self, control_file: str | Path):
        self.control_file = Path(control_file)
        self.detections = []
        self.sx = 1.0
        self.sy = 1.0
        self.change_thresh = 0
        self.interval = 0.0
        self.conf = 1.0
        self.iou = 1.0
        self.hide_overlay_during_capture = False
        self.overlay_window = None
        self._latest_det_dicts = []
        self._prev_det_dicts = []
        self._on_change = None
        self._with_frame = False
        self._last_payload_key = None
        self._state = "inactive"
        self._payload = {}

    def set_callback(self, fn, with_frame=False, diff_iou=0.5):
        self._on_change = fn if callable(fn) else None
        self._with_frame = bool(with_frame)

    def start(self):
        self._load()

    def stop(self):
        return

    def is_active(self) -> bool:
        self._load()
        return self._state == "active"

    def get_detections(self):
        self._load()
        return list(self.detections)

    def get_detection_dicts(self):
        self._load()
        return [dict(det) for det in self._latest_det_dicts]

    def get_control_payload(self):
        self._load()
        return dict(self._payload)

    @staticmethod
    def _compact_detection(det):
        if det is None:
            return None
        compact = {
            "id": det.get("id"),
            "x": det.get("x"),
            "y": det.get("y"),
            "w": det.get("width"),
            "h": det.get("height"),
            "score": round(float(det.get("score", 0.0)), 4),
            "class": det.get("class_name"),
            "class_id": det.get("class_id"),
            "target_index": det.get("target_index"),
            "source_line_number": det.get("source_line_number"),
            "source_line": det.get("source_line"),
        }
        for key in (
            "role",
            "synthetic_widget_bbox",
            "synthetic_widget_center",
            "synthetic_distance",
            "synthetic_fitts_id",
        ):
            if key in det:
                compact[key] = det.get(key)
        return compact

    def find_detection_for_point(self, x, y, *, include_text=True, fallback_nearest=False):
        self._load()
        candidates = []
        for det in self._latest_det_dicts:
            if not include_text and det.get("class_id") == 3:
                continue
            dx = float(det["x"])
            dy = float(det["y"])
            dw = float(det["width"])
            dh = float(det["height"])
            if dx <= x <= dx + dw and dy <= y <= dy + dh:
                candidates.append((dw * dh, -float(det.get("score", 0.0)), det))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            return self._compact_detection(candidates[0][2])

        if not fallback_nearest:
            return None

        nearest = []
        for det in self._latest_det_dicts:
            if not include_text and det.get("class_id") == 3:
                continue
            cx = float(det["x"]) + float(det["width"]) / 2.0
            cy = float(det["y"]) + float(det["height"]) / 2.0
            center_dist = (cx - x) ** 2 + (cy - y) ** 2
            area = float(det["width"]) * float(det["height"])
            nearest.append((center_dist, area, -float(det.get("score", 0.0)), det))

        if not nearest:
            return None
        nearest.sort(key=lambda item: (item[0], item[1], item[2]))
        return self._compact_detection(nearest[0][3])

    def find_detection_by_geometry(self, cx, cy, w, h, *, class_id=None, tolerance=3.0):
        self._load()
        matches = []
        for det in self._latest_det_dicts:
            if class_id is not None and det.get("class_id") != class_id:
                continue
            det_cx = float(det["x"]) + float(det["width"]) / 2.0
            det_cy = float(det["y"]) + float(det["height"]) / 2.0
            if (
                abs(det_cx - cx) <= tolerance
                and abs(det_cy - cy) <= tolerance
                and abs(float(det["width"]) - w) <= tolerance * 2.0
                and abs(float(det["height"]) - h) <= tolerance * 2.0
            ):
                area = float(det["width"]) * float(det["height"])
                matches.append((area, -float(det.get("score", 0.0)), det))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1]))
        return self._compact_detection(matches[0][2])

    def _load(self):
        try:
            stat = self.control_file.stat()
            payload_key = (stat.st_mtime_ns, stat.st_size)
            if payload_key == self._last_payload_key:
                return
            payload = json.loads(self.control_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self._payload = dict(payload)
        self._state = str(payload.get("state") or "active").strip().lower()
        if self._state != "active":
            previous = self._latest_det_dicts
            self._latest_det_dicts = []
            self.detections = []
            self._last_payload_key = payload_key
            if self._on_change is not None and previous:
                self._on_change([], [], [dict(det) for det in previous], None)
            return

        dets = []
        for idx, det in enumerate(payload.get("detections", []) or [], start=1):
            class_id = int(det.get("class_id", 0))
            x = float(det.get("x", 0.0))
            y = float(det.get("y", 0.0))
            width = float(det.get("width", 0.0))
            height = float(det.get("height", 0.0))
            if width <= 0.0 or height <= 0.0:
                continue
            det_dict = {
                "id": int(det.get("id", idx)),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": float(det.get("score", 1.0)),
                "class_id": class_id,
                "class_name": det.get("class_name") or CLASS_NAMES.get(class_id, str(class_id)),
                "target_index": det.get("target_index"),
                "source_line_number": det.get("source_line_number"),
                "source_line": det.get("source_line"),
            }
            for key in (
                "role",
                "synthetic_widget_bbox",
                "synthetic_widget_center",
                "synthetic_distance",
                "synthetic_fitts_id",
            ):
                if key in det:
                    det_dict[key] = det.get(key)
            dets.append(det_dict)

        previous = self._latest_det_dicts
        self._latest_det_dicts = dets
        self.detections = [
            (
                det["x"],
                det["y"],
                det["width"],
                det["height"],
                det["score"],
                det["class_id"],
            )
            for det in dets
        ]
        self._last_payload_key = payload_key

        if self._on_change is not None:
            previous_ids = {det.get("id") for det in previous}
            current_ids = {det.get("id") for det in dets}
            added = [dict(det) for det in dets if det.get("id") not in previous_ids]
            removed = [dict(det) for det in previous if det.get("id") not in current_ids]
            self._on_change([dict(det) for det in dets], added, removed, None)


class FakeTargetFinder(AnnotationDetector):
    """Semantic alias for generated/known targets.

    This class intentionally keeps the same API as AnnotationDetector and the
    TargetFinder subset used by the interaction techniques. It makes synthetic
    tasks explicit: the detector does not capture the screen or run YOLO; it
    simply returns already-known target boxes generated by the task.
    """

    pass
