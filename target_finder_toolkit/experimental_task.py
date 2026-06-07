"""
Controlled target-selection experimental task prototype.

This module turns the annotated TargetFinder dataset into simple Fitts-style
click trials:

- load screenshot images and YOLO-format target annotations,
- convert normalized annotations to pixel-space bounding boxes,
- sample targets by index of difficulty: log2(1 + D / W),
- display the screenshot and highlighted target,
- reset the cursor, run a countdown, collect clicks, and log trial results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from target_finder_toolkit.filters import add_filter_arguments
from target_finder_toolkit.logging_utils import make_default_log_path
from target_finder_toolkit.mouse_utils import restore_default_cursors
from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui

try:
    from target_finder_toolkit.targetfinder import CLASS_NAMES
except Exception:  # pragma: no cover - targetfinder deps may be unavailable.
    CLASS_NAMES = {}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/Users/tangxinqi/Desktop/stage/data/web")
DIFFICULTY_BINS = {
    "easy": (0.0, 3.0),
    "medium": (3.0, 5.0),
    "hard": (5.0, 8.5),
}
TECHNIQUES = [
    "mouse",
    "targetfinder",
    "bubble",
    "semantic",
    "dynaspot",
    "ninja_cursors",
]
TECHNIQUE_MODULES = {
    "targetfinder": "target_finder_toolkit.targetfinder",
    "bubble": "target_finder_toolkit.bubblecursor",
    "semantic": "target_finder_toolkit.semanticpointing",
    "dynaspot": "target_finder_toolkit.dynaspot",
    "ninja_cursors": "target_finder_toolkit.ninjacursors",
}
IN_PROCESS_TASK_TECHNIQUES: set[str] = set()
DEFAULT_CHANGE_THRESH = 100
DEFAULT_CAPTURE_INTERVAL = 1 / 30
DEFAULT_CONFIDENCE = 0.28
DEFAULT_IOU = 0.3
DEFAULT_DYNASPOT_MIN_SPEED = 100.0
DEFAULT_DYNASPOT_SPOT_WIDTH = 128.0
DEFAULT_DYNASPOT_LAG = 0.300
DEFAULT_DYNASPOT_REDUCE_TIME = 0.500
DEFAULT_NINJA_CAMERA_INDEX = 0
DEFAULT_NINJA_SPACING = 320.0
DEFAULT_NINJA_GAZE_SMOOTHING = 0.35
DEFAULT_NINJA_GAZE_GAIN_X = 1.0
DEFAULT_NINJA_GAZE_GAIN_Y = 1.0
DEFAULT_NINJA_GAZE_OFFSET_X = 0.0
DEFAULT_NINJA_GAZE_OFFSET_Y = -200.0
DEFAULT_NINJA_SELECTION_HOLD = 2.0


@dataclass(frozen=True)
class TargetAnnotation:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    distance: float
    width_metric: float
    fitts_id: float
    source_line_number: int
    source_line: str


@dataclass(frozen=True)
class DatasetImage:
    image_path: Path
    label_path: Path
    width: int
    height: int
    targets: tuple[TargetAnnotation, ...]


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    technique: str
    difficulty: str
    image_path: str
    label_path: str
    image_size: tuple[int, int]
    target_index: int
    target_class_id: int
    target_class_name: str
    target_bbox: tuple[float, float, float, float]
    target_center: tuple[float, float]
    start_position: tuple[float, float]
    distance: float
    width_metric: float
    fitts_id: float
    source_line_number: int
    source_line: str


def class_name_for_id(class_id: int) -> str:
    if isinstance(CLASS_NAMES, dict):
        return str(CLASS_NAMES.get(class_id, class_id))
    try:
        return str(CLASS_NAMES[class_id])
    except Exception:
        return str(class_id)


def yolo_to_bbox(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    width = width_norm * image_width
    height = height_norm * image_height
    x = (x_center_norm - width_norm / 2.0) * image_width
    y = (y_center_norm - height_norm / 2.0) * image_height
    x = max(0.0, min(float(image_width), x))
    y = max(0.0, min(float(image_height), y))
    width = max(0.0, min(float(image_width) - x, width))
    height = max(0.0, min(float(image_height) - y, height))
    return x, y, width, height


def compute_fitts_id(
    start: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], float, float, float]:
    x, y, width, height = bbox
    center = (x + width / 2.0, y + height / 2.0)
    distance = math.hypot(center[0] - start[0], center[1] - start[1])
    width_metric = max(1.0, min(width, height))
    fitts_id = math.log2(1.0 + distance / width_metric)
    return center, distance, width_metric, fitts_id


def load_dataset(data_dir: Path, *, min_target_size: float = 4.0) -> list[DatasetImage]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths = sorted(
        path
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for path in data_dir.glob(ext)
        if path.with_suffix(".txt").is_file()
    )
    dataset: list[DatasetImage] = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        image = QtGui.QImage(str(image_path))
        if image.isNull():
            continue
        image_width = image.width()
        image_height = image.height()
        start = (image_width / 2.0, image_height / 2.0)
        targets: list[TargetAnnotation] = []

        for source_line_number, line in enumerate(
            label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                continue
            bbox = yolo_to_bbox(x_center, y_center, width, height, image_width, image_height)
            if bbox[2] < min_target_size or bbox[3] < min_target_size:
                continue
            center, distance, width_metric, fitts_id = compute_fitts_id(start, bbox)
            targets.append(
                TargetAnnotation(
                    class_id=class_id,
                    class_name=class_name_for_id(class_id),
                    bbox=bbox,
                    center=center,
                    distance=distance,
                    width_metric=width_metric,
                    fitts_id=fitts_id,
                    source_line_number=source_line_number,
                    source_line=line,
                )
            )

        if targets:
            dataset.append(
                DatasetImage(
                    image_path=image_path,
                    label_path=label_path,
                    width=image_width,
                    height=image_height,
                    targets=tuple(targets),
                )
            )

    if not dataset:
        raise RuntimeError(f"No annotated images found in {data_dir}")
    return dataset


def sample_trials(
    dataset: list[DatasetImage],
    *,
    technique: str,
    count: int,
    difficulty: str,
    seed: int | None = None,
) -> list[TrialSpec]:
    rng = random.Random(seed)
    candidates: list[tuple[str, DatasetImage, int, TargetAnnotation]] = []
    difficulty_names = list(DIFFICULTY_BINS) if difficulty == "mixed" else [difficulty]

    for difficulty_name in difficulty_names:
        low, high = DIFFICULTY_BINS[difficulty_name]
        bucket = [
            (difficulty_name, item, idx, target)
            for item in dataset
            for idx, target in enumerate(item.targets)
            if low <= target.fitts_id < high
        ]
        candidates.extend(bucket)

    if not candidates:
        raise RuntimeError(f"No targets found for difficulty={difficulty!r}")

    trials: list[TrialSpec] = []
    for trial_id in range(1, count + 1):
        difficulty_name, item, target_index, target = rng.choice(candidates)
        trials.append(
            TrialSpec(
                trial_id=trial_id,
                technique=technique,
                difficulty=difficulty_name,
                image_path=str(item.image_path),
                label_path=str(item.label_path),
                image_size=(item.width, item.height),
                target_index=target_index,
                target_class_id=target.class_id,
                target_class_name=target.class_name,
                target_bbox=target.bbox,
                target_center=target.center,
                start_position=(item.width / 2.0, item.height / 2.0),
                distance=target.distance,
                width_metric=target.width_metric,
                fitts_id=target.fitts_id,
                source_line_number=target.source_line_number,
                source_line=target.source_line,
            )
        )
    return trials


def safe_log_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def default_log_path(*, technique: str, difficulty: str) -> Path:
    logs_dir = PROJECT_ROOT / "task_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    technique_name = safe_log_name(technique)
    difficulty_name = safe_log_name(difficulty)
    return logs_dir / f"{stamp}_task_trials_{technique_name}_{difficulty_name}.jsonl"


def default_technique_log_path(technique: str) -> Path:
    return make_default_log_path(PROJECT_ROOT, f"{technique}_during_task")


def build_technique_command(
    args,
    technique_log_file: Path | None,
    annotation_control_file: Path | None,
) -> list[str] | None:
    if args.technique == "mouse":
        return None
    module_name = TECHNIQUE_MODULES.get(args.technique)
    if module_name is None:
        raise ValueError(f"Unsupported launch technique: {args.technique}")

    cmd = [sys.executable, "-m", module_name]
    cmd += [
        "--change-thresh",
        str(args.change_thresh),
        "--capture-interval",
        str(args.capture_interval),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
        "--filter",
        args.filter,
        "--filter-freq",
        str(args.filter_freq),
        "--filter-min-cutoff",
        str(args.filter_min_cutoff),
        "--filter-beta",
        str(args.filter_beta),
        "--filter-d-cutoff",
        str(args.filter_d_cutoff),
    ]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if technique_log_file is not None:
        cmd += ["--log-file", str(technique_log_file), "--log-cursor-hz", str(args.technique_log_cursor_hz)]
    if annotation_control_file is not None and args.technique != "targetfinder":
        cmd += ["--annotation-control-file", str(annotation_control_file)]

    if args.technique == "bubble":
        cmd.append("--include-text-targets")
    elif args.technique == "semantic":
        if args.semantic_display:
            cmd.append("--display")
        if args.semantic_disable_accel:
            cmd.append("--disable-accel")
    elif args.technique == "dynaspot":
        cmd += [
            "--min-speed",
            str(args.dynaspot_min_speed),
            "--spot-width",
            str(args.dynaspot_spot_width),
            "--lag",
            str(args.dynaspot_lag),
            "--reduce-time",
            str(args.dynaspot_reduce_time),
        ]
        cmd.append("--include-text-targets")
    elif args.technique == "ninja_cursors":
        cmd += [
            "--camera-index",
            str(args.ninja_camera_index),
            "--ninja-spacing",
            str(args.ninja_spacing),
            "--gaze-smoothing",
            str(args.ninja_gaze_smoothing),
            "--gaze-gain-x",
            str(args.ninja_gaze_gain_x),
            "--gaze-gain-y",
            str(args.ninja_gaze_gain_y),
            "--gaze-offset-x",
            str(args.ninja_gaze_offset_x),
            "--gaze-offset-y",
            str(args.ninja_gaze_offset_y),
            "--selection-hold",
            str(args.ninja_selection_hold),
            "--calib-points",
            str(args.ninja_calib_points),
        ]
        if args.ninja_screen_width_cm is not None:
            cmd += ["--screen-width-cm", str(args.ninja_screen_width_cm)]
        if args.ninja_screen_height_cm is not None:
            cmd += ["--screen-height-cm", str(args.ninja_screen_height_cm)]
        if args.ninja_lock_on_dwell:
            cmd.append("--lock-on-dwell")
        if args.ninja_hide_gaze_point:
            cmd.append("--hide-gaze-point")
        if args.ninja_auto_calibrate:
            cmd.append("--auto-calibrate")
        if args.ninja_without_targetfinder:
            cmd.append("--without-targetfinder")

    return cmd


def bbox_contains_point(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def point_to_bbox_distance(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = bbox
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    dx = max(0.0, abs(x - cx) - bw / 2.0)
    dy = max(0.0, abs(y - cy) - bh / 2.0)
    return math.hypot(dx, dy)


def target_to_log_dict(target: TargetAnnotation | None) -> dict | None:
    if target is None:
        return None
    return {
        "class_id": target.class_id,
        "class_name": target.class_name,
        "bbox": [round(value, 3) for value in target.bbox],
        "center": [round(value, 3) for value in target.center],
        "fitts_id": round(target.fitts_id, 4),
    }


class TrialCanvas(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(dict)
    TOP_BAR_HEIGHT = 30
    TARGET_RED = QtGui.QColor(185, 32, 32)
    MESSAGE_RED = QtGui.QColor(190, 45, 45)

    def __init__(self):
        super().__init__()
        self._image = QtGui.QImage()
        self._target_bbox: tuple[float, float, float, float] | None = None
        self._show_all_targets = False
        self._all_targets: tuple[TargetAnnotation, ...] = ()
        self._image_rect = QtCore.QRectF()
        self._status_text = ""
        self._message_text = ""
        self._message_style = "bottom"
        self._technique = "mouse"
        self._cursor_image_point: tuple[float, float] | None = None
        self._bubble_target: TargetAnnotation | None = None
        self._bubble_radius = 0.0
        self._attention_started_at: float | None = None
        self._attention_duration = 1.0
        self._attention_timer = QtCore.QTimer(self)
        self._attention_timer.timeout.connect(self._tick_attention_cue)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 420)

    def set_trial(
        self,
        image_path: str,
        target_bbox: tuple[float, float, float, float],
        *,
        all_targets: tuple[TargetAnnotation, ...] = (),
        show_all_targets: bool = False,
        technique: str = "mouse",
    ):
        self._image = QtGui.QImage(image_path)
        self._target_bbox = target_bbox
        self._all_targets = all_targets
        self._show_all_targets = show_all_targets
        self._technique = technique
        self._cursor_image_point = None
        self._bubble_target = None
        self._bubble_radius = 0.0
        self.clear_attention_cue()
        self.update()

    def set_status_text(self, text: str):
        self._status_text = text
        self.update()

    def set_message_text(self, text: str, *, style: str = "bottom"):
        self._message_text = text
        self._message_style = style
        self.update()

    def start_attention_cue(self, *, duration: float = 1.0):
        self._attention_duration = max(0.1, float(duration))
        self._attention_started_at = time.monotonic()
        self._attention_timer.start(16)
        self.update()

    def clear_attention_cue(self):
        self._attention_started_at = None
        if self._attention_timer.isActive():
            self._attention_timer.stop()
        self.update()

    def _tick_attention_cue(self):
        if self._attention_started_at is None:
            self._attention_timer.stop()
            return
        if time.monotonic() - self._attention_started_at >= self._attention_duration:
            self._attention_timer.stop()
        self.update()

    def image_to_widget_rect(self, bbox: tuple[float, float, float, float]) -> QtCore.QRectF:
        if self._image.isNull() or self._image_rect.isNull():
            return QtCore.QRectF()
        scale_x = self._image_rect.width() / self._image.width()
        scale_y = self._image_rect.height() / self._image.height()
        x, y, width, height = bbox
        return QtCore.QRectF(
            self._image_rect.left() + x * scale_x,
            self._image_rect.top() + y * scale_y,
            width * scale_x,
            height * scale_y,
        )

    def image_to_widget_point(self, point: tuple[float, float]) -> QtCore.QPointF:
        if self._image.isNull() or self._image_rect.isNull():
            return QtCore.QPointF()
        scale_x = self._image_rect.width() / self._image.width()
        scale_y = self._image_rect.height() / self._image.height()
        return QtCore.QPointF(
            self._image_rect.left() + point[0] * scale_x,
            self._image_rect.top() + point[1] * scale_y,
        )

    def image_to_widget_scale(self) -> float:
        if self._image.isNull() or self._image_rect.isNull():
            return 1.0
        return float(self._image_rect.width() / self._image.width())

    def widget_to_image_point(self, point: QtCore.QPointF) -> tuple[float, float] | None:
        if self._image.isNull() or not self._image_rect.contains(point):
            return None
        scale_x = self._image.width() / self._image_rect.width()
        scale_y = self._image.height() / self._image_rect.height()
        return (
            (point.x() - self._image_rect.left()) * scale_x,
            (point.y() - self._image_rect.top()) * scale_y,
        )

    def image_center_global(self) -> QtCore.QPoint:
        if self._image_rect.isNull():
            return self.mapToGlobal(self.rect().center())
        center = self._image_rect.center()
        return self.mapToGlobal(QtCore.QPoint(int(center.x()), int(center.y())))

    def current_image_rect(self) -> QtCore.QRectF:
        if self._image.isNull():
            return QtCore.QRectF()
        image_area = QtCore.QRectF(
            0,
            self.TOP_BAR_HEIGHT,
            self.width(),
            max(1.0, self.height() - self.TOP_BAR_HEIGHT),
        )
        target_size = self._image.size().scaled(
            image_area.size().toSize(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        left = image_area.left() + (image_area.width() - target_size.width()) / 2.0
        top = image_area.top() + (image_area.height() - target_size.height()) / 2.0
        return QtCore.QRectF(left, top, target_size.width(), target_size.height())

    def _target_at_point(self, x: float, y: float) -> TargetAnnotation | None:
        containing = [
            target
            for target in self._all_targets
            if bbox_contains_point(target.bbox, x, y)
        ]
        if not containing:
            return None
        return min(containing, key=lambda target: target.bbox[2] * target.bbox[3])

    def _update_bubble_target(self, point: tuple[float, float] | None):
        self._cursor_image_point = point
        self._bubble_target = None
        self._bubble_radius = 0.0
        if self._technique != "bubble" or point is None or not self._all_targets:
            return

        px, py = point
        distances = sorted(
            # The standalone bubblecursor.py skips class_id == 3 (Text).
            # The controlled experiment intentionally keeps Text selectable.
            (point_to_bbox_distance(px, py, target.bbox), index, target)
            for index, target in enumerate(self._all_targets)
        )
        if not distances:
            return

        nearest_distance, _, nearest = distances[0]
        self._bubble_target = nearest
        x, y, width, height = nearest.bbox
        corners = ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
        containment_distance = max(math.hypot(px - cx, py - cy) for cx, cy in corners)
        if len(distances) > 1:
            second_distance = distances[1][0]
            radius = min(containment_distance, second_distance)
            if radius == second_distance:
                radius = max(0.0, radius - 0.01 * radius)
        else:
            radius = containment_distance
        self._bubble_radius = max(float(radius), float(nearest_distance), 8.0)

    def _bubble_click_payload(self, raw_x: float, raw_y: float) -> dict:
        direct_target = self._target_at_point(raw_x, raw_y)
        selected_target = self._bubble_target
        redirected = (
            selected_target is not None
            and not bbox_contains_point(selected_target.bbox, raw_x, raw_y)
        )
        effective_x = raw_x
        effective_y = raw_y
        if redirected and selected_target is not None:
            effective_x, effective_y = selected_target.center
        return {
            "raw_position": [raw_x, raw_y],
            "effective_position": [effective_x, effective_y],
            "redirected": redirected,
            "selected_target": target_to_log_dict(selected_target),
            "direct_target": target_to_log_dict(direct_target),
            "interaction_backend": "in_process_bubble",
        }

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0))

        top_bar = QtCore.QRectF(0, 0, self.width(), self.TOP_BAR_HEIGHT)
        image_area = QtCore.QRectF(
            0,
            top_bar.bottom(),
            self.width(),
            max(1.0, self.height() - top_bar.bottom()),
        )

        if self._image.isNull():
            painter.setPen(QtGui.QColor(80, 80, 80))
            painter.drawText(image_area, QtCore.Qt.AlignmentFlag.AlignCenter, "Aucune image")
            self._draw_bar_text(painter, top_bar, image_area)
            painter.end()
            return

        self._image_rect = self.current_image_rect()
        painter.drawImage(self._image_rect, self._image)

        if self._show_all_targets:
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 180, 80, 130), 1.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for target in self._all_targets:
                painter.drawRect(self.image_to_widget_rect(target.bbox))

        if self._target_bbox is not None:
            rect = self.image_to_widget_rect(self._target_bbox)
            painter.setPen(QtGui.QPen(self.TARGET_RED, 4))
            painter.setBrush(QtGui.QColor(185, 32, 32, 38))
            painter.drawRoundedRect(rect, 8, 8)
            self._draw_attention_cue(painter, rect)

        self._draw_bubble_overlay(painter)

        self._draw_bar_text(painter, top_bar, image_area)
        painter.end()

    def _draw_attention_cue(self, painter: QtGui.QPainter, target_rect: QtCore.QRectF):
        if self._attention_started_at is None or target_rect.isNull():
            return
        elapsed = time.monotonic() - self._attention_started_at
        progress = max(0.0, min(elapsed / self._attention_duration, 1.0))
        if progress >= 1.0:
            return

        eased = 1.0 - pow(1.0 - progress, 3)
        target_radius = math.hypot(target_rect.width(), target_rect.height()) / 2.0
        end_radius = max(28.0, target_radius + 14.0)
        start_radius = max(end_radius + 90.0, min(260.0, end_radius * 4.5))
        radius = start_radius + (end_radius - start_radius) * eased
        alpha = int(210 * (1.0 - 0.35 * progress))

        painter.save()
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 54, 44, alpha), 7))
        painter.drawEllipse(target_rect.center(), radius, radius)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 210, 120, max(80, alpha // 2)), 2))
        painter.drawEllipse(target_rect.center(), max(end_radius, radius * 0.74), max(end_radius, radius * 0.74))
        painter.restore()

    def _draw_bubble_overlay(self, painter: QtGui.QPainter):
        if self._technique != "bubble" or self._cursor_image_point is None:
            return
        cursor = self.image_to_widget_point(self._cursor_image_point)
        if self._bubble_target is not None:
            scale = self.image_to_widget_scale()
            radius = max(6.0, self._bubble_radius * scale)
            target_rect = self.image_to_widget_rect(self._bubble_target.bbox)

            main_path = QtGui.QPainterPath()
            main_path.addEllipse(cursor, radius, radius)
            env_path = QtGui.QPainterPath()
            corner = min(target_rect.width(), target_rect.height()) / 2.0
            d = math.hypot(corner, corner) - corner
            env_path.addRoundedRect(
                target_rect.adjusted(-d, -d, d, d),
                corner + d,
                corner + d,
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 60, 220), 3))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(main_path.united(env_path))
        self._draw_fake_cursor(painter, cursor)

    def _draw_fake_cursor(self, painter: QtGui.QPainter, cursor: QtCore.QPointF):
        cx = cursor.x()
        cy = cursor.y()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        radius_cursor = 6
        painter.drawEllipse(cursor, radius_cursor, radius_cursor)
        line_len = 4
        gap = radius_cursor + 1
        painter.drawLine(QtCore.QLineF(cx, cy - gap - line_len, cx, cy - gap))
        painter.drawLine(QtCore.QLineF(cx, cy + gap, cx, cy + gap + line_len))
        painter.drawLine(QtCore.QLineF(cx + gap, cy, cx + gap + line_len, cy))
        painter.drawLine(QtCore.QLineF(cx - gap - line_len, cy, cx - gap, cy))

    def _draw_bar_text(
        self,
        painter: QtGui.QPainter,
        top_bar: QtCore.QRectF,
        image_area: QtCore.QRectF,
    ):
        margin = 16
        if self._status_text:
            painter.setPen(QtGui.QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                top_bar.adjusted(margin, 0, -margin, 0),
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                self._status_text,
            )

        if self._message_text:
            font = painter.font()
            is_countdown = self._message_text.strip().isdigit()
            is_centered = self._message_style == "center"
            if is_countdown:
                font.setPointSize(72)
            elif is_centered and len(self._message_text) <= 12:
                font.setPointSize(56)
            elif is_centered:
                font.setPointSize(30)
            else:
                font.setPointSize(20)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(self.MESSAGE_RED)
            if is_countdown or is_centered:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(0, 0, 0, 120))
                if is_countdown or len(self._message_text) <= 12:
                    metrics = painter.fontMetrics()
                    text_rect = metrics.boundingRect(self._message_text)
                    message_box = QtCore.QRectF(
                        0,
                        0,
                        min(max(220.0, text_rect.width() + 96.0), image_area.width() * 0.9),
                        150,
                    )
                else:
                    message_box = QtCore.QRectF(
                        0,
                        0,
                        min(900.0, image_area.width() * 0.78),
                        190,
                    )
                message_box.moveCenter(image_area.center())
                painter.drawRoundedRect(message_box, 26, 26)
                painter.setPen(self.MESSAGE_RED)
                painter.drawText(
                    message_box.adjusted(24, 10, -24, -10),
                    QtCore.Qt.AlignmentFlag.AlignCenter
                    | QtCore.Qt.TextFlag.TextWordWrap,
                    self._message_text,
                )
                return

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        point = self.widget_to_image_point(event.position())
        if point is not None:
            self._update_bubble_target(point)
            if self._technique == "bubble":
                self.clicked.emit(self._bubble_click_payload(point[0], point[1]))
            else:
                self.clicked.emit(
                    {
                        "raw_position": [point[0], point[1]],
                        "effective_position": [point[0], point[1]],
                        "redirected": False,
                        "selected_target": None,
                        "direct_target": target_to_log_dict(self._target_at_point(point[0], point[1])),
                        "interaction_backend": "direct_click",
                    }
                )

    def mouseMoveEvent(self, event):
        self._update_bubble_target(self.widget_to_image_point(event.position()))
        if self._technique == "bubble":
            self.update()
        super().mouseMoveEvent(event)


class ExperimentalTaskWindow(QtWidgets.QWidget):
    def __init__(
        self,
        trials: list[TrialSpec],
        dataset_by_image: dict[str, DatasetImage],
        *,
        log_file: Path,
        countdown_sec: int = 3,
        max_clicks: int = 1,
        show_all_targets: bool = False,
        technique_command: list[str] | None = None,
        technique_log_file: Path | None = None,
        annotation_control_file: Path | None = None,
        ninja_control_file: Path | None = None,
        external_technique_active: bool = False,
        cleanup_control_files: bool = True,
        technique_start_delay_sec: float = 3.0,
        cursor_log_hz: float = 30.0,
        fullscreen: bool = True,
        emit_session_events: bool = True,
        session_metadata: dict | None = None,
    ):
        super().__init__()
        self.trials = trials
        self.dataset_by_image = dataset_by_image
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.countdown_sec = max(0, int(countdown_sec))
        self.max_clicks = max(1, int(max_clicks))
        self.show_all_targets = show_all_targets
        self.technique_command = list(technique_command) if technique_command else None
        self.technique_log_file = Path(technique_log_file) if technique_log_file else None
        self.annotation_control_file = Path(annotation_control_file) if annotation_control_file else None
        self.external_technique_active = bool(external_technique_active)
        self.cleanup_control_files = bool(cleanup_control_files)
        self.technique_start_delay_sec = max(0.0, float(technique_start_delay_sec))
        self.cursor_log_interval_ms = max(1, int(round(1000.0 / max(float(cursor_log_hz), 1.0))))
        self.fullscreen = bool(fullscreen)
        self.emit_session_events = bool(emit_session_events)
        self.session_metadata = dict(session_metadata or {})
        self.technique_process: subprocess.Popen | None = None
        self.current_index = -1
        self.current_trial: TrialSpec | None = None
        self._start_monotonic = time.monotonic()
        self.trial_started_at = 0.0
        self.click_count = 0
        self.miss_count = 0
        self._countdown_remaining = 0
        self._accept_clicks = False
        self._session_ended = False
        self._process_output_buffer = ""
        self._process_output_lines: list[str] = []
        self._waiting_for_ninja_calibration = False
        self._pending_external_click_payload: dict | None = None
        self._exit_code = 0
        self._aborting = False
        self.technique_name = self.trials[0].technique if self.trials else None
        self.ninja_control_file: Path | None = Path(ninja_control_file) if ninja_control_file else None
        if self.technique_command is not None and self._uses_ninja_cursors():
            if self.ninja_control_file is None:
                self.ninja_control_file = self.log_file.with_suffix(".ninja_control")
            self.technique_command += ["--experiment-control-file", str(self.ninja_control_file)]
            self._set_ninja_control_state("paused")
        elif self.ninja_control_file is not None:
            self._set_ninja_control_state("paused")

        self.setWindowTitle("Tache experimentale")
        self.resize(1200, 820)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = TrialCanvas()
        self.canvas.clicked.connect(self._handle_click)
        layout.addWidget(self.canvas, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick_countdown)

        self.cursor_lock_timer = QtCore.QTimer(self)
        self.cursor_lock_timer.setInterval(16)
        self.cursor_lock_timer.timeout.connect(self._set_cursor_to_start)

        self.technique_watch_timer = QtCore.QTimer(self)
        self.technique_watch_timer.setInterval(100)
        self.technique_watch_timer.timeout.connect(self._check_technique_process)

        self.external_click_timer = QtCore.QTimer(self)
        self.external_click_timer.setSingleShot(True)
        self.external_click_timer.setInterval(180)
        self.external_click_timer.timeout.connect(self._flush_pending_external_click)

        self.cursor_sample_timer = QtCore.QTimer(self)
        self.cursor_sample_timer.setInterval(self.cursor_log_interval_ms)
        self.cursor_sample_timer.timeout.connect(self._write_cursor_sample)

        self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._escape_shortcut.activated.connect(lambda: self._abort_experiment("escape"))
        self._q_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
        self._q_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._q_shortcut.activated.connect(lambda: self._abort_experiment("keyboard_q"))

        if self.emit_session_events:
            self._write_event(
                {
                    "type": "session_start",
                    "log_kind": "experimental_task_trials",
                    "task": "controlled_target_selection",
                    "trial_count": len(self.trials),
                    "countdown_sec": self.countdown_sec,
                    "max_clicks": self.max_clicks,
                    "technique": self.technique_name,
                    "interaction_backend": (
                        "in_process_bubble"
                        if self.technique_name in IN_PROCESS_TASK_TECHNIQUES
                        else (
                            "external_process"
                            if self.technique_command is not None or self.external_technique_active
                            else "direct_click"
                        )
                    ),
                    "technique_process_enabled": self.technique_command is not None,
                    "external_technique_active": self.external_technique_active,
                    "technique_command": self.technique_command,
                    "technique_log_file": str(self.technique_log_file) if self.technique_log_file else None,
                    "annotation_control_file": (
                        str(self.annotation_control_file) if self.annotation_control_file else None
                    ),
                    **self.session_metadata,
                }
            )
        QtCore.QTimer.singleShot(200, self.next_trial)

    def closeEvent(self, event):
        self.timer.stop()
        self.cursor_lock_timer.stop()
        self.cursor_sample_timer.stop()
        self.external_click_timer.stop()
        self._stop_technique_process(wait=not self._aborting)
        self._cleanup_ninja_control_file()
        self._cleanup_annotation_control_file()
        self._end_session("window_close")
        super().closeEvent(event)

    def _write_event(self, payload: dict):
        payload = {
            "timestamp": time.time(),
            "t": round(time.monotonic() - self._start_monotonic, 6),
            **payload,
        }
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _end_session(self, reason: str):
        if self._session_ended:
            return
        self._session_ended = True
        if self.emit_session_events:
            self._write_event({"type": "session_end", "reason": reason})

    def _abort_experiment(self, reason: str):
        if self._aborting:
            return
        self._aborting = True
        self._exit_code = 130
        self.timer.stop()
        self.cursor_lock_timer.stop()
        self.cursor_sample_timer.stop()
        self.external_click_timer.stop()
        self._stop_technique_process(wait=False)
        self._cleanup_ninja_control_file()
        self._cleanup_annotation_control_file()
        self._end_session(reason)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.exit(self._exit_code)

    def _start_technique_process(self):
        if self.technique_command is None or self.technique_process is not None:
            return
        popen_kwargs = {
            "cwd": str(PROJECT_ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        try:
            self.technique_process = subprocess.Popen(self.technique_command, **popen_kwargs)
            if self.technique_process.stdout is not None:
                try:
                    os.set_blocking(self.technique_process.stdout.fileno(), False)
                except Exception:
                    pass
        except Exception as exc:
            self._write_event(
                {
                    "type": "technique_process_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": self.technique_command,
                }
            )
            self._set_status_text(f"Echec du lancement de la technique : {exc}")
            return
        self._write_event(
            {
                "type": "technique_process_start",
                "pid": self.technique_process.pid,
                "command": self.technique_command,
                "technique_log_file": str(self.technique_log_file) if self.technique_log_file else None,
            }
        )
        self.technique_watch_timer.start()

    def _technique_command_has(self, value: str) -> bool:
        return bool(self.technique_command and value in self.technique_command)

    def _should_wait_for_ninja_calibration(self) -> bool:
        return (
            self._technique_command_has("target_finder_toolkit.ninjacursors")
            and self._technique_command_has("--auto-calibrate")
        )

    def _drain_technique_output(self):
        if self.technique_process is None or self.technique_process.stdout is None:
            return
        fd = self.technique_process.stdout.fileno()
        chunks = []
        while True:
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        if not chunks:
            return
        self._process_output_buffer += "".join(chunks)
        while True:
            newline_idx = self._process_output_buffer.find("\n")
            if newline_idx < 0:
                break
            line = self._process_output_buffer[:newline_idx].rstrip("\r")
            self._process_output_buffer = self._process_output_buffer[newline_idx + 1:]
            if line:
                self._process_output_lines.append(line)
                self._process_output_lines = self._process_output_lines[-40:]
                self._write_event({"type": "technique_process_output", "line": line})
            self._handle_technique_output_line(line)

    def _handle_technique_output_line(self, line: str):
        prefix = "__NINJA_CALIB__ "
        if not line.startswith(prefix):
            return
        try:
            payload = json.loads(line[len(prefix):])
        except Exception:
            return
        event = payload.get("event")
        self._write_event({"type": "ninja_calibration_event", **payload})
        if event == "started":
            points = payload.get("num_points")
            suffix = f" ({points} points)" if points else ""
            self._set_message_text(f"Calibration Ninja Cursors{suffix}...", style="center")
            return
        if event not in {"calibrated", "failed", "cancelled"}:
            return
        if not self._waiting_for_ninja_calibration:
            return
        self._waiting_for_ninja_calibration = False
        if event == "calibrated":
            self._set_message_text("Calibration terminee", style="center")
        elif event == "failed":
            self._set_message_text("Calibration echouee", style="center")
        else:
            self._set_message_text("Calibration annulee", style="center")
        QtCore.QTimer.singleShot(900, self._show_trial_intro)

    def _check_technique_process(self):
        if self.technique_process is None:
            self.technique_watch_timer.stop()
            return
        self._drain_technique_output()
        exit_code = self.technique_process.poll()
        if exit_code is None:
            return
        self._drain_technique_output()
        if self._process_output_buffer.strip():
            line = self._process_output_buffer.strip()
            self._process_output_lines.append(line)
            self._process_output_lines = self._process_output_lines[-40:]
            self._write_event({"type": "technique_process_output", "line": line})
            self._handle_technique_output_line(line)
            self._process_output_buffer = ""
        self._write_event({"type": "technique_process_exit", "exit_code": exit_code})
        self.technique_process = None
        self.technique_watch_timer.stop()
        self._set_status_text(f"{self._status_text()} | Technique arretee ({exit_code})")
        if self._waiting_for_ninja_calibration:
            self._waiting_for_ninja_calibration = False
            self._set_message_text(f"Ninja Cursors arrete ({exit_code})", style="center")
            QtCore.QTimer.singleShot(900, self._show_trial_intro)

    def _stop_technique_process(self, *, wait: bool = True):
        if self.technique_process is None:
            self.technique_watch_timer.stop()
            restore_default_cursors()
            return
        proc = self.technique_process
        self._drain_technique_output()
        self.technique_process = None
        self.technique_watch_timer.stop()
        try:
            if proc.poll() is None:
                if sys.platform.startswith("win"):
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.send_signal(signal.SIGTERM)
                if wait:
                    proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        finally:
            restore_default_cursors()
            self._write_event(
                {
                    "type": "technique_process_stop",
                    "exit_code": proc.poll(),
                }
            )

    def _set_cursor_to_start(self):
        QtGui.QCursor.setPos(self.canvas.image_center_global())

    def _uses_ninja_cursors(self) -> bool:
        return self.technique_name == "ninja_cursors" or self._technique_command_has(
            "target_finder_toolkit.ninjacursors"
        )

    def _set_ninja_control_state(self, state: str):
        if self.ninja_control_file is None:
            return
        try:
            self.ninja_control_file.write_text(state, encoding="utf-8")
            self._write_event({"type": "ninja_control_state", "state": state})
        except OSError as exc:
            self._write_event({"type": "ninja_control_error", "error": str(exc)})

    def _ninja_pretrial_state(self) -> str:
        return "ready" if self._uses_ninja_cursors() else "paused"

    def _cleanup_ninja_control_file(self):
        if self.ninja_control_file is None or not self.cleanup_control_files:
            return
        try:
            self.ninja_control_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _cleanup_annotation_control_file(self):
        if self.annotation_control_file is None or not self.cleanup_control_files:
            return
        try:
            self.annotation_control_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _write_annotation_control_state(self, state: str):
        if self.annotation_control_file is None or self.current_trial is None:
            return
        item = self.dataset_by_image[self.current_trial.image_path]
        image_rect = self.canvas.current_image_rect()
        if image_rect.isNull() or self.canvas._image.isNull():
            return
        canvas_origin = self.canvas.mapToGlobal(QtCore.QPoint(0, 0))
        scale_x = image_rect.width() / max(float(item.width), 1.0)
        scale_y = image_rect.height() / max(float(item.height), 1.0)
        detections = []
        for index, target in enumerate(item.targets):
            x, y, width, height = target.bbox
            screen_x = canvas_origin.x() + image_rect.left() + x * scale_x
            screen_y = canvas_origin.y() + image_rect.top() + y * scale_y
            detections.append(
                {
                    "id": index + 1,
                    "target_index": index,
                    "class_id": target.class_id,
                    "class_name": target.class_name,
                    "x": screen_x,
                    "y": screen_y,
                    "width": width * scale_x,
                    "height": height * scale_y,
                    "score": 1.0,
                    "image_bbox": list(target.bbox),
                    "image_center": list(target.center),
                    "source_line_number": target.source_line_number,
                    "source_line": target.source_line,
                }
            )
        payload = {
            "version": 1,
            "state": state,
            "trial_id": self.current_trial.trial_id,
            "technique": self.current_trial.technique,
            "image_path": self.current_trial.image_path,
            "image_size": [item.width, item.height],
            "image_rect_global": [
                canvas_origin.x() + image_rect.left(),
                canvas_origin.y() + image_rect.top(),
                image_rect.width(),
                image_rect.height(),
            ],
            "detections": detections,
        }
        tmp_path = self.annotation_control_file.with_suffix(
            self.annotation_control_file.suffix + ".tmp"
        )
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.annotation_control_file)
            self._write_event(
                {
                    "type": "annotation_control_state",
                    "state": state,
                    "trial_id": self.current_trial.trial_id,
                    "target_count": len(detections),
                }
            )
        except OSError as exc:
            self._write_event({"type": "annotation_control_error", "error": str(exc)})

    def _start_cursor_lock_if_needed(self):
        if self._uses_ninja_cursors():
            self.cursor_lock_timer.stop()
            return
        self._set_cursor_to_start()
        self.cursor_lock_timer.start()

    def _set_cursor_to_start_if_needed(self):
        if not self._uses_ninja_cursors():
            self._set_cursor_to_start()

    def _status_text(self) -> str:
        return self.canvas._status_text

    def show_desktop_fullscreen(self):
        """Cover the screen without entering the macOS fullscreen Space."""
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.show()
        raise_macos_window_above_system_ui(self, level_offset=0)
        self.raise_()
        self.activateWindow()

    def _set_status_text(self, text: str):
        self.canvas.set_status_text(text)

    def _set_message_text(self, text: str, *, style: str = "bottom"):
        self.canvas.set_message_text(text, style=style)

    def next_trial(self):
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self._set_status_text(f"Termine. Journal : {self.log_file}")
            self._set_message_text("Termine", style="center")
            self._stop_technique_process()
            self._cleanup_ninja_control_file()
            self._cleanup_annotation_control_file()
            self._end_session("completed")
            QtCore.QTimer.singleShot(1200, QtWidgets.QApplication.instance().quit)
            return

        self.current_trial = self.trials[self.current_index]
        self._set_ninja_control_state(self._ninja_pretrial_state())
        item = self.dataset_by_image[self.current_trial.image_path]
        self.click_count = 0
        self.miss_count = 0
        self._accept_clicks = False
        self.canvas.set_trial(
            self.current_trial.image_path,
            self.current_trial.target_bbox,
            all_targets=item.targets,
            show_all_targets=self.show_all_targets,
            technique=(
                self.current_trial.technique
                if self.current_trial.technique in IN_PROCESS_TASK_TECHNIQUES
                else "mouse"
            ),
        )
        self._write_annotation_control_state("paused")
        self._set_status_text(
            f"Essai {self.current_trial.trial_id}/{len(self.trials)} | "
            f"Technique: {self.current_trial.technique} | "
            f"Difficulte: {self.current_trial.difficulty} | "
            f"ID={self.current_trial.fitts_id:.2f} | "
            f"Cible: {self.current_trial.target_class_name}"
        )
        self._write_event(
            {
                "type": "trial_start",
                **self.session_metadata,
                "global_trial_id": self._global_trial_id(),
                **asdict(self.current_trial),
            }
        )
        if self.current_index == 0 and self.technique_command is not None and self.technique_process is None:
            self._set_message_text("Preparation de la technique...", style="center")
            self._start_technique_process()
            if self._should_wait_for_ninja_calibration():
                self._waiting_for_ninja_calibration = True
                self._set_message_text("Initialisation de Ninja Cursors...", style="center")
                return
            countdown_delay = int(max(0.5, self.technique_start_delay_sec) * 1000)
        else:
            countdown_delay = 100
        QtCore.QTimer.singleShot(countdown_delay, self._show_trial_intro)

    def _show_trial_intro(self):
        if self.current_trial is None:
            return
        self._accept_clicks = False
        self._set_ninja_control_state(self._ninja_pretrial_state())
        self._start_cursor_lock_if_needed()
        if self.current_index == 0:
            self._set_message_text(
                f"Après un compte à rebours de {self.countdown_sec} s,\n"
                "cliquez sur la cible rouge.",
                style="center",
            )
            QtCore.QTimer.singleShot(5000, self._start_countdown)
            return
        self._set_message_text(f"Image {self.current_index + 1}", style="center")
        QtCore.QTimer.singleShot(800, self._start_countdown)

    def _start_countdown(self):
        self._accept_clicks = False
        self._set_ninja_control_state(self._ninja_pretrial_state())
        self._start_cursor_lock_if_needed()
        self._write_annotation_control_state("paused")
        self._countdown_remaining = self.countdown_sec
        if self._countdown_remaining <= 0:
            self._begin_click_phase()
            return
        self.canvas.start_attention_cue(duration=1.0)
        self._set_message_text(str(self._countdown_remaining), style="center")
        self.timer.start()

    def _tick_countdown(self):
        self._set_cursor_to_start_if_needed()
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            self.timer.stop()
            self._begin_click_phase()
        else:
            self._set_message_text(str(self._countdown_remaining), style="center")

    def _begin_click_phase(self):
        self.cursor_lock_timer.stop()
        self._set_cursor_to_start_if_needed()
        self._set_ninja_control_state("active")
        self._write_annotation_control_state("active")
        self.canvas.clear_attention_cue()
        self._set_message_text("")
        self._accept_clicks = True
        self.trial_started_at = time.monotonic()
        self._write_cursor_sample()
        self.cursor_sample_timer.start()

    def _write_cursor_sample(self):
        if not self._accept_clicks or self.current_trial is None:
            return
        global_pos = QtGui.QCursor.pos()
        widget_pos = self.canvas.mapFromGlobal(global_pos)
        image_point = self.canvas.widget_to_image_point(QtCore.QPointF(widget_pos))
        payload = {
            "type": "cursor_sample",
            "trial_id": self.current_trial.trial_id,
            "global_trial_id": self._global_trial_id(),
            **self.session_metadata,
            "technique": self.current_trial.technique,
            "difficulty": self.current_trial.difficulty,
            "screen_position": [round(float(global_pos.x()), 3), round(float(global_pos.y()), 3)],
            "widget_position": [round(float(widget_pos.x()), 3), round(float(widget_pos.y()), 3)],
            "inside_image": image_point is not None,
            "target_bbox": list(self.current_trial.target_bbox),
        }
        if image_point is not None:
            payload["image_position"] = [round(float(image_point[0]), 3), round(float(image_point[1]), 3)]
        else:
            payload["image_position"] = None
        self._write_event(payload)

    def _target_contains(self, x: float, y: float) -> bool:
        if self.current_trial is None:
            return False
        return bbox_contains_point(self.current_trial.target_bbox, x, y)

    def _handle_click(self, click_payload: dict):
        if self._should_delay_external_click(click_payload):
            self._pending_external_click_payload = click_payload
            if not self.external_click_timer.isActive():
                self.external_click_timer.start()
            return
        self._process_click_payload(click_payload)

    def _should_delay_external_click(self, click_payload: dict) -> bool:
        return (
            (self.technique_process is not None or self.external_technique_active)
            and click_payload.get("interaction_backend") == "direct_click"
            and self.current_trial is not None
            and self._accept_clicks
        )

    def _flush_pending_external_click(self):
        payload = self._pending_external_click_payload
        self._pending_external_click_payload = None
        if payload is not None:
            self._process_click_payload(payload)

    def _process_click_payload(self, click_payload: dict):
        if not self._accept_clicks or self.current_trial is None:
            return
        raw_x, raw_y = click_payload.get("raw_position", [None, None])
        x, y = click_payload.get("effective_position", [raw_x, raw_y])
        if x is None or y is None or raw_x is None or raw_y is None:
            return
        self.click_count += 1
        success = self._target_contains(x, y)
        elapsed_ms = (time.monotonic() - self.trial_started_at) * 1000.0
        if not success:
            self.miss_count += 1

        self._write_event(
            {
                "type": "click",
                "trial_id": self.current_trial.trial_id,
                "global_trial_id": self._global_trial_id(),
                **self.session_metadata,
                "technique": self.current_trial.technique,
                "click_index": self.click_count,
                "click_position": [round(x, 3), round(y, 3)],
                "raw": [round(raw_x, 3), round(raw_y, 3)],
                "effective": [round(x, 3), round(y, 3)],
                "raw_click_position": [round(raw_x, 3), round(raw_y, 3)],
                "effective_click_position": [round(x, 3), round(y, 3)],
                "redirected": bool(click_payload.get("redirected", False)),
                "interaction_backend": click_payload.get("interaction_backend"),
                "selected_target": click_payload.get("selected_target"),
                "direct_target": click_payload.get("direct_target"),
                "success": success,
                "movement_time_ms": round(elapsed_ms, 3),
                "miss_count": self.miss_count,
                "target_bbox": list(self.current_trial.target_bbox),
                "target_class_id": self.current_trial.target_class_id,
                "target_class_name": self.current_trial.target_class_name,
                "fitts_id": round(self.current_trial.fitts_id, 4),
                "distance": round(self.current_trial.distance, 3),
                "width_metric": round(self.current_trial.width_metric, 3),
            }
        )

        if success or self.click_count >= self.max_clicks:
            self._accept_clicks = False
            self.cursor_sample_timer.stop()
            self._write_event(
                {
                    "type": "trial_end",
                    "trial_id": self.current_trial.trial_id,
                    "global_trial_id": self._global_trial_id(),
                    **self.session_metadata,
                    "technique": self.current_trial.technique,
                    "success": success,
                    "movement_time_ms": round(elapsed_ms, 3),
                    "click_count": self.click_count,
                    "miss_count": self.miss_count,
                }
            )
            self._set_message_text("Reussi" if success else "Echec", style="center")
            QtCore.QTimer.singleShot(700, self.next_trial)
        else:
            self._set_message_text("Echec - recommencez", style="center")

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            reason = "escape" if event.key() == QtCore.Qt.Key.Key_Escape else "keyboard_q"
            self._abort_experiment(reason)
            return
        super().keyPressEvent(event)

    def _global_trial_id(self) -> int | None:
        offset = self.session_metadata.get("trial_offset")
        if offset is None or self.current_trial is None:
            return None
        try:
            return int(offset) + int(self.current_trial.trial_id)
        except (TypeError, ValueError):
            return None


def print_dataset_summary(dataset: list[DatasetImage]):
    all_targets = [target for item in dataset for target in item.targets]
    ids = [target.fitts_id for target in all_targets]
    print(f"images={len(dataset)} targets={len(all_targets)}")
    print(f"fitts_id min={min(ids):.2f} median={sorted(ids)[len(ids)//2]:.2f} max={max(ids):.2f}")
    for name, (low, high) in DIFFICULTY_BINS.items():
        count = sum(1 for value in ids if low <= value < high)
        print(f"{name}: {count} targets in [{low}, {high})")


def main():
    parser = argparse.ArgumentParser(description="Run a controlled target-selection experimental task prototype")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing .png/.txt annotated pairs")
    parser.add_argument("--technique", choices=TECHNIQUES, default="mouse", help="Technique label to store in task logs")
    parser.add_argument("--trials", "--trial", dest="trials", type=int, default=12, help="Number of trials to generate")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed", help="Target difficulty bin")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible trial sampling")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before each trial starts")
    parser.add_argument("--max-clicks", type=int, default=1, help="Maximum clicks allowed per trial")
    parser.add_argument("--log-file", default=None, help="JSONL log path")
    parser.add_argument("--participant-id", default=None, help="Participant identifier stored in trial logs")
    parser.add_argument("--session-id", default=None, help="Experimental session identifier stored in trial logs")
    parser.add_argument("--block-index", type=int, default=None, help="1-based block position in the session order")
    parser.add_argument("--block-count", type=int, default=None, help="Total number of blocks in the session")
    parser.add_argument("--block-id", default=None, help="Stable block identifier stored in trial logs")
    parser.add_argument("--block-order", default=None, help="Comma-separated ordered block ids stored in logs")
    parser.add_argument("--trial-offset", type=int, default=0, help="Number of trials before this block in the session")
    parser.add_argument("--show-all-targets", action="store_true", help="Draw all annotated targets in green for debugging")
    parser.add_argument("--windowed", action="store_true", help="Run in a normal window instead of fullscreen")
    parser.add_argument("--summary-only", action="store_true", help="Only print dataset and trial summary; do not open GUI")
    parser.add_argument("--no-launch-technique", action="store_true", help="Do not start the selected overlay process")
    parser.add_argument("--technique-start-delay", type=float, default=3.0, help="Seconds to wait before the first trial after launching a technique")
    parser.add_argument("--technique-log-file", default=None, help="Optional JSONL log path for the launched technique process")
    parser.add_argument("--no-technique-log", action="store_true", help="Do not write a separate technique runtime log")
    parser.add_argument("--technique-log-cursor-hz", type=float, default=30.0, help="Cursor sampling rate for the technique runtime log")
    parser.add_argument("--cursor-log-hz", type=float, default=30.0, help="Experiment-level cursor sampling rate")
    parser.add_argument("--no-task-session-events", action="store_true", help="Do not write task-level session_start/session_end events")
    parser.add_argument("--annotation-control-file", default=None, help="Existing annotation control file updated by this task")
    parser.add_argument("--ninja-control-file", default=None, help="Existing Ninja experiment control file updated by this task")
    parser.add_argument("--keep-control-files", action="store_true", help="Do not delete external control files on task exit")
    parser.add_argument("--model-path", default=None, help="Optional YOLO model path for launched TargetFinder-based techniques")
    parser.add_argument("--change-thresh", type=int, default=DEFAULT_CHANGE_THRESH, help="Screen-change threshold for launched techniques")
    parser.add_argument("--capture-interval", type=float, default=DEFAULT_CAPTURE_INTERVAL, help="Screen capture interval for launched techniques")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="YOLO confidence for launched techniques")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="YOLO IoU for launched techniques")
    add_filter_arguments(parser)
    parser.add_argument("--semantic-display", action="store_true", help="Show Semantic Pointing visual guides")
    parser.add_argument("--semantic-disable-accel", action="store_true", help="Disable system mouse acceleration for Semantic Pointing")
    parser.add_argument("--dynaspot-min-speed", type=float, default=DEFAULT_DYNASPOT_MIN_SPEED, help="DynaSpot minimum speed threshold")
    parser.add_argument("--dynaspot-spot-width", type=float, default=DEFAULT_DYNASPOT_SPOT_WIDTH, help="DynaSpot maximum spot diameter")
    parser.add_argument("--dynaspot-lag", type=float, default=DEFAULT_DYNASPOT_LAG, help="DynaSpot shrink lag")
    parser.add_argument("--dynaspot-reduce-time", type=float, default=DEFAULT_DYNASPOT_REDUCE_TIME, help="DynaSpot reduction time")
    parser.add_argument("--ninja-camera-index", type=int, default=DEFAULT_NINJA_CAMERA_INDEX, help="Ninja webcam index")
    parser.add_argument("--ninja-screen-width-cm", type=float, default=None, help="Ninja physical screen width in cm")
    parser.add_argument("--ninja-screen-height-cm", type=float, default=None, help="Ninja physical screen height in cm")
    parser.add_argument("--ninja-spacing", type=float, default=DEFAULT_NINJA_SPACING, help="Ninja cursor spacing in pixels")
    parser.add_argument("--ninja-gaze-smoothing", type=float, default=DEFAULT_NINJA_GAZE_SMOOTHING, help="Ninja gaze smoothing")
    parser.add_argument("--ninja-gaze-gain-x", type=float, default=DEFAULT_NINJA_GAZE_GAIN_X, help="Ninja gaze horizontal gain")
    parser.add_argument("--ninja-gaze-gain-y", type=float, default=DEFAULT_NINJA_GAZE_GAIN_Y, help="Ninja gaze vertical gain")
    parser.add_argument("--ninja-gaze-offset-x", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_X, help="Ninja gaze horizontal offset")
    parser.add_argument("--ninja-gaze-offset-y", type=float, default=DEFAULT_NINJA_GAZE_OFFSET_Y, help="Ninja gaze vertical offset")
    parser.add_argument("--ninja-selection-hold", type=float, default=DEFAULT_NINJA_SELECTION_HOLD, help="Ninja dwell duration before lock")
    parser.add_argument("--ninja-lock-on-dwell", action="store_true", help="Require Ninja dwell lock before click")
    parser.add_argument("--ninja-hide-gaze-point", action="store_true", help="Hide Ninja red gaze point")
    parser.add_argument("--ninja-calib-points", type=int, choices=[5, 9, 13], default=5, help="Ninja calibration point count")
    parser.add_argument("--ninja-auto-calibrate", action="store_true", help="Start Ninja calibration automatically")
    parser.add_argument("--ninja-with-targetfinder", dest="ninja_without_targetfinder", action="store_false", help="Enable TargetFinder detections inside Ninja Cursors")
    parser.set_defaults(ninja_without_targetfinder=True)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.data_dir))
    trials = sample_trials(
        dataset,
        technique=args.technique,
        count=args.trials,
        difficulty=args.difficulty,
        seed=args.seed,
    )

    print_dataset_summary(dataset)
    print("sample_trials:")
    for trial in trials[: min(5, len(trials))]:
        print(
            f"  trial={trial.trial_id} difficulty={trial.difficulty} "
            f"ID={trial.fitts_id:.2f} image={Path(trial.image_path).name} "
            f"target_index={trial.target_index} "
            f"source_line_number={trial.source_line_number} "
            f"source_line={trial.source_line!r} "
            f"bbox={[round(v, 1) for v in trial.target_bbox]}"
        )

    if args.summary_only:
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dataset_by_image = {str(item.image_path): item for item in dataset}
    log_file = (
        Path(args.log_file).expanduser()
        if args.log_file
        else default_log_path(technique=args.technique, difficulty=args.difficulty)
    )
    launch_technique = (
        args.technique != "mouse"
        and args.technique not in IN_PROCESS_TASK_TECHNIQUES
        and not args.no_launch_technique
    )
    technique_log_file = None
    if launch_technique and not args.no_technique_log:
        technique_log_file = (
            Path(args.technique_log_file).expanduser()
            if args.technique_log_file
            else default_technique_log_path(args.technique)
        )
    annotation_control_file = (
        Path(args.annotation_control_file).expanduser()
        if args.annotation_control_file
        else (log_file.with_suffix(".annotations.json") if launch_technique else None)
    )
    technique_command = (
        build_technique_command(args, technique_log_file, annotation_control_file)
        if launch_technique
        else None
    )
    window = ExperimentalTaskWindow(
        trials,
        dataset_by_image,
        log_file=log_file,
        countdown_sec=args.countdown,
        max_clicks=args.max_clicks,
        show_all_targets=args.show_all_targets,
        technique_command=technique_command,
        technique_log_file=technique_log_file,
        annotation_control_file=annotation_control_file,
        ninja_control_file=Path(args.ninja_control_file).expanduser() if args.ninja_control_file else None,
        external_technique_active=bool(args.no_launch_technique and annotation_control_file is not None and args.technique != "mouse"),
        cleanup_control_files=not args.keep_control_files,
        technique_start_delay_sec=args.technique_start_delay,
        cursor_log_hz=args.cursor_log_hz,
        fullscreen=not args.windowed,
        emit_session_events=not args.no_task_session_events,
        session_metadata={
            "participant_id": args.participant_id,
            "session_id": args.session_id,
            "block_index": args.block_index,
            "block_count": args.block_count,
            "block_id": args.block_id,
            "block_order": args.block_order,
            "trial_offset": args.trial_offset,
        },
    )
    if args.windowed:
        window.show()
    elif sys.platform == "darwin" and launch_technique:
        window.show_desktop_fullscreen()
    else:
        window.showFullScreen()
    exit_code = app.exec()
    sys.exit(window._exit_code or exit_code)


if __name__ == "__main__":
    main()
