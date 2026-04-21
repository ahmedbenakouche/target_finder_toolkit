"""
Rake Cursor Demo
================

This module demonstrates a webcam-based gaze + mouse interaction technique
inspired by Rake Cursor. A small rake of candidate cursors follows the mouse,
and the gaze point selects which candidate becomes active.
"""

import argparse
import math
import os
import signal
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "target_finder_toolkit_mpl"))

import cv2
import pyautogui
from PyQt6 import QtCore, QtGui, QtWidgets
from pynput import keyboard, mouse

from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors
from target_finder_toolkit.targetfinder import TargetFinder

try:
    from webeyetrack import WebEyeTrack, WebEyeTrackConfig
    from webeyetrack.data_protocols import TrackingStatus
except Exception as exc:  # pragma: no cover - optional dependency
    TrackingStatus = None
    WebEyeTrack = None
    WebEyeTrackConfig = None
    _WEBEYETRACK_IMPORT_ERROR = exc
else:
    _WEBEYETRACK_IMPORT_ERROR = None

__all__ = ["rake_cursor", "main"]


def _create_webeyetrack(config):
    """Create WebEyeTrack with CPU MediaPipe delegate when GPU setup is unavailable."""
    try:
        import webeyetrack.webeyetrack as wet_module
    except Exception:
        return WebEyeTrack(config)

    original_base_options = wet_module.python.BaseOptions
    original_delegate = getattr(original_base_options, "Delegate", None)
    cpu_delegate = getattr(original_delegate, "CPU", None)

    def base_options_with_cpu_delegate(*args, **kwargs):
        if cpu_delegate is not None:
            kwargs.setdefault("delegate", cpu_delegate)
        return original_base_options(*args, **kwargs)

    if original_delegate is not None:
        base_options_with_cpu_delegate.Delegate = original_delegate
    wet_module.python.BaseOptions = base_options_with_cpu_delegate
    try:
        return WebEyeTrack(config)
    except Exception as exc:
        if _can_use_legacy_face_mesh_fallback(wet_module, exc):
            _patch_legacy_face_landmarker(wet_module)
            return WebEyeTrack(config)
        raise
    finally:
        wet_module.python.BaseOptions = original_base_options


def _can_use_legacy_face_mesh_fallback(wet_module, exc) -> bool:
    if not sys.platform == "darwin":
        return False
    message = str(exc)
    if "kGpuService" not in message and "FaceLandmarkerOptions" not in message:
        return False
    try:
        import mediapipe as mp
    except Exception:
        return False
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def _patch_legacy_face_landmarker(wet_module):
    import types

    import mediapipe as mp
    import numpy as np

    class LegacyFaceLandmarkerOptions:
        def __init__(self, base_options=None, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1):
            self.base_options = base_options
            self.output_face_blendshapes = output_face_blendshapes
            self.output_facial_transformation_matrixes = output_facial_transformation_matrixes
            self.num_faces = num_faces

    class LegacyFaceLandmarker:
        @classmethod
        def create_from_options(cls, options):
            return cls(options)

        def __init__(self, options):
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max(1, int(getattr(options, "num_faces", 1))),
                refine_landmarks=True,
            )

        def detect(self, mp_image):
            frame = mp_image.numpy_view()
            results = self._face_mesh.process(frame)
            landmarks = [
                list(face_landmarks.landmark)
                for face_landmarks in (results.multi_face_landmarks or [])
            ]
            transforms = [np.eye(4, dtype=np.float32) for _ in landmarks]
            return types.SimpleNamespace(
                face_landmarks=landmarks,
                facial_transformation_matrixes=transforms,
                face_blendshapes=[],
            )

    wet_module.vision.FaceLandmarkerOptions = LegacyFaceLandmarkerOptions
    wet_module.vision.FaceLandmarker = LegacyFaceLandmarker


class RakeCursor(QtWidgets.QWidget):
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_SCREEN_WIDTH_CM = 34.0
    DEFAULT_SCREEN_HEIGHT_CM = 19.0
    DEFAULT_RAKE_SPACING = 72.0
    DEFAULT_GAZE_SMOOTHING = 0.35
    DEFAULT_GAZE_GAIN = 2.0
    FIXED_DIRECTION_THRESHOLD = 35.0
    DEFAULT_SELECTION_HOLD = 0.5
    DEFAULT_SHOW_GAZE = True
    CURSOR_RADIUS = 6.0
    ACTIVE_RADIUS = 12.0
    CLICK_EPSILON = 3.0
    DEBUG_TEXT_REFRESH_SEC = 1.0
    GAZE_VALID_TTL_SEC = 0.6

    @classmethod
    def resolve_screen_size_cm(cls, screen_width_cm=None, screen_height_cm=None, screen=None) -> tuple[float, float]:
        width_cm = cls._valid_screen_dimension(screen_width_cm)
        height_cm = cls._valid_screen_dimension(screen_height_cm)
        if width_cm is not None and height_cm is not None:
            return width_cm, height_cm

        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            physical_size = screen.physicalSize()
            detected_width_cm = cls._valid_screen_dimension(physical_size.width() / 10.0)
            detected_height_cm = cls._valid_screen_dimension(physical_size.height() / 10.0)
            width_cm = width_cm if width_cm is not None else detected_width_cm
            height_cm = height_cm if height_cm is not None else detected_height_cm

        return (
            width_cm if width_cm is not None else cls.DEFAULT_SCREEN_WIDTH_CM,
            height_cm if height_cm is not None else cls.DEFAULT_SCREEN_HEIGHT_CM,
        )

    @staticmethod
    def _valid_screen_dimension(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0.0 or not math.isfinite(value):
            return None
        return value

    def __init__(
        self,
        detector: TargetFinder,
        cursor_filter=None,
        logger=None,
        *,
        camera_index: int = 0,
        screen_width_cm: float | None = None,
        screen_height_cm: float | None = None,
        rake_spacing: float = DEFAULT_RAKE_SPACING,
        gaze_smoothing: float = DEFAULT_GAZE_SMOOTHING,
        gaze_gain: float = DEFAULT_GAZE_GAIN,
        selection_hold: float = DEFAULT_SELECTION_HOLD,
        show_gaze: bool = DEFAULT_SHOW_GAZE,
    ):
        if WebEyeTrack is None:
            raise RuntimeError(
                "WebEyeTrack is not available. Install it with `pip install webeyetrack`."
            ) from _WEBEYETRACK_IMPORT_ERROR

        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.camera_index = int(camera_index)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("Could not detect a primary screen for Rake Cursor.")
        self.screen_width_cm, self.screen_height_cm = self.resolve_screen_size_cm(
            screen_width_cm,
            screen_height_cm,
            screen,
        )
        self.rake_spacing = float(rake_spacing)
        self.gaze_smoothing = max(0.0, min(float(gaze_smoothing), 0.95))
        self.gaze_gain = max(0.1, float(gaze_gain))
        self.direction_threshold = self.FIXED_DIRECTION_THRESHOLD
        self.selection_hold = max(0.0, float(selection_hold))
        self.show_gaze = bool(show_gaze)

        self._mouse_listener = None
        self._keyboard_listener = None
        self._gaze_timer = None
        self._tracking_ok = False
        self._gaze_point = None
        self._active_index = 0
        self._active_point = None
        self._active_target = None
        self._simulating_click = False
        self._pending_click_point = None
        self._pending_click_target = None
        self._last_gaze_status = "waiting for webcam"
        self._last_gaze_debug_t = 0.0
        self._last_successful_gaze_t = 0.0
        self._held_active_index = 0
        self._held_active_until = 0.0
        self._selection_magnitude = 0.0
        self._amplified_gaze_point = None

        geom = screen.geometry()
        self.setGeometry(geom)
        self._screen_rect = geom
        self._screen_px_dimensions = (geom.width(), geom.height())

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

        cfg = WebEyeTrackConfig(
            screen_px_dimensions=self._screen_px_dimensions,
            screen_cm_dimensions=(self.screen_width_cm, self.screen_height_cm),
        )
        self._tracker = _create_webeyetrack(cfg)
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.camera_index}.")

        self._start_mouse_listener()
        self._start_keyboard_listener()
        self.detector.start()

        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.timeout.connect(self.update)
        self._paint_timer.start(10)

        self._gaze_timer = QtCore.QTimer(self)
        self._gaze_timer.timeout.connect(self._update_gaze)
        self._gaze_timer.start(33)

    def _screen_for_point(self, x: int, y: int):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return QtWidgets.QApplication.primaryScreen()
        return app.screenAt(QtCore.QPoint(int(x), int(y))) or QtWidgets.QApplication.primaryScreen()

    def _in_system_reserved_area(self, x: int, y: int) -> bool:
        if sys.platform != "darwin":
            return False
        screen = self._screen_for_point(x, y)
        if screen is None:
            return False
        return not screen.availableGeometry().contains(QtCore.QPoint(int(x), int(y)))

    def _candidate_points(self, cx: float, cy: float) -> list[tuple[float, float]]:
        points = [(cx, cy)]
        for angle_deg in range(0, 360, 60):
            angle = math.radians(angle_deg)
            px = cx + self.rake_spacing * math.cos(angle)
            py = cy + self.rake_spacing * math.sin(angle)
            points.append((px, py))
        return points

    def _direction_candidate_index(self, points: list[tuple[float, float]]) -> int:
        self._selection_magnitude = 0.0
        self._amplified_gaze_point = None
        if not self._gaze_is_recent():
            return 0
        cx, cy = points[0]
        gx, gy = self._gaze_point
        dx = (gx - cx) * self.gaze_gain
        dy = (gy - cy) * self.gaze_gain
        self._selection_magnitude = math.hypot(dx, dy)
        self._amplified_gaze_point = (cx + dx, cy + dy)
        if self._selection_magnitude < self.direction_threshold:
            return 0

        best_index = 0
        best_score = -float("inf")
        for idx, (px, py) in enumerate(points[1:], start=1):
            candidate_dx = px - cx
            candidate_dy = py - cy
            candidate_len = math.hypot(candidate_dx, candidate_dy)
            if candidate_len <= 0:
                continue
            score = (dx * candidate_dx + dy * candidate_dy) / candidate_len
            if score > best_score:
                best_score = score
                best_index = idx
        return best_index

    def _select_active_index(self, points: list[tuple[float, float]]) -> int:
        now = time.time()
        next_index = self._direction_candidate_index(points)
        if now < self._held_active_until:
            return self._held_active_index

        self._held_active_index = next_index
        if self.selection_hold > 0.0:
            self._held_active_until = now + self.selection_hold
        else:
            self._held_active_until = 0.0
        return next_index

    def _apply_gaze_smoothing(self, x: float, y: float):
        if self._gaze_point is None:
            self._gaze_point = (x, y)
            return
        keep = self.gaze_smoothing
        old_x, old_y = self._gaze_point
        self._gaze_point = (
            old_x * keep + x * (1.0 - keep),
            old_y * keep + y * (1.0 - keep),
        )

    def _gaze_is_recent(self) -> bool:
        return (
            self._gaze_point is not None
            and time.time() - self._last_successful_gaze_t <= self.GAZE_VALID_TTL_SEC
        )

    def _set_gaze_status(self, message: str):
        self._last_gaze_status = message
        now = time.time()
        if now - self._last_gaze_debug_t >= self.DEBUG_TEXT_REFRESH_SEC:
            print(f"[rake] {message}", flush=True)
            self._last_gaze_debug_t = now

    @QtCore.pyqtSlot()
    def _update_gaze(self):
        if self._capture is None:
            self._tracking_ok = False
            self._set_gaze_status("camera not initialized")
            return
        ok, frame = self._capture.read()
        if not ok:
            self._tracking_ok = False
            self._set_gaze_status("camera frame failed")
            return

        try:
            status, gaze_result, _ = self._tracker.process_frame(frame)
        except Exception as exc:
            self._tracking_ok = False
            self._set_gaze_status(f"tracking error: {type(exc).__name__}: {exc}")
            return

        norm_pog = getattr(gaze_result, "norm_pog", None) if gaze_result is not None else None
        if status == TrackingStatus.SUCCESS and norm_pog is not None:
            gx = self._screen_rect.left() + (float(norm_pog[0]) + 0.5) * self._screen_rect.width()
            gy = self._screen_rect.top() + (float(norm_pog[1]) + 0.5) * self._screen_rect.height()
            gx = max(float(self._screen_rect.left()), min(float(self._screen_rect.right()), gx))
            gy = max(float(self._screen_rect.top()), min(float(self._screen_rect.bottom()), gy))
            self._apply_gaze_smoothing(gx, gy)
            self._tracking_ok = True
            self._last_successful_gaze_t = time.time()
            self._set_gaze_status(f"tracking ok gaze=({gx:.0f}, {gy:.0f})")
        else:
            self._tracking_ok = False
            self._set_gaze_status(f"not tracking status={status}")

    def _draw_debug_status(self, painter: QtGui.QPainter):
        painter.save()
        box_width = max(360.0, min(820.0, float(self.width()) - 32.0))
        box = QtCore.QRectF(16.0, 16.0, box_width, 74.0)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.drawRoundedRect(box, 8.0, 8.0)

        title = f"Rake gaze: {self._last_gaze_status}"
        if self._active_point is not None:
            freshness = "recent" if self._gaze_is_recent() else "stale/default"
            held = "held" if time.time() < self._held_active_until else "free"
            title += f" | active={self._active_index} | gaze={freshness} | {held} | mag={self._selection_magnitude:.0f}"

        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 235), 1))
        painter.drawText(
            QtCore.QRectF(28.0, 24.0, box.width() - 24.0, 22.0),
            QtCore.Qt.AlignmentFlag.AlignLeft,
            title[:130],
        )

        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 240, 255, 230), 1))
        painter.drawText(
            QtCore.QRectF(28.0, 50.0, box.width() - 24.0, 30.0),
            QtCore.Qt.AlignmentFlag.AlignLeft,
            "Move the mouse to place the rake; gaze selects the active candidate.",
        )
        painter.restore()

    def paintEvent(self, event):
        if self._simulating_click:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pos = QtGui.QCursor.pos()
        raw_x, raw_y = float(pos.x()), float(pos.y())
        cx, cy = raw_x, raw_y
        if self.cursor_filter is not None:
            cx, cy = self.cursor_filter.filter(raw_x, raw_y)

        points = self._candidate_points(cx, cy)
        self._active_index = self._select_active_index(points)
        self._active_point = points[self._active_index]
        self._active_target = self.detector.find_detection_for_point(
            float(self._active_point[0]),
            float(self._active_point[1]),
            include_text=False,
            fallback_nearest=True,
        )

        center_x, center_y = points[0]
        connector_pen = QtGui.QPen(QtGui.QColor(120, 220, 255, 70), 1)
        painter.setPen(connector_pen)
        for px, py in points[1:]:
            painter.drawLine(QtCore.QLineF(center_x, center_y, px, py))

        for idx, (px, py) in enumerate(points):
            is_active = idx == self._active_index
            outer_pen = QtGui.QPen(
                QtGui.QColor(255, 210, 40, 255) if is_active else QtGui.QColor(70, 255, 120, 110),
                5 if is_active else 2,
            )
            painter.setPen(outer_pen)
            painter.setBrush(
                QtGui.QColor(255, 210, 40, 42) if is_active else QtGui.QColor(70, 255, 120, 8)
            )
            radius = self.ACTIVE_RADIUS if is_active else self.CURSOR_RADIUS
            painter.drawEllipse(QtCore.QPointF(px, py), radius, radius)
            cross_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 240 if is_active else 150), 2)
            painter.setPen(cross_pen)
            line_len = 4
            gap = radius + 1
            painter.drawLine(QtCore.QLineF(px, py - gap - line_len, px, py - gap))
            painter.drawLine(QtCore.QLineF(px, py + gap, px, py + gap + line_len))
            painter.drawLine(QtCore.QLineF(px + gap, py, px + gap + line_len, py))
            painter.drawLine(QtCore.QLineF(px - gap - line_len, py, px - gap, py))

        if self.show_gaze and self._gaze_point is not None:
            gx, gy = self._gaze_point
            if self._gaze_is_recent():
                gaze_color = QtGui.QColor(255, 90, 90, 220)
                gaze_fill = QtGui.QColor(255, 90, 90, 24)
            else:
                gaze_color = QtGui.QColor(150, 150, 150, 180)
                gaze_fill = QtGui.QColor(150, 150, 150, 18)
            painter.setPen(QtGui.QPen(gaze_color, 2))
            painter.setBrush(gaze_fill)
            painter.drawEllipse(QtCore.QPointF(gx, gy), 12, 12)

        self._draw_debug_status(painter)

        if self.logger is not None:
            fields = {
                "technique": "rake",
                "filter_name": self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                "active_index": int(self._active_index),
                "active": [round(float(self._active_point[0]), 3), round(float(self._active_point[1]), 3)],
                "tracking_ok": bool(self._tracking_ok),
                "gaze_gain": round(float(self.gaze_gain), 3),
                "selection_hold": round(float(self.selection_hold), 3),
                "selection_magnitude": round(float(self._selection_magnitude), 3),
                "detection_count": len(self.detector.get_detections()),
            }
            if self._gaze_point is not None:
                fields["gaze"] = [round(float(self._gaze_point[0]), 3), round(float(self._gaze_point[1]), 3)]
            if self._amplified_gaze_point is not None:
                fields["amplified_gaze"] = [
                    round(float(self._amplified_gaze_point[0]), 3),
                    round(float(self._amplified_gaze_point[1]), 3),
                ]
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=cx,
                filtered_y=cy,
                **fields,
            )
        painter.end()

    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        self._cleanup(runtime_reason="quit")
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event):
        self._cleanup(runtime_reason=None)
        super().closeEvent(event)

    def _cleanup(self, runtime_reason: str | None):
        if self._gaze_timer is not None:
            self._gaze_timer.stop()
        if self._paint_timer is not None:
            self._paint_timer.stop()
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
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        restore_default_cursors()
        if self.logger is not None and runtime_reason is not None:
            self.logger.log_session_end(reason=runtime_reason)
            self.logger.close()

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == "q":
                    self._queue_quit()
            except AttributeError:
                pass

        kwargs = {"on_press": on_press}
        if sys.platform == "darwin":
            kwargs["darwin_intercept"] = self._intercept_keyboard_event
        self._keyboard_listener = keyboard.Listener(**kwargs)
        self._keyboard_listener.start()

    def _queue_quit(self):
        QtCore.QMetaObject.invokeMethod(
            self,
            "stop_and_quit",
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def _intercept_keyboard_event(self, event_type, event):
        try:
            import Quartz
        except Exception:
            return event
        if event_type != Quartz.kCGEventKeyDown:
            return event
        try:
            keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            _length, chars = Quartz.CGEventKeyboardGetUnicodeString(event, 100, None, None)
        except Exception:
            return event
        if keycode == 12 or chars == "q":
            print("[rake] q pressed, quitting", flush=True)
            self._queue_quit()
            return None
        return event

    def _start_mouse_listener(self):
        def on_click(x, y, button, pressed):
            if sys.platform == "darwin":
                return
            if pressed or button != button.left:
                return
            QtCore.QMetaObject.invokeMethod(
                self,
                "_simulate_click",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(int, x),
                QtCore.Q_ARG(int, y),
            )

        kwargs = {"on_click": on_click}
        if sys.platform == "darwin":
            kwargs["darwin_intercept"] = self._intercept_mouse_event
        self._mouse_listener = mouse.Listener(**kwargs)
        self._mouse_listener.start()

    def _intercept_mouse_event(self, event_type, event):
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
            self._pending_click_point = self._active_point
            self._pending_click_target = self._active_target

        target_point = self._pending_click_point or self._active_point
        if target_point is None:
            return event

        tx, ty = float(target_point[0]), float(target_point[1])
        Quartz.CGEventSetLocation(event, Quartz.CGPointMake(tx, ty))
        redirected = math.hypot(float(px) - tx, float(py) - ty) > self.CLICK_EPSILON
        if event_type == Quartz.kCGEventLeftMouseDown:
            print(
                f"[rake] retarget native click raw=({float(px):.1f}, {float(py):.1f}) "
                f"active_index={self._active_index} effective=({tx:.1f}, {ty:.1f}) "
                f"redirected={redirected} gaze_recent={self._gaze_is_recent()}",
                flush=True,
            )
        elif event_type == Quartz.kCGEventLeftMouseUp:
            if self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[round(float(px), 3), round(float(py), 3)],
                    effective=[round(tx, 3), round(ty, 3)],
                    redirected=redirected,
                    target=self._pending_click_target,
                )
            self._pending_click_point = None
            self._pending_click_target = None
        return event

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        if self._in_system_reserved_area(orig_x, orig_y):
            if self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=None,
                )
            return

        if self._active_point is None:
            if self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=None,
                )
            if sys.platform == "darwin":
                self._send_click(orig_x, orig_y, orig_x, orig_y)
            return

        tx, ty = self._active_point
        redirected = math.hypot(float(orig_x) - tx, float(orig_y) - ty) > self.CLICK_EPSILON
        print(
            f"[rake] click raw=({orig_x}, {orig_y}) active_index={self._active_index} "
            f"effective=({tx:.1f}, {ty:.1f}) redirected={redirected} gaze_recent={self._gaze_is_recent()}",
            flush=True,
        )
        if not redirected and sys.platform != "darwin":
            if self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._active_target,
                )
            return

        self._send_click(orig_x, orig_y, tx, ty)
        if self.logger is not None:
            self.logger.log_click(
                technique="rake",
                raw=[orig_x, orig_y],
                effective=[round(float(tx), 3), round(float(ty), 3)],
                redirected=redirected,
                target=self._active_target,
            )

    def _send_click(self, orig_x, orig_y, tx, ty):
        self._simulating_click = True
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        try:
            if sys.platform == "darwin":
                self._send_macos_click(float(orig_x), float(orig_y), float(tx), float(ty))
                backend = "quartz"
            else:
                pyautogui.mouseUp(button="left")
                pyautogui.moveTo(tx, ty)
                pyautogui.click()
                pyautogui.moveTo(orig_x, orig_y)
                backend = "pyautogui"
            print(
                f"[rake] sent click backend={backend} effective=({float(tx):.1f}, {float(ty):.1f})",
                flush=True,
            )
        except pyautogui.FailSafeException:
            print("[rake] click skipped by pyautogui fail-safe", flush=True)
        except Exception as exc:
            print(f"[rake] click send error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            hide_cursor_everywhere()
            self._simulating_click = False
            self._start_mouse_listener()

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


def rake_cursor(
    detector: TargetFinder,
    cursor_filter=None,
    logger=None,
    **kwargs,
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    overlay = RakeCursor(detector, cursor_filter=cursor_filter, logger=logger, **kwargs)
    overlay.show()
    if sys.platform == "darwin":
        QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
        QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
    else:
        hide_cursor_everywhere()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if logger is not None:
        logger.log_session_end(reason="app_exit")
        logger.close()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description="Launch the Rake Cursor gaze overlay")
    parser.add_argument("--model-path", default=None, help="Path to the YOLO model .pt file")
    parser.add_argument("--change-thresh", type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument("--capture-interval", type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument("--confidence", type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument("--iou", type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    parser.add_argument("--filter", choices=sorted(FILTER_OPTIONS.keys()), default="none", help="Optional pointer filter")
    parser.add_argument("--log-file", default=None, help="Optional JSONL log file path")
    parser.add_argument("--log-cursor-hz", type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument("--camera-index", type=int, default=RakeCursor.DEFAULT_CAMERA_INDEX, help="Webcam index used for gaze tracking")
    parser.add_argument("--screen-width-cm", type=float, default=None, help="Approximate physical screen width in centimeters; auto-detected when omitted")
    parser.add_argument("--screen-height-cm", type=float, default=None, help="Approximate physical screen height in centimeters; auto-detected when omitted")
    parser.add_argument("--rake-spacing", type=float, default=RakeCursor.DEFAULT_RAKE_SPACING, help="Distance in pixels between the center cursor and the outer rake cursors")
    parser.add_argument("--gaze-smoothing", type=float, default=RakeCursor.DEFAULT_GAZE_SMOOTHING, help="Smoothing factor applied to the gaze point (0 = no smoothing, higher = steadier gaze)")
    parser.add_argument("--gaze-gain", type=float, default=RakeCursor.DEFAULT_GAZE_GAIN, help="Multiplier applied to the gaze direction vector around the mouse center")
    parser.add_argument("--selection-hold", type=float, default=RakeCursor.DEFAULT_SELECTION_HOLD, help="Seconds to keep a selected outer rake cursor before switching again")
    parser.add_argument("--hide-gaze-point", action="store_true", help="Hide the red on-screen gaze feedback marker")
    args = parser.parse_args()

    if WebEyeTrack is None:
        raise SystemExit(
            "WebEyeTrack is not available in this environment. Install it with `pip install webeyetrack`."
        )

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "best.pt")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    args.screen_width_cm, args.screen_height_cm = RakeCursor.resolve_screen_size_cm(
        args.screen_width_cm,
        args.screen_height_cm,
    )

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    cursor_filter = PointFilter2D(args.filter) if args.filter != "none" else None
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="rake",
            filter_name=args.filter,
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
            camera_index=args.camera_index,
            screen_width_cm=args.screen_width_cm,
            screen_height_cm=args.screen_height_cm,
            rake_spacing=args.rake_spacing,
            gaze_smoothing=args.gaze_smoothing,
            gaze_gain=args.gaze_gain,
            selection_hold=args.selection_hold,
            show_gaze=not args.hide_gaze_point,
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    rake_cursor(
        det,
        cursor_filter=cursor_filter,
        logger=logger,
        camera_index=args.camera_index,
        screen_width_cm=args.screen_width_cm,
        screen_height_cm=args.screen_height_cm,
        rake_spacing=args.rake_spacing,
        gaze_smoothing=args.gaze_smoothing,
        gaze_gain=args.gaze_gain,
        selection_hold=args.selection_hold,
        show_gaze=not args.hide_gaze_point,
    )


if __name__ == "__main__":
    main()
