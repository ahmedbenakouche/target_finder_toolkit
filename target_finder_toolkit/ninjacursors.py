"""
Ninja Cursors(gaze) Demo
========================

This module implements a gaze-assisted multi-cursor technique inspired by
Ninja Cursor and gaze-driven cursor selection:

- eight distributed cursors move synchronously with the mouse,
- gaze selects which cursor is currently active,
- keeping gaze on the same cursor locks it,
- the mouse then performs the final local adjustment and click.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import signal
import sys
import tempfile
import time
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "target_finder_toolkit_mpl"))

import cv2
import pyautogui
from PyQt6 import QtCore, QtGui, QtWidgets
from pynput import keyboard, mouse

from target_finder_toolkit.eye_calibration import EyeCalibration
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors
from target_finder_toolkit.targetfinder import TargetFinder


def _ensure_mediapipe_python_alias():
    """Provide a compatibility alias for WebEyeTrack on newer MediaPipe builds."""
    try:
        import mediapipe as mp
    except Exception:
        return
    if "mediapipe.python" not in sys.modules:
        sys.modules["mediapipe.python"] = mp
    if not hasattr(mp, "python"):
        mp.python = mp


_ensure_mediapipe_python_alias()

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

__all__ = ["ninja_cursors", "main"]


_WEIGHT_FILE_CACHE: dict[str, Optional[str]] = {}

_MODEL_URLS = {
    "face_landmarker_v2_with_blendshapes.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "blazegaze_mpiifacegaze.keras":
        "https://github.com/RedForestAI/WebEyeTrack/raw/main/python/webeyetrack/model_weights/blazegaze_mpiifacegaze.keras",
}


def _download_model_weight(filename: str, dest_dir: pathlib.Path) -> Optional[str]:
    url = _MODEL_URLS.get(filename)
    if not url:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.is_file():
        return str(dest)
    print(f"[gaze] Downloading {filename}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        print(f"[gaze] Saved to {dest}")
        return str(dest)
    except Exception as e:
        print(f"[gaze] Download failed: {e}")
        return None


def _normalize_webeyetrack_config_paths(obj):
    """Convert Path-like values inside WebEyeTrack config objects to strings."""
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return obj
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _normalize_webeyetrack_config_paths(value)
        return obj
    if isinstance(obj, tuple):
        return tuple(_normalize_webeyetrack_config_paths(value) for value in obj)
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _normalize_webeyetrack_config_paths(value)
        return obj
    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            setattr(obj, key, _normalize_webeyetrack_config_paths(value))
    return obj


def _patch_webeyetrack_model_paths(config, wet_module):
    """Override broken default weight paths with files shipped in the package."""
    try:
        package_dir = pathlib.Path(wet_module.__file__).resolve().parent
    except Exception:
        return config

    def resolve_weight(filename: str) -> Optional[str]:
        cached = _WEIGHT_FILE_CACHE.get(filename)
        if cached is not None:
            return cached

        candidate_paths = [
            package_dir / "model_weights" / filename,
            package_dir.parent.parent / "python" / "webeyetrack" / "model_weights" / filename,
            package_dir.parent.parent.parent / "python" / "webeyetrack" / "model_weights" / filename,
            pathlib.Path(sys.prefix) / "Lib" / "python" / "webeyetrack" / "model_weights" / filename,
            pathlib.Path(sys.prefix) / "lib" / "python" / "webeyetrack" / "model_weights" / filename,
        ]
        for conda_env in pathlib.Path("/opt/homebrew/Caskroom/miniforge/base/envs").glob("*/lib/python*/site-packages/webeyetrack/model_weights"):
            candidate_paths.append(conda_env / filename)
        for path in candidate_paths:
            if path.is_file():
                resolved = str(path)
                _WEIGHT_FILE_CACHE[filename] = resolved
                return resolved

        search_roots = [package_dir, pathlib.Path(sys.prefix)]
        for root in search_roots:
            if not root.exists():
                continue
            try:
                match = next(root.rglob(filename), None)
            except Exception:
                match = None
            if match is not None and match.is_file():
                resolved = str(match)
                _WEIGHT_FILE_CACHE[filename] = resolved
                return resolved

        downloaded = _download_model_weight(filename, package_dir / "model_weights")
        if downloaded:
            _WEIGHT_FILE_CACHE[filename] = downloaded
            return downloaded

        _WEIGHT_FILE_CACHE[filename] = None
        return None

    face_landmarker = resolve_weight("face_landmarker_v2_with_blendshapes.task")
    blazegaze = resolve_weight("blazegaze_mpiifacegaze.keras")

    current_face = pathlib.Path(str(getattr(config, "mediapipe_flm_model_fp", "")))
    current_blaze = pathlib.Path(str(getattr(config, "blazegaze_mlp_fp", "")))

    if face_landmarker and (not str(current_face) or not current_face.is_file()):
        config.mediapipe_flm_model_fp = face_landmarker
    if blazegaze and (not str(current_blaze) or not current_blaze.is_file()):
        config.blazegaze_mlp_fp = blazegaze

    # Patch module-level defaults as well because WebEyeTrackConfig stores class attributes.
    try:
        wet_module.FACE_LANDMARKER_PATH = pathlib.Path(str(config.mediapipe_flm_model_fp))
        wet_module.BLAZEGAZE_PATH = pathlib.Path(str(config.blazegaze_mlp_fp))
        if hasattr(wet_module, "WebEyeTrackConfig"):
            wet_module.WebEyeTrackConfig.mediapipe_flm_model_fp = str(config.mediapipe_flm_model_fp)
            wet_module.WebEyeTrackConfig.blazegaze_mlp_fp = str(config.blazegaze_mlp_fp)
        try:
            import webeyetrack.constants as constants_module
            constants_module.FACE_LANDMARKER_PATH = pathlib.Path(str(config.mediapipe_flm_model_fp))
            constants_module.BLAZEGAZE_PATH = pathlib.Path(str(config.blazegaze_mlp_fp))
        except Exception:
            pass
    except Exception:
        pass
    return config


def _create_webeyetrack(config):
    """Create WebEyeTrack with CPU MediaPipe delegate when GPU setup is unavailable."""
    config = _normalize_webeyetrack_config_paths(config)
    try:
        import webeyetrack.webeyetrack as wet_module
    except Exception:
        return WebEyeTrack(config)
    config = _patch_webeyetrack_model_paths(config, wet_module)
    _patch_webeyetrack_failure_reporting(wet_module)
    _patch_webeyetrack_opencv_compat(wet_module)

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


def _patch_webeyetrack_failure_reporting(wet_module):
    """Expose the internal step failure reason instead of swallowing it inside WebEyeTrack."""
    webeyetrack_cls = getattr(wet_module, "WebEyeTrack", None)
    if webeyetrack_cls is None or getattr(webeyetrack_cls, "_ninja_failure_patch", False):
        return

    original_step = webeyetrack_cls.step

    def patched_step(self, *args, **kwargs):
        self._ninja_last_step_error = None
        try:
            return original_step(self, *args, **kwargs)
        except Exception as exc:
            self._ninja_last_step_error = f"{type(exc).__name__}: {exc}"
            raise

    original_prepare_input = webeyetrack_cls.prepare_input

    def patched_prepare_input(self, *args, **kwargs):
        try:
            return original_prepare_input(self, *args, **kwargs)
        except Exception as exc:
            self._ninja_last_step_error = f"{type(exc).__name__}: {exc}"
            raise

    webeyetrack_cls.step = patched_step
    webeyetrack_cls.prepare_input = patched_prepare_input
    webeyetrack_cls._ninja_failure_patch = True


def _patch_webeyetrack_opencv_compat(wet_module):
    """Coerce WebEyeTrack preprocessing inputs to numeric arrays for OpenCV on Windows."""
    if getattr(wet_module, "_ninja_opencv_patch", False):
        return
    try:
        import numpy as np
        import webeyetrack.model_based as model_based_module
    except Exception:
        return

    original_obtain_eyepatch = model_based_module.obtain_eyepatch

    def patched_obtain_eyepatch(frame, face_landmarks, *args, **kwargs):
        frame = np.asarray(frame, dtype=np.uint8)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        face_landmarks = np.asarray(face_landmarks, dtype=np.float32)
        return original_obtain_eyepatch(frame, face_landmarks, *args, **kwargs)

    model_based_module.obtain_eyepatch = patched_obtain_eyepatch
    wet_module.obtain_eyepatch = patched_obtain_eyepatch
    wet_module._ninja_opencv_patch = True


def _can_use_legacy_face_mesh_fallback(wet_module, exc) -> bool:
    if sys.platform != "darwin":
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
        def __init__(
            self,
            base_options=None,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        ):
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


class NinjaCursors(QtWidgets.QWidget):
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_SCREEN_WIDTH_CM = 34.0
    DEFAULT_SCREEN_HEIGHT_CM = 19.0
    DEFAULT_RAKE_SPACING = 320.0
    DEFAULT_GAZE_SMOOTHING = 0.35
    DEFAULT_GAZE_OFFSET_X = 0.0
    DEFAULT_GAZE_OFFSET_Y = -200.0
    DEFAULT_GAZE_GAIN_X = 1.0
    DEFAULT_GAZE_GAIN_Y = 1.0
    DEFAULT_TOP_HALF_EXTRA_Y = -200.0
    DEFAULT_SELECTION_HOLD = 2.0
    DEFAULT_SHOW_GAZE = True
    UPWARD_SMOOTHING_CAP = 0.05
    UPWARD_DELTA_GAIN = 1.28
    CURSOR_ROWS = 2
    CURSOR_COLS = 4
    PAPER_COL_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
    PAPER_ROW_FRACTIONS = (0.25, 0.75)
    CURSOR_RADIUS = 7.0
    ACTIVE_RADIUS = 12.0
    CLICK_EPSILON = 3.0
    DEBUG_TEXT_REFRESH_SEC = 1.0
    GAZE_VALID_TTL_SEC = 0.6

    @classmethod
    def resolve_screen_size_cm(
        cls,
        screen_width_cm=None,
        screen_height_cm=None,
        screen=None,
    ) -> tuple[float, float]:
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
        detector: Optional[TargetFinder],
        cursor_filter=None,
        logger=None,
        *,
        camera_index: int = 0,
        screen_width_cm: float | None = None,
        screen_height_cm: float | None = None,
        rake_spacing: float = DEFAULT_RAKE_SPACING,
        gaze_smoothing: float = DEFAULT_GAZE_SMOOTHING,
        gaze_offset_x: float = DEFAULT_GAZE_OFFSET_X,
        gaze_offset_y: float = DEFAULT_GAZE_OFFSET_Y,
        gaze_gain_x: float = DEFAULT_GAZE_GAIN_X,
        gaze_gain_y: float = DEFAULT_GAZE_GAIN_Y,
        selection_hold: float = DEFAULT_SELECTION_HOLD,
        show_gaze: bool = DEFAULT_SHOW_GAZE,
        calib_points: int = 5,
        auto_calibrate: bool = False,
    ):
        if WebEyeTrack is None:
            raise RuntimeError(
                "WebEyeTrack is not available. Install it with `pip install webeyetrack`."
            ) from _WEBEYETRACK_IMPORT_ERROR

        super().__init__()
        self.detector = detector
        if self.detector is not None:
            self.detector.overlay_window = self
        self.cursor_filter = cursor_filter
        self.logger = logger
        self.camera_index = int(camera_index)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("Could not detect a primary screen for Ninja Cursors(gaze).")
        self.screen_width_cm, self.screen_height_cm = self.resolve_screen_size_cm(
            screen_width_cm,
            screen_height_cm,
            screen,
        )
        self.rake_spacing = max(40.0, float(rake_spacing))
        self.gaze_smoothing = max(0.0, min(float(gaze_smoothing), 0.95))
        self.gaze_offset_x = float(gaze_offset_x)
        self.gaze_offset_y = float(gaze_offset_y)
        self.gaze_gain_x = max(0.1, min(float(gaze_gain_x), 10.0))
        self.gaze_gain_y = max(0.1, min(float(gaze_gain_y), 10.0))
        self.top_half_extra_y = float(self.DEFAULT_TOP_HALF_EXTRA_Y)
        self.selection_hold = max(0.0, float(selection_hold))
        self.show_gaze = bool(show_gaze)
        self._calib_points = calib_points if calib_points in (5, 9, 13) else 5
        self._auto_calibrate = auto_calibrate

        self._mouse_listener = None
        self._keyboard_listener = None
        self._gaze_timer = None
        self._paint_timer = None
        self._cursor_refresh_timer = None
        self._quit_shortcut = None
        self._tracking_ok = False
        self._gaze_point: Optional[tuple[float, float]] = None
        self._active_cursor_id: Optional[tuple[int, int]] = None
        self._active_point: Optional[tuple[float, float]] = None
        self._active_target = None
        self._candidate_cursor_id: Optional[tuple[int, int]] = None
        self._candidate_since = 0.0
        self._cursor_locked = False
        self._locked_cursor_id: Optional[tuple[int, int]] = None
        self._simulating_click = False
        self._pending_click_point = None
        self._pending_click_target = None
        self._last_gaze_status = "waiting for webcam"
        self._last_gaze_debug_t = 0.0
        self._last_successful_gaze_t = 0.0
        self._raw_offset_x = 0.0
        self._raw_offset_y = 0.0
        self._filtered_offset_x = 0.0
        self._filtered_offset_y = 0.0
        self._last_observed_mouse = None
        self._ignore_next_mouse_delta = True

        self._calibration = None
        self._calib_status_text = ""
        self._calib_status_until = 0.0

        geom = screen.geometry()
        self.setGeometry(geom)
        self._screen_rect = geom
        self._screen_px_dimensions = (geom.width(), geom.height())
        self._anchor_point = QtCore.QPointF(geom.center())
        self._prev_real = QtCore.QPointF(self._anchor_point)
        self._last_observed_mouse = (self._anchor_point.x(), self._anchor_point.y())

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
        self._install_quit_shortcut()
        if self.detector is not None:
            self.detector.start()

        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.timeout.connect(self.update)
        self._paint_timer.start(10)

        self._gaze_timer = QtCore.QTimer(self)
        self._gaze_timer.timeout.connect(self._update_gaze)
        self._gaze_timer.start(33)

        if sys.platform == "darwin":
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(hide_cursor_everywhere)
            self._cursor_refresh_timer.start(16)

        QtCore.QTimer.singleShot(0, self._prime_hidden_cursor)

        if self._auto_calibrate:
            QtCore.QTimer.singleShot(1500, self._start_calibration)

    def _prime_hidden_cursor(self):
        QtGui.QCursor.setPos(int(self._anchor_point.x()), int(self._anchor_point.y()))
        self._prev_real = QtCore.QPointF(self._anchor_point)
        hide_cursor_everywhere()

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

    def _observe_mouse_delta(self) -> tuple[float, float, float, float, bool]:
        if self._simulating_click:
            return self._last_observed_mouse[0], self._last_observed_mouse[1], 0.0, 0.0, False

        pos = QtGui.QCursor.pos()
        observed_x = float(pos.x())
        observed_y = float(pos.y())
        dx = observed_x - float(self._prev_real.x())
        dy = observed_y - float(self._prev_real.y())
        self._last_observed_mouse = (observed_x, observed_y)
        moved = dx != 0.0 or dy != 0.0

        QtGui.QCursor.setPos(int(self._anchor_point.x()), int(self._anchor_point.y()))
        self._prev_real = QtCore.QPointF(self._anchor_point)
        if self._ignore_next_mouse_delta:
            self._ignore_next_mouse_delta = False
            return self._last_observed_mouse[0], self._last_observed_mouse[1], 0.0, 0.0, False
        return self._last_observed_mouse[0], self._last_observed_mouse[1], dx, dy, moved

    def _apply_mouse_delta(self, dx: float, dy: float):
        self._raw_offset_x += dx
        self._raw_offset_y += dy
        if self.cursor_filter is not None:
            self._filtered_offset_x, self._filtered_offset_y = self.cursor_filter.filter(
                self._raw_offset_x,
                self._raw_offset_y,
            )
        else:
            self._filtered_offset_x = self._raw_offset_x
            self._filtered_offset_y = self._raw_offset_y

    def _wrapped_screen_point(self, x: float, y: float) -> tuple[float, float]:
        left = float(self._screen_rect.left())
        top = float(self._screen_rect.top())
        width = float(self._screen_rect.width())
        height = float(self._screen_rect.height())
        wrapped_x = left + ((x - left) % width)
        wrapped_y = top + ((y - top) % height)
        return wrapped_x, wrapped_y

    def _base_cursor_points(self) -> list[tuple[tuple[int, int], tuple[float, float]]]:
        left = float(self._screen_rect.left())
        top = float(self._screen_rect.top())
        width = float(self._screen_rect.width())
        height = float(self._screen_rect.height())
        spread_scale = max(0.25, float(self.rake_spacing) / float(self.DEFAULT_RAKE_SPACING))
        center_x = left + width * 0.5
        center_y = top + height * 0.5

        points: list[tuple[tuple[int, int], tuple[float, float]]] = []
        for row, fraction_y in enumerate(self.PAPER_ROW_FRACTIONS):
            base_y = top + height * fraction_y
            y = center_y + (base_y - center_y) * spread_scale
            for col, fraction_x in enumerate(self.PAPER_COL_FRACTIONS):
                base_x = left + width * fraction_x
                x = center_x + (base_x - center_x) * spread_scale
                points.append(((row, col), (x, y)))
        return points

    def _grid_points(self) -> list[tuple[tuple[int, int], tuple[float, float]]]:
        points: list[tuple[tuple[int, int], tuple[float, float]]] = []
        for cursor_id, (base_x, base_y) in self._base_cursor_points():
            x = base_x + self._filtered_offset_x
            y = base_y + self._filtered_offset_y
            points.append((cursor_id, self._wrapped_screen_point(x, y)))
        return points

    def _active_cursor_from_gaze(
        self,
        points: list[tuple[tuple[int, int], tuple[float, float]]],
    ) -> tuple[int, int]:
        point_map = {cursor_id: point for cursor_id, point in points}
        now = time.time()

        if self._cursor_locked and self._locked_cursor_id in point_map:
            self._active_cursor_id = self._locked_cursor_id
            return self._locked_cursor_id

        if self._gaze_is_recent() and self._gaze_point is not None:
            gx, gy = self._gaze_point
            candidate_id, _candidate_point = min(
                points,
                key=lambda item: (item[1][0] - gx) ** 2 + (item[1][1] - gy) ** 2,
            )
            if candidate_id != self._candidate_cursor_id:
                self._candidate_cursor_id = candidate_id
                self._candidate_since = now
            dwell_time = max(0.0, float(self.selection_hold))
            if dwell_time <= 0.0 or (now - self._candidate_since) >= dwell_time:
                self._lock_active_cursor(candidate_id)
                return candidate_id
            self._active_cursor_id = candidate_id
            return candidate_id

        self._candidate_cursor_id = None
        self._candidate_since = 0.0

        if self._active_cursor_id is not None and self._active_cursor_id in point_map:
            return self._active_cursor_id

        if not points:
            raise RuntimeError("Ninja Cursors(gaze) needs at least one candidate cursor.")

        else:
            ax = float(self._anchor_point.x())
            ay = float(self._anchor_point.y())
            candidate_id, _candidate_point = min(
                points,
                key=lambda item: (item[1][0] - ax) ** 2 + (item[1][1] - ay) ** 2,
            )
        self._active_cursor_id = candidate_id
        return candidate_id

    def _lock_active_cursor(self, active_id: tuple[int, int] | None):
        if active_id is None:
            return
        self._cursor_locked = True
        self._locked_cursor_id = active_id
        self._active_cursor_id = active_id
        self._candidate_cursor_id = active_id
        self._candidate_since = time.time()

    def _unlock_cursor_selection(self):
        self._cursor_locked = False
        self._locked_cursor_id = None
        self._candidate_cursor_id = None
        self._candidate_since = 0.0
        self._active_cursor_id = None
        self._active_target = None

    def _apply_gaze_smoothing(self, x: float, y: float):
        if self._gaze_point is None:
            self._gaze_point = (x, y)
            return
        keep = self.gaze_smoothing
        old_x, old_y = self._gaze_point
        keep_x = keep
        keep_y = keep
        if y < old_y:
            # Make upward gaze movements react more readily, because webcam
            # tracking tends to compress eye-only upward motion.
            y = old_y + (y - old_y) * self.UPWARD_DELTA_GAIN
            keep_y = min(keep_y, self.UPWARD_SMOOTHING_CAP)
        self._gaze_point = (
            old_x * keep_x + x * (1.0 - keep_x),
            old_y * keep_y + y * (1.0 - keep_y),
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
            # OpenCV returns BGR frames, while WebEyeTrack / MediaPipe expects SRGB.
            frame_for_tracking = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self._tracking_ok = False
            self._set_gaze_status(f"frame conversion failed: {type(exc).__name__}: {exc}")
            return

        try:
            status, gaze_result, detection_results = self._tracker.process_frame(frame_for_tracking)
        except Exception as exc:
            self._tracking_ok = False
            self._set_gaze_status(f"tracking error: {type(exc).__name__}: {exc}")
            return

        if self._calibration and self._calibration.is_calibrating:
            if status == TrackingStatus.SUCCESS and gaze_result is not None:
                self._calibration.feed(gaze_result)
            else:
                print(f"[calib] no gaze: status={status}, gaze_result={'None' if gaze_result is None else 'exists'}")
            return

        norm_pog = getattr(gaze_result, "norm_pog", None) if gaze_result is not None else None
        if status == TrackingStatus.SUCCESS and norm_pog is not None:
            gx = self._screen_rect.left() + (float(norm_pog[0]) + 0.5) * self._screen_rect.width()
            gy = self._screen_rect.top() + (float(norm_pog[1]) + 0.5) * self._screen_rect.height()
            calibrated = self._calibration and self._calibration.is_calibrated
            if not calibrated:
                center_x = self._screen_rect.left() + self._screen_rect.width() * 0.5
                center_y = self._screen_rect.top() + self._screen_rect.height() * 0.5
                gx = center_x + (gx - center_x) * self.gaze_gain_x
                gy = center_y + (gy - center_y) * self.gaze_gain_y
                assist_start_y = self._screen_rect.top() + self._screen_rect.height() * 0.68
                if gy < assist_start_y:
                    assist_span = max(1.0, assist_start_y - float(self._screen_rect.top()))
                    assist_ratio = max(0.0, min(1.0, (assist_start_y - gy) / assist_span))
                    assist_strength = math.pow(assist_ratio, 1.35)
                    gy += self.top_half_extra_y * assist_strength
                gx += self.gaze_offset_x
                gy += self.gaze_offset_y
            gx = max(float(self._screen_rect.left()), min(float(self._screen_rect.right()), gx))
            gy = max(float(self._screen_rect.top()), min(float(self._screen_rect.bottom()), gy))
            self._apply_gaze_smoothing(gx, gy)
            self._tracking_ok = True
            self._last_successful_gaze_t = time.time()
            self._set_gaze_status(f"tracking ok gaze=({gx:.0f}, {gy:.0f}) {'[calibrated]' if calibrated else ''}")
        else:
            self._tracking_ok = False
            face_count = 0
            try:
                face_count = len(getattr(detection_results, "face_landmarks", []) or [])
            except Exception:
                face_count = 0
            detail = getattr(self._tracker, "_ninja_last_step_error", None)
            suffix = f" detail={detail}" if detail else ""
            self._set_gaze_status(
                f"not tracking status={status} faces={face_count} norm_pog={'yes' if norm_pog is not None else 'no'}{suffix}"
            )

    def _paint_calibration(self):
        calib = self._calibration
        if not calib or not calib.targets:
            return
        idx = calib.current_point_idx
        if idx >= len(calib.targets):
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 180))
        painter.drawRect(self.rect())

        tx, ty = calib.targets[idx]
        elapsed = time.time() - calib._start_time
        progress = min(elapsed / calib.HOLD_SEC, 1.0)

        ring_r = 12 + 28 * (1 - progress)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80, 180), 3))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QtCore.QPointF(tx, ty), ring_r, ring_r)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 50, 50, 240))
        painter.drawEllipse(QtCore.QPointF(tx, ty), 8, 8)

        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255, 220))
        text = f"Calibration point {idx + 1}/{len(calib.targets)} - Look at the red dot"
        text_rect = QtCore.QRectF(0, self.height() - 60, self.width(), 40)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

        bar_w = 300
        bar_x = (self.width() - bar_w) / 2
        bar_y = self.height() - 30
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(60, 60, 60, 200))
        painter.drawRoundedRect(int(bar_x), int(bar_y), bar_w, 8, 4, 4)
        painter.setBrush(QtGui.QColor(80, 200, 120, 230))
        painter.drawRoundedRect(int(bar_x), int(bar_y), int(bar_w * progress), 8, 4, 4)

        painter.end()

    def _paint_calib_status(self, painter: QtGui.QPainter):
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(80, 220, 120, 240))
        text_rect = QtCore.QRectF(0, self.height() / 2 - 20, self.width(), 40)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._calib_status_text)

    def _draw_debug_status(self, painter: QtGui.QPainter):
        painter.save()
        box_width = max(420.0, min(980.0, float(self.width()) - 32.0))
        box = QtCore.QRectF(16.0, 16.0, box_width, 80.0)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.drawRoundedRect(box, 8.0, 8.0)

        title = f"Ninja gaze: {self._last_gaze_status}"
        if self._active_point is not None:
            freshness = "recent" if self._gaze_is_recent() else "stale"
            if self._cursor_locked:
                cursor_state = "locked"
            elif self._candidate_cursor_id is not None and self._candidate_cursor_id == self._active_cursor_id:
                dwell_elapsed = max(0.0, time.time() - self._candidate_since)
                cursor_state = f"candidate {dwell_elapsed:.2f}/{self.selection_hold:.2f}s"
            else:
                cursor_state = "free"
            title += f" | active={self._active_cursor_id} | gaze={freshness} | {cursor_state}"

        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 235), 1))
        painter.drawText(
            QtCore.QRectF(28.0, 24.0, box.width() - 24.0, 22.0),
            QtCore.Qt.AlignmentFlag.AlignLeft,
            title[:140],
        )

        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 240, 255, 230), 1))
        painter.drawText(
            QtCore.QRectF(28.0, 50.0, box.width() - 24.0, 30.0),
            QtCore.Qt.AlignmentFlag.AlignLeft,
            "Look at one cursor and keep your gaze there to lock it, then use the mouse for local refinement.",
        )
        painter.restore()

    def paintEvent(self, event):
        if self._calibration and self._calibration.is_calibrating:
            self._paint_calibration()
            return

        if self._calib_status_text and time.time() < self._calib_status_until:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self._paint_calib_status(painter)
            painter.end()

        if self._simulating_click:
            return

        observed_x, observed_y, dx, dy, mouse_moved = self._observe_mouse_delta()
        points = self._grid_points()
        if not points:
            return

        active_id = self._active_cursor_from_gaze(points)
        if mouse_moved and self._cursor_locked:
            self._apply_mouse_delta(dx, dy)
            points = self._grid_points()
            active_id = self._active_cursor_from_gaze(points)
        point_map = {cursor_id: point for cursor_id, point in points}
        self._active_point = point_map.get(active_id)
        if self._active_point is None:
            return

        ax, ay = self._active_point
        if self._cursor_locked and self.detector is not None:
            self._active_target = self.detector.find_detection_for_point(
                float(ax),
                float(ay),
                include_text=True,
                fallback_nearest=False,
            )
        else:
            self._active_target = None

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if self._active_target is not None:
            x = self._active_target["x"]
            y = self._active_target["y"]
            w = self._active_target["w"]
            h = self._active_target["h"]
            rect = QtCore.QRectF(x, y, w, h)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 210, 40, 230), 3))
            painter.setBrush(QtGui.QColor(255, 210, 40, 22))
            painter.drawRoundedRect(rect, 10, 10)

        for cursor_id, (px, py) in points:
            is_active = cursor_id == active_id
            is_candidate = is_active and not self._cursor_locked
            is_locked_out = self._cursor_locked and not is_active
            outer_pen = QtGui.QPen(
                QtGui.QColor(0, 214, 96, 255)
                if self._cursor_locked and is_active
                else QtGui.QColor(255, 210, 40, 230)
                if is_candidate
                else QtGui.QColor(160, 160, 160, 110 if is_locked_out else 150),
                5 if is_active else 2,
            )
            painter.setPen(outer_pen)
            painter.setBrush(
                QtGui.QColor(0, 214, 96, 48)
                if self._cursor_locked and is_active
                else QtGui.QColor(255, 210, 40, 34)
                if is_candidate
                else QtGui.QColor(120, 120, 120, 16 if is_locked_out else 10)
            )
            radius = self.ACTIVE_RADIUS if is_active else self.CURSOR_RADIUS
            painter.drawEllipse(QtCore.QPointF(px, py), radius, radius)
            cross_pen = QtGui.QPen(
                QtGui.QColor(255, 255, 255, 240 if is_active else (110 if is_locked_out else 150)),
                2,
            )
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
            candidate_elapsed_ms = 0.0
            if self._candidate_cursor_id is not None and self._candidate_cursor_id == active_id:
                candidate_elapsed_ms = max(0.0, (time.time() - self._candidate_since) * 1000.0)
            fields = {
                "technique": "rake",
                "filter_name": self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                "active_cursor_id": list(active_id),
                "active": [round(float(ax), 3), round(float(ay), 3)],
                "tracking_ok": bool(self._tracking_ok),
                "dwell_lock_time": round(float(self.selection_hold), 3),
                "cursor_locked": bool(self._cursor_locked),
                "candidate_cursor_id": list(self._candidate_cursor_id) if self._candidate_cursor_id is not None else None,
                "candidate_elapsed_ms": round(float(candidate_elapsed_ms), 3),
                "rake_spacing": round(float(self.rake_spacing), 3),
                "gaze_offset_x": round(float(self.gaze_offset_x), 3),
                "gaze_offset_y": round(float(self.gaze_offset_y), 3),
                "selection_mode": "nearest_gaze_cursor_with_dwell_lock",
                "cursor_count": self.CURSOR_ROWS * self.CURSOR_COLS,
                "rake_layout": "ninja8_grid",
                "detection_count": len(self.detector.get_detections()) if self.detector is not None else 0,
            }
            if self._gaze_point is not None:
                fields["gaze"] = [round(float(self._gaze_point[0]), 3), round(float(self._gaze_point[1]), 3)]
            self.logger.log_cursor_sample(
                raw_x=round(float(self._raw_offset_x), 3),
                raw_y=round(float(self._raw_offset_y), 3),
                filtered_x=round(float(self._filtered_offset_x), 3),
                filtered_y=round(float(self._filtered_offset_y), 3),
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
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        if self.detector is not None:
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
        self._pressed_keys = set()

        def on_press(key):
            self._pressed_keys.add(key)
            key_char = getattr(key, "char", None)
            if isinstance(key_char, str) and key_char.lower() == "q":
                self._queue_quit()
            if key == keyboard.Key.esc:
                if self._calibration and self._calibration.is_calibrating:
                    self._calibration.abort()
                    self._calib_status_text = "Calibration cancelled"
                    self._calib_status_until = time.time() + 2.0
            if isinstance(key_char, str) and key_char.lower() == "c":
                has_cmd = keyboard.Key.cmd in self._pressed_keys or keyboard.Key.cmd_r in self._pressed_keys
                has_shift = keyboard.Key.shift in self._pressed_keys or keyboard.Key.shift_r in self._pressed_keys
                if has_cmd and has_shift:
                    QtCore.QMetaObject.invokeMethod(
                        self, "_start_calibration",
                        QtCore.Qt.ConnectionType.QueuedConnection,
                    )

        def on_release(key):
            self._pressed_keys.discard(key)

        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._keyboard_listener.start()

    def _install_quit_shortcut(self):
        self._quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("q"), self)
        self._quit_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._quit_shortcut.activated.connect(self.stop_and_quit)

        calib_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+C"), self
        )
        calib_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        calib_shortcut.activated.connect(self._start_calibration)
        self._calib_shortcut = calib_shortcut

    def _queue_quit(self):
        QtCore.QMetaObject.invokeMethod(
            self,
            "stop_and_quit",
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    @QtCore.pyqtSlot()
    def _start_calibration(self):
        sw, sh = self._screen_px_dimensions
        self._calibration = EyeCalibration(
            sw, sh,
            num_points=self._calib_points,
            on_progress=self._on_calib_progress,
            on_done=self._on_calib_done,
        )
        self._calibration.start(self._tracker)
        print(f"Eye calibration started ({self._calibration.num_points} points). "
              "Look at each red dot. Press ESC to cancel.")

    def _on_calib_progress(self, point_idx, progress):
        pass

    def _on_calib_done(self, success, mean_error_px):
        if success:
            self._calib_status_text = f"Calibrated! Error: {mean_error_px:.0f}px"
        else:
            self._calib_status_text = "Calibration failed"
        self._calib_status_until = time.time() + 3.0

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
        if event_type not in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            return event
        try:
            px, py = Quartz.CGEventGetLocation(event)
        except Exception:
            return event
        if self._in_system_reserved_area(int(px), int(py)):
            return event
        if not self._cursor_locked or self._active_point is None:
            if event_type == Quartz.kCGEventLeftMouseUp and self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[round(float(px), 3), round(float(py), 3)],
                    effective=[round(float(px), 3), round(float(py), 3)],
                    redirected=False,
                    target=None,
                    ignored=True,
                    reason="cursor_not_locked",
                )
            return None

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
                f"active={self._active_cursor_id} effective=({tx:.1f}, {ty:.1f}) "
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
            self._unlock_cursor_selection()
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

        if not self._cursor_locked or self._active_point is None:
            if self.logger is not None:
                self.logger.log_click(
                    technique="rake",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=None,
                    ignored=True,
                    reason="cursor_not_locked",
                )
            return

        tx, ty = self._active_point
        redirected = math.hypot(float(orig_x) - tx, float(orig_y) - ty) > self.CLICK_EPSILON
        print(
            f"[rake] click raw=({orig_x}, {orig_y}) active={self._active_cursor_id} "
            f"effective=({tx:.1f}, {ty:.1f}) redirected={redirected} gaze_recent={self._gaze_is_recent()}",
            flush=True,
        )

        self._send_click(orig_x, orig_y, tx, ty)
        if self.logger is not None:
            self.logger.log_click(
                technique="rake",
                raw=[orig_x, orig_y],
                effective=[round(float(tx), 3), round(float(ty), 3)],
                redirected=redirected,
                target=self._active_target,
            )
        self._unlock_cursor_selection()

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
            QtGui.QCursor.setPos(int(self._anchor_point.x()), int(self._anchor_point.y()))
            self._prev_real = QtCore.QPointF(self._anchor_point)
            self._ignore_next_mouse_delta = True
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


def ninja_cursors(detector: Optional[TargetFinder], cursor_filter=None, logger=None, **kwargs):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    overlay = NinjaCursors(detector, cursor_filter=cursor_filter, logger=logger, **kwargs)
    overlay.show()
    hide_cursor_everywhere()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if logger is not None:
        logger.log_session_end(reason="app_exit")
        logger.close()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(description="Launch the Ninja Cursors(gaze) overlay")
    parser.add_argument("--model-path", default=None, help="Path to the YOLO model .pt file")
    parser.add_argument("--change-thresh", type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument("--capture-interval", type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument("--confidence", type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument("--iou", type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    parser.add_argument("--filter", choices=sorted(FILTER_OPTIONS.keys()), default="none", help="Optional pointer filter")
    parser.add_argument("--log-file", default=None, help="Optional JSONL log file path")
    parser.add_argument("--log-cursor-hz", type=float, default=30.0, help="Cursor sampling rate for logging")
    parser.add_argument("--camera-index", type=int, default=NinjaCursors.DEFAULT_CAMERA_INDEX, help="Webcam index used for gaze tracking")
    parser.add_argument("--screen-width-cm", type=float, default=None, help="Approximate physical screen width in centimeters; auto-detected when omitted")
    parser.add_argument("--screen-height-cm", type=float, default=None, help="Approximate physical screen height in centimeters; auto-detected when omitted")
    parser.add_argument("--rake-spacing", type=float, default=NinjaCursors.DEFAULT_RAKE_SPACING, help="Center-to-center spacing in pixels between neighboring cursors in the 8-cursor Ninja layout")
    parser.add_argument("--gaze-smoothing", type=float, default=NinjaCursors.DEFAULT_GAZE_SMOOTHING, help="Smoothing factor applied to the gaze point (0 = no smoothing, higher = steadier gaze)")
    parser.add_argument("--gaze-offset-x", type=float, default=NinjaCursors.DEFAULT_GAZE_OFFSET_X, help="Horizontal pixel offset applied to the webcam-estimated gaze point before cursor selection")
    parser.add_argument("--gaze-offset-y", type=float, default=NinjaCursors.DEFAULT_GAZE_OFFSET_Y, help="Vertical pixel offset applied to the webcam-estimated gaze point before cursor selection")
    parser.add_argument("--gaze-gain-x", type=float, default=NinjaCursors.DEFAULT_GAZE_GAIN_X, help="Horizontal gain applied around the screen center before gaze offset and cursor selection")
    parser.add_argument("--gaze-gain-y", type=float, default=NinjaCursors.DEFAULT_GAZE_GAIN_Y, help="Vertical gain applied around the screen center before gaze offset and cursor selection")
    parser.add_argument("--gaze-gain", type=float, default=2.0, help="Deprecated compatibility option from the old local-direction rake; ignored by this 8-cursor version")
    parser.add_argument("--selection-hold", type=float, default=NinjaCursors.DEFAULT_SELECTION_HOLD, help="Seconds gaze must remain on the same cursor before it locks automatically")
    parser.add_argument("--hide-gaze-point", action="store_true", help="Hide the red on-screen gaze feedback marker")
    parser.add_argument("--calib-points", type=int, choices=[5, 9, 13], default=5, help="Number of calibration points (5, 9, or 13)")
    parser.add_argument("--auto-calibrate", action="store_true", help="Start eye calibration immediately on launch")
    parser.add_argument("--without-targetfinder", action="store_true", help="Run Ninja Cursors(gaze) without TargetFinder detection or target highlighting")
    args = parser.parse_args()

    if WebEyeTrack is None:
        raise SystemExit(
            "WebEyeTrack is not available in this environment. Install it with `pip install webeyetrack`."
        )

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "yolo26s_1280.pt")

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    args.screen_width_cm, args.screen_height_cm = NinjaCursors.resolve_screen_size_cm(
        args.screen_width_cm,
        args.screen_height_cm,
    )

    det = None
    if not args.without_targetfinder:
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
            gaze_offset_x=args.gaze_offset_x,
            gaze_offset_y=args.gaze_offset_y,
            gaze_gain_x=args.gaze_gain_x,
            gaze_gain_y=args.gaze_gain_y,
            selection_hold=args.selection_hold,
            show_gaze=not args.hide_gaze_point,
            without_targetfinder=bool(args.without_targetfinder),
            rake_layout="ninja8_grid",
            cursor_count=NinjaCursors.CURSOR_ROWS * NinjaCursors.CURSOR_COLS,
            selection_mode="nearest_gaze_cursor_then_mouse_lock",
        )
        if det is not None:
            det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    ninja_cursors(
        det,
        cursor_filter=cursor_filter,
        logger=logger,
        camera_index=args.camera_index,
        screen_width_cm=args.screen_width_cm,
        screen_height_cm=args.screen_height_cm,
        rake_spacing=args.rake_spacing,
        gaze_smoothing=args.gaze_smoothing,
        gaze_offset_x=args.gaze_offset_x,
        gaze_offset_y=args.gaze_offset_y,
        gaze_gain_x=args.gaze_gain_x,
        gaze_gain_y=args.gaze_gain_y,
        selection_hold=args.selection_hold,
        show_gaze=not args.hide_gaze_point,
        calib_points=args.calib_points,
        auto_calibrate=args.auto_calibrate,
    )


if __name__ == "__main__":
    main()
