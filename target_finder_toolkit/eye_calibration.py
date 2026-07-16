"""
Eye tracking calibration module using WebEyeTrack's MAML few-shot adaptation.

Provides multi-point calibration (5, 9, or 13 points) that fits an affine
transform to map raw gaze estimates to accurate screen coordinates.
Designed to integrate with an existing WebEyeTrack instance and gaze loop.
"""

import json
import pathlib
import time

import numpy as np


class EyeCalibration:
    HOLD_SEC = 3.0
    SETTLE_SEC = 1.5
    GAZE_RESULTS_PER_POINT = 30

    POINT_LAYOUTS = {
        5: lambda sw, sh, mx, my, cx, cy: [
            (cx, cy),
            (mx, my), (sw - mx, my),
            (mx, sh - my), (sw - mx, sh - my),
        ],
        9: lambda sw, sh, mx, my, cx, cy: [
            (cx, cy),
            (mx, my), (sw - mx, my),
            (mx, sh - my), (sw - mx, sh - my),
            (cx, my), (mx, cy),
            (sw - mx, cy), (cx, sh - my),
        ],
        13: lambda sw, sh, mx, my, cx, cy: [
            (cx, cy),
            (mx, my), (sw - mx, my),
            (mx, sh - my), (sw - mx, sh - my),
            (cx, my), (mx, cy),
            (sw - mx, cy), (cx, sh - my),
            (cx // 2, cy // 2), (cx + cx // 2, cy // 2),
            (cx // 2, cy + cy // 2), (cx + cx // 2, cy + cy // 2),
        ],
    }

    SAVE_DIR = pathlib.Path.home() / ".target_finder_toolkit" / "eye_calibration"

    def __init__(self, screen_w, screen_h, num_points=5,
                 on_progress=None, on_done=None):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._num_points = num_points if num_points in self.POINT_LAYOUTS else 5
        self.on_progress = on_progress
        self.on_done = on_done

        self._calibrating = False
        self._point_idx = 0
        self._start_time = 0.0
        self._gaze_results: list[list] = []
        self._frame_count = 0
        self._targets = []
        self._calibrated = False
        self._correction_values = None

    @property
    def is_calibrating(self):
        return self._calibrating

    @property
    def is_calibrated(self):
        return self._calibrated

    @property
    def correction_values(self):
        return self._correction_values

    @property
    def current_point_idx(self):
        return self._point_idx

    @property
    def targets(self):
        return self._targets

    @property
    def num_points(self):
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        if value in self.POINT_LAYOUTS:
            self._num_points = value

    def get_targets(self):
        sw, sh = self.screen_w, self.screen_h
        mx, my = int(sw * 0.1), int(sh * 0.1)
        cx, cy = sw // 2, sh // 2
        return self.POINT_LAYOUTS[self._num_points](sw, sh, mx, my, cx, cy)

    def start(self, tracker):
        """Begin calibration. Clears any existing affine_matrix on the tracker."""
        self._tracker_ref = tracker
        tracker.affine_matrix = None
        tracker.affine_matrix_tf = None
        self._calibrated = False
        self._correction_values = None
        self._calibrating = True
        self._point_idx = 0
        self._start_time = time.time()
        self._gaze_results = []
        self._frame_count = 0
        self._targets = self.get_targets()

    def abort(self):
        if not self._calibrating:
            return
        self._calibrating = False
        self._gaze_results = []
        self._frame_count = 0
        print("Eye calibration aborted.")

    def feed(self, gaze_result):
        """Feed a single gaze_result from process_frame during calibration.

        Call this every frame while is_calibrating is True.
        Returns True while calibration is still in progress, False when done.
        """
        if not self._calibrating:
            return False

        now = time.time()
        elapsed = now - self._start_time
        progress = min(elapsed / self.HOLD_SEC, 1.0)

        if self.on_progress:
            self.on_progress(self._point_idx, progress)

        if elapsed > self.SETTLE_SEC and self._frame_count < self.GAZE_RESULTS_PER_POINT:
            
            if self._frame_count % 5 == 0:
                print(
                    f"[raw gaze] point={self._point_idx}, "
                    f"norm_pog={getattr(gaze_result, 'norm_pog', None)}"
                )

            if len(self._gaze_results) <= self._point_idx:
                self._gaze_results.append([])
            self._gaze_results[self._point_idx].append(gaze_result)
            self._frame_count += 1

        if elapsed >= self.HOLD_SEC:
            samples = len(self._gaze_results[self._point_idx]) if self._point_idx < len(self._gaze_results) else 0
            print(f"[calib] point {self._point_idx} done, collected {samples} samples")
            self._point_idx += 1
            self._frame_count = 0
            self._start_time = now

            if self._point_idx >= len(self._targets):
                self._calibrating = False
                print(f"[calib] total points with data: {len(self._gaze_results)}")
                self._fit()
                return False

        return True

    def _fit(self):
        tracker = self._tracker_ref
        calib_gaze_results = []
        calib_norm_pogs = []

        for i, gaze_results in enumerate(self._gaze_results):
            if not gaze_results:
                print(f"WARNING: No samples for calibration point {i}")
                if self.on_done:
                    self.on_done(False, None)
                return

            tx, ty = self._targets[i]
            norm_x = tx / self.screen_w - 0.5
            norm_y = ty / self.screen_h - 0.5

            # Outlier rejection: drop samples > 2 std from mean
            pogs = np.array([gr.norm_pog for gr in gaze_results])
            mean_pog = np.mean(pogs, axis=0)
            distances = np.linalg.norm(pogs - mean_pog, axis=1)
            threshold = np.mean(distances) + 2 * np.std(distances)
            for j, gr in enumerate(gaze_results):
                if distances[j] <= threshold:
                    calib_gaze_results.append(gr)
                    calib_norm_pogs.append((norm_x, norm_y))

        print(f"[calib] fitting with {len(calib_gaze_results)} total samples")

        try:
            # Fit one transform from the exact coordinates produced at runtime.
            # WebEyeTrack's adapt(..., affine_transform=True) computes affine
            # *before* updating the MAML gaze head.  The subsequent weight update
            # changes the source coordinate space and can make that affine matrix
            # invalid (often pushing every result above/left of the screen).
            # Keeping the model fixed makes calibration deterministic.
            raw_pogs = np.asarray(
                [gr.norm_pog for gr in calib_gaze_results],
                dtype=np.float64,
            )
            targets = np.asarray(calib_norm_pogs, dtype=np.float64)
            if raw_pogs.shape != targets.shape or raw_pogs.ndim != 2 or raw_pogs.shape[1] != 2:
                raise RuntimeError(
                    f"Invalid calibration samples: raw={raw_pogs.shape}, targets={targets.shape}"
                )
            source_augmented = np.column_stack(
                [raw_pogs, np.ones(raw_pogs.shape[0], dtype=np.float64)]
            )
            if np.linalg.matrix_rank(source_augmented) < 3:
                raise RuntimeError(
                    "Calibration samples do not span the screen; keep the head still "
                    "and look directly at every point"
                )
            coefficients, _, _, _ = np.linalg.lstsq(
                source_augmented,
                targets,
                rcond=None,
            )
            m = coefficients.T.astype(np.float32)
            if m.shape != (2, 3) or not np.all(np.isfinite(m)):
                raise RuntimeError(f"Invalid affine calibration matrix: shape={m.shape}")
            tracker.affine_matrix = m
            try:
                import tensorflow as tf
                tracker.affine_matrix_tf = tf.convert_to_tensor(m, dtype=tf.float32)
            except Exception:
                tracker.affine_matrix_tf = None

            # The Kalman state was built from uncalibrated coordinates.  Reset
            # it before the first affine-transformed measurement; otherwise the
            # old position/velocity can pin gaze to a screen edge for seconds.
            kalman = getattr(tracker, "kalman_filter", None)
            if kalman is not None:
                if hasattr(kalman, "x"):
                    kalman.x = np.zeros_like(kalman.x)
                if hasattr(kalman, "P"):
                    kalman.P = np.eye(kalman.P.shape[0], dtype=kalman.P.dtype)

            # Manual correction must be neutral because WebEyeTrack now applies
            # the complete affine transform to norm_pog on every frame.
            self._correction_values = {
                "gaze_gain_x": 1.0,
                "gaze_gain_y": 1.0,
                "gaze_offset_x": 0.0,
                "gaze_offset_y": 0.0,
                "affine_matrix": m.tolist(),
            }
            self._calibrated = True
            print(f"[calib] affine matrix kept for runtime:\n{m}")

            preds = source_augmented @ m.T
            diff_px = (preds - targets) * np.array([self.screen_w, self.screen_h])
            errors_px = np.sqrt(np.sum(diff_px ** 2, axis=1))
            mean_err_px = float(np.mean(errors_px))

            self._save_result(mean_err_px)
            print(f"Eye calibration complete! Mean error: {mean_err_px:.1f}px")
            if self.on_done:
                self.on_done(True, mean_err_px)

        except Exception as e:
            self._calibrated = False
            self._correction_values = None
            print(f"Eye calibration error: {e}")
            if self.on_done:
                self.on_done(False, None)

    def _save_result(self, mean_error_px):
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = self.SAVE_DIR / "last_calibration.json"
        data = {
            "num_points": self._num_points,
            "mean_error_px": mean_error_px,
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "correction_values": self._correction_values,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, indent=2))
