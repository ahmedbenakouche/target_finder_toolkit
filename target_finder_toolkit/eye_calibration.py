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
            returned_calib = tracker.adapt_from_gaze_results(
                calib_gaze_results,
                np.array(calib_norm_pogs),
                affine_transform=True,
                steps_inner=10,
                inner_lr=1e-4,
                pt_type='calib',
            )
            self._calibrated = True

            if tracker.affine_matrix is not None:
                m = tracker.affine_matrix.copy()
                scale_x = np.linalg.norm(m[0, :2])
                scale_y = np.linalg.norm(m[1, :2])
                max_scale = 1.25
                if scale_x > max_scale:
                    m[0, :2] *= max_scale / scale_x
                if scale_y > max_scale:
                    m[1, :2] *= max_scale / scale_y
                tracker.affine_matrix = m
                tracker.affine_matrix_tf = None
                try:
                    import tensorflow as tf
                    tracker.affine_matrix_tf = tf.convert_to_tensor(m, dtype=tf.float32)
                except Exception:
                    pass

                clamped_sx = np.linalg.norm(m[0, :2])
                clamped_sy = np.linalg.norm(m[1, :2])
                offset_x_px = float(m[0, 2] * self.screen_w)
                offset_y_px = float(m[1, 2] * self.screen_h)
                self._correction_values = {
                    "gaze_gain_x": float(clamped_sx),
                    "gaze_gain_y": float(clamped_sy),
                    "gaze_offset_x": offset_x_px,
                    "gaze_offset_y": offset_y_px,
                    "affine_matrix": m.tolist(),
                }
                print(f"[calib] scale: raw=({scale_x:.3f}, {scale_y:.3f}) clamped=({clamped_sx:.3f}, {clamped_sy:.3f})")
                print(f"[calib] offset_px=({offset_x_px:.0f}, {offset_y_px:.0f})")

            preds = np.array(returned_calib)
            targets = np.array(calib_norm_pogs)
            diff_px = (preds - targets) * np.array([self.screen_w, self.screen_h])
            errors_px = np.sqrt(np.sum(diff_px ** 2, axis=1))
            mean_err_px = float(np.mean(errors_px))

            self._save_result(mean_err_px)
            # The panel uses calibration as an automatic initializer for the
            # editable gain/offset fields. Disable WebEyeTrack's affine state so
            # the runtime does not apply both affine and manual corrections.
            tracker.affine_matrix = None
            tracker.affine_matrix_tf = None
            print(f"Eye calibration complete! Mean error: {mean_err_px:.1f}px")
            if self.on_done:
                self.on_done(True, mean_err_px)

        except Exception as e:
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
