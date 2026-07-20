"""
Eye tracking calibration module using WebEyeTrack's MAML few-shot adaptation.

Provides multi-point calibration (5, 9, or 13 points) that fits an affine
transform to map raw gaze estimates to accurate screen coordinates.
Designed to integrate with an existing WebEyeTrack instance and gaze loop.
"""

import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np


class EyeCalibration:
    HOLD_SEC = 3.0
    SETTLE_SEC = 1.5
    GAZE_RESULTS_PER_POINT = 30
    MAX_ACCEPTED_AFFINE_ERROR_FRACTION_OF_HALF_COLUMN = 1.0
    MAX_ACCEPTED_POINT_ERROR_FRACTION_OF_HALF_COLUMN = 1.5
    MIN_VALID_GAZE_RESULTS_PER_POINT = 3
    POINT_OUTLIER_MIN_THRESHOLD_NORM = 0.05

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

    # Calibration must be shared by standalone Ninja, tester tasks, and full
    # experiments. Keep it outside project logs/output directories.
    SAVE_DIR = pathlib.Path.home() / ".target_finder_toolkit" / "eye_calibration"
    LAST_SUCCESS_FILENAME = "last_calibration.json"
    LAST_FAILED_FILENAME = "last_failed_calibration.json"

    @classmethod
    def last_calibration_path(cls):
        return cls.SAVE_DIR / cls.LAST_SUCCESS_FILENAME

    @classmethod
    def last_failed_calibration_path(cls):
        return cls.SAVE_DIR / cls.LAST_FAILED_FILENAME

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
        self._diagnostics = None

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
        self._diagnostics = None
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
        point_summaries = []

        for i, gaze_results in enumerate(self._gaze_results):
            if not gaze_results:
                print(f"WARNING: No samples for calibration point {i}")
                if self.on_done:
                    self.on_done(False, None)
                return

            tx, ty = self._targets[i]
            norm_x = tx / self.screen_w - 0.5
            norm_y = ty / self.screen_h - 0.5

            # Per-point robust gaze estimate:
            # 1) discard invalid frames,
            # 2) remove within-point outliers around the point median,
            # 3) use exactly one median gaze sample for this target.
            # This keeps the current timing unchanged while preventing one bad
            # frame from pulling the calibration fit away from the true point.
            valid_pogs = self._valid_norm_pogs(gaze_results)
            kept_pogs, outlier_threshold = self._filter_point_outliers(valid_pogs)

            if kept_pogs.shape[0] < self.MIN_VALID_GAZE_RESULTS_PER_POINT:
                print(
                    f"WARNING: Too few valid samples for calibration point {i}: "
                    f"{kept_pogs.shape[0]} kept"
                )
                if self.on_done:
                    self.on_done(False, None)
                return
            median_pog = np.median(kept_pogs, axis=0)
            calib_gaze_results.append(SimpleNamespace(norm_pog=median_pog))
            calib_norm_pogs.append((norm_x, norm_y))
            point_summaries.append(
                {
                    "index": int(i),
                    "target_px": [float(tx), float(ty)],
                    "target_norm": [float(norm_x), float(norm_y)],
                    "sample_count_raw": int(len(gaze_results)),
                    "sample_count_valid": int(valid_pogs.shape[0]),
                    "sample_count_kept": int(kept_pogs.shape[0]),
                    "sample_count_rejected": int(valid_pogs.shape[0] - kept_pogs.shape[0]),
                    "outlier_threshold_norm": float(outlier_threshold),
                    "raw_median_norm": [
                        float(median_pog[0]),
                        float(median_pog[1]),
                    ],
                    "raw_mean_norm": [
                        float(np.mean(kept_pogs[:, 0])),
                        float(np.mean(kept_pogs[:, 1])),
                    ],
                    "raw_std_norm": [
                        float(np.std(kept_pogs[:, 0])),
                        float(np.std(kept_pogs[:, 1])),
                    ],
                }
            )

        print(f"[calib] fitting with {len(calib_gaze_results)} median point samples")

        try:
            fit = self._fit_affine(calib_gaze_results, calib_norm_pogs, point_summaries)
        except Exception as e:
            self._calibrated = False
            self._correction_values = None
            self._diagnostics = None
            tracker.affine_matrix = None
            tracker.affine_matrix_tf = None
            print(f"Eye calibration error: {e}")
            if self.on_done:
                self.on_done(False, None)
            return

        m, diagnostics, mean_err_px, manual_mean_err_px = fit
        self._correction_values = {
            "gaze_gain_x": 1.0,
            "gaze_gain_y": 1.0,
            "gaze_offset_x": 0.0,
            "gaze_offset_y": 0.0,
            "affine_matrix": m.tolist(),
            "ninja_affine_matrix": m.tolist(),
        }
        self._diagnostics = diagnostics

        # The panel uses calibration as an automatic initializer for the
        # editable gain/offset fields. Disable WebEyeTrack's affine state so
        # the runtime does not apply both affine and manual corrections.
        tracker.affine_matrix = None
        tracker.affine_matrix_tf = None
        self._reset_tracker_kalman(tracker)

        max_accepted_error_px = self._max_accepted_affine_error_px()
        max_accepted_point_error_px = self._max_accepted_point_error_px()
        max_point_err_px = float(diagnostics["affine_point_max_error_px"])

        if mean_err_px > max_accepted_error_px:
            failure_reason = (
                f"affine calibration error too high: "
                f"{mean_err_px:.1f}px > {max_accepted_error_px:.1f}px"
            )
            self._save_result(
                mean_err_px,
                diagnostics,
                accepted=False,
                failure_reason=failure_reason,
                manual_mean_error_px=manual_mean_err_px,
            )
            self._calibrated = False
            self._correction_values = None
            print(f"Eye calibration rejected: {failure_reason}")
            if self.on_done:
                self.on_done(False, mean_err_px)
            return
        if max_point_err_px > max_accepted_point_error_px:
            failure_reason = (
                f"one calibration point error too high: "
                f"{max_point_err_px:.1f}px > {max_accepted_point_error_px:.1f}px; "
                f"redo calibration"
            )
            self._save_result(
                mean_err_px,
                diagnostics,
                accepted=False,
                failure_reason=failure_reason,
                manual_mean_error_px=manual_mean_err_px,
            )
            self._calibrated = False
            self._correction_values = None
            print(f"Eye calibration rejected: {failure_reason}")
            if self.on_done:
                self.on_done(False, mean_err_px)
            return

        self._calibrated = True
        self._save_result(
            mean_err_px,
            diagnostics,
            accepted=True,
            manual_mean_error_px=manual_mean_err_px,
        )
        print(f"Eye calibration complete! Mean error: {mean_err_px:.1f}px")
        if self.on_done:
            self.on_done(True, mean_err_px)

    def _fit_affine(self, calib_gaze_results, calib_norm_pogs, point_summaries):
        """Fit the affine transform from median per-point gaze samples.

        Returns (affine_matrix, diagnostics, mean_error_px, manual_mean_error_px).
        Raises RuntimeError/ValueError on invalid input; callers handle logging
        and on_done reporting.
        """
        # Fit the affine transform from the exact norm_pog coordinates that
        # process_frame produces at runtime.  Do not call WebEyeTrack's
        # adapt_from_gaze_results here: it computes its affine matrix and
        # then changes the gaze-model weights, so clearing the affine matrix
        # afterwards leaves runtime output in a different coordinate space.
        # That mismatch can make every corrected point clip to the top-left.
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
        m = coefficients.T
        if m.shape != (2, 3) or not np.all(np.isfinite(m)):
            raise RuntimeError(f"Invalid affine calibration matrix: shape={m.shape}")

        # Ninja applies the full affine matrix itself at runtime.  The
        # editable gain/offset fields remain available as an extra manual
        # fine-tuning layer after affine calibration, so keep them neutral.
        gain_x, offset_x_norm = self._extract_axis_manual_correction(m, raw_pogs, 0)
        gain_y, offset_y_norm = self._extract_axis_manual_correction(m, raw_pogs, 1)
        offset_x_px = float(offset_x_norm * self.screen_w)
        offset_y_px = float(offset_y_norm * self.screen_h)
        print(f"[calib] affine matrix (Ninja runtime calibration):\n{m}")
        print(
            f"[calib] manual-equivalent fit for diagnostics only: "
            f"gain=({gain_x:.3f}, {gain_y:.3f}) "
            f"offset_px=({offset_x_px:.0f}, {offset_y_px:.0f})"
        )

        manual_preds = np.column_stack(
            [
                raw_pogs[:, 0] * gain_x + offset_x_norm,
                raw_pogs[:, 1] * gain_y + offset_y_norm,
            ]
        )
        manual_diff_px = (manual_preds - targets) * np.array([self.screen_w, self.screen_h])
        manual_errors_px = np.sqrt(np.sum(manual_diff_px ** 2, axis=1))
        manual_mean_err_px = float(np.mean(manual_errors_px))
        affine_preds = source_augmented @ m.T
        affine_diff_px = (affine_preds - targets) * np.array([self.screen_w, self.screen_h])
        affine_errors_px = np.sqrt(np.sum(affine_diff_px ** 2, axis=1))
        affine_sample_mean_err_px = float(np.mean(affine_errors_px))
        diagnostics = self._build_diagnostics(
            point_summaries,
            m,
            gain_x,
            gain_y,
            offset_x_norm,
            offset_y_norm,
        )
        diagnostics["affine_sample_mean_error_px"] = affine_sample_mean_err_px
        mean_err_px = float(diagnostics["affine_point_mean_error_px"])
        return m, diagnostics, mean_err_px, manual_mean_err_px

    @staticmethod
    def _extract_axis_manual_correction(affine_matrix, raw_pogs, axis: int):
        other_axis = 1 - axis
        gain = float(affine_matrix[axis, axis])
        other_median = float(np.median(raw_pogs[:, other_axis]))
        offset = float(affine_matrix[axis, 2] + affine_matrix[axis, other_axis] * other_median)
        if not np.isfinite(gain):
            gain = 1.0
        if not np.isfinite(offset):
            offset = 0.0
        gain = max(-10.0, min(gain, 10.0))
        if abs(gain) < 0.1:
            gain = 0.1 if gain >= 0.0 else -0.1
        return gain, offset

    @staticmethod
    def _valid_norm_pogs(gaze_results):
        pogs = []
        for gr in gaze_results:
            norm_pog = getattr(gr, "norm_pog", None)
            if norm_pog is None:
                continue
            arr = np.asarray(norm_pog, dtype=np.float64)
            if arr.shape != (2,) or not np.all(np.isfinite(arr)):
                continue
            pogs.append(arr)
        if not pogs:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(pogs, dtype=np.float64)

    def _filter_point_outliers(self, pogs):
        if pogs.shape[0] == 0:
            return pogs, 0.0
        median_pog = np.median(pogs, axis=0)
        distances = np.linalg.norm(pogs - median_pog, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        robust_sigma = 1.4826 * mad
        threshold = median_distance + 3.0 * robust_sigma
        threshold = max(threshold, self.POINT_OUTLIER_MIN_THRESHOLD_NORM)
        kept = pogs[distances <= threshold]
        return kept, threshold

    @staticmethod
    def _reset_tracker_kalman(tracker):
        """Discard calibration-point history before normal gaze resumes."""
        kalman = getattr(tracker, "kalman_filter", None)
        if kalman is None:
            return
        if hasattr(kalman, "x"):
            kalman.x = np.zeros_like(kalman.x)
        if hasattr(kalman, "P"):
            kalman.P = np.eye(kalman.P.shape[0], dtype=kalman.P.dtype)

    def _build_diagnostics(
        self,
        point_summaries,
        affine_matrix,
        gain_x,
        gain_y,
        offset_x_norm,
        offset_y_norm,
    ):
        points = []
        manual_errors = []
        affine_errors = []
        for summary in point_summaries:
            raw = np.asarray(summary.get("raw_median_norm", summary["raw_mean_norm"]), dtype=np.float64)
            target = np.asarray(summary["target_norm"], dtype=np.float64)
            raw_augmented = np.array([raw[0], raw[1], 1.0], dtype=np.float64)
            manual_pred = np.array(
                [
                    raw[0] * gain_x + offset_x_norm,
                    raw[1] * gain_y + offset_y_norm,
                ],
                dtype=np.float64,
            )
            affine_pred = affine_matrix @ raw_augmented
            manual_error_xy_px = (manual_pred - target) * np.array([self.screen_w, self.screen_h])
            affine_error_xy_px = (affine_pred - target) * np.array([self.screen_w, self.screen_h])
            manual_error_px = float(np.linalg.norm(manual_error_xy_px))
            affine_error_px = float(np.linalg.norm(affine_error_xy_px))
            manual_errors.append(manual_error_px)
            affine_errors.append(affine_error_px)
            point = dict(summary)
            point.update(
                {
                    "manual_pred_px": [
                        float((manual_pred[0] + 0.5) * self.screen_w),
                        float((manual_pred[1] + 0.5) * self.screen_h),
                    ],
                    "manual_error_xy_px": [
                        float(manual_error_xy_px[0]),
                        float(manual_error_xy_px[1]),
                    ],
                    "manual_error_px": manual_error_px,
                    "affine_pred_px": [
                        float((affine_pred[0] + 0.5) * self.screen_w),
                        float((affine_pred[1] + 0.5) * self.screen_h),
                    ],
                    "affine_error_xy_px": [
                        float(affine_error_xy_px[0]),
                        float(affine_error_xy_px[1]),
                    ],
                    "affine_error_px": affine_error_px,
                }
            )
            points.append(point)

        manual_errors_arr = np.asarray(manual_errors, dtype=np.float64)
        affine_errors_arr = np.asarray(affine_errors, dtype=np.float64)
        return {
            "manual_point_mean_error_px": float(np.mean(manual_errors_arr)),
            "manual_point_max_error_px": float(np.max(manual_errors_arr)),
            "manual_point_rmse_px": float(np.sqrt(np.mean(manual_errors_arr ** 2))),
            "affine_point_mean_error_px": float(np.mean(affine_errors_arr)),
            "affine_point_max_error_px": float(np.max(affine_errors_arr)),
            "affine_point_rmse_px": float(np.sqrt(np.mean(affine_errors_arr ** 2))),
            "points": points,
        }

    def _max_accepted_affine_error_px(self):
        half_ninja_column_gap_px = float(self.screen_w) * 0.125
        return half_ninja_column_gap_px * self.MAX_ACCEPTED_AFFINE_ERROR_FRACTION_OF_HALF_COLUMN

    def _max_accepted_point_error_px(self):
        half_ninja_column_gap_px = float(self.screen_w) * 0.125
        return half_ninja_column_gap_px * self.MAX_ACCEPTED_POINT_ERROR_FRACTION_OF_HALF_COLUMN

    def _save_result(
        self,
        mean_error_px,
        diagnostics=None,
        *,
        accepted=True,
        failure_reason=None,
        manual_mean_error_px=None,
    ):
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "num_points": self._num_points,
            "mean_error_px": mean_error_px,
            "error_model": "ninja_affine",
            "manual_mean_error_px": manual_mean_error_px,
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "correction_values": self._correction_values,
            "diagnostics": diagnostics,
            "accepted": bool(accepted),
            "failure_reason": failure_reason,
            "max_accepted_affine_error_px": self._max_accepted_affine_error_px(),
            "max_accepted_point_error_px": self._max_accepted_point_error_px(),
            "timestamp": time.time(),
        }
        filename = (
            self.LAST_SUCCESS_FILENAME
            if accepted
            else self.LAST_FAILED_FILENAME
        )
        path = self.SAVE_DIR / filename
        path.write_text(json.dumps(data, indent=2))
