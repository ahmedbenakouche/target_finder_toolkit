import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from target_finder_toolkit.eye_calibration import EyeCalibration


class _FakeTracker:
    def __init__(self):
        self.affine_matrix = np.ones((2, 3), dtype=np.float64)
        self.affine_matrix_tf = object()
        self.kalman_filter = SimpleNamespace(
            x=np.ones((4, 1), dtype=np.float64),
            P=np.full((4, 4), 2.0, dtype=np.float64),
        )
        self.adapt_called = False

    def adapt_from_gaze_results(self, *_args, **_kwargs):
        self.adapt_called = True
        raise AssertionError("calibration must not change WebEyeTrack model weights")


class EyeCalibrationTest(unittest.TestCase):
    def test_fit_initializes_manual_correction_without_adapting_tracker(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        # Build raw samples whose known affine mapping reaches every target.
        expected_gain_x = 1.2
        expected_gain_y = 0.8
        expected_bias_x = -0.05
        expected_bias_y = 0.04
        calibration._gaze_results = []
        for target_x, target_y in calibration.targets:
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            raw_x = (target_norm_x - expected_bias_x) / expected_gain_x
            raw_y = (target_norm_y - expected_bias_y) / expected_gain_y
            calibration._gaze_results.append(
                [SimpleNamespace(norm_pog=np.array([raw_x, raw_y])) for _ in range(5)]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()

        self.assertFalse(tracker.adapt_called)
        self.assertTrue(calibration.is_calibrated)
        self.assertIsNone(tracker.affine_matrix)
        self.assertIsNone(tracker.affine_matrix_tf)
        self.assertTrue(np.allclose(tracker.kalman_filter.x, 0.0))
        self.assertTrue(np.allclose(tracker.kalman_filter.P, np.eye(4)))
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0][0])
        self.assertLess(done[0][1], 1e-6)

        correction = calibration.correction_values
        self.assertAlmostEqual(correction["gaze_gain_x"], 1.0, places=6)
        self.assertAlmostEqual(correction["gaze_gain_y"], 1.0, places=6)
        self.assertAlmostEqual(correction["gaze_offset_x"], 0.0, places=6)
        self.assertAlmostEqual(correction["gaze_offset_y"], 0.0, places=6)
        affine = np.array(correction["ninja_affine_matrix"], dtype=np.float64)
        self.assertAlmostEqual(affine[0, 0], expected_gain_x, places=6)
        self.assertAlmostEqual(affine[1, 1], expected_gain_y, places=6)
        self.assertAlmostEqual(affine[0, 2], expected_bias_x, places=6)
        self.assertAlmostEqual(affine[1, 2], expected_bias_y, places=6)

    def test_fit_rejects_bad_runtime_manual_model_when_affine_has_shear(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        raw_points = [
            (-0.42, -0.35),
            (0.38, -0.30),
            (0.0, 0.0),
            (-0.37, 0.34),
            (0.41, 0.32),
        ]
        calibration._gaze_results = []
        for (target_x, target_y), (raw_x, raw_y) in zip(calibration.targets, raw_points):
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            # Add a cross-axis term to the samples.  The full affine can model
            # it, but Ninja runtime can only use diagonal gain plus offset.
            sample_x = raw_x + raw_y * 0.45
            sample_y = raw_y - raw_x * 0.30
            calibration._gaze_results.append(
                [SimpleNamespace(norm_pog=np.array([sample_x, sample_y])) for _ in range(6)]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()
            saved = json.loads((Path(tmpdir) / "last_calibration.json").read_text())

        self.assertFalse(calibration.is_calibrated)
        self.assertIsNone(calibration.correction_values)
        self.assertEqual(len(done), 1)
        self.assertFalse(done[0][0])
        self.assertGreater(done[0][1], saved["max_accepted_affine_error_px"])
        self.assertFalse(saved["accepted"])
        self.assertIn("affine calibration error too high", saved["failure_reason"])
        self.assertIn("diagnostics", saved)
        self.assertIn("correction_values", saved)


if __name__ == "__main__":
    unittest.main()
