import math
import random
import unittest

from target_finder_toolkit.fitts_distractors_task import generate_trial


def distractor_area(trial):
    return sum(math.pi * (obj.diameter / 2.0) ** 2 for obj in trial.distractors)


class FittsDistractorDensityTests(unittest.TestCase):
    def test_id8_density_conditions_are_visually_distinct(self):
        trials = [
            generate_trial(
                trial_id=index,
                technique="mouse",
                difficulty=f"ID 8 rho {rho:g}",
                density=density,
                id_value=8.0,
                widget_size=(1440, 900),
                rng=random.Random(1234),
                condition_metadata={"rho": rho},
            )
            for index, (density, rho) in enumerate(
                [("low", 0.1), ("medium", 0.3), ("high", 0.6)],
                start=1,
            )
        ]

        counts = [len(trial.distractors) for trial in trials]
        areas = [distractor_area(trial) for trial in trials]
        self.assertGreater(counts[0], 0)
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])
        self.assertLess(areas[0], areas[1])
        self.assertLess(areas[1], areas[2])
        for trial in trials:
            self.assertEqual(trial.layout_metadata["fallback_layout"]["layout"], "viewport_density_fallback")

    def test_condition_metadata_rho_overrides_density_label(self):
        low_label_high_rho = generate_trial(
            trial_id=1,
            technique="mouse",
            difficulty="ID 8",
            density="low",
            id_value=8.0,
            widget_size=(1440, 900),
            rng=random.Random(1),
            condition_metadata={"rho": 0.6},
        )

        self.assertEqual(low_label_high_rho.rho, 0.6)
        self.assertGreater(len(low_label_high_rho.distractors), 0)

    def test_id8_fallback_layout_is_deterministic(self):
        first = generate_trial(
            trial_id=1,
            technique="mouse",
            difficulty="ID 8",
            density="high",
            id_value=8.0,
            widget_size=(1440, 900),
            rng=random.Random(1),
            condition_metadata={"rho": 0.6},
        )
        second = generate_trial(
            trial_id=2,
            technique="mouse",
            difficulty="ID 8",
            density="high",
            id_value=8.0,
            widget_size=(1440, 900),
            rng=random.Random(999),
            condition_metadata={"rho": 0.6},
        )

        first_layout = [(round(obj.center[0], 3), round(obj.center[1], 3), round(obj.diameter, 3)) for obj in first.distractors]
        second_layout = [(round(obj.center[0], 3), round(obj.center[1], 3), round(obj.diameter, 3)) for obj in second.distractors]
        self.assertEqual(first_layout, second_layout)
        self.assertTrue(first.layout_metadata["fallback_layout"]["deterministic"])


if __name__ == "__main__":
    unittest.main()
