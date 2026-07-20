import math
import random
import unittest

from target_finder_toolkit.fitts_distractors_task import generate_trial


def distractor_area(trial):
    return sum(math.pi * (obj.diameter / 2.0) ** 2 for obj in trial.distractors)


class FittsDistractorDensityTests(unittest.TestCase):
    def test_id7_density_conditions_use_strict_blanch_ortega_layout(self):
        trials = [
            generate_trial(
                trial_id=index,
                technique="mouse",
                difficulty=f"ID 7 rho {rho:g}",
                density=density,
                id_value=7.0,
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
            self.assertEqual(trial.layout_metadata["layout"], "blanch_ortega_2011_2d")
            self.assertNotIn("fallback_layout", trial.layout_metadata)

    def test_condition_metadata_rho_overrides_density_label(self):
        low_label_high_rho = generate_trial(
            trial_id=1,
            technique="mouse",
            difficulty="ID 7",
            density="low",
            id_value=7.0,
            widget_size=(1440, 900),
            rng=random.Random(1),
            condition_metadata={"rho": 0.6},
        )

        self.assertEqual(low_label_high_rho.rho, 0.6)
        self.assertGreater(len(low_label_high_rho.distractors), 0)

    def test_blanch_ortega_layout_is_deterministic(self):
        first = generate_trial(
            trial_id=1,
            technique="mouse",
            difficulty="ID 7",
            density="high",
            id_value=7.0,
            widget_size=(1440, 900),
            rng=random.Random(1),
            condition_metadata={"rho": 0.6},
        )
        second = generate_trial(
            trial_id=2,
            technique="mouse",
            difficulty="ID 7",
            density="high",
            id_value=7.0,
            widget_size=(1440, 900),
            rng=random.Random(999),
            condition_metadata={"rho": 0.6},
        )

        first_layout = [(round(obj.center[0], 3), round(obj.center[1], 3), round(obj.diameter, 3)) for obj in first.distractors]
        second_layout = [(round(obj.center[0], 3), round(obj.center[1], 3), round(obj.diameter, 3)) for obj in second.distractors]
        self.assertEqual(first_layout, second_layout)
        self.assertEqual(first.layout_metadata["layout"], "blanch_ortega_2011_2d")
        self.assertNotIn("fallback_layout", first.layout_metadata)


if __name__ == "__main__":
    unittest.main()
