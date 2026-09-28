from __future__ import annotations

import math
from pathlib import Path
import unittest

from experiments.preserve_change_destroy.run_fb_three_layer_bridge import (
    FULL_PROFILE,
    INITIAL_BATCH,
    REGIMES,
    SCHEDULES,
    ETA_COEFFICIENT,
    learning_rate,
    maximum_intrinsic_time,
)


class FBThreeLayerBridgeContractTest(unittest.TestCase):
    def test_regimes_are_in_the_declared_open_fb_cells(self) -> None:
        fb1, fb2 = REGIMES
        self.assertEqual(fb1.name, "FB1")
        self.assertLess(0.0, fb1.alpha)
        self.assertLess(fb1.alpha, 0.25)
        self.assertLess(0.5 - fb1.alpha, fb1.beta)
        self.assertLess(fb1.beta, 0.5)

        self.assertEqual(fb2.name, "FB2")
        self.assertLess(0.0, fb2.alpha)
        self.assertLess(fb2.alpha, 0.25)
        self.assertGreater(fb2.beta, 0.5)
        self.assertGreater(fb1.p, 0.0)
        self.assertGreater(fb2.p, 0.0)

    def test_triangular_horizon_is_strict_source_window_sequence(self) -> None:
        for regime in REGIMES:
            ratios = [
                maximum_intrinsic_time(regime, width)
                / float(width) ** (2.0 * regime.alpha)
                for width in (128, 512, 2048)
            ]
            expected = [2.0 / math.log(float(width)) for width in (128, 512, 2048)]
            for observed, target in zip(ratios, expected, strict=True):
                self.assertAlmostEqual(observed, target, places=14)
            self.assertGreater(ratios[0], ratios[1])
            self.assertGreater(ratios[1], ratios[2])

    def test_learning_rate_has_the_finite_bulk_peak_scaling(self) -> None:
        for regime in REGIMES:
            for width in FULL_PROFILE.widths:
                scaled_inverse_ratio = (
                    learning_rate(regime, width)
                    / INITIAL_BATCH
                    * float(width) ** (1.0 - 2.0 * regime.alpha)
                )
                self.assertAlmostEqual(
                    scaled_inverse_ratio, ETA_COEFFICIENT, places=14
                )

    def test_schedule_labels_describe_finite_bulk_accumulation(self) -> None:
        self.assertEqual(
            SCHEDULES,
            (
                (0.0, "linear accumulation"),
                (1.0, "logarithmic accumulation"),
                (2.0, "summable injection"),
            ),
        )

    def test_full_profile_uses_nested_feature_seed_sets(self) -> None:
        self.assertTrue(
            set(FULL_PROFILE.true_sgd_seeds).issubset(FULL_PROFILE.finite_w_seeds)
        )
        self.assertEqual(FULL_PROFILE.display_width, 512)

    def test_smoke_artifact_if_present_is_not_a_paper_claim(self) -> None:
        summary = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "fb_three_layer_bridge_smoke"
            / "summary.json"
        )
        if summary.exists():
            text = summary.read_text(encoding="utf-8")
            self.assertIn('"profile": "smoke"', text)
            self.assertIn("not a uniform random-matrix theorem", text)


if __name__ == "__main__":
    unittest.main()
