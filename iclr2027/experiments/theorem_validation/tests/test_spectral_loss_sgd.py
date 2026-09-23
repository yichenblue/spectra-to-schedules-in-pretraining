from __future__ import annotations

import unittest

import numpy as np

from iclr2027.experiments.theorem_validation.spectral_loss_sgd import (
    CASES,
    build_case,
    evaluation_steps,
    local_exponent_curve,
    train_one,
)


class SpectralLossSgdTests(unittest.TestCase):
    def test_all_cases_are_finite_normalized_and_ordered(self) -> None:
        for case in CASES:
            for width in (128, 4096):
                eigenvalues, target = build_case(case, width)
                self.assertTrue(np.all(np.isfinite(eigenvalues)))
                self.assertTrue(np.all(eigenvalues > 0.0))
                self.assertTrue(np.all(np.diff(eigenvalues) <= 0.0))
                self.assertAlmostEqual(
                    float(eigenvalues @ np.square(target)), 1.0, places=12
                )

    def test_training_is_reproducible_and_positive(self) -> None:
        eigenvalues, target = build_case(CASES[0], 64)
        checkpoints = evaluation_steps(48, 12)
        first = train_one(
            eigenvalues,
            target,
            eta=0.05,
            batch_size=4,
            steps=48,
            seed=123,
            eval_steps=checkpoints,
        )
        second = train_one(
            eigenvalues,
            target,
            eta=0.05,
            batch_size=4,
            steps=48,
            seed=123,
            eval_steps=checkpoints,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0.0))
        self.assertAlmostEqual(float(first[0]), 1.0, places=12)
        self.assertLess(float(first[-1]), float(first[0]))

    def test_trace_driven_instability_is_rejected(self) -> None:
        width = 4096
        eigenvalues = np.ones(width)
        target = np.ones(width) / np.sqrt(width)
        with self.assertRaises(ValueError):
            train_one(
                eigenvalues,
                target,
                eta=0.05,
                batch_size=16,
                steps=2,
                seed=1,
                eval_steps=np.asarray([0, 1, 2]),
            )

    def test_local_exponent_recovers_an_exact_power(self) -> None:
        times = np.geomspace(0.1, 100.0, 101)
        values = 3.0 * times ** (-0.7)
        _, exponents = local_exponent_curve(times, values)
        self.assertGreater(exponents.size, 20)
        np.testing.assert_allclose(exponents, 0.7, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
