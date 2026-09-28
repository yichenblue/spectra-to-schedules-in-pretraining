from __future__ import annotations

import unittest

import numpy as np

from experiments.theorem_validation.spectral_width_time import (
    exact_forcing,
    evaluation_steps,
    fit_curve,
    modal_data,
)


class SpectralWidthTimeTests(unittest.TestCase):
    def test_modal_data_are_normalized_and_positive(self) -> None:
        for case in ("polynomial_canonical", "polynomial_sparse_target"):
            eigenvalues, weights = modal_data(case, 4096)
            self.assertTrue(np.all(eigenvalues > 0.0))
            self.assertTrue(np.all(weights > 0.0))
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=13)

    def test_exact_forcing_starts_at_one_and_decreases(self) -> None:
        eigenvalues, weights = modal_data("polynomial_canonical", 1024)
        steps = evaluation_steps(2.0, 0.05, 20)
        forcing = exact_forcing(
            eigenvalues,
            weights,
            eta=0.05,
            batch_size=16,
            steps=steps,
        )
        self.assertAlmostEqual(float(forcing[0]), 1.0, places=13)
        self.assertTrue(np.all(np.diff(forcing) < 0.0))

    def test_fit_curve_recovers_exact_power(self) -> None:
        times = np.geomspace(1.0, 100.0, 101)
        values = 2.0 * times ** (-0.5)
        fit = fit_curve(times, values, 2.0, 80.0)
        self.assertAlmostEqual(float(fit["power_exponent"]), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
