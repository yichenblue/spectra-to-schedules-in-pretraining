import unittest

import numpy as np

from experiments.theorem_validation.spectral_loss_sgd import build_case, evaluation_steps
from experiments.theorem_validation.spectral_rapid_target_mechanism import (
    RAPID_CASE,
    population_gd_control,
    sampled_train_with_bands,
    spectral_slices,
)


class RapidTargetMechanismTests(unittest.TestCase):
    def test_sampled_risk_bands_are_exact_and_positive(self) -> None:
        eigenvalues, target = build_case(RAPID_CASE, 64)
        eval_steps = evaluation_steps(48, 17)
        bands = spectral_slices(64, [0, 4, 16, 64])
        risks, band_risks = sampled_train_with_bands(
            eigenvalues, target, eta=0.05, batch_size=8, steps=48,
            seed=7, eval_steps=eval_steps, bands=bands,
        )
        self.assertTrue(np.all(risks > 0.0))
        np.testing.assert_allclose(risks, np.sum(band_risks, axis=1), rtol=2e-13, atol=2e-15)

    def test_population_control_decreases(self) -> None:
        eigenvalues, target = build_case(RAPID_CASE, 64)
        eval_steps = evaluation_steps(48, 17)
        risks = population_gd_control(
            eigenvalues, target, eta=0.05, steps=48, eval_steps=eval_steps
        )
        self.assertTrue(np.all(np.diff(risks) < 0.0))


if __name__ == "__main__":
    unittest.main()
