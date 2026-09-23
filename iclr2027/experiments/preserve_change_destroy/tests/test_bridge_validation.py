from __future__ import annotations

import unittest

import numpy as np

from iclr2027.experiments.preserve_change_destroy.bridge_config import (
    BRIDGE_FINITE_SIZE_TRAIN_PROFILE,
    BRIDGE_LARGE_PROFILE,
    BRIDGE_NOISE_SWEEP_UNIT_PROFILE,
)
from iclr2027.experiments.preserve_change_destroy.de_config import (
    DE_FINITE_SIZE_PROFILE,
)
from iclr2027.experiments.preserve_change_destroy.bridge_validation import (
    analytic_power_law_spectrum,
    simulate_true_gaussian_sgd,
)
from iclr2027.experiments.preserve_change_destroy.de_quadrature import (
    head_preserving_quadrature,
    run_scheduled_continuum_modal_dynamics,
)
from iclr2027.experiments.preserve_change_destroy.dynamics import (
    batch_schedule,
    run_exact_dynamics,
)
from iclr2027.experiments.preserve_change_destroy.run_experiment_one_m10000 import (
    _predicted_total_exponent,
)
from iclr2027.experiments.preserve_change_destroy.run_experiment_one_sigma100 import (
    _rescale_components,
)


class AuthenticityBridgeTests(unittest.TestCase):
    def test_finite_size_protocol_keeps_train_and_blind_test_disjoint(self) -> None:
        self.assertEqual(
            BRIDGE_FINITE_SIZE_TRAIN_PROFILE.widths, (512, 1024, 2048, 4096)
        )
        self.assertNotIn(10_000, BRIDGE_FINITE_SIZE_TRAIN_PROFILE.widths)
        self.assertEqual(DE_FINITE_SIZE_PROFILE.effective_widths[0], 10_000)
        self.assertEqual(DE_FINITE_SIZE_PROFILE.effective_widths[-1], 10**48)
        self.assertEqual(
            BRIDGE_FINITE_SIZE_TRAIN_PROFILE.theta_values,
            DE_FINITE_SIZE_PROFILE.theta_values,
        )

    def test_large_profile_reaches_ten_thousand_without_changing_contract(self) -> None:
        self.assertEqual(BRIDGE_LARGE_PROFILE.widths[-1], 10_000)
        self.assertEqual(BRIDGE_LARGE_PROFILE.eta, 1.0 / 8.0)
        self.assertEqual(BRIDGE_LARGE_PROFILE.initial_batch, 16)
        self.assertEqual(BRIDGE_LARGE_PROFILE.sigma2, 10.0)

    def test_noise_sweep_unit_profile_is_frozen(self) -> None:
        self.assertEqual(BRIDGE_NOISE_SWEEP_UNIT_PROFILE.widths, (10_000,))
        self.assertEqual(
            BRIDGE_NOISE_SWEEP_UNIT_PROFILE.theta_values, (0.25, 0.50, 0.90)
        )
        self.assertEqual(BRIDGE_NOISE_SWEEP_UNIT_PROFILE.sigma2, 1.0)
        self.assertEqual(BRIDGE_NOISE_SWEEP_UNIT_PROFILE.trajectories, 128)

    def test_sigma100_experiment_rescales_only_the_gap(self) -> None:
        clean = np.asarray([3.0, 2.0])
        gap = np.asarray([0.1, 0.2])
        scaled_clean, scaled_gap, total = _rescale_components(clean, gap)
        np.testing.assert_array_equal(scaled_clean, clean)
        np.testing.assert_allclose(scaled_gap, [1.0, 2.0])
        np.testing.assert_allclose(total, [4.0, 4.0])

    def test_m10000_experiment_one_theory_curve_is_frozen(self) -> None:
        expected = {
            0.25: 0.0,
            0.35: 0.10,
            0.50: 0.25,
            0.65: 0.40,
            0.75: 0.50,
            0.90: 0.50,
            1.00: 0.50,
            1.10: 0.50,
        }
        for theta, exponent in expected.items():
            self.assertAlmostEqual(
                _predicted_total_exponent(theta), exponent, places=12
            )

    def test_analytic_spectrum_preserves_teacher_energy(self) -> None:
        spectrum = analytic_power_law_spectrum(0.4, 0.3, 128)
        reconstructed = (
            np.sum(spectrum.teacher_overlaps) + spectrum.approximation_floor
        )
        self.assertAlmostEqual(reconstructed, spectrum.teacher_energy, places=12)
        self.assertAlmostEqual(spectrum.eigenvalues[0], 1.0, places=15)

    def test_collapsed_true_sgd_one_step_matches_exact_moments(self) -> None:
        spectrum = analytic_power_law_spectrum(0.4, 0.3, 4)
        eta = 0.125
        batches = np.asarray([3], dtype=np.int64)
        sampled = simulate_true_gaussian_sgd(
            spectrum=spectrum,
            eta=eta,
            batches=batches,
            sigma2=10.0,
            trajectories=100_000,
            seed=1917,
            trajectory_chunk_size=5_000,
        )
        exact = run_exact_dynamics(
            spectrum=spectrum,
            eta=eta,
            theta=0.0,
            initial_batch=3,
            sigma2=10.0,
            horizon=1,
        )
        clean_z = abs(sampled.clean[-1] - exact.clean_total[-1]) / (
            sampled.clean_standard_error[-1]
        )
        gap_z = abs(sampled.gap[-1] - exact.noise_gap[-1]) / (
            sampled.gap_standard_error[-1]
        )
        self.assertLess(clean_z, 4.0)
        self.assertLess(gap_z, 4.0)

    def test_collapsed_true_sgd_noise_response_is_pathwise_linear(self) -> None:
        spectrum = analytic_power_law_spectrum(0.4, 0.3, 4)
        batches = np.asarray([3, 4], dtype=np.int64)
        unit = simulate_true_gaussian_sgd(
            spectrum, 0.125, batches, 1.0, 64, 9917
        )
        scaled = simulate_true_gaussian_sgd(
            spectrum, 0.125, batches, 1000.0, 64, 9917
        )
        np.testing.assert_allclose(scaled.clean, unit.clean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            scaled.gap, 1000.0 * unit.gap, rtol=2.0e-14, atol=1.0e-14
        )

    def test_scheduled_continuum_uses_identical_grid_and_batches(self) -> None:
        width = 16
        spectrum = analytic_power_law_spectrum(0.4, 0.3, width)
        quadrature = head_preserving_quadrature(
            0.4, 0.3, width, relative_bin_width=0.5 / width
        )
        eta = 0.05
        times, batches = batch_schedule(eta, 0.5, 16, 20)
        exact = run_exact_dynamics(
            spectrum, eta, 0.5, 16, 10.0, horizon=20
        )
        continuum = run_scheduled_continuum_modal_dynamics(
            quadrature,
            times,
            batches,
            eta,
            0.5,
            10.0,
            0.8,
        )
        self.assertTrue(np.array_equal(exact.times, continuum.times))
        self.assertLess(
            np.linalg.norm(continuum.noise_gap - exact.noise_gap)
            / np.linalg.norm(exact.noise_gap),
            0.03,
        )


if __name__ == "__main__":
    unittest.main()
