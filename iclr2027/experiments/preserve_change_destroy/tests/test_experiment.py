from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from iclr2027.experiments.preserve_change_destroy.config import SMOKE_PROFILE
from iclr2027.experiments.preserve_change_destroy.dynamics import (
    dense_covariance_risk,
    one_step_polynomial,
    run_exact_dynamics,
    solve_two_time_volterra,
)
from iclr2027.experiments.preserve_change_destroy.monte_carlo import (
    compare_to_exact,
    paired_antithetic_gap,
)
from iclr2027.experiments.preserve_change_destroy.run import run_experiment
from iclr2027.experiments.preserve_change_destroy.slopes import audit_fixed_window
from iclr2027.experiments.preserve_change_destroy.spectrum import (
    build_empirical_spectrum,
    select_global_learning_rate,
)


class PreserveChangeDestroyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spectrum = build_empirical_spectrum(
            alpha=0.4,
            beta=0.3,
            width=8,
            ambient_to_width_ratio=2,
            seed=17,
        )

    def test_theory_values_and_phase_labels(self) -> None:
        self.assertAlmostEqual(SMOKE_PROFILE.q_clean, 0.5)
        self.assertAlmostEqual(SMOKE_PROFILE.q_kernel, 0.75)
        self.assertAlmostEqual(SMOKE_PROFILE.destroy_boundary, 0.25)
        self.assertAlmostEqual(SMOKE_PROFILE.preservation_boundary, 0.75)
        self.assertEqual(SMOKE_PROFILE.phase_label(0.25), "destroy")
        self.assertEqual(SMOKE_PROFILE.phase_label(0.50), "change")
        self.assertEqual(SMOKE_PROFILE.phase_label(0.75), "critical")
        self.assertEqual(SMOKE_PROFILE.phase_label(0.90), "preserve")
        self.assertAlmostEqual(SMOKE_PROFILE.total_exponent(0.50), 0.25)
        self.assertAlmostEqual(SMOKE_PROFILE.total_exponent(0.90), 0.50)

    def test_spectrum_parseval_and_current_dimensions(self) -> None:
        self.assertEqual(self.spectrum.ambient_dimension, 16)
        self.assertEqual(self.spectrum.width, 8)
        self.assertEqual(self.spectrum.eigenvalues.shape, (8,))
        self.assertEqual(self.spectrum.teacher_overlaps.shape, (8,))
        self.assertGreater(self.spectrum.approximation_floor, 0.0)
        self.assertLess(self.spectrum.parseval_residual, 1.0e-11)
        reconstructed = (
            self.spectrum.approximation_floor
            + float(np.sum(self.spectrum.teacher_overlaps))
        )
        self.assertAlmostEqual(reconstructed, self.spectrum.teacher_energy, places=11)

    def test_fixed_batch_summed_and_averaged_parameterizations_agree(self) -> None:
        batch = 4
        gamma = 0.03
        eta = batch * gamma
        values = self.spectrum.eigenvalues
        averaged = one_step_polynomial(values, eta, batch)
        old_summed = (
            1.0
            - 2.0 * gamma * batch * values
            + gamma * gamma * batch * (batch + 1) * values * values
        )
        np.testing.assert_allclose(averaged, old_summed, atol=2.0e-15, rtol=0.0)
        np.testing.assert_allclose(
            eta * eta * values * values / batch,
            gamma * gamma * batch * values * values,
            atol=2.0e-15,
            rtol=0.0,
        )

    def test_modal_and_two_time_volterra_agree_for_varying_batch(self) -> None:
        eta = 0.08
        theta = 0.5
        batch = 4
        sigma2 = 0.35
        horizon = 9
        modal = run_exact_dynamics(
            self.spectrum, eta, theta, batch, sigma2, horizon
        )
        volterra = solve_two_time_volterra(
            self.spectrum, eta, theta, batch, sigma2, horizon
        )
        np.testing.assert_allclose(
            modal.clean_total, volterra.clean_total, atol=2.0e-12, rtol=2.0e-12
        )
        np.testing.assert_allclose(
            modal.noisy_total, volterra.noisy_total, atol=2.0e-12, rtol=2.0e-12
        )
        np.testing.assert_allclose(
            modal.noise_gap, volterra.noise_gap, atol=2.0e-12, rtol=2.0e-12
        )
        np.testing.assert_allclose(
            modal.row_mass,
            np.sum(volterra.kernel, axis=1),
            atol=2.0e-12,
            rtol=2.0e-12,
        )

    def test_modal_and_dense_covariance_routes_agree(self) -> None:
        eta = 0.06
        theta = 0.9
        batch = 3
        horizon = 7
        for sigma2 in (0.0, 0.4):
            modal = run_exact_dynamics(
                self.spectrum, eta, theta, batch, sigma2, horizon
            )
            dense = dense_covariance_risk(
                self.spectrum, eta, theta, batch, sigma2, horizon
            )
            np.testing.assert_allclose(
                modal.noisy_total,
                dense,
                atol=3.0e-12,
                rtol=3.0e-12,
            )

    def test_gap_is_direct_and_linear_in_noise_variance(self) -> None:
        common = dict(
            spectrum=self.spectrum,
            eta=0.07,
            theta=0.65,
            initial_batch=4,
            horizon=8,
        )
        unit = run_exact_dynamics(sigma2=1.0, **common)
        scaled = run_exact_dynamics(sigma2=0.17, **common)
        np.testing.assert_allclose(
            scaled.noise_gap,
            0.17 * unit.noise_gap,
            atol=2.0e-14,
            rtol=2.0e-13,
        )
        np.testing.assert_allclose(
            scaled.noisy_centered,
            scaled.clean_centered + scaled.noise_gap,
            atol=0.0,
            rtol=0.0,
        )

    def test_global_learning_rate_aligns_time_and_passes_both_gates(self) -> None:
        spectra = [
            self.spectrum,
            build_empirical_spectrum(0.4, 0.3, 12, 2, 29),
        ]
        selected = select_global_learning_rate(
            spectra,
            initial_batch=8,
            pointwise_product_cap=0.8,
            row_certificate_cap=0.2,
        )
        self.assertLessEqual(selected.pointwise_product, 0.8 + 1.0e-14)
        self.assertLess(selected.row_mass_certificate, 1.0)
        trajectories = [
            run_exact_dynamics(item, selected.eta, 0.5, 8, 0.2, 10)
            for item in spectra
        ]
        np.testing.assert_array_equal(trajectories[0].times, trajectories[1].times)
        self.assertTrue(all(item.pointwise_stable for item in trajectories))
        self.assertTrue(all(item.row_stable for item in trajectories))

    def test_fixed_window_slope_audit(self) -> None:
        times = np.geomspace(1.0, 1.0e4, 401)
        exact_power = 2.3 * times ** (-0.5)
        audit = audit_fixed_window(
            times,
            exact_power,
            observable="synthetic",
            lower_time=10.0,
            upper_time=1000.0,
            minimum_decades=1.0,
            minimum_points=12,
            half_window_decades=0.25,
            maximum_local_variation=0.1,
        )
        self.assertEqual(audit.status, "PASS")
        self.assertAlmostEqual(audit.exponent, 0.5, places=12)
        curved = np.exp(-np.sqrt(times))
        curved_audit = audit_fixed_window(
            times,
            curved,
            observable="curved",
            lower_time=10.0,
            upper_time=1000.0,
            minimum_decades=1.0,
            minimum_points=12,
            half_window_decades=0.25,
            maximum_local_variation=0.1,
        )
        self.assertEqual(curved_audit.status, "INCONCLUSIVE")

    def test_paired_true_sgd_matches_exact_gap(self) -> None:
        spectrum = build_empirical_spectrum(0.4, 0.3, 4, 2, 31)
        exact = run_exact_dynamics(
            spectrum=spectrum,
            eta=0.05,
            theta=0.5,
            initial_batch=4,
            sigma2=0.3,
            horizon=5,
        )
        sampled = paired_antithetic_gap(
            eigenvalues=spectrum.eigenvalues,
            eta=0.05,
            batches=exact.batches,
            sigma2=0.3,
            trajectories=12000,
            seed=101,
        )
        comparison = compare_to_exact(sampled, exact.noise_gap, maximum_allowed_z=5.0)
        self.assertTrue(comparison.passed)

    def test_tiny_end_to_end_run_writes_contract_and_figure(self) -> None:
        tiny = replace(
            SMOKE_PROFILE,
            name="tiny_test",
            widths=(8, 16),
            seeds=(7, 9),
            theta_values=(0.25, 0.50, 0.75, 0.90),
            horizon_factor=0.20,
            monte_carlo_steps=3,
            monte_carlo_trajectories=128,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_experiment(
                profile=tiny,
                output_dir=output,
                run_monte_carlo=False,
            )
            self.assertTrue(summary["numerical_contract_pass"])
            self.assertFalse(summary["slope_claim_ready"])
            for name in (
                "summary.json",
                "curves.csv",
                "slope_audits.csv",
                "paired_sgd_validation.csv",
                "preserve_change_destroy.png",
                "preserve_change_destroy.pdf",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
