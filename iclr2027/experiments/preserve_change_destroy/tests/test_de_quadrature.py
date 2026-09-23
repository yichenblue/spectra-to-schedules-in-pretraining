from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from iclr2027.experiments.preserve_change_destroy.de_config import (
    DE_SMOKE_PROFILE,
)
from iclr2027.experiments.preserve_change_destroy.de_quadrature import (
    head_preserving_quadrature,
    run_continuum_modal_dynamics,
    spectral_profiles,
    triangular_time_grid,
)
from iclr2027.experiments.preserve_change_destroy.run_de import (
    run_de_experiment,
)


class DESpectralQuadratureTests(unittest.TestCase):
    def test_leading_modes_are_exact_and_mass_is_accounted_for(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 32, 0.02)
        indices = np.arange(1, 33, dtype=float)
        np.testing.assert_array_equal(quadrature.multiplicities, np.ones(32))
        np.testing.assert_allclose(
            quadrature.eigenvalues,
            indices ** (-0.8),
            atol=0.0,
            rtol=2.0e-15,
        )
        np.testing.assert_allclose(
            quadrature.initial_clean_mass,
            indices ** (-1.4),
            atol=0.0,
            rtol=2.0e-15,
        )
        reconstructed = (
            float(np.sum(quadrature.initial_clean_mass))
            + quadrature.approximation_floor
        )
        self.assertAlmostEqual(reconstructed, quadrature.total_teacher_energy, places=13)

    def test_log_tail_compression_matches_dense_spectral_profiles(self) -> None:
        width = 10000
        quadrature = head_preserving_quadrature(0.4, 0.3, width, 0.02)
        times = np.geomspace(1.0, 100.0, 31)
        compressed_kernel, compressed_forcing = spectral_profiles(quadrature, times)
        indices = np.arange(1, width + 1, dtype=float)
        eigenvalues = indices ** (-0.8)
        survival = np.exp(-2.0 * eigenvalues[:, None] * times[None, :])
        dense_kernel = (eigenvalues**2) @ survival
        dense_forcing = (indices ** (-1.4)) @ survival
        np.testing.assert_allclose(compressed_kernel, dense_kernel, rtol=2.0e-4)
        np.testing.assert_allclose(compressed_forcing, dense_forcing, rtol=2.0e-4)

    def test_time_grid_is_strictly_pre_saturation(self) -> None:
        width = 10**12
        times = triangular_time_grid(width, 0.56, 1.0e-3, 12)
        self.assertEqual(times[0], 0.0)
        self.assertTrue(np.all(np.diff(times) > 0.0))
        self.assertLess(times[-1], width**0.8)
        self.assertAlmostEqual(np.log(times[-1]) / np.log(width), 0.56, places=12)

    def test_zero_label_noise_has_zero_gap_and_positive_row_mass(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 1000, 0.02)
        times = triangular_time_grid(1000, 0.4, 1.0e-3, 12)
        result = run_continuum_modal_dynamics(
            quadrature,
            times,
            (0.25, 0.5, 0.9),
            sigma2=0.0,
            inverse_ratio_amplitude=1.0 / 128.0,
            row_mass_cap=0.2,
        )
        for trajectory in result.values():
            np.testing.assert_array_equal(
                trajectory.noise_gap, np.zeros_like(trajectory.noise_gap)
            )
            self.assertTrue(np.all(trajectory.clean_centered > 0.0))
            self.assertGreater(trajectory.maximum_row_mass, 0.0)
            self.assertTrue(trajectory.stable)

    def test_tiny_end_to_end_de_run_writes_contract(self) -> None:
        tiny = replace(
            DE_SMOKE_PROFILE,
            name="de_tiny_test",
            effective_widths=(1000, 10000),
            theta_values=(0.25, 0.5, 0.9),
            representative_thetas=(0.25, 0.5, 0.9),
            fit_windows=(("nominal", 0.10, 0.45),),
            minimum_fit_decades=0.5,
            minimum_fit_points=6,
            local_slope_half_window_decades=0.5,
            time_points_per_decade=8,
            maximum_resolution_curve_error=0.08,
            maximum_resolution_slope_error=0.05,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_de_experiment(tiny, output)
            self.assertTrue(summary["numerical_contract_pass"])
            self.assertFalse(summary["slope_claim_ready"])
            for name in (
                "summary.json",
                "curves.csv",
                "spectral_profiles.csv",
                "slope_audits.csv",
                "resolution_checks.csv",
                "preserve_change_destroy_de.png",
                "preserve_change_destroy_de.pdf",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
