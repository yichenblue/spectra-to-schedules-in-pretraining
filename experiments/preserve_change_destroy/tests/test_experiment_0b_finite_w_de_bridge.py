from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.preserve_change_destroy.dynamics import run_exact_dynamics
from experiments.preserve_change_destroy.resolvent_de import (
    compute_density_level,
    density_to_modes,
)
from experiments.preserve_change_destroy.run_experiment_0b_finite_w_de_bridge import (
    DE_DENSITY_CACHE,
    REGIMES,
    _de_modes,
    run_de_schedule,
)
from experiments.preserve_change_destroy.spectrum import EmpiricalSpectrum


class Experiment0bTests(unittest.TestCase):
    def test_checked_in_density_cache_reconstructs_audited_modes(self) -> None:
        regime = next(item for item in REGIMES if item.name == "LM")
        density = compute_density_level(
            phase="schedule_LM",
            alpha=regime.alpha,
            beta=regime.beta,
            width=512,
            level=1,
            cache_dir=DE_DENSITY_CACHE,
        )
        self.assertEqual(density["cache_source"], "density_cache")
        modes, diagnostics = _de_modes(regime, width=512, level=1)
        self.assertEqual(diagnostics["density_points"], 80_001)
        self.assertEqual(len(modes["kernel_nodes"]), 234)
        self.assertEqual(len(modes["forcing_nodes"]), 240)
        np.testing.assert_allclose(
            [
                diagnostics["raw_trace_mass_relative_error"],
                diagnostics["raw_target_mass_relative_error"],
                diagnostics["raw_trace_first_moment_relative_error"],
                diagnostics["raw_target_first_moment_relative_error"],
            ],
            [
                7.144591784857823e-08,
                1.6461423860832282e-06,
                1.7491019346220728e-07,
                6.290342608702222e-06,
            ],
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        self.assertLess(diagnostics["max_m_residual"], 5.0e-11)
        self.assertLess(diagnostics["trace_mass_identity_error"], 1.0e-12)
        self.assertLess(diagnostics["target_mass_identity_error"], 1.0e-12)

    def test_cache_miss_compiles_local_backend_and_round_trips(self) -> None:
        with TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            computed = compute_density_level(
                phase="tiny_test",
                alpha=0.4,
                beta=0.3,
                width=8,
                level=0,
                cache_dir=cache_dir,
                initial_intervals=128,
            )
            self.assertEqual(computed["cache_source"], "computed")
            self.assertLess(computed["max_m_residual"], 5.0e-11)
            self.assertGreater(np.count_nonzero(computed["trace_density"]), 0)

            cached = compute_density_level(
                phase="tiny_test",
                alpha=0.4,
                beta=0.3,
                width=8,
                level=0,
                cache_dir=cache_dir,
                initial_intervals=128,
            )
            self.assertEqual(cached["cache_source"], "density_cache")
            for field in ("x", "trace_density", "target_density"):
                np.testing.assert_array_equal(cached[field], computed[field])

            modes = density_to_modes(0.4, 0.3, 8, cached)
            self.assertAlmostEqual(
                float(np.sum(modes["kernel_trace_weights"])), 8.0, places=12
            )
            self.assertAlmostEqual(
                float(np.sum(modes["forcing_weights"]))
                + float(modes["null_weight"]),
                float(modes["target_energy"]),
                places=12,
            )

    def test_de_schedule_reduces_to_exact_modal_recursion(self) -> None:
        eigenvalues = np.asarray([0.11, 0.37, 0.82])
        overlaps = np.asarray([0.19, 0.31, 0.23])
        floor = 0.17
        spectrum = EmpiricalSpectrum(
            alpha=0.4,
            beta=0.3,
            ambient_dimension=6,
            width=3,
            seed=1,
            eigenvalues=eigenvalues,
            teacher_overlaps=overlaps,
            approximation_floor=floor,
            teacher_energy=floor + float(np.sum(overlaps)),
            trace=float(np.sum(eigenvalues)),
            lambda_max=float(np.max(eigenvalues)),
            rank_tolerance=0.0,
            parseval_residual=0.0,
        )
        modes = {
            "kernel_nodes": eigenvalues,
            "kernel_trace_weights": np.ones_like(eigenvalues),
            "forcing_nodes": eigenvalues,
            "forcing_weights": overlaps,
            "null_weight": floor,
            "target_energy": spectrum.teacher_energy,
        }
        exact = run_exact_dynamics(
            spectrum,
            eta=0.08,
            theta=0.5,
            initial_batch=4,
            sigma2=0.7,
            horizon=11,
        )
        de = run_de_schedule(
            modes,
            eta=0.08,
            theta=0.5,
            initial_batch=4,
            sigma2=0.7,
            horizon=11,
        )
        np.testing.assert_allclose(
            de.clean_centered, exact.clean_centered, rtol=2.0e-13, atol=2.0e-14
        )
        np.testing.assert_allclose(
            de.noise_gap, exact.noise_gap, rtol=2.0e-13, atol=2.0e-14
        )
        np.testing.assert_allclose(
            de.noisy_centered, exact.noisy_centered, rtol=2.0e-13, atol=2.0e-14
        )
        np.testing.assert_allclose(
            de.row_mass, exact.row_mass, rtol=2.0e-13, atol=2.0e-14
        )


if __name__ == "__main__":
    unittest.main()
