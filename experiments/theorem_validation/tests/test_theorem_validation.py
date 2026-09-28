from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.theorem_validation.common import (
    finite_json,
    log_binned_indices,
)
from experiments.theorem_validation.ratio_control import (
    SIGMA2,
    ModalSpectrum,
    build_spectrum,
    equal_sample_quantiles,
    ratio_profiles,
    run_unit_batch_modal_recursion,
)
from experiments.theorem_validation.spectral_criterion import (
    BATCH,
    ETA,
    Case,
    exact_cutoff_tail,
    modal_transform,
    theorem_scaled_transform,
)


class SpectralCriterionTests(unittest.TestCase):
    def test_cutoff_and_transform_are_positive(self) -> None:
        case = Case("forcing", 0.5, 0.8)
        steps = np.asarray([20, 40, 80, 160], dtype=np.int64)
        times = 0.05 * steps
        tail = exact_cutoff_tail(case, 10000, times)
        transform, mode_count = modal_transform(case, 10000, steps, 0.01)
        scaled = theorem_scaled_transform(case, transform)
        self.assertGreater(mode_count, 100)
        self.assertTrue(np.all(tail > 0.0))
        self.assertTrue(np.all(scaled > 0.0))
        self.assertTrue(np.all(np.diff(tail) <= 0.0))
        self.assertTrue(np.all(np.diff(scaled) < 0.0))

    def test_memory_prefactor_route_is_finite(self) -> None:
        case = Case("memory", 0.75, 0.8)
        steps = np.asarray([200, 400, 800], dtype=np.int64)
        width = 100000
        relative_bin_width = 0.005
        transform, _ = modal_transform(
            case, width, steps, relative_bin_width
        )
        scaled = theorem_scaled_transform(case, transform)
        _, _, nodes, multiplicity = log_binned_indices(
            width, relative_bin_width
        )
        eigenvalues = nodes ** (-case.eigenvalue_power)
        decrement = (
            2.0 * ETA * eigenvalues
            - (1.0 + 1.0 / BATCH)
            * ETA
            * ETA
            * eigenvalues
            * eigenvalues
        )
        manual_transform = np.asarray(
            [
                ETA
                * ETA
                / BATCH
                * np.sum(
                    multiplicity
                    * eigenvalues
                    * eigenvalues
                    * (1.0 - decrement) ** step
                )
                for step in steps
            ]
        )
        manual_scaled = (
            manual_transform
            * BATCH
            / (ETA * ETA)
            * 2.0**case.exponent
            / math.gamma(case.exponent + 1.0)
        )
        self.assertTrue(np.all(np.isfinite(scaled)))
        self.assertTrue(np.all(scaled > 0.0))
        np.testing.assert_allclose(scaled, manual_scaled, rtol=2.0e-13)


class RatioControlTests(unittest.TestCase):
    def test_square_root_profile_has_exact_budget_and_minimum(self) -> None:
        spectrum = build_spectrum(256, 0.01)
        profile = ratio_profiles(spectrum, 128, 8.0, 512)
        self.assertAlmostEqual(profile["budgets"]["optimal"], 128.0, places=10)
        self.assertAlmostEqual(
            profile["objectives"]["optimal"],
            profile["j_star_formula"],
            places=11,
        )
        self.assertLessEqual(
            profile["objectives"]["optimal"],
            profile["objectives"]["constant"],
        )
        self.assertLessEqual(
            profile["objectives"]["optimal"],
            profile["objectives"]["reversed"],
        )

    def test_unit_batch_quantiles_preserve_budget_and_horizon(self) -> None:
        spectrum = build_spectrum(256, 0.01)
        profile = ratio_profiles(spectrum, 128, 8.0, 512)
        schedules = {}
        for name, ratio in profile["profiles"].items():
            steps, discrepancy = equal_sample_quantiles(
                profile["edges"], ratio, 128
            )
            schedules[name] = steps
            self.assertLessEqual(discrepancy, 2.0e-15)
            self.assertEqual(steps.size, 128)
            self.assertAlmostEqual(float(np.sum(steps)), 8.0, places=12)
        result = run_unit_batch_modal_recursion(spectrum, schedules)
        for value in result.values():
            self.assertEqual(value["processed_samples"], 128)
            self.assertTrue(value["pointwise_stable"])
            self.assertTrue(value["row_stable"])
            self.assertGreater(value["total_risk"], 0.0)

    def test_one_step_direct_gap_and_modal_prefactor(self) -> None:
        eigenvalue = 0.7
        multiplicity = 1.3
        clean_mass = 0.4
        floor = 0.2
        eta = 0.1
        spectrum = ModalSpectrum(
            width=1,
            relative_bin_width=0.1,
            eigenvalues=np.asarray([eigenvalue]),
            multiplicities=np.asarray([multiplicity]),
            initial_clean_mass=np.asarray([clean_mass]),
            approximation_floor=floor,
            mode_count=1,
        )
        result = run_unit_batch_modal_recursion(
            spectrum, {"one_step": np.asarray([eta])}
        )["one_step"]
        q = 1.0 - 2.0 * eta * eigenvalue + 2.0 * eta * eta * eigenvalue**2
        injection = eta * eta * multiplicity * eigenvalue**2
        expected_clean = q * clean_mass + injection * (floor + clean_mass)
        expected_gap = injection * SIGMA2
        self.assertAlmostEqual(result["clean_centered"], expected_clean, places=14)
        self.assertAlmostEqual(result["noise_gap"], expected_gap, places=14)
        self.assertAlmostEqual(
            result["total_risk"], floor + expected_clean + expected_gap, places=14
        )
        self.assertEqual(result["processed_samples"], 1)
        self.assertAlmostEqual(result["intrinsic_horizon"], eta, places=15)


class SerializationTests(unittest.TestCase):
    def test_json_contract_preserves_booleans(self) -> None:
        payload = finite_json({"passed": True, "failed": np.bool_(False)})
        self.assertIs(payload["passed"], True)
        self.assertIs(payload["failed"], False)


if __name__ == "__main__":
    unittest.main()
