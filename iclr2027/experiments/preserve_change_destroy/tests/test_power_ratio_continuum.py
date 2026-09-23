from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest
from unittest import mock

import numpy as np

from iclr2027.experiments.preserve_change_destroy.de_quadrature import (
    head_preserving_quadrature,
    run_continuum_modal_dynamics,
)
from iclr2027.experiments.preserve_change_destroy.power_ratio_continuum import (
    exact_power_inverse_ratio_cell_integrals,
    run_exact_power_ratio_continuum_modal_dynamics,
)
from iclr2027.experiments.preserve_change_destroy.run_exact_power_continuum_m100000 import (
    CONTROL_THETAS,
    PAPER_FIGURE_LAYOUT,
    PAPER_FIGURE_SIZE_INCHES,
    PAPER_THETAS,
    PRIMARY_THETAS,
    SENTINEL_THETAS,
    THETAS,
    _admissible_decay_theory,
    _compare_resolution,
    _component_guide_exponent,
    _destroy_boundary_theory,
    _process,
)
from iclr2027.experiments.preserve_change_destroy import (
    run_exact_power_continuum_m100000_sigma25 as sigma25_runner,
)
from iclr2027.experiments.preserve_change_destroy.run_exact_power_continuum_m100000_sigma25 import (
    DISPLAY_ONLY_THETAS as SIGMA25_DISPLAY_ONLY_THETAS,
    MAKE_COMPACT_TRIPTYCH as SIGMA25_MAKE_COMPACT_TRIPTYCH,
    OUTPUT_DIR as SIGMA25_OUTPUT_DIR,
    SIGMA2 as SIGMA25,
)


class ExactPowerRatioContinuumTests(unittest.TestCase):
    def test_sigma25_wrapper_adds_only_a_display_paper_trajectory(self) -> None:
        self.assertEqual(SIGMA25, 25.0)
        self.assertEqual(
            SIGMA25_OUTPUT_DIR.name,
            "exact_power_continuum_m100000_sigma25",
        )
        self.assertEqual(SIGMA25_DISPLAY_ONLY_THETAS, (2.0,))
        self.assertTrue(SIGMA25_MAKE_COMPACT_TRIPTYCH)

    def test_compact_figure_theta_two_does_not_change_frozen_gates(self) -> None:
        self.assertEqual(
            THETAS,
            CONTROL_THETAS + PRIMARY_THETAS + SENTINEL_THETAS,
        )
        self.assertEqual(PAPER_THETAS, (0.0, 0.25, 0.5, 2.0))
        self.assertFalse(set(SIGMA25_DISPLAY_ONLY_THETAS).intersection(THETAS))
        self.assertNotIn(2.0, PRIMARY_THETAS)
        self.assertNotIn(2.0, SENTINEL_THETAS)

    def test_paper_triptych_uses_wide_single_row_layout(self) -> None:
        self.assertAlmostEqual(
            PAPER_FIGURE_SIZE_INCHES[0] / PAPER_FIGURE_SIZE_INCHES[1],
            12.40 / 4.15,
        )
        self.assertIn("single-row", PAPER_FIGURE_LAYOUT)
        self.assertIn("side by side", PAPER_FIGURE_LAYOUT)
        self.assertIn("legend below", PAPER_FIGURE_LAYOUT)

    def test_sigma25_main_passes_display_only_options_explicitly(self) -> None:
        fake_summary = {
            "status": "PILOT_UNPROMOTED",
            "protocol": {"sigma2": 25.0},
            "numerical_contract_pass": True,
            "primary_phase_gate_pass": True,
            "sentinel_phase_gate_pass": True,
            "overall_observable_gate_pass": True,
            "destroy_control": {},
            "elapsed_seconds": 0.0,
            "artifacts": {},
        }
        with mock.patch.object(
            sigma25_runner, "run_experiment", return_value=fake_summary
        ) as mocked_run:
            with redirect_stdout(io.StringIO()):
                sigma25_runner.main()
        mocked_run.assert_called_once_with(
            output_dir=SIGMA25_OUTPUT_DIR,
            sigma2=SIGMA25,
            display_only_thetas=SIGMA25_DISPLAY_ONLY_THETAS,
            make_compact_triptych=SIGMA25_MAKE_COMPACT_TRIPTYCH,
        )

    def test_display_only_theta_does_not_change_frozen_trajectories(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 48, 0.03)
        times = np.concatenate(
            [np.asarray([0.0]), np.geomspace(1.0e-3, 6.0, 61)]
        )
        frozen = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature, times, THETAS, 25.0, 1.0 / 128.0, 0.2
        )
        extended = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature, times, THETAS + (2.0,), 25.0, 1.0 / 128.0, 0.2
        )
        for theta in THETAS:
            np.testing.assert_allclose(
                extended[theta].clean_centered, frozen[theta].clean_centered
            )
            np.testing.assert_allclose(
                extended[theta].noise_gap, frozen[theta].noise_gap
            )
            np.testing.assert_allclose(
                extended[theta].row_mass, frozen[theta].row_mass
            )
        processed = _process(
            quadrature,
            extended,
            np.geomspace(0.2, 6.0, 21),
            theta_values=THETAS + (2.0,),
            display_only_thetas=(2.0,),
        )
        self.assertEqual(processed["metrics"][2.0]["group"], "display-only")
        self.assertFalse(
            processed["metrics"][2.0]["included_in_overall_observable_gate"]
        )
        self.assertFalse(
            processed["metrics"][2.0]["included_in_frozen_numerical_contract"]
        )

    def test_display_only_resolution_failure_is_excluded_from_frozen_gate(self) -> None:
        times = np.geomspace(1.0, 10.0, 9)

        def payload(theta: float, values: np.ndarray) -> dict[str, object]:
            return {
                "target_times": times,
                "pooled_clean": values,
                "pooled_centered_clean": values,
                "gap": {theta: values},
                "contrasts": {theta: values},
                "envelopes": {theta: values},
                "total": {theta: values},
            }

        reference = times ** (-0.25)
        frozen_rows = _compare_resolution(
            "frozen",
            payload(0.25, reference),
            payload(0.25, reference),
            theta_values=(0.25,),
            included_in_numerical_gate=True,
        )
        display_rows = _compare_resolution(
            "display",
            payload(2.0, times ** (-0.7)),
            payload(2.0, times ** (-0.4)),
            theta_values=(2.0,),
            included_in_numerical_gate=False,
        )
        rows = frozen_rows + display_rows
        self.assertTrue(
            all(
                row["status"] == "PASS"
                for row in rows
                if row["included_in_numerical_gate"]
            )
        )
        self.assertTrue(
            any(
                row["status"] == "FAIL"
                for row in rows
                if not row["included_in_numerical_gate"]
            )
        )

    def test_below_destroy_boundary_is_not_clipped_to_a_decay_phase(self) -> None:
        self.assertAlmostEqual(_destroy_boundary_theory(), 0.25)
        self.assertIsNone(_admissible_decay_theory(0.0))
        self.assertEqual(_admissible_decay_theory(0.25), (0.0, 0.0))
        self.assertAlmostEqual(_component_guide_exponent(0.0), -0.25)

    def test_cell_integrals_handle_constant_crossing_and_log_branches(self) -> None:
        amplitude = 1.0 / 128.0
        times = np.asarray([0.0, 0.4, 0.5, 2.0, 4.0])
        theta = (0.5, 1.0, 1.5)
        values = exact_power_inverse_ratio_cell_integrals(
            times, theta, amplitude
        )
        np.testing.assert_allclose(values[:, 0], 0.4 * amplitude)
        expected_crossing = np.asarray(
            [
                0.5 + 2.0 * (np.sqrt(2.0) - 1.0),
                0.5 + np.log(2.0),
                0.5 + 2.0 * (1.0 - 1.0 / np.sqrt(2.0)),
            ]
        )
        np.testing.assert_allclose(
            values[:, 2], amplitude * expected_crossing, rtol=2.0e-14
        )

    def test_cell_integrals_are_additive_and_near_one_is_finite(self) -> None:
        amplitude = 1.0 / 128.0
        theta = (1.0 - 1.0e-10, 1.0, 1.0 + 1.0e-10)
        whole = exact_power_inverse_ratio_cell_integrals(
            np.asarray([0.0, 4.0]), theta, amplitude
        )[:, 0]
        pieces = exact_power_inverse_ratio_cell_integrals(
            np.asarray([0.0, 0.7, 1.0, 1.9, 4.0]), theta, amplitude
        ).sum(axis=1)
        np.testing.assert_allclose(whole, pieces, rtol=2.0e-14, atol=1.0e-15)
        self.assertTrue(np.all(np.isfinite(whole)))
        self.assertTrue(np.all(whole > 0.0))

    def test_theta_zero_matches_historical_constant_ratio_solver(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 128, 0.02)
        times = np.concatenate(
            [np.asarray([0.0]), np.geomspace(1.0e-3, 20.0, 121)]
        )
        old = run_continuum_modal_dynamics(
            quadrature=quadrature,
            times=times,
            theta_values=(0.0,),
            sigma2=10.0,
            inverse_ratio_amplitude=1.0 / 128.0,
            row_mass_cap=0.2,
        )[0.0]
        new = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature=quadrature,
            times=times,
            theta_values=(0.0,),
            sigma2=10.0,
            inverse_ratio_amplitude=1.0 / 128.0,
            row_mass_cap=0.2,
        )[0.0]
        np.testing.assert_allclose(new.clean_centered, old.clean_centered)
        np.testing.assert_allclose(new.noise_gap, old.noise_gap)
        np.testing.assert_allclose(new.row_mass, old.row_mass)

    def test_one_cell_gap_uses_cell_mean_not_cell_mass(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 1, 0.02)
        times = np.asarray([0.0, 0.5])
        amplitude = 1.0 / 128.0
        sigma2 = 10.0
        result = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature=quadrature,
            times=times,
            theta_values=(0.9,),
            sigma2=sigma2,
            inverse_ratio_amplitude=amplitude,
            row_mass_cap=0.2,
        )[0.9]
        survival = np.exp(-2.0 * 0.5)
        base_injection = 0.5 * (1.0 - survival)
        coupling = amplitude * base_injection
        expected_gap = coupling * sigma2 / (1.0 - 0.5 * coupling)
        self.assertAlmostEqual(result.noise_gap[1], expected_gap, places=14)

    def test_noise_linearity_and_common_warmup(self) -> None:
        quadrature = head_preserving_quadrature(0.4, 0.3, 96, 0.02)
        times = np.concatenate(
            [np.asarray([0.0]), np.geomspace(1.0e-3, 8.0, 101)]
        )
        theta = (0.0, 0.25, 0.5, 0.9, 1.5, 2.5)
        unit = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature, times, theta, 1.0, 1.0 / 128.0, 0.2
        )
        ten = run_exact_power_ratio_continuum_modal_dynamics(
            quadrature, times, theta, 10.0, 1.0 / 128.0, 0.2
        )
        before_one = times <= 1.0
        reference = ten[theta[0]]
        for theta_value in theta:
            np.testing.assert_allclose(
                ten[theta_value].noise_gap, 10.0 * unit[theta_value].noise_gap
            )
            np.testing.assert_allclose(
                ten[theta_value].clean_centered,
                unit[theta_value].clean_centered,
            )
            np.testing.assert_allclose(
                ten[theta_value].noise_gap[before_one],
                reference.noise_gap[before_one],
            )
            np.testing.assert_allclose(
                ten[theta_value].clean_centered[before_one],
                reference.clean_centered[before_one],
            )


if __name__ == "__main__":
    unittest.main()
