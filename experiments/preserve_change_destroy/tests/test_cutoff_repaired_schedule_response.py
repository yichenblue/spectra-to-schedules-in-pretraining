from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.preserve_change_destroy import (
    plot_cutoff_repaired_schedule_response as figure_module,
)


ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "cutoff_repaired_schedule_response"
)


def _rows(name: str) -> list[dict[str, str]]:
    with (ARTIFACT_ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CutoffRepairedScheduleResponseTest(unittest.TestCase):
    def test_theory_response_has_expected_boundaries_and_ceilings(self) -> None:
        lm_gap, lm_total = figure_module._theory(
            "LM", np.array([0.25, 0.50, 0.75, 1.00, 1.20])
        )
        np.testing.assert_allclose(lm_gap, [0.0, 0.25, 0.50, 0.75, 0.75])
        np.testing.assert_allclose(lm_total, [0.0, 0.25, 0.50, 0.50, 0.50])
        im_gap, im_total = figure_module._theory(
            "IM", np.array([0.0, 0.25, 0.50, 7.0 / 6.0, 1.30])
        )
        np.testing.assert_allclose(im_gap, [0.0, 0.25, 0.50, 7.0 / 6.0, 7.0 / 6.0])
        np.testing.assert_allclose(im_total, [0.0, 0.25, 0.50, 0.50, 0.50])

    def test_reviewed_statuses_are_preserved(self) -> None:
        counts: dict[str, int] = {}
        for row in _rows("layer_b_response_summary.csv"):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        self.assertEqual(counts, {"PASS": 88, "CENSORED": 14, "INCONCLUSIVE": 4})

    def test_cutoff_only_refinement_is_independent_and_passes(self) -> None:
        cutoff_rows = [
            row
            for row in _rows("layer_b_separate_resolution.csv")
            if row["refinement_axis"] == "spectral_cutoff"
        ]
        self.assertEqual(len(cutoff_rows), 106)
        self.assertTrue(all(row["status"] == "PASS" for row in cutoff_rows))
        self.assertTrue(
            all(row["reference_spectral_cutoff"] == "1000000000000" for row in cutoff_rows)
        )
        self.assertTrue(
            all(row["candidate_spectral_cutoff"] == "100000000000000" for row in cutoff_rows)
        )
        self.assertLessEqual(
            max(float(row["nominal_window_curve_relative_l2"]) for row in cutoff_rows),
            0.002,
        )
        self.assertLessEqual(
            max(float(row["nominal_window_exponent_difference"]) for row in cutoff_rows),
            0.002,
        )

    def test_true_sgd_bridge_is_bounded_and_not_exponent_evidence(self) -> None:
        rows = _rows("layer_a_true_sgd_anchor_metrics.csv")
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["passed"] == "True" for row in rows))
        self.assertLess(max(float(row["relative_l2_error"]) for row in rows), 0.05)
        self.assertGreaterEqual(
            min(float(row["fraction_within_three_standard_errors"]) for row in rows),
            0.98,
        )
        summary = json.loads((ARTIFACT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["objective_scope"]["population_to_empirical_claim_made"])
        self.assertFalse(summary["layer_A"]["fixed_width_terminal_tail_used_for_theorem"])

    def test_plotter_has_no_workstream_runtime_dependency(self) -> None:
        source = Path(figure_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("comath-codex/workstreams", source)


if __name__ == "__main__":
    unittest.main()
