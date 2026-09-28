from __future__ import annotations

import unittest

import numpy as np

from experiments.theorem_validation.spectral_width_time_full_sgd import (
    aggregate_exponentials,
    exponential_series,
    full_risk_from_volterra,
    geometric_bin_edges,
)


class SpectralWidthTimeFullSgdTests(unittest.TestCase):
    def test_geometric_bins_cover_domain(self) -> None:
        edges = geometric_bin_edges(1000, 64)
        self.assertEqual(int(edges[0]), 0)
        self.assertEqual(int(edges[-1]), 1000)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_singleton_bins_recover_exact_exponential_sum(self) -> None:
        weights = np.asarray([1.0, 2.0, 3.0])
        q = np.asarray([0.9, 0.8, 0.7])
        binned_weights, binned_q = aggregate_exponentials(weights, np.log(q), 3)
        series = exponential_series(binned_weights, binned_q, 4)
        expected = np.asarray([np.sum(weights * q**step) for step in range(5)])
        np.testing.assert_allclose(series, expected, rtol=1e-14, atol=1e-14)

    def test_volterra_recurrence_retains_feedback(self) -> None:
        forcing = np.asarray([1.0, 0.8, 0.64, 0.512])
        kernel = np.asarray([2.0, 1.0, 0.5, 0.25])
        risk = full_risk_from_volterra(forcing, kernel, 0.1)
        self.assertAlmostEqual(float(risk[1]), 1.0)
        self.assertAlmostEqual(float(risk[2]), 0.94)
        self.assertTrue(np.all(risk >= forcing))


if __name__ == "__main__":
    unittest.main()
