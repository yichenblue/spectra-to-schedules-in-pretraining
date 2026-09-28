import unittest

import numpy as np

from experiments.theorem_validation.spectral_long_horizon_counterexamples import (
    local_exponent_curve,
    periodic_power_fit,
)


class LongHorizonCounterexampleTests(unittest.TestCase):
    def test_narrow_local_slope_recovers_power(self) -> None:
        times = np.geomspace(1.0, 1.0e4, 1001)
        values = 3.0 * times ** -1.75
        _, slopes = local_exponent_curve(
            times, values, half_window_decades=0.035, minimum_points=7
        )
        np.testing.assert_allclose(slopes, 1.75, rtol=0.0, atol=2e-12)

    def test_fixed_period_fit_recovers_log_periodic_residual(self) -> None:
        times = np.geomspace(2.0, 1.0e4, 2001)
        log_period = 0.35
        phase = 2.0 * np.pi * np.log(times) / log_period
        values = 2.0 * times ** -1.75 * np.exp(0.08 * np.sin(phase))
        fit = periodic_power_fit(times, values, log_period=log_period)
        self.assertAlmostEqual(fit["power_exponent"], 1.75, places=3)
        self.assertAlmostEqual(fit["periodic_amplitude"], 0.08, places=10)
        self.assertGreater(fit["periodic_residual_fraction_explained"], 0.999999)


if __name__ == "__main__":
    unittest.main()
