import json
from pathlib import Path
import tempfile
import unittest

from experiments.theorem_validation.spectral_loss_sgd import CASES
from experiments.theorem_validation.spectral_loss_sgd_width_time import run


class TrueSampledWidthTimeTests(unittest.TestCase):
    def test_tiny_run_executes_all_six_cases_without_proxy(self) -> None:
        config = {
            "run_id": "tiny-real-sgd-test", "stage": "unit_test",
            "primary_prediction_id": "P2_PHASE", "claim_bearing": False,
            "eta": 0.05, "batch_size": 4,
            "width_steps": [{"width": 32, "steps": 128}],
            "seeds": [91], "evaluation_points": 49,
            "fit_intrinsic_time": [0.5, 6.4],
        }
        with tempfile.TemporaryDirectory() as temporary:
            summary = run(config, Path(temporary))
            self.assertEqual(len(summary["metrics"]), len(CASES))
            self.assertFalse(summary["uses_analytic_recurrence"])
            self.assertFalse(summary["uses_spectral_binning"])
            self.assertEqual(summary["training_mode"], "fresh_dense_gaussian_minibatches_and_explicit_parameter_updates")
            stored = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(stored["estimand"], "sampled_complete_population_excess_risk")


if __name__ == "__main__":
    unittest.main()
