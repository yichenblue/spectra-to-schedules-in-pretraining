from __future__ import annotations

import unittest

from experiments.preserve_change_destroy.plot_experiment_0c_three_layer_bridge import (
    merge_layers,
)


class Experiment0cTests(unittest.TestCase):
    def test_three_layers_share_the_declared_discrete_grid(self) -> None:
        curves, metrics, audit = merge_layers()
        self.assertEqual(len(metrics), 6)
        self.assertEqual(
            len({(row["regime"], row["theta"], row["response"]) for row in curves}),
            6,
        )
        self.assertTrue(all(audit["contract_checks"].values()))
        self.assertLess(
            max(
                max(
                    row["true_finite_total_relative_l2"],
                    row["true_finite_gap_relative_l2"],
                )
                for row in metrics
            ),
            0.04,
        )
        self.assertLess(
            max(
                max(
                    row["finite_de_total_relative_l2"],
                    row["finite_de_gap_relative_l2"],
                )
                for row in metrics
            ),
            0.04,
        )


if __name__ == "__main__":
    unittest.main()
