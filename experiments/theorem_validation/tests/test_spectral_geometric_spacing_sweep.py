import unittest

import numpy as np

from experiments.theorem_validation.spectral_geometric_spacing_sweep import (
    build_geometric_case,
)


class GeometricSpacingSweepTests(unittest.TestCase):
    def test_geometric_case_is_normalized(self) -> None:
        for spacing in (1.05, 1.4, 1.75):
            eigenvalues, target = build_geometric_case(128, spacing)
            self.assertTrue(np.all(eigenvalues > 0.0))
            self.assertTrue(np.all(np.diff(eigenvalues) <= 0.0))
            self.assertAlmostEqual(
                float(eigenvalues @ np.square(target)), 1.0, places=12
            )

    def test_larger_spacing_has_smaller_second_eigenvalue(self) -> None:
        small, _ = build_geometric_case(64, 1.05)
        large, _ = build_geometric_case(64, 1.75)
        self.assertLess(large[1], small[1])


if __name__ == "__main__":
    unittest.main()
