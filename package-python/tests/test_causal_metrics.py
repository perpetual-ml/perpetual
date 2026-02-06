import unittest

import numpy as np
from perpetual.causal_metrics import (
    auuc,
    cumulative_gain_curve,
    qini_coefficient,
    qini_curve,
)


class TestCausalMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 500
        self.w = np.random.randint(0, 2, n)
        # Outcome has both a baseline component and a heterogeneous
        # treatment effect that depends on a latent "uplift" dimension.
        x_uplift = np.random.randn(n)
        true_tau = 0.3 * (x_uplift > 0).astype(float)  # uplift only when x>0
        baseline = 0.3
        p_y = np.clip(baseline + true_tau * self.w, 0.01, 0.99)
        self.y = (np.random.rand(n) < p_y).astype(float)
        # "Good" score correlates with true effect
        self.uplift_good = true_tau + np.random.randn(n) * 0.05
        self.uplift_random = np.random.randn(n)

    # ---------------------------------------------------------------
    # cumulative_gain_curve
    # ---------------------------------------------------------------
    def test_cumulative_gain_curve_shape(self):
        fracs, gains = cumulative_gain_curve(self.y, self.w, self.uplift_random)
        self.assertEqual(len(fracs), len(self.y))
        self.assertEqual(len(gains), len(self.y))
        # Fractions should go from ~0 to 1
        self.assertAlmostEqual(fracs[-1], 1.0)
        self.assertTrue(fracs[0] > 0)

    def test_cumulative_gain_curve_finite(self):
        fracs, gains = cumulative_gain_curve(self.y, self.w, self.uplift_good)
        self.assertTrue(np.all(np.isfinite(gains)))

    # ---------------------------------------------------------------
    # AUUC
    # ---------------------------------------------------------------
    def test_auuc_good_vs_random(self):
        """A good model should have higher AUUC than a random model."""
        auuc_good = auuc(self.y, self.w, self.uplift_good, normalize=True)
        auuc_rand = auuc(self.y, self.w, self.uplift_random, normalize=True)
        self.assertGreater(
            auuc_good,
            auuc_rand,
            f"Good model AUUC ({auuc_good:.4f}) should exceed random ({auuc_rand:.4f})",
        )

    def test_auuc_returns_float(self):
        result = auuc(self.y, self.w, self.uplift_random)
        self.assertIsInstance(result, float)

    def test_auuc_normalized_random_near_zero(self):
        """Random uplift scores should yield AUUC near 0 when normalized."""
        np.random.seed(0)
        n = 5000
        w = np.random.randint(0, 2, n)
        y = np.random.randint(0, 2, n).astype(float)
        score = np.random.randn(n)
        result = auuc(y, w, score, normalize=True)
        self.assertAlmostEqual(result, 0.0, places=1)

    # ---------------------------------------------------------------
    # Qini curve
    # ---------------------------------------------------------------
    def test_qini_curve_shape(self):
        fracs, q = qini_curve(self.y, self.w, self.uplift_random)
        # Prepended origin → n+1 elements
        self.assertEqual(len(fracs), len(self.y) + 1)
        self.assertEqual(len(q), len(self.y) + 1)
        self.assertAlmostEqual(fracs[0], 0.0)
        self.assertAlmostEqual(fracs[-1], 1.0)
        self.assertAlmostEqual(q[0], 0.0)

    def test_qini_curve_finite(self):
        fracs, q = qini_curve(self.y, self.w, self.uplift_good)
        self.assertTrue(np.all(np.isfinite(q)))

    # ---------------------------------------------------------------
    # Qini coefficient
    # ---------------------------------------------------------------
    def test_qini_coefficient_good_vs_random(self):
        qc_good = qini_coefficient(self.y, self.w, self.uplift_good)
        qc_rand = qini_coefficient(self.y, self.w, self.uplift_random)
        self.assertGreater(
            qc_good,
            qc_rand,
            f"Good model Qini ({qc_good:.4f}) should exceed random ({qc_rand:.4f})",
        )

    def test_qini_coefficient_returns_float(self):
        result = qini_coefficient(self.y, self.w, self.uplift_random)
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
