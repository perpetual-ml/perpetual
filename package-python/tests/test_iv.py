import unittest

import numpy as np
from perpetual.iv import BraidedBooster


class TestBraidedBooster(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        U = np.random.randn(n)  # Unobserved confounder
        z = np.random.randn(n)  # Instrument
        w = 0.8 * z + 0.5 * U + np.random.randn(n) * 0.1
        y = 2.0 * w + 1.2 * U + np.random.randn(n) * 0.1

        self.X = np.random.randn(n, 2)
        self.Z = z.reshape(-1, 1)
        self.y = y
        self.w = w

    def test_initialization(self):
        model = BraidedBooster(stage1_budget=0.2, stage2_budget=0.3)
        self.assertEqual(model.stage1_budget, 0.2)
        self.assertEqual(model.stage2_budget, 0.3)

    def test_fit_predict(self):
        model = BraidedBooster(stage1_budget=0.1, stage2_budget=0.1)
        model.fit(self.X, self.Z, self.y, self.w)

        # Standard prediction
        preds = model.predict(self.X, w_counterfactual=self.w)
        self.assertEqual(len(preds), 100)

        # Counterfactual prediction
        preds_cf = model.predict(self.X, w_counterfactual=np.ones(100))
        self.assertEqual(len(preds_cf), 100)

    def test_serialization(self):
        model = BraidedBooster(stage1_budget=0.1, stage2_budget=0.1)
        model.fit(self.X, self.Z, self.y, self.w)

        json_str = model.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("stage1", json_str)
        self.assertIn("stage2", json_str)

        new_model = BraidedBooster.from_json(json_str)
        new_model = BraidedBooster.from_json(json_str)
        preds_old = model.predict(self.X, w_counterfactual=self.w)
        preds_new = new_model.predict(self.X, w_counterfactual=self.w)
        np.testing.assert_allclose(preds_old, preds_new, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
