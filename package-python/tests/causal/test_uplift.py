import unittest

import numpy as np
import pandas as pd
from perpetual.uplift import UpliftBooster


class TestUpliftBooster(unittest.TestCase):
    def setUp(self):
        # Generate small synthetic data for testing
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.w = np.random.randint(0, 2, 100)
        self.y = (
            0.5 * self.X[:, 0]
            + self.w * (0.2 * self.X[:, 1])
            + np.random.randn(100) * 0.1
        )

    def test_initialization(self):
        model = UpliftBooster(
            outcome_budget=0.2, propensity_budget=0.05, effect_budget=0.1
        )
        self.assertEqual(model.outcome_budget, 0.2)
        self.assertEqual(model.propensity_budget, 0.05)
        self.assertEqual(model.effect_budget, 0.1)

    def test_fit_predict(self):
        model = UpliftBooster(effect_budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_fit_pandas(self):
        X_df = pd.DataFrame(self.X, columns=[f"f{i}" for i in range(5)])
        model = UpliftBooster(effect_budget=0.1)
        model.fit(X_df, self.w, self.y)
        preds = model.predict(X_df)
        self.assertEqual(len(preds), 100)

    def test_serialization(self):
        model = UpliftBooster(effect_budget=0.1)
        model.fit(self.X, self.w, self.y)

        json_str = model.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("outcome_model", json_str)
        self.assertIn("propensity_model", json_str)
        self.assertIn("effect_model", json_str)

        new_model = UpliftBooster.from_json(json_str)
        preds_old = model.predict(self.X)
        preds_new = new_model.predict(self.X)
        np.testing.assert_allclose(preds_old, preds_new, atol=1e-5)

    def test_invalid_inputs(self):
        model = UpliftBooster()
        with self.assertRaises(Exception):
            # w should be binary
            model.fit(self.X, np.random.randn(100), self.y)


if __name__ == "__main__":
    unittest.main()
