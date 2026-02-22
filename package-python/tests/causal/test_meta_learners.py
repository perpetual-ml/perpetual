import unittest

import numpy as np
import pandas as pd
from perpetual.meta_learners import DRLearner, SLearner, TLearner, XLearner


class TestMetaLearners(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.w = np.random.randint(0, 2, 100)
        # Outcome with some treatment effect
        self.y = 0.5 * self.X[:, 0] + self.w * 0.3 + np.random.randn(100) * 0.01

    # ---------------------------------------------------------------
    # S-Learner
    # ---------------------------------------------------------------
    def test_s_learner(self):
        model = SLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_s_learner_feature_importances(self):
        model = SLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        self.assertIsNotNone(model.feature_importances_)
        self.assertEqual(len(model.feature_importances_), 3)

    # ---------------------------------------------------------------
    # T-Learner
    # ---------------------------------------------------------------
    def test_t_learner(self):
        model = TLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)

    def test_t_learner_feature_importances(self):
        model = TLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        self.assertIsNotNone(model.feature_importances_)
        self.assertEqual(len(model.feature_importances_), 3)

    # ---------------------------------------------------------------
    # X-Learner
    # ---------------------------------------------------------------
    def test_x_learner(self):
        model = XLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)

    def test_x_learner_feature_importances(self):
        model = XLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        self.assertIsNotNone(model.feature_importances_)
        self.assertEqual(len(model.feature_importances_), 3)

    # ---------------------------------------------------------------
    # DR-Learner
    # ---------------------------------------------------------------
    def test_dr_learner(self):
        model = DRLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_dr_learner_feature_importances(self):
        model = DRLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        self.assertIsNotNone(model.feature_importances_)
        self.assertEqual(len(model.feature_importances_), 3)

    def test_dr_learner_positive_effect(self):
        """DRLearner should recover a positive treatment effect."""
        np.random.seed(123)
        X = np.random.randn(200, 2)
        w = np.random.randint(0, 2, 200)
        y = X[:, 0] + 1.0 * w + np.random.randn(200) * 0.1  # true CATE = 1.0

        model = DRLearner(budget=0.1)
        model.fit(X, w, y)
        preds = model.predict(X)
        avg_pred = preds.mean()
        self.assertTrue(
            avg_pred > 0.3,
            f"Average predicted CATE should be positive, got {avg_pred:.3f}",
        )

    # ---------------------------------------------------------------
    # Pandas inputs
    # ---------------------------------------------------------------
    def test_pandas_inputs(self):
        X_df = pd.DataFrame(self.X, columns=["a", "b", "c"])
        for Learner in [SLearner, TLearner, XLearner, DRLearner]:
            model = Learner(budget=0.1)
            model.fit(X_df, self.w, self.y)
            preds = model.predict(X_df)
            self.assertEqual(len(preds), 100)

    # ---------------------------------------------------------------
    # Invalid inputs
    # ---------------------------------------------------------------
    def test_invalid_treatment(self):
        for Learner in [SLearner, TLearner, XLearner, DRLearner]:
            model = Learner(budget=0.1)
            with self.assertRaises(ValueError):
                model.fit(self.X, np.random.randn(100), self.y)


if __name__ == "__main__":
    unittest.main()
