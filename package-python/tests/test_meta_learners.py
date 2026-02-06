import unittest

import numpy as np
import pandas as pd
from perpetual.meta_learners import SLearner, TLearner, XLearner


class TestMetaLearners(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.w = np.random.randint(0, 2, 100)
        # Outcome with some treatment effect
        self.y = 0.5 * self.X[:, 0] + self.w * 0.3 + np.random.randn(100) * 0.01

    def test_s_learner(self):
        model = SLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_t_learner(self):
        model = TLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)

    def test_x_learner(self):
        model = XLearner(budget=0.1)
        model.fit(self.X, self.w, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)

    def test_pandas_inputs(self):
        X_df = pd.DataFrame(self.X, columns=["a", "b", "c"])
        for Learner in [SLearner, TLearner, XLearner]:
            model = Learner(budget=0.05)
            model.fit(X_df, self.w, self.y)
            preds = model.predict(X_df)
            self.assertEqual(len(preds), 100)


if __name__ == "__main__":
    unittest.main()
