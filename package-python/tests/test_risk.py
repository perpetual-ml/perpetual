import unittest

import numpy as np
import pandas as pd
from perpetual import PerpetualBooster, PerpetualRiskEngine


class TestPerpetualRiskEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 4), columns=["v1", "v2", "v3", "v4"])
        # Target: v1 + v2 > 0
        self.y = (self.X["v1"] + self.X["v2"] > 0).astype(int)

        self.model = PerpetualBooster(budget=0.5)
        self.model.fit(self.X, self.y)

    def test_initialization(self):
        engine = PerpetualRiskEngine(self.model)
        self.assertEqual(engine.booster, self.model)

    def test_generate_reason_codes(self):
        engine = PerpetualRiskEngine(self.model)
        # Select an applicant with high risk
        applicant = self.X.iloc[[0]]
        reasons = engine.generate_reason_codes(applicant, threshold=1.0)

        self.assertIsInstance(reasons, list)
        self.assertEqual(len(reasons), 1)
        self.assertIsInstance(reasons[0], list)
        # Check that it identifies some features
        self.assertTrue(len(reasons[0]) > 0)
        self.assertIn(reasons[0][0].split(":")[0].strip(), self.X.columns)

    def test_multiple_applicants(self):
        engine = PerpetualRiskEngine(self.model)
        applicants = self.X.iloc[0:5]
        reasons = engine.generate_reason_codes(applicants, threshold=0.5)
        self.assertEqual(len(reasons), 5)


if __name__ == "__main__":
    unittest.main()
