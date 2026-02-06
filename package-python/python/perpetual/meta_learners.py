from typing import Optional

import numpy as np
from typing_extensions import Self

from perpetual.booster import PerpetualBooster


class SLearner:
    """
    S-Learner (Single Learner) for Heterogeneous Treatment Effect (HTE) estimation.

    Uses a single model to estimate the outcome:
    Y ~ M(X, W)

    CATE = M(X, 1) - M(X, 0)
    """

    def __init__(self, budget: float = 0.5, **kwargs):
        self.learner = PerpetualBooster(budget=budget, **kwargs)

    def fit(self, X, w, y) -> Self:
        w_arr = np.array(w)
        if not np.all(np.isin(w_arr, [0, 1])):
            raise ValueError("Treatment 'w' must be binary (0 or 1).")
        # Augment X with W
        X_aug = np.column_stack([X, w])
        self.learner.fit(X_aug, y)
        return self

    def predict(self, X) -> np.ndarray:
        # Predict M(X, 1)
        X_1 = np.column_stack([X, np.ones(X.shape[0])])
        preds_1 = self.learner.predict(X_1)

        # Predict M(X, 0)
        X_0 = np.column_stack([X, np.zeros(X.shape[0])])
        preds_0 = self.learner.predict(X_0)

        return preds_1 - preds_0


class TLearner:
    """
    T-Learner (Two Learners) for Heterogeneous Treatment Effect (HTE) estimation.

    Uses two separate models:
    M0(X) ~ Y[W=0]
    M1(X) ~ Y[W=1]

    CATE = M1(X) - M0(X)
    """

    def __init__(self, budget: float = 0.5, **kwargs):
        self.m0 = PerpetualBooster(budget=budget, **kwargs)
        self.m1 = PerpetualBooster(budget=budget, **kwargs)

    def fit(self, X, w, y) -> Self:
        w_arr = np.array(w)
        if not np.all(np.isin(w_arr, [0, 1])):
            raise ValueError("Treatment 'w' must be binary (0 or 1).")
        mask_0 = w == 0
        mask_1 = w == 1

        self.m0.fit(X[mask_0], y[mask_0])
        self.m1.fit(X[mask_1], y[mask_1])
        return self

    def predict(self, X) -> np.ndarray:
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    """
    X-Learner for HTE estimation (typically better for imbalanced treatment groups).

    Stage 1: Estimate T-Learner (M0, M1).
    Stage 2: Impute separate treatment effects derived from Stage 1.
             D1 = Y[W=1] - M0(X[W=1])
             D0 = M1(X[W=0]) - Y[W=0]
    Stage 3: Fit models for imputed effects.
             tau1(X) ~ D1
             tau0(X) ~ D0
    Stage 4: Combine using propensity score g(X).
             CATE(X) = g(X)*tau0(X) + (1-g(X))*tau1(X)
    """

    def __init__(
        self,
        budget: float = 0.5,
        propensity_model: Optional[PerpetualBooster] = None,
        **kwargs,
    ):
        self.m0 = PerpetualBooster(budget=budget, **kwargs)
        self.m1 = PerpetualBooster(budget=budget, **kwargs)
        self.tau0 = PerpetualBooster(budget=budget, objective="SquaredLoss", **kwargs)
        self.tau1 = PerpetualBooster(budget=budget, objective="SquaredLoss", **kwargs)

        if propensity_model is None:
            self.g = PerpetualBooster(budget=budget, objective="LogLoss", **kwargs)
        else:
            self.g = propensity_model

    def fit(self, X, w, y) -> Self:
        w_arr = np.array(w)
        if not np.all(np.isin(w_arr, [0, 1])):
            raise ValueError("Treatment 'w' must be binary (0 or 1).")
        mask_0 = w == 0
        mask_1 = w == 1

        # Stage 1
        self.m0.fit(X[mask_0], y[mask_0])
        self.m1.fit(X[mask_1], y[mask_1])

        # Stage 2
        pred_m0_on_1 = self.m0.predict(X[mask_1])
        pred_m1_on_0 = self.m1.predict(X[mask_0])

        d1 = y[mask_1] - pred_m0_on_1
        d0 = pred_m1_on_0 - y[mask_0]

        # Stage 3
        self.tau1.fit(X[mask_1], d1)
        self.tau0.fit(X[mask_0], d0)

        # Propensity
        self.g.fit(X, w)

        return self

    def predict(self, X) -> np.ndarray:
        p_score = self.g.predict_proba(X)[:, 1]

        # Stage 4 combination
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)

        return p_score * tau0_pred + (1 - p_score) * tau1_pred
