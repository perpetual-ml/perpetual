"""Meta-learner strategies for heterogeneous treatment effect estimation.

Provides S-Learner, T-Learner, X-Learner, and DR-Learner wrappers around
`PerpetualBooster` for estimating the Conditional Average Treatment
Effect (CATE) from observational data.

All learners follow a consistent API:
    - ``fit(X, w, y)`` – fit on covariates, binary treatment, and outcome.
    - ``predict(X)`` – return estimated CATE for each row.
    - ``feature_importances_`` – feature importance from the final effect
      model (available after ``fit``).
"""

from typing import Optional

import numpy as np
from typing_extensions import Self

from perpetual.booster import PerpetualBooster


def _validate_binary_treatment(w):
    """Validate that *w* is a binary {0, 1} array."""
    w_arr = np.asarray(w)
    if not np.all(np.isin(w_arr, [0, 1])):
        raise ValueError("Treatment 'w' must be binary (0 or 1).")
    return w_arr


# ---------------------------------------------------------------------------
# S-Learner
# ---------------------------------------------------------------------------


class SLearner:
    """S-Learner (Single Learner) for Heterogeneous Treatment Effect (HTE) estimation.

    Uses a single model to estimate the outcome:
    ``Y ~ M(X, W)``

    The CATE is obtained by contrasting predictions under treatment and
    control: ``CATE = M(X, 1) - M(X, 0)``.
    """

    def __init__(self, budget: float = 0.5, **kwargs):
        """Create an S-Learner.

        Parameters
        ----------
        budget : float, default=0.5
            Fitting budget forwarded to `PerpetualBooster`.
        **kwargs
            Additional keyword arguments forwarded to `PerpetualBooster`.
        """
        self.budget = budget
        self._kwargs = kwargs
        self.learner = PerpetualBooster(budget=budget, **kwargs)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, w, y) -> Self:
        """Fit the single model on covariates augmented with treatment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcome.

        Returns
        -------
        self
            Fitted estimator.
        """
        w_arr = _validate_binary_treatment(w)
        X_arr = np.asarray(X)
        X_aug = np.column_stack([X_arr, w_arr])
        self.learner.fit(X_aug, y)
        # Feature importances: drop the appended treatment column.
        self.feature_importances_ = self.learner.feature_importances_[:-1]
        return self

    def predict(self, X) -> np.ndarray:
        """Estimate the CATE as ``M(X, 1) - M(X, 0)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated treatment effect for each sample.
        """
        X_arr = np.asarray(X)
        X_1 = np.column_stack([X_arr, np.ones(X_arr.shape[0])])
        X_0 = np.column_stack([X_arr, np.zeros(X_arr.shape[0])])
        return self.learner.predict(X_1) - self.learner.predict(X_0)


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------


class TLearner:
    """T-Learner (Two Learners) for Heterogeneous Treatment Effect (HTE) estimation.

    Uses two separate models::

        M0(X) ~ Y[W=0]
        M1(X) ~ Y[W=1]

    The CATE is ``M1(X) - M0(X)``.
    """

    def __init__(self, budget: float = 0.5, **kwargs):
        """Create a T-Learner.

        Parameters
        ----------
        budget : float, default=0.5
            Fitting budget forwarded to each `PerpetualBooster`.
        **kwargs
            Additional keyword arguments forwarded to `PerpetualBooster`.
        """
        self.budget = budget
        self._kwargs = kwargs
        self.m0 = PerpetualBooster(budget=budget, **kwargs)
        self.m1 = PerpetualBooster(budget=budget, **kwargs)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, w, y) -> Self:
        """Fit separate models on control and treated groups.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcome.

        Returns
        -------
        self
            Fitted estimator.
        """
        w_arr = _validate_binary_treatment(w)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        mask_0 = w_arr == 0
        mask_1 = w_arr == 1

        self.m0.fit(X_arr[mask_0], y_arr[mask_0])
        self.m1.fit(X_arr[mask_1], y_arr[mask_1])

        # Average feature importances across both arms.
        self.feature_importances_ = (
            self.m0.feature_importances_ + self.m1.feature_importances_
        ) / 2.0
        return self

    def predict(self, X) -> np.ndarray:
        """Estimate the CATE as ``M1(X) - M0(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated treatment effect for each sample.
        """
        return self.m1.predict(X) - self.m0.predict(X)


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------


class XLearner:
    """X-Learner for HTE estimation (typically better for imbalanced treatment groups).

    Stage 1: Estimate T-Learner (M0, M1).
    Stage 2: Impute separate treatment effects from Stage 1::

        D1 = Y[W=1] - M0(X[W=1])
        D0 = M1(X[W=0]) - Y[W=0]

    Stage 3: Fit models for imputed effects::

        tau1(X) ~ D1
        tau0(X) ~ D0

    Stage 4: Combine using propensity score g(X)::

        CATE(X) = g(X) * tau0(X) + (1 - g(X)) * tau1(X)
    """

    def __init__(
        self,
        budget: float = 0.5,
        propensity_model: Optional[PerpetualBooster] = None,
        **kwargs,
    ):
        """Create an X-Learner.

        Parameters
        ----------
        budget : float, default=0.5
            Fitting budget forwarded to each `PerpetualBooster`.
        propensity_model : PerpetualBooster, optional
            Pre-configured booster for propensity estimation.  If ``None``, a
            default ``LogLoss`` booster is created.
        **kwargs
            Additional keyword arguments forwarded to `PerpetualBooster`.
        """
        self.budget = budget
        self._kwargs = kwargs
        self.m0 = PerpetualBooster(budget=budget, **kwargs)
        self.m1 = PerpetualBooster(budget=budget, **kwargs)
        self.tau0 = PerpetualBooster(budget=budget, objective="SquaredLoss", **kwargs)
        self.tau1 = PerpetualBooster(budget=budget, objective="SquaredLoss", **kwargs)

        if propensity_model is None:
            self.g = PerpetualBooster(budget=budget, objective="LogLoss", **kwargs)
        else:
            self.g = propensity_model

        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, w, y) -> Self:
        """Fit all stages of the X-Learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcome.

        Returns
        -------
        self
            Fitted estimator.
        """
        w_arr = _validate_binary_treatment(w)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        mask_0 = w_arr == 0
        mask_1 = w_arr == 1

        # Stage 1
        self.m0.fit(X_arr[mask_0], y_arr[mask_0])
        self.m1.fit(X_arr[mask_1], y_arr[mask_1])

        # Stage 2
        pred_m0_on_1 = self.m0.predict(X_arr[mask_1])
        pred_m1_on_0 = self.m1.predict(X_arr[mask_0])

        d1 = y_arr[mask_1] - pred_m0_on_1
        d0 = pred_m1_on_0 - y_arr[mask_0]

        # Stage 3
        self.tau1.fit(X_arr[mask_1], d1)
        self.tau0.fit(X_arr[mask_0], d0)

        # Propensity
        self.g.fit(X_arr, w_arr)

        # Average feature importances from the two effect models.
        self.feature_importances_ = (
            self.tau0.feature_importances_ + self.tau1.feature_importances_
        ) / 2.0
        return self

    def predict(self, X) -> np.ndarray:
        """Estimate the CATE by combining ``tau0`` and ``tau1`` via propensity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated treatment effect for each sample.
        """
        # Propensity score as P(W=1|X).
        # Use predict (raw log-odds) and apply sigmoid for robustness,
        # since predict_proba may not be available for every objective.
        log_odds = self.g.predict(X)
        p_score = 1.0 / (1.0 + np.exp(-log_odds))

        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)

        return p_score * tau0_pred + (1 - p_score) * tau1_pred


# ---------------------------------------------------------------------------
# DR-Learner  (Doubly Robust / AIPW)
# ---------------------------------------------------------------------------


class DRLearner:
    """Doubly Robust (DR) Learner for heterogeneous treatment effect estimation.

    The DR-Learner constructs an *augmented IPW* (AIPW) pseudo-outcome
    and regresses it on covariates to estimate the CATE directly.

    AIPW Pseudo-outcome
    --------------------
    For each sample *i*:

    .. math::

        \\Gamma_i = \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{W_i (Y_i - \\hat{\\mu}_1(X_i))}{\\hat{p}(X_i)}
            - \\frac{(1-W_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-\\hat{p}(X_i)}

    The DR-Learner is "doubly robust" because the resulting CATE estimate
    is consistent as long as *either* the outcome models or the propensity
    model is correctly specified.

    Stages
    ------
    1. Fit outcome models per arm: ``mu0(X)`` and ``mu1(X)``.
    2. Fit propensity model: ``p(X) = P(W=1|X)``.
    3. Compute AIPW pseudo-outcomes.
    4. Fit a final regression of pseudo-outcomes on covariates.

    Parameters
    ----------
    budget : float, default=0.5
        Fitting budget forwarded to each internal `PerpetualBooster`.
    propensity_budget : float, optional
        Separate budget for the propensity model (defaults to ``budget``).
    clip : float, default=0.01
        Propensity scores are clipped to ``[clip, 1-clip]`` for stability.
    **kwargs
        Additional keyword arguments forwarded to `PerpetualBooster`.
    """

    def __init__(
        self,
        budget: float = 0.5,
        propensity_budget: Optional[float] = None,
        clip: float = 0.01,
        **kwargs,
    ):
        self.budget = budget
        self.propensity_budget = propensity_budget or budget
        self.clip = clip
        self._kwargs = kwargs

        self.mu0 = PerpetualBooster(budget=budget, **kwargs)
        self.mu1 = PerpetualBooster(budget=budget, **kwargs)
        self.propensity = PerpetualBooster(
            budget=self.propensity_budget, objective="LogLoss", **kwargs
        )
        self.effect = PerpetualBooster(budget=budget, objective="SquaredLoss", **kwargs)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, w, y) -> Self:
        """Fit the DR-Learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcome.

        Returns
        -------
        self
            Fitted estimator.
        """
        w_arr = _validate_binary_treatment(w)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask_0 = w_arr == 0
        mask_1 = w_arr == 1

        # Stage 1: Outcome models per arm.
        self.mu0.fit(X_arr[mask_0], y_arr[mask_0])
        self.mu1.fit(X_arr[mask_1], y_arr[mask_1])

        mu0_hat = self.mu0.predict(X_arr)
        mu1_hat = self.mu1.predict(X_arr)

        # Stage 2: Propensity model.
        self.propensity.fit(X_arr, w_arr)
        log_odds = self.propensity.predict(X_arr)
        p_hat = np.clip(1.0 / (1.0 + np.exp(-log_odds)), self.clip, 1.0 - self.clip)

        # Stage 3: AIPW pseudo-outcomes.
        gamma = (
            mu1_hat
            - mu0_hat
            + w_arr * (y_arr - mu1_hat) / p_hat
            - (1 - w_arr) * (y_arr - mu0_hat) / (1 - p_hat)
        )

        # Stage 4: Final effect model.
        self.effect.fit(X_arr, gamma)
        self.feature_importances_ = self.effect.feature_importances_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict CATE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated treatment effect for each sample.
        """
        return self.effect.predict(X)
