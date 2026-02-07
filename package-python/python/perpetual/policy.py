"""Policy learning via Inverse Propensity Weighting (IPW / AIPW).

Implements the Athey & Wager (2021) policy-learning framework using
gradient boosting.  The learned policy assigns treatment when the
boosted score is positive.

Two modes are supported:

* **IPW** — Standard Inverse Propensity Weighting.
* **AIPW** (Augmented / Doubly Robust) — Reduces variance by
  incorporating a baseline outcome model.
"""

from typing import Optional

import numpy as np
from typing_extensions import Self

from perpetual.booster import PerpetualBooster

# ---------------------------------------------------------------------------
# Custom objective helpers  (mirror Rust PolicyObjective)
# ---------------------------------------------------------------------------

_PROPENSITY_CLIP = 1e-3


def _pseudo_outcomes_ipw(y, treatment, propensity):
    """Compute IPW pseudo-outcomes."""
    w = treatment.astype(float)
    p = np.clip(propensity, _PROPENSITY_CLIP, 1.0 - _PROPENSITY_CLIP)
    return y * w / p - y * (1.0 - w) / (1.0 - p)


def _pseudo_outcomes_aipw(y, treatment, propensity, mu_hat):
    """Compute AIPW (doubly robust) pseudo-outcomes."""
    w = treatment.astype(float)
    p = np.clip(propensity, _PROPENSITY_CLIP, 1.0 - _PROPENSITY_CLIP)
    return w * (y - mu_hat) / p - (1.0 - w) * (y - mu_hat) / (1.0 - p)


def _policy_loss(gamma):
    """Return a weighted logistic loss callable."""

    def loss(y, yhat, sample_weight, group):
        yhat = np.asarray(yhat)
        sigma = 1.0 / (1.0 + np.exp(-yhat))
        target = (gamma >= 0).astype(float)
        weight = np.abs(gamma)
        ll = -(
            target * np.log(np.maximum(sigma, 1e-15))
            + (1.0 - target) * np.log(np.maximum(1.0 - sigma, 1e-15))
        )
        return (weight * ll).tolist()

    return loss


def _policy_gradient(gamma):
    """Return a weighted logistic gradient callable."""

    def gradient(y, yhat, sample_weight, group):
        yhat = np.asarray(yhat)
        sigma = 1.0 / (1.0 + np.exp(-yhat))
        target = (gamma >= 0).astype(float)
        weight = np.abs(gamma)

        grad = weight * (sigma - target)
        hess = weight * sigma * (1.0 - sigma)
        return grad.tolist(), hess.tolist()

    return gradient


def _policy_initial_value():
    """Return an initial-value callable."""

    def initial_value(y, sample_weight, group):
        return 0.0

    return initial_value


# ---------------------------------------------------------------------------
# PolicyLearner
# ---------------------------------------------------------------------------


class PolicyLearner:
    """Policy learner via Inverse Propensity Weighting.

    Learns a treatment-assignment policy :math:`\\pi(X)` that maximizes
    expected reward using the Athey & Wager (2021) policy-learning
    framework.

    The learned policy assigns :math:`W = 1` when the boosted
    score :math:`F(X) > 0`.

    Parameters
    ----------
    budget : float, default=0.5
        Fitting budget forwarded to ``PerpetualBooster``.
    mode : str, default="ipw"
        ``"ipw"`` for standard Inverse Propensity Weighting or
        ``"aipw"`` for Augmented (Doubly Robust) IPW.
    propensity_budget : float, optional
        Separate budget for the propensity model.  If ``None``, defaults to
        ``budget``.  Only used when ``propensity`` is not supplied to
        :meth:`fit`.
    **kwargs
        Additional keyword arguments forwarded to ``PerpetualBooster``.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from the policy model.

    Examples
    --------
    >>> from perpetual.policy import PolicyLearner
    >>> import numpy as np
    >>> n = 500
    >>> X = np.random.randn(n, 5)
    >>> w = np.random.binomial(1, 0.5, n)
    >>> y = X[:, 0] * w + np.random.randn(n) * 0.5
    >>> pl = PolicyLearner(budget=0.3)
    >>> pl.fit(X, w, y)  # doctest: +SKIP
    >>> policy = pl.predict(X)  # doctest: +SKIP

    References
    ----------
    Athey, S., & Wager, S. (2021). *Policy learning with observational data*.
    Econometrica, 89(1), 133-161.
    """

    def __init__(
        self,
        budget: float = 0.5,
        mode: str = "ipw",
        propensity_budget: Optional[float] = None,
        **kwargs,
    ):
        if mode not in ("ipw", "aipw"):
            raise ValueError(f"mode must be 'ipw' or 'aipw', got {mode!r}.")
        self.budget = budget
        self.mode = mode
        self.propensity_budget = propensity_budget or budget
        self._kwargs = kwargs
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X,
        w,
        y,
        propensity: Optional[np.ndarray] = None,
        mu_hat: Optional[np.ndarray] = None,
    ) -> Self:
        """Fit the policy learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Observed binary treatment assignment (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcome.
        propensity : array-like of shape (n_samples,), optional
            Estimated :math:`P(W=1|X)`.  If ``None``, a propensity model is
            fitted internally.
        mu_hat : array-like of shape (n_samples,), optional
            Predicted baseline outcome :math:`\\hat{\\mu}(X)`.  Required when
            ``mode="aipw"``.  If ``None`` and ``mode="aipw"``, an outcome
            model is fitted internally.

        Returns
        -------
        self
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        w_arr = np.asarray(w, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Propensity scores
        if propensity is not None:
            p_hat = np.asarray(propensity, dtype=float)
        else:
            prop_model = PerpetualBooster(
                budget=self.propensity_budget, objective="LogLoss", **self._kwargs
            )
            prop_model.fit(X_arr, w_arr)
            log_odds = prop_model.predict(X_arr)
            p_hat = 1.0 / (1.0 + np.exp(-log_odds))

        # Pseudo-outcomes
        if self.mode == "ipw":
            gamma = _pseudo_outcomes_ipw(y_arr, w_arr, p_hat)
        else:
            # AIPW
            if mu_hat is not None:
                mu = np.asarray(mu_hat, dtype=float)
            else:
                mu_model = PerpetualBooster(
                    budget=self.budget, objective="SquaredLoss", **self._kwargs
                )
                mu_model.fit(X_arr, y_arr)
                mu = mu_model.predict(X_arr)
            gamma = _pseudo_outcomes_aipw(y_arr, w_arr, p_hat, mu)

        # Policy model via custom objective
        objective = (
            _policy_loss(gamma),
            _policy_gradient(gamma),
            _policy_initial_value(),
        )

        self._policy_model = PerpetualBooster(
            budget=self.budget, objective=objective, **self._kwargs
        )
        # The target passed to the booster is unused by the custom objective.
        self._policy_model.fit(X_arr, (gamma >= 0).astype(float))
        self.feature_importances_ = self._policy_model.feature_importances_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict the optimal treatment assignment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Binary treatment policy (1 = treat, 0 = do not treat).
        """
        scores = self.decision_function(X)
        return (scores > 0).astype(int)

    def decision_function(self, X) -> np.ndarray:
        """Return raw policy scores.

        Positive values indicate treatment is beneficial.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Raw boosted policy scores.
        """
        return self._policy_model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probability that treatment is beneficial.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Probability of treatment being beneficial (sigmoid of score).
        """
        scores = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
