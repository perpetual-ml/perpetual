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
from perpetual.perpetual import PolicyObjective

# ---------------------------------------------------------------------------
# Custom objective helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PolicyLearner
# ---------------------------------------------------------------------------


class PolicyLearner:
    r"""Policy learner via Inverse Propensity Weighting.

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
        mu_hat_1: Optional[np.ndarray] = None,
        mu_hat_0: Optional[np.ndarray] = None,
    ) -> Self:
        r"""Fit the policy learner.

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
        mu_hat_1 : array-like of shape (n_samples,), optional
            Predicted outcome under treatment :math:`\hat{\mu}_1(X)`.
            Required when ``mode="aipw"``.
        mu_hat_0 : array-like of shape (n_samples,), optional
            Predicted outcome under control :math:`\hat{\mu}_0(X)`.
            Required when ``mode="aipw"``.

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

        # Baseline outcome models for AIPW
        m1 = None
        m0 = None
        if self.mode == "aipw":
            if mu_hat_1 is not None and mu_hat_0 is not None:
                m1 = np.asarray(mu_hat_1, dtype=float)
                m0 = np.asarray(mu_hat_0, dtype=float)
            else:
                # Fit separate outcome models for treatment and control
                m1_model = PerpetualBooster(
                    budget=self.budget, objective="SquaredLoss", **self._kwargs
                )
                m0_model = PerpetualBooster(
                    budget=self.budget, objective="SquaredLoss", **self._kwargs
                )
                mask1 = w_arr == 1
                mask0 = w_arr == 0
                if mask1.any():
                    m1_model.fit(X_arr[mask1], y_arr[mask1])
                if mask0.any():
                    m0_model.fit(X_arr[mask0], y_arr[mask0])

                m1 = m1_model.predict(X_arr)
                m0 = m0_model.predict(X_arr)

        # Policy model via Rust PolicyObjective
        objective = PolicyObjective(
            treatment=w_arr.astype("uint8"),
            propensity=p_hat,
            mode=self.mode,
            mu_hat_1=m1,
            mu_hat_0=m0,
        )

        self._policy_model = PerpetualBooster(
            budget=self.budget, objective=objective, **self._kwargs
        )
        # Pass observed outcome y_arr. The definition of Gamma is handled internally
        # by the RustPolicyObjective, which implements pseudo_outcome using y.
        self._policy_model.fit(X_arr, y_arr)
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
