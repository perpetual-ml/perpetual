"""Fairness-aware gradient boosting classifier.

Provides ``FairClassifier``, a wrapper around ``PerpetualBooster`` that adds
an in-processing fairness regularization penalty to the log-loss gradient.
Two fairness criteria are supported:

* **Demographic Parity** — penalizes correlation between predictions and a
  sensitive attribute :math:`S`.
* **Equalized Odds** — penalizes correlation between predictions and
  :math:`S` *conditionally* within each class of the true label :math:`Y`.

The objective is:

.. math::

    L = \\text{LogLoss} + \\lambda \\cdot \\text{Penalty}(\\hat{Y}, S)
"""

from typing import Optional

import numpy as np
from typing_extensions import Self

from perpetual.booster import PerpetualBooster

# ---------------------------------------------------------------------------
# Custom objective helpers  (mirror Rust FairnessObjective)
# ---------------------------------------------------------------------------

_VALID_FAIRNESS_TYPES = ("demographic_parity", "equalized_odds")


def _fair_loss(sensitive_attr):
    """Return a loss callable (standard log-loss; penalty is gradient-only)."""

    def loss(y, yhat, sample_weight, group):
        y = np.asarray(y)
        yhat = np.asarray(yhat)
        p = 1.0 / (1.0 + np.exp(-yhat))
        score = -(
            y * np.log(np.maximum(p, 1e-15))
            + (1.0 - y) * np.log(np.maximum(1.0 - p, 1e-15))
        )
        return score.tolist()

    return loss


def _fair_gradient_dp(sensitive_attr, lam):
    """Gradient for Demographic Parity fairness."""
    s = np.asarray(sensitive_attr)

    def gradient(y, yhat, sample_weight, group):
        y = np.asarray(y)
        yhat = np.asarray(yhat)
        p = 1.0 / (1.0 + np.exp(-yhat))
        dp = p * (1.0 - p)

        # Group means
        mask1 = s == 1
        mask0 = ~mask1
        n1 = float(mask1.sum()) or 1.0
        n0 = float(mask0.sum()) or 1.0
        mean1 = p[mask1].sum() / n1
        mean0 = p[mask0].sum() / n0
        diff = mean1 - mean0

        grad = p - y  # standard log-loss gradient

        # Fairness penalty gradient
        fair_grad = np.where(
            mask1,
            2.0 * lam * diff * (1.0 / n1) * dp,
            2.0 * lam * diff * (-1.0 / n0) * dp,
        )
        grad = grad + fair_grad
        hess = dp

        return grad.tolist(), hess.tolist()

    return gradient


def _fair_gradient_eo(sensitive_attr, lam):
    """Gradient for Equalized Odds fairness."""
    s = np.asarray(sensitive_attr)

    def gradient(y, yhat, sample_weight, group):
        y = np.asarray(y)
        yhat = np.asarray(yhat)
        p = 1.0 / (1.0 + np.exp(-yhat))
        dp = p * (1.0 - p)
        label = (y >= 0.5).astype(int)

        grad = p - y  # standard log-loss gradient
        hess = dp.copy()

        for lbl in (0, 1):
            lbl_mask = label == lbl
            s1_lbl = (s == 1) & lbl_mask
            s0_lbl = (s == 0) & lbl_mask
            n1 = float(s1_lbl.sum()) or 1.0
            n0 = float(s0_lbl.sum()) or 1.0
            mean1 = p[s1_lbl].sum() / n1 if s1_lbl.any() else 0.0
            mean0 = p[s0_lbl].sum() / n0 if s0_lbl.any() else 0.0
            diff = mean1 - mean0

            fair_grad = np.where(
                s1_lbl,
                2.0 * lam * diff * (1.0 / n1) * dp,
                np.where(
                    s0_lbl,
                    2.0 * lam * diff * (-1.0 / n0) * dp,
                    0.0,
                ),
            )
            grad = grad + fair_grad

        return grad.tolist(), hess.tolist()

    return gradient


def _fair_initial_value():
    """Return an initial-value callable."""

    def initial_value(y, sample_weight, group):
        return 0.0

    return initial_value


# ---------------------------------------------------------------------------
# FairClassifier
# ---------------------------------------------------------------------------


class FairClassifier:
    """Fairness-aware gradient boosting classifier.

    Wraps a ``PerpetualBooster`` with an in-processing fairness penalty that
    regularizes the log-loss gradient to reduce dependence of predictions on
    a sensitive attribute.

    Parameters
    ----------
    sensitive_feature : int
        Column index of the sensitive attribute in *X*.  The column must be
        binary (0 or 1).
    fairness_type : str, default="demographic_parity"
        Fairness criterion.  One of:

        - ``"demographic_parity"`` — penalize overall disparity.
        - ``"equalized_odds"`` — penalize disparity within each label class.
    lam : float, default=1.0
        Strength of the fairness penalty (:math:`\\lambda`).
    budget : float, default=0.5
        Fitting budget forwarded to ``PerpetualBooster``.
    **kwargs
        Additional keyword arguments forwarded to ``PerpetualBooster``.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from the fitted model.

    Examples
    --------
    >>> from perpetual.fairness import FairClassifier
    >>> import numpy as np
    >>> X = np.column_stack([np.random.randn(200, 3),
    ...                      np.random.binomial(1, 0.5, 200)])
    >>> y = (X[:, 0] > 0).astype(float)
    >>> clf = FairClassifier(sensitive_feature=3, lam=2.0)
    >>> clf.fit(X, y)  # doctest: +SKIP
    >>> probs = clf.predict_proba(X)  # doctest: +SKIP

    Notes
    -----
    The fairness penalty is applied only through the gradient; the reported
    loss is standard log-loss.  This mirrors the Rust ``FairnessObjective``
    implementation.
    """

    def __init__(
        self,
        sensitive_feature: int,
        fairness_type: str = "demographic_parity",
        lam: float = 1.0,
        budget: float = 0.5,
        **kwargs,
    ):
        if fairness_type not in _VALID_FAIRNESS_TYPES:
            raise ValueError(
                f"fairness_type must be one of {_VALID_FAIRNESS_TYPES}, "
                f"got {fairness_type!r}."
            )
        self.sensitive_feature = sensitive_feature
        self.fairness_type = fairness_type
        self.lam = lam
        self.budget = budget
        self._kwargs = kwargs
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, y) -> Self:
        """Fit the fair classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.  The column at index ``sensitive_feature`` must
            contain binary (0/1) values.
        y : array-like of shape (n_samples,)
            Binary target variable (0 or 1).

        Returns
        -------
        self
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        sensitive_attr = X_arr[:, self.sensitive_feature].astype(int)

        if self.fairness_type == "demographic_parity":
            grad_fn = _fair_gradient_dp(sensitive_attr, self.lam)
        else:
            grad_fn = _fair_gradient_eo(sensitive_attr, self.lam)

        objective = (
            _fair_loss(sensitive_attr),
            grad_fn,
            _fair_initial_value(),
        )

        self._model = PerpetualBooster(
            budget=self.budget, objective=objective, **self._kwargs
        )
        self._model.fit(X_arr, y_arr)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        scores = self._model.predict(X)
        return (scores > 0).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Predicted probabilities for class 0 and class 1.
        """
        scores = self._model.predict(X)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p1, p1])

    def decision_function(self, X) -> np.ndarray:
        """Return raw log-odds scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Raw boosted scores (log-odds).
        """
        return self._model.predict(X)
