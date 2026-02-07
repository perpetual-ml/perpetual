"""Double / Debiased Machine Learning (DML) estimator.

Implements the Chernozhukov et al. (2018) partial-linear model for
heterogeneous treatment effect estimation using cross-fitting and
Perpetual's self-generalizing gradient boosting.

Partial-linear model
--------------------

.. math::

    Y = \\theta(X) \\cdot W + g(X) + \\epsilon

where :math:`\\theta(X)` is the heterogeneous treatment effect.

The DML approach orthogonalizes both the treatment and outcome:

.. math::

    \\tilde{Y} = Y - \\hat{g}(X), \\quad \\tilde{W} = W - \\hat{m}(X)

and the effect model minimizes the R-Learner / DML objective:

.. math::

    L = \\bigl(\\tilde{Y} - \\theta(X) \\cdot \\tilde{W}\\bigr)^2
"""

from typing import Optional

import numpy as np
from typing_extensions import Self

from perpetual.booster import PerpetualBooster

# ---------------------------------------------------------------------------
# Custom objective helpers  (mirror Rust DMLObjective)
# ---------------------------------------------------------------------------

_HESSIAN_FLOOR = 1e-6


def _dml_loss(y_residual, w_residual):
    """Return a loss callable for the DML objective."""

    def loss(y, yhat, sample_weight, group):
        diff = y_residual - yhat * w_residual
        return (diff * diff).tolist()

    return loss


def _dml_gradient(y_residual, w_residual):
    """Return a gradient callable for the DML objective."""

    def gradient(y, yhat, sample_weight, group):
        # g = -w_res * y_res + theta * w_res^2
        grad = -w_residual * y_residual + yhat * w_residual * w_residual
        hess = np.maximum(w_residual * w_residual, _HESSIAN_FLOOR)
        return grad.tolist(), hess.tolist()

    return gradient


def _dml_initial_value(y_residual, w_residual):
    """Return an initial-value callable for the DML objective."""

    def initial_value(y, sample_weight, group):
        num = float(np.sum(y_residual * w_residual))
        den = float(np.sum(w_residual * w_residual))
        if abs(den) < _HESSIAN_FLOOR:
            return 0.0
        return num / den

    return initial_value


# ---------------------------------------------------------------------------
# DMLEstimator
# ---------------------------------------------------------------------------


class DMLEstimator:
    """Double Machine Learning (DML) estimator for heterogeneous treatment effects.

    Uses three gradient boosting stages with K-fold cross-fitting to learn
    :math:`\\theta(X)` — the Conditional Average Treatment Effect (CATE).

    Stages
    ------
    1. **Outcome nuisance** — fit :math:`g(X) \\approx E[Y|X]` via cross-fitting.
    2. **Treatment nuisance** — fit :math:`m(X) \\approx E[W|X]` via cross-fitting.
    3. **Effect model** — regress :math:`\\tilde{Y}` on :math:`X` using a
       custom DML objective weighted by :math:`\\tilde{W}`.

    Parameters
    ----------
    budget : float, default=0.5
        Fitting budget forwarded to each internal ``PerpetualBooster``.
    n_folds : int, default=2
        Number of cross-fitting folds (must be >= 2).
    clip : float, default=0.01
        Treatment residuals are clipped to ``[-1/clip, 1/clip]`` range to
        avoid extreme weights.
    **kwargs
        Additional keyword arguments forwarded to ``PerpetualBooster``.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from the final effect model.

    Examples
    --------
    >>> from perpetual.dml import DMLEstimator
    >>> import numpy as np
    >>> X = np.random.randn(500, 5)
    >>> w = np.random.binomial(1, 0.5, 500).astype(float)
    >>> y = X[:, 0] * w + np.random.randn(500) * 0.5
    >>> dml = DMLEstimator(budget=0.3, n_folds=2)
    >>> dml.fit(X, w, y)  # doctest: +SKIP
    >>> cate = dml.predict(X)  # doctest: +SKIP

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). *Double/debiased machine learning for
    treatment and structural parameters*. The Econometrics Journal, 21(1).
    """

    def __init__(
        self,
        budget: float = 0.5,
        n_folds: int = 2,
        clip: float = 0.01,
        **kwargs,
    ):
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2.")
        self.budget = budget
        self.n_folds = n_folds
        self.clip = clip
        self._kwargs = kwargs
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X, w, y) -> Self:
        """Fit the DML estimator with cross-fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        w : array-like of shape (n_samples,)
            Treatment variable (continuous or binary).
        y : array-like of shape (n_samples,)
            Outcome variable.

        Returns
        -------
        self
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        w_arr = np.asarray(w, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = X_arr.shape[0]

        # ---- Cross-fitted residuals ----
        y_residual = np.zeros(n)
        w_residual = np.zeros(n)

        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        for fold_idx in range(self.n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_folds) if j != fold_idx]
            )

            # Outcome nuisance g(X) = E[Y|X]
            g_model = PerpetualBooster(
                budget=self.budget, objective="SquaredLoss", **self._kwargs
            )
            g_model.fit(X_arr[train_idx], y_arr[train_idx])
            y_residual[test_idx] = y_arr[test_idx] - g_model.predict(X_arr[test_idx])

            # Treatment nuisance m(X) = E[W|X]
            m_model = PerpetualBooster(
                budget=self.budget, objective="SquaredLoss", **self._kwargs
            )
            m_model.fit(X_arr[train_idx], w_arr[train_idx])
            w_residual[test_idx] = w_arr[test_idx] - m_model.predict(X_arr[test_idx])

        # Clip treatment residuals for numerical stability.
        w_residual = np.clip(w_residual, -1.0 / self.clip, 1.0 / self.clip)

        # ---- Effect model with DML objective ----
        objective = (
            _dml_loss(y_residual, w_residual),
            _dml_gradient(y_residual, w_residual),
            _dml_initial_value(y_residual, w_residual),
        )

        self._effect_model = PerpetualBooster(
            budget=self.budget, objective=objective, **self._kwargs
        )
        # The "y" passed to the booster is unused by the custom objective;
        # pass y_residual as a placeholder.
        self._effect_model.fit(X_arr, y_residual)
        self.feature_importances_ = self._effect_model.feature_importances_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict CATE (heterogeneous treatment effect).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated :math:`\\theta(X)` for each sample.
        """
        return self._effect_model.predict(X)
