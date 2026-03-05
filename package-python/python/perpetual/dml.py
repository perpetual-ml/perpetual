r"""Double / Debiased Machine Learning (DML) estimator.

Implements the Chernozhukov et al. (2018) partial-linear model for
heterogeneous treatment effect estimation using cross-fitting and
Perpetual's self-generalizing gradient boosting natively in Rust.

Partial-linear model
--------------------

.. math::

    Y = \theta(X) \cdot W + g(X) + \epsilon

where :math:`\theta(X)` is the heterogeneous treatment effect.

The DML approach orthogonalizes both the treatment and outcome:

.. math::

    \tilde{Y} = Y - \hat{g}(X), \quad \tilde{W} = W - \hat{m}(X)

and the effect model minimizes the R-Learner / DML objective:

.. math::

    L = \bigl(\tilde{Y} - \theta(X) \cdot \tilde{W}\bigr)^2
"""

from typing import Dict, Optional

import numpy as np
from typing_extensions import Self

from perpetual.meta_learners import _prepare_params
from perpetual.perpetual import DMLEstimator as RustDMLEstimator


class DMLEstimator:
    r"""Double Machine Learning (DML) estimator for heterogeneous treatment effects.

    Uses three gradient boosting stages with K-fold cross-fitting to learn
    :math:`\theta(X)` — the Conditional Average Treatment Effect (CATE).
    Computation is fully native in Rust to eliminate crossover overhead.

    Stages
    ------
    1. **Outcome nuisance** — fit :math:`g(X) \approx E[Y|X]` via cross-fitting.
    2. **Treatment nuisance** — fit :math:`m(X) \approx E[W|X]` via cross-fitting.
    3. **Effect model** — regress :math:`\tilde{Y}` on :math:`X` using a
       custom DML objective weighted by :math:`\tilde{W}`.

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
    >>> inference = dml.ate_inference()  # doctest: +SKIP

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
        self._params = _prepare_params(budget, kwargs)
        self._params["n_folds"] = n_folds
        self._params["clip"] = clip
        self.learner = RustDMLEstimator(**self._params)
        self._n_features: Optional[int] = None

    def fit(self, X, w, y) -> Self:
        """Fit the DML estimator natively with cross-fitting.

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
        x_arr = np.asarray(X, dtype=float, order="F")
        self._n_features = x_arr.shape[1]
        self.learner.fit(
            x_arr.ravel(),
            x_arr.shape[0],
            x_arr.shape[1],
            np.asarray(w, dtype=float),
            np.asarray(y, dtype=float),
        )
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance of the final effect model."""
        importances_map = self.learner.calculate_feature_importance("Gain", True)
        if self._n_features is None:
            return np.zeros(0)
        return np.array([importances_map.get(i, 0.0) for i in range(self._n_features)])

    def predict(self, X) -> np.ndarray:
        """Predict CATE (heterogeneous treatment effect).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated :math:`\theta(X)` for each sample.
        """
        x_arr = np.asarray(X, dtype=float, order="F")
        return self.learner.predict(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1])

    def ate_inference(self) -> Dict[str, float]:
        """Compute the Average Treatment Effect (ATE) and statistical inference.

        Returns
        -------
        dict
            Contains 'ate', 'std_err', 'ci_lower', and 'ci_upper'.
        """
        return {
            "ate": self.learner.ate,
            "std_err": self.learner.ate_se,
            "ci_lower": self.learner.ate_ci_lower,
            "ci_upper": self.learner.ate_ci_upper,
        }
