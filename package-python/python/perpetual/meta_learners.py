"""Meta-learner strategies for heterogeneous treatment effect estimation.

Provides S-Learner, T-Learner, X-Learner, and DR-Learner wrappers around
optimized Rust implementations for estimating the Conditional Average Treatment
Effect (CATE) from observational data.

All learners follow a consistent API:
    - ``fit(X, w, y)`` - fit on covariates, binary treatment, and outcome.
    - ``predict(X)`` - return estimated CATE for each row.
    - ``feature_importances_`` - feature importance from the final effect
      model (available after ``fit``).
"""

from typing import Any, Dict, Optional

import numpy as np
from typing_extensions import Self

from perpetual.perpetual import (
    DRLearner as RustDRLearner,
)
from perpetual.perpetual import (
    SLearner as RustSLearner,
)
from perpetual.perpetual import (
    TLearner as RustTLearner,
)
from perpetual.perpetual import (
    XLearner as RustXLearner,
)


def _validate_binary_treatment(w):
    """Validate that *w* is a binary {0, 1} array."""
    w_arr = np.asarray(w)
    if not np.all(np.isin(w_arr, [0, 1])):
        raise ValueError("Treatment 'w' must be binary (0 or 1).")
    return w_arr.astype(np.float64)


def _prepare_params(budget: float, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare parameters for the Rust backend with defaults matching PerpetualBooster."""
    params = {
        "budget": budget,
        "max_bin": 255,
        "num_threads": None,
        "monotone_constraints": {},
        "interaction_constraints": None,
        "force_children_to_bound_parent": False,
        "missing": np.nan,
        "allow_missing_splits": True,
        "create_missing_branch": False,
        "terminate_missing_features": set(),
        "missing_node_treatment": "AssignToParent",
        "log_iterations": 0,
        "seed": 42,
        "quantile": None,
        "reset": None,
        "categorical_features": None,
        "timeout": None,
        "iteration_limit": None,
        "memory_limit": None,
        "stopping_rounds": None,
    }
    params.update(kwargs)
    return params


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
            Fitting budget forwarded to the Rust backend.
        **kwargs
            Additional keyword arguments forwarded to `PerpetualBooster`.
        """
        self.budget = budget
        self._params = _prepare_params(budget, kwargs)
        self.learner = RustSLearner(**self._params)
        self._n_features: Optional[int] = None

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
        x_arr = np.asarray(X, dtype=float, order="F")
        y_arr = np.asarray(y, dtype=float)

        self._n_features = x_arr.shape[1]
        self.learner.fit(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1], w_arr, y_arr)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance of the single model (excluding treatment feature)."""
        importances_map = self.learner.calculate_feature_importance("Gain", True)
        if self._n_features is None:
            return np.zeros(0)
        # Return only the FIRST n_features (the covariates).
        # We assume treatment was appended at index n_features.
        return np.array([importances_map.get(i, 0.0) for i in range(self._n_features)])

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
        x_arr = np.asarray(X, dtype=float, order="F")
        return self.learner.predict(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1])


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
            Fitting budget forwarded to the Rust backend.
        **kwargs
            Additional keyword arguments forwarded to `PerpetualBooster`.
        """
        self.budget = budget
        self._params = _prepare_params(budget, kwargs)
        self.learner = RustTLearner(**self._params)
        self._n_features: Optional[int] = None

    def fit(self, X, w, y) -> Self:
        x_arr = np.asarray(X, dtype=float, order="F")
        self._n_features = x_arr.shape[1]
        self.learner.fit(
            x_arr.ravel(),
            x_arr.shape[0],
            x_arr.shape[1],
            _validate_binary_treatment(w),
            np.asarray(y, dtype=float),
        )
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Aggregated feature importance from mu0 and mu1."""
        importances_map = self.learner.calculate_feature_importance("Gain", True)
        n = self._n_features if self._n_features is not None else 0
        return np.array([importances_map.get(i, 0.0) for i in range(n)])

    def predict(self, X) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float, order="F")
        return self.learner.predict(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1])


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------


class XLearner:
    """X-Learner for HTE estimation (typically better for imbalanced treatment groups)."""

    def __init__(
        self,
        budget: float = 0.5,
        propensity_budget: Optional[float] = None,
        **kwargs,
    ):
        self.budget = budget
        self._params = _prepare_params(budget, kwargs)
        self._params["propensity_budget"] = propensity_budget
        self.learner = RustXLearner(**self._params)
        self._n_features: Optional[int] = None

    def fit(self, X, w, y) -> Self:
        x_arr = np.asarray(X, dtype=float, order="F")
        self._n_features = x_arr.shape[1]
        self.learner.fit(
            x_arr.ravel(),
            x_arr.shape[0],
            x_arr.shape[1],
            _validate_binary_treatment(w),
            np.asarray(y, dtype=float),
        )
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Aggregated feature importance from the second-stage effect models (tau0, tau1)."""
        importances_map = self.learner.calculate_feature_importance("Gain", True)
        n = self._n_features if self._n_features is not None else 0
        return np.array([importances_map.get(i, 0.0) for i in range(n)])

    def predict(self, X) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float, order="F")
        return self.learner.predict(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1])


# ---------------------------------------------------------------------------
# DR-Learner  (Doubly Robust / AIPW)
# ---------------------------------------------------------------------------


class DRLearner:
    """Doubly Robust (DR) Learner for heterogeneous treatment effect estimation."""

    def __init__(
        self,
        budget: float = 0.5,
        propensity_budget: Optional[float] = None,
        **kwargs,
    ):
        self.budget = budget
        self._params = _prepare_params(budget, kwargs)
        self._params["propensity_budget"] = propensity_budget
        self.learner = RustDRLearner(**self._params)
        self._n_features: Optional[int] = None

    def fit(self, X, w, y) -> Self:
        x_arr = np.asarray(X, dtype=float, order="F")
        self._n_features = x_arr.shape[1]
        self.learner.fit(
            x_arr.ravel(),
            x_arr.shape[0],
            x_arr.shape[1],
            _validate_binary_treatment(w),
            np.asarray(y, dtype=float),
        )
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance from the outcome model fitted on pseudo-outcomes."""
        importances_map = self.learner.calculate_feature_importance("Gain", True)
        n = self._n_features if self._n_features is not None else 0
        return np.array([importances_map.get(i, 0.0) for i in range(n)])

    def predict(self, X) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float, order="F")
        return self.learner.predict(x_arr.ravel(), x_arr.shape[0], x_arr.shape[1])
