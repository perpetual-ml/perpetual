import warnings
from types import FunctionType
from typing import Any, Dict, Optional, Tuple, Union

from perpetual.booster import PerpetualBooster
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from typing_extensions import Self


class PerpetualClassifier(PerpetualBooster, ClassifierMixin):
    """
    A scikit-learn compatible classifier based on PerpetualBooster.
    Uses 'LogLoss' as the default objective.
    """

    # Expose the objective explicitly in the __init__ signature to allow
    # scikit-learn to correctly discover and set it via set_params.
    def __init__(
        self,
        *,
        objective: Union[
            str, Tuple[FunctionType, FunctionType, FunctionType]
        ] = "LogLoss",
        budget: float = 0.5,
        num_threads: Optional[int] = None,
        monotone_constraints: Union[Dict[Any, int], None] = None,
        # ... other parameters ...
        max_bin: int = 256,
        max_cat: int = 1000,
        # Capture all parameters in a way that BaseEstimator can handle
        **kwargs,
    ):
        # Ensure the objective is one of the valid classification objectives
        valid_objectives = {
            "LogLoss"
        }  # Assuming only LogLoss for classification for simplicity
        if isinstance(objective, str) and objective not in valid_objectives:
            # Custom objectives are allowed via the tuple form
            pass

        super().__init__(
            objective=objective,
            budget=budget,
            num_threads=num_threads,
            monotone_constraints=monotone_constraints,
            # ... pass all other parameters ...
            max_bin=max_bin,
            max_cat=max_cat,
            **kwargs,  # Catch-all for any other parameters passed by user or set_params
        )

    # fit, predict, predict_proba, and predict_log_proba are inherited
    # and properly adapted in PerpetualBooster.

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels."""
        preds = self.predict(X)
        return accuracy_score(y, preds, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None, **fit_params) -> Self:
        """A wrapper for the base fit method."""
        # Check if objective is appropriate for classification if it's a string
        if isinstance(self.objective, str) and self.objective not in ["LogLoss"]:
            warnings.warn(
                f"Objective '{self.objective}' is typically for regression/ranking but used in PerpetualClassifier. Consider 'LogLoss'."
            )

        # In classification, the labels (classes_) are set in the base fit.
        return super().fit(X, y, sample_weight=sample_weight, **fit_params)


class PerpetualRegressor(PerpetualBooster, RegressorMixin):
    """
    A scikit-learn compatible regressor based on PerpetualBooster.
    Uses 'SquaredLoss' as the default objective.
    """

    def __init__(
        self,
        *,
        objective: Union[
            str, Tuple[FunctionType, FunctionType, FunctionType]
        ] = "SquaredLoss",
        budget: float = 0.5,
        num_threads: Optional[int] = None,
        monotone_constraints: Union[Dict[Any, int], None] = None,
        # ... other parameters ...
        max_bin: int = 256,
        max_cat: int = 1000,
        **kwargs,
    ):
        # Enforce or warn about regression objectives
        valid_objectives = {
            "SquaredLoss",
            "QuantileLoss",
            "HuberLoss",
            "AdaptiveHuberLoss",
        }
        if isinstance(objective, str) and objective not in valid_objectives:
            pass  # Allow for custom string or tuple objective

        super().__init__(
            objective=objective,
            budget=budget,
            num_threads=num_threads,
            monotone_constraints=monotone_constraints,
            # ... pass all other parameters ...
            max_bin=max_bin,
            max_cat=max_cat,
            **kwargs,
        )

    def fit(self, X, y, sample_weight=None, **fit_params) -> Self:
        """A wrapper for the base fit method."""
        # For regression, we typically enforce len(self.classes_) == 0 after fit
        if isinstance(self.objective, str) and self.objective not in [
            "SquaredLoss",
            "QuantileLoss",
            "HuberLoss",
            "AdaptiveHuberLoss",
        ]:
            warnings.warn(
                f"Objective '{self.objective}' may not be suitable for PerpetualRegressor. Consider 'SquaredLoss' or a quantile/huber loss."
            )

        return super().fit(X, y, sample_weight=sample_weight, **fit_params)

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination ($R^2$) of the prediction."""
        preds = self.predict(X)
        return r2_score(y, preds, sample_weight=sample_weight)


class PerpetualRanker(
    PerpetualBooster, RegressorMixin
):  # Ranking models sometimes inherit from RegressorMixin for compatibility
    """
    A scikit-learn compatible ranker based on PerpetualBooster.
    Uses 'ListNetLoss' as the default objective.
    Requires the 'group' parameter to be passed to fit.
    """

    def __init__(
        self,
        *,
        objective: Union[
            str, Tuple[FunctionType, FunctionType, FunctionType]
        ] = "ListNetLoss",
        budget: float = 0.5,
        num_threads: Optional[int] = None,
        monotone_constraints: Union[Dict[Any, int], None] = None,
        # ... other parameters ...
        max_bin: int = 256,
        max_cat: int = 1000,
        **kwargs,
    ):
        if isinstance(objective, str) and objective not in {"ListNetLoss"}:
            warnings.warn(
                f"Objective '{objective}' may not be suitable for PerpetualRanker. Consider 'ListNetLoss'."
            )

        super().__init__(
            objective=objective,
            budget=budget,
            num_threads=num_threads,
            monotone_constraints=monotone_constraints,
            # ... pass all other parameters ...
            max_bin=max_bin,
            max_cat=max_cat,
            **kwargs,
        )

    def fit(self, X, y, group=None, sample_weight=None, **fit_params) -> Self:
        """
        Fit the ranker. Requires the 'group' parameter.

        Args:
            X: Training data.
            y: Target relevance scores.
            group: Group lengths to use for a ranking objective. (Required for ListNetLoss).
            sample_weight: Instance weights.
        """
        if (
            group is None
            and isinstance(self.objective, str)
            and self.objective == "ListNetLoss"
        ):
            raise ValueError(
                "The 'group' parameter must be provided when using the 'ListNetLoss' objective for ranking."
            )

        return super().fit(X, y, sample_weight=sample_weight, group=group, **fit_params)
