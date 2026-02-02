import warnings
from types import FunctionType
from typing import Any, Dict, Optional, Tuple, Union

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from typing_extensions import Self

from perpetual.booster import PerpetualBooster


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
        """
              Gradient Boosting Machine with Perpetual Learning.

              A self-generalizing gradient boosting machine that doesn't need hyperparameter
              optimization. It automatically finds the best configuration based on the provided budget.

              Parameters
              ----------
              objective : str or tuple, default="LogLoss"
                  Learning objective function to be used for optimization. Valid options are:

                  - "LogLoss": logistic loss for binary classification.
                  - custom objective: a tuple of (loss, gradient, initial_value) functions.
        Each function should have the following signature:

        - ``loss(y, pred, weight, group)`` : returns the loss value for each sample.
        - ``gradient(y, pred, weight, group)`` : returns a tuple of (gradient, hessian).
          If the hessian is constant (e.g., 1.0 for SquaredLoss), return ``None`` to improve performance.
        - ``initial_value(y, weight, group)`` : returns the initial value for the booster.

              budget : float, default=0.5
                  A positive number for fitting budget. Increasing this number will more likely result
                  in more boosting rounds and increased predictive power.
              num_threads : int, optional
                  Number of threads to be used during training and prediction.
              monotone_constraints : dict, optional
                  Constraints to enforce a specific relationship between features and target.
                  Keys are feature indices or names, values are -1, 1, or 0.
              force_children_to_bound_parent : bool, default=False
                  Whether to restrict children nodes to be within the parent's range.
              missing : float, default=np.nan
                  Value to consider as missing data.
              allow_missing_splits : bool, default=True
                  Whether to allow splits that separate missing from non-missing values.
              create_missing_branch : bool, default=False
                  Whether to create a separate branch for missing values (ternary trees).
              terminate_missing_features : iterable, optional
                  Features for which missing branches will always be terminated if
                  ``create_missing_branch`` is True.
              missing_node_treatment : str, default="None"
                  How to handle weights for missing nodes if ``create_missing_branch`` is True.
                  Options: "None", "AssignToParent", "AverageLeafWeight", "AverageNodeWeight".
              log_iterations : int, default=0
                  Logging frequency (every N iterations). 0 disables logging.
              feature_importance_method : str, default="Gain"
                  Method for calculating feature importance. Options: "Gain", "Weight", "Cover",
                  "TotalGain", "TotalCover".
              quantile : float, optional
                  Target quantile for quantile regression (objective="QuantileLoss").
              reset : bool, optional
                  Whether to reset the model or continue training on subsequent calls to fit.
              categorical_features : str or iterable, default="auto"
                  Feature indices or names to treat as categorical.
              timeout : float, optional
                  Time limit for fitting in seconds.
              iteration_limit : int, optional
                  Maximum number of boosting iterations.
              memory_limit : float, optional
                  Memory limit for training in GB.
              stopping_rounds : int, optional
                  Early stopping rounds.
              max_bin : int, default=256
                  Maximum number of bins for feature discretization.
              max_cat : int, default=1000
                  Maximum unique categories before a feature is treated as numerical.
              **kwargs
                  Arbitrary keyword arguments to be passed to the base class.
        """
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
        """
              Gradient Boosting Machine with Perpetual Learning.

              A self-generalizing gradient boosting machine that doesn't need hyperparameter
              optimization. It automatically finds the best configuration based on the provided budget.

              Parameters
              ----------
              objective : str or tuple, default="SquaredLoss"
                  Learning objective function to be used for optimization. Valid options are:

                  - "SquaredLoss": squared error for regression.
                  - "QuantileLoss": quantile error for quantile regression.
                  - "HuberLoss": Huber loss for robust regression.
                  - "AdaptiveHuberLoss": adaptive Huber loss for robust regression.
                  - custom objective: a tuple of (loss, gradient, initial_value) functions.
        Each function should have the following signature:

        - ``loss(y, pred, weight, group)`` : returns the loss value for each sample.
        - ``gradient(y, pred, weight, group)`` : returns a tuple of (gradient, hessian).
          If the hessian is constant (e.g., 1.0 for SquaredLoss), return ``None`` to improve performance.
        - ``initial_value(y, weight, group)`` : returns the initial value for the booster.

              budget : float, default=0.5
                  A positive number for fitting budget. Increasing this number will more likely result
                  in more boosting rounds and increased predictive power.
              num_threads : int, optional
                  Number of threads to be used during training and prediction.
              monotone_constraints : dict, optional
                  Constraints to enforce a specific relationship between features and target.
                  Keys are feature indices or names, values are -1, 1, or 0.
              force_children_to_bound_parent : bool, default=False
                  Whether to restrict children nodes to be within the parent's range.
              missing : float, default=np.nan
                  Value to consider as missing data.
              allow_missing_splits : bool, default=True
                  Whether to allow splits that separate missing from non-missing values.
              create_missing_branch : bool, default=False
                  Whether to create a separate branch for missing values (ternary trees).
              terminate_missing_features : iterable, optional
                  Features for which missing branches will always be terminated if
                  ``create_missing_branch`` is True.
              missing_node_treatment : str, default="None"
                  How to handle weights for missing nodes if ``create_missing_branch`` is True.
                  Options: "None", "AssignToParent", "AverageLeafWeight", "AverageNodeWeight".
              log_iterations : int, default=0
                  Logging frequency (every N iterations). 0 disables logging.
              feature_importance_method : str, default="Gain"
                  Method for calculating feature importance. Options: "Gain", "Weight", "Cover",
                  "TotalGain", "TotalCover".
              quantile : float, optional
                  Target quantile for quantile regression (objective="QuantileLoss").
              reset : bool, optional
                  Whether to reset the model or continue training on subsequent calls to fit.
              categorical_features : str or iterable, default="auto"
                  Feature indices or names to treat as categorical.
              timeout : float, optional
                  Time limit for fitting in seconds.
              iteration_limit : int, optional
                  Maximum number of boosting iterations.
              memory_limit : float, optional
                  Memory limit for training in GB.
              stopping_rounds : int, optional
                  Early stopping rounds.
              max_bin : int, default=256
                  Maximum number of bins for feature discretization.
              max_cat : int, default=1000
                  Maximum unique categories before a feature is treated as numerical.
              **kwargs
                  Arbitrary keyword arguments to be passed to the base class.
        """
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
        """
              Gradient Boosting Machine with Perpetual Learning.

              A self-generalizing gradient boosting machine that doesn't need hyperparameter
              optimization. It automatically finds the best configuration based on the provided budget.

              Parameters
              ----------
              objective : str or tuple, default="ListNetLoss"
                  Learning objective function to be used for optimization. Valid options are:

                  - "ListNetLoss": ListNet loss for ranking.
                  - custom objective: a tuple of (loss, gradient, initial_value) functions.
        Each function should have the following signature:

        - ``loss(y, pred, weight, group)`` : returns the loss value for each sample.
        - ``gradient(y, pred, weight, group)`` : returns a tuple of (gradient, hessian).
          If the hessian is constant (e.g., 1.0 for SquaredLoss), return ``None`` to improve performance.
        - ``initial_value(y, weight, group)`` : returns the initial value for the booster.

              budget : float, default=0.5
                  A positive number for fitting budget. Increasing this number will more likely result
                  in more boosting rounds and increased predictive power.
              num_threads : int, optional
                  Number of threads to be used during training and prediction.
              monotone_constraints : dict, optional
                  Constraints to enforce a specific relationship between features and target.
                  Keys are feature indices or names, values are -1, 1, or 0.
              force_children_to_bound_parent : bool, default=False
                  Whether to restrict children nodes to be within the parent's range.
              missing : float, default=np.nan
                  Value to consider as missing data.
              allow_missing_splits : bool, default=True
                  Whether to allow splits that separate missing from non-missing values.
              create_missing_branch : bool, default=False
                  Whether to create a separate branch for missing values (ternary trees).
              terminate_missing_features : iterable, optional
                  Features for which missing branches will always be terminated if
                  ``create_missing_branch`` is True.
              missing_node_treatment : str, default="None"
                  How to handle weights for missing nodes if ``create_missing_branch`` is True.
                  Options: "None", "AssignToParent", "AverageLeafWeight", "AverageNodeWeight".
              log_iterations : int, default=0
                  Logging frequency (every N iterations). 0 disables logging.
              feature_importance_method : str, default="Gain"
                  Method for calculating feature importance. Options: "Gain", "Weight", "Cover",
                  "TotalGain", "TotalCover".
              quantile : float, optional
                  Target quantile for quantile regression (objective="QuantileLoss").
              reset : bool, optional
                  Whether to reset the model or continue training on subsequent calls to fit.
              categorical_features : str or iterable, default="auto"
                  Feature indices or names to treat as categorical.
              timeout : float, optional
                  Time limit for fitting in seconds.
              iteration_limit : int, optional
                  Maximum number of boosting iterations.
              memory_limit : float, optional
                  Memory limit for training in GB.
              stopping_rounds : int, optional
                  Early stopping rounds.
              max_bin : int, default=256
                  Maximum number of bins for feature discretization.
              max_cat : int, default=1000
                  Maximum unique categories before a feature is treated as numerical.
              **kwargs
                  Arbitrary keyword arguments to be passed to the base class.
        """
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
