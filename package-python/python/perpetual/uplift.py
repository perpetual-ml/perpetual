"""Uplift modelling via the R-Learner approach.

Estimates the Conditional Average Treatment Effect (CATE) using a three-stage
gradient boosting pipeline: outcome model, propensity model, and effect model.
"""

from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from typing_extensions import Self

from perpetual.perpetual import UpliftBooster as CrateUpliftBooster
from perpetual.utils import convert_input_array, convert_input_frame


class UpliftBooster:
    """R-Learner uplift model for estimating heterogeneous treatment effects.

    Learns the Conditional Average Treatment Effect (CATE)
    ``tau(x) = E[Y | X, W=1] - E[Y | X, W=0]`` using three sequentially
    fitted gradient boosting models: an outcome model, a propensity model,
    and an effect model.
    """

    def __init__(
        self,
        outcome_budget: float = 0.5,
        propensity_budget: float = 0.5,
        effect_budget: float = 0.5,
        num_threads: Optional[int] = None,
        monotone_constraints: Union[Dict[Any, int], None] = None,
        force_children_to_bound_parent: bool = False,
        missing: float = np.nan,
        allow_missing_splits: bool = True,
        create_missing_branch: bool = False,
        terminate_missing_features: Optional[Iterable[Any]] = None,
        missing_node_treatment: str = "None",
        log_iterations: int = 0,
        quantile: Optional[float] = None,
        reset: Optional[bool] = None,
        categorical_features: Union[Iterable[int], Iterable[str], str, None] = "auto",
        timeout: Optional[float] = None,
        iteration_limit: Optional[int] = None,
        memory_limit: Optional[float] = None,
        stopping_rounds: Optional[int] = None,
        max_bin: int = 256,
        max_cat: int = 1000,
        interaction_constraints: Optional[list[list[int]]] = None,
    ):
        """
        Uplift Boosting Machine (R-Learner).

        Estimates the Conditional Average Treatment Effect (CATE): tau(x) = E[Y | X, W=1] - E[Y | X, W=0].

        Parameters
        ----------
        outcome_budget : float, default=0.5
            Fitting budget for the outcome model ``mu(x)``. Higher values allow
            more boosting rounds.
        propensity_budget : float, default=0.5
            Fitting budget for the propensity model ``p(x)``. Higher values allow
            more boosting rounds.
        effect_budget : float, default=0.5
            Fitting budget for the effect model ``tau(x)``. Higher values allow
            more boosting rounds.
        num_threads : int, optional
            Number of threads to use during training and prediction.
        monotone_constraints : dict, optional
            Constraints mapping feature indices/names to -1, 1, or 0.
        force_children_to_bound_parent : bool, default=False
            Whether to restrict children nodes to be within the parent's range.
        missing : float, default=np.nan
            Value to consider as missing data.
        allow_missing_splits : bool, default=True
            Whether to allow splits that separate missing from non-missing values.
        create_missing_branch : bool, default=False
            Whether to create a separate branch for missing values (ternary trees).
        terminate_missing_features : iterable, optional
            Features for which missing branches are always terminated when
            ``create_missing_branch`` is True.
        missing_node_treatment : str, default="None"
            How to handle weights for missing nodes. Options: ``"None"``,
            ``"AssignToParent"``, ``"AverageLeafWeight"``, ``"AverageNodeWeight"``.
        log_iterations : int, default=0
            Logging frequency (every N iterations). 0 disables logging.
        quantile : float, optional
            Target quantile when using ``"QuantileLoss"``.
        reset : bool, optional
            Whether to reset the model or continue training on subsequent fits.
        categorical_features : iterable or str, default="auto"
            Feature indices or names to treat as categorical.
        timeout : float, optional
            Time limit for fitting in seconds.
        iteration_limit : int, optional
            Maximum number of boosting iterations.
        memory_limit : float, optional
            Memory limit for training in GB.
        stopping_rounds : int, optional
            Number of rounds without improvement before stopping.
        max_bin : int, default=256
            Maximum number of bins for feature discretization.
        max_cat : int, default=1000
            Maximum unique categories before a feature is treated as numerical.
        interaction_constraints : list of list of int, optional
            Groups of feature indices allowed to interact.
        """
        self.outcome_budget = outcome_budget
        self.propensity_budget = propensity_budget
        self.effect_budget = effect_budget
        self.num_threads = num_threads
        self.monotone_constraints = monotone_constraints
        self.force_children_to_bound_parent = force_children_to_bound_parent
        self.missing = missing
        self.allow_missing_splits = allow_missing_splits
        self.create_missing_branch = create_missing_branch
        self.terminate_missing_features = terminate_missing_features
        self.missing_node_treatment = missing_node_treatment
        self.log_iterations = log_iterations
        self.quantile = quantile
        self.reset = reset
        self.categorical_features = categorical_features
        self.timeout = timeout
        self.iteration_limit = iteration_limit
        self.memory_limit = memory_limit
        self.stopping_rounds = stopping_rounds
        self.max_bin = max_bin
        self.max_cat = max_cat
        self.interaction_constraints = interaction_constraints

        self.booster = CrateUpliftBooster(
            outcome_budget=outcome_budget,
            propensity_budget=propensity_budget,
            effect_budget=effect_budget,
            max_bin=max_bin,
            num_threads=num_threads,
            monotone_constraints={},
            interaction_constraints=interaction_constraints,
            force_children_to_bound_parent=force_children_to_bound_parent,
            missing=missing,
            allow_missing_splits=allow_missing_splits,
            create_missing_branch=create_missing_branch,
            terminate_missing_features=set(),
            missing_node_treatment=missing_node_treatment,
            log_iterations=log_iterations,
            quantile=quantile,
            reset=reset,
            categorical_features=set(),
            timeout=timeout,
            iteration_limit=iteration_limit,
            memory_limit=memory_limit,
            stopping_rounds=stopping_rounds,
        )

    def fit(self, X, w, y) -> Self:
        """
        Fit the Uplift model.

        Parameters
        ----------
        X : array-like
            Covariates.
        w : array-like
            Treatment indicator (0 or 1).
        y : array-like
            Outcome variable.
        """
        # Feature conversion
        features_, flat_data, rows, cols, categorical_features_, cat_mapping = (
            convert_input_frame(X, "auto", 1000)
        )
        self.feature_names_in_ = features_

        # Treatment conversion
        w_arr = np.array(w)
        if not np.all(np.isin(w_arr, [0, 1])):
            raise ValueError("Treatment indicator 'w' must be binary (0 or 1).")
        w_, _ = convert_input_array(w_arr, "LogLoss")  # Treat as binary

        # Outcome conversion
        y_, _ = convert_input_array(y, "SquaredLoss")  # Treat as regression by default

        self.booster.fit(flat_data, rows, cols, w_, y_)
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict CATE.

        Parameters
        ----------
        X : array-like
            Covariates.

        Returns
        -------
        cate : ndarray
            Predicted Conditional Average Treatment Effect.
        """
        features_, flat_data, rows, cols, _, _ = convert_input_frame(X, "auto", 1000)

        preds = self.booster.predict(flat_data, rows, cols)
        return preds

    def to_json(self) -> str:
        """Serialize model to JSON string."""
        return self.booster.json_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "UpliftBooster":
        """Deserialize model from JSON string."""
        obj = cls.__new__(cls)
        obj.booster = CrateUpliftBooster.from_json(json_str)
        obj.outcome_budget = None
        obj.propensity_budget = None
        obj.effect_budget = None
        return obj
