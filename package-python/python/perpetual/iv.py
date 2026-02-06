from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from typing_extensions import Self

from perpetual.perpetual import IVBooster as CrateIVBooster
from perpetual.utils import convert_input_array, convert_input_frame


class BraidedBooster:
    def __init__(
        self,
        treatment_objective: str = "SquaredLoss",
        outcome_objective: str = "SquaredLoss",
        stage1_budget: float = 0.5,
        stage2_budget: float = 0.5,
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
        Boosted Instrumental Variable (BoostIV) Estimator.

        Implements a 2-Stage Least Squares (2SLS) approach using Gradient Boosting.

        Parameters
        ----------
        treatment_objective : str
            Objective for Stage 1 (Treatment Model). e.g., "SquaredLoss" or "LogLoss".
        outcome_objective : str
            Objective for Stage 2 (Outcome Model). e.g., "SquaredLoss".
        stage1_budget : float
            Budget for Stage 1.
        stage2_budget : float
            Budget for Stage 2.
        num_threads : int, optional
            Number of threads.
        monotone_constraints : dict, optional
            Monotone constraints.
        force_children_to_bound_parent : bool, default=False
            Force children to be bounded by parent.
        missing : float, default=np.nan
            Missing value.
        allow_missing_splits : bool, default=True
            Allow missing splits.
        create_missing_branch : bool, default=False
            Create missing branch.
        terminate_missing_features : iterable, optional
            Terminate missing features.
        missing_node_treatment : str, default="None"
            Missing node treatment.
        log_iterations : int, default=0
            Log iterations.
        quantile : float, optional
            Quantile for QuantileLoss.
        reset : bool, optional
            Reset model on fit.
        categorical_features : iterable or str, default="auto"
            Categorical features.
        timeout : float, optional
            Timeout in seconds.
        iteration_limit : int, optional
            Iteration limit.
        memory_limit : float, optional
            Memory limit in GB.
        stopping_rounds : int, optional
            Stopping rounds.
        max_bin : int, default=256
            Maximum number of bins.
        max_cat : int, default=1000
            Maximum unique categories.
        interaction_constraints : list of list of int, optional
            Interaction constraints.
        """
        self.treatment_objective = treatment_objective
        self.outcome_objective = outcome_objective
        self.stage1_budget = stage1_budget
        self.stage2_budget = stage2_budget
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

        # Placeholder for standardized constraints and categorical features.
        # These are usually calculated in fit(), but we need to pass something to __init__
        # for CrateIVBooster if we follow the PerpetualBooster pattern of recreating it.
        # However, IVBooster's fit currently uses the same instance.

        self.booster = CrateIVBooster(
            treatment_objective=treatment_objective,
            outcome_objective=outcome_objective,
            stage1_budget=stage1_budget,
            stage2_budget=stage2_budget,
            max_bin=max_bin,
            num_threads=num_threads,
            monotone_constraints={},  # Passed during fit in PerpetualBooster
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

    def fit(self, X, Z, y, w) -> Self:
        """
        Fit the IV model.

        Parameters
        ----------
        X : array-like
            Covariates (Controls).
        Z : array-like
            Instruments.
        y : array-like
            Outcome variable.
        w : array-like
            Treatment received.
        """
        # Feature conversion
        features_x, flat_x, rows_x, cols_x, _, _ = convert_input_frame(X, "auto", 1000)
        self.feature_names_in_ = features_x

        # Z conversion (features)
        _, flat_z, rows_z, cols_z, _, _ = convert_input_frame(Z, "auto", 1000)

        if rows_x != rows_z:
            raise ValueError("X and Z must have the same number of rows.")

        # Treatment conversion
        w_, _ = convert_input_array(w, self.treatment_objective)

        # Outcome conversion
        y_, _ = convert_input_array(y, self.outcome_objective)

        self.booster.fit(flat_x, rows_x, cols_x, flat_z, rows_z, cols_z, y_, w_)
        return self

    def predict(self, X, w_counterfactual) -> np.ndarray:
        """
        Predict Outcome given X and a counterfactual W.

        Parameters
        ----------
        X : array-like
            Covariates.
        w_counterfactual : array-like
            Treatment value to simulate.

        Returns
        -------
        preds : ndarray
            Predicted Outcome.
        """
        _, flat_x, rows_x, cols_x, _, _ = convert_input_frame(X, "auto", 1000)

        w_cf_, _ = convert_input_array(
            w_counterfactual, self.treatment_objective
        )  # Use same encoding logic as W

        preds = self.booster.predict(flat_x, rows_x, cols_x, w_cf_)
        return preds

    def to_json(self) -> str:
        """Serialize model to JSON string."""
        return self.booster.json_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "BraidedBooster":
        """Deserialize model from JSON string."""
        obj = cls.__new__(cls)
        obj.booster = CrateIVBooster.from_json(json_str)
        params = obj.booster.get_params()
        obj.treatment_objective = params["treatment_objective"]
        obj.outcome_objective = params["outcome_objective"]
        obj.stage1_budget = params["stage1_budget"]
        obj.stage2_budget = params["stage2_budget"]
        return obj
