import inspect
import json
import warnings
from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import numpy as np
from typing_extensions import Self

from perpetual.data import Node
from perpetual.perpetual import (
    MultiOutputBooster as CrateMultiOutputBooster,  # type: ignore
)
from perpetual.perpetual import (
    PerpetualBooster as CratePerpetualBooster,  # type: ignore
)
from perpetual.serialize import BaseSerializer, ObjectSerializer
from perpetual.types import BoosterType, MultiOutputBoosterType
from perpetual.utils import (
    CONTRIBUTION_METHODS,
    convert_input_array,
    convert_input_frame,
    convert_input_frame_columnar,
    transform_input_frame,
    transform_input_frame_columnar,
    type_df,
)


class PerpetualBooster:
    # Define the metadata parameters
    # that are present on all instances of this class
    # this is useful for parameters that should be
    # attempted to be loaded in and set
    # as attributes on the booster after it is loaded.
    metadata_attributes: Dict[str, BaseSerializer] = {
        "feature_names_in_": ObjectSerializer(),
        "n_features_": ObjectSerializer(),
        "feature_importance_method": ObjectSerializer(),
        "cat_mapping": ObjectSerializer(),
        "classes_": ObjectSerializer(),
    }

    def __init__(
        self,
        *,
        objective: Union[
            str, Tuple[FunctionType, FunctionType, FunctionType]
        ] = "LogLoss",
        budget: float = 0.5,
        num_threads: Optional[int] = None,
        monotone_constraints: Union[Dict[Any, int], None] = None,
        force_children_to_bound_parent: bool = False,
        missing: float = np.nan,
        allow_missing_splits: bool = True,
        create_missing_branch: bool = False,
        terminate_missing_features: Optional[Iterable[Any]] = None,
        missing_node_treatment: str = "None",
        log_iterations: int = 0,
        feature_importance_method: str = "Gain",
        quantile: Optional[float] = None,
        reset: Optional[bool] = None,
        categorical_features: Union[Iterable[int], Iterable[str], str, None] = "auto",
        timeout: Optional[float] = None,
        iteration_limit: Optional[int] = None,
        memory_limit: Optional[float] = None,
        stopping_rounds: Optional[int] = None,
        max_bin: int = 256,
        max_cat: int = 1000,
    ):
        """
        Gradient Boosting Machine with Perpetual Learning.

        A self-generalizing gradient boosting machine that doesn't need hyperparameter optimization.
        It automatically finds the best configuration based on the provided budget.

        Parameters
        ----------
        objective : str or tuple, default="LogLoss"
            Learning objective function to be used for optimization. Valid options are:

            - "LogLoss": logistic loss for binary classification.
            - "SquaredLoss": squared error for regression.
            - "QuantileLoss": quantile error for quantile regression.
            - "HuberLoss": Huber loss for robust regression.
            - "AdaptiveHuberLoss": adaptive Huber loss for robust regression.
            - "ListNetLoss": ListNet loss for ranking.
            - custom objective: a tuple of (loss, gradient, initial_value) functions.
              Each function should have the following signature:

              - **loss(y, pred, weight, group)** : returns the loss value for each sample.
              - **gradient(y, pred, weight, group)** : returns a tuple of (gradient, hessian).
                If the hessian is constant (e.g., 1.0 for SquaredLoss), return ``None`` to improve performance.
              - **initial_value(y, weight, group)** : returns the initial value for the booster.

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

        Attributes
        ----------
        feature_names_in_ : list of str
            Names of features seen during :meth:`fit`.
        n_features_ : int
            Number of features seen during :meth:`fit`.
        classes_ : list
            Class labels for classification tasks.
        feature_importances_ : ndarray of shape (n_features,)
            Feature importances calculated via ``feature_importance_method``.

        See Also
        --------
        perpetual.sklearn.PerpetualClassifier : Scikit-learn compatible classifier.
        perpetual.sklearn.PerpetualRegressor : Scikit-learn compatible regressor.

        Examples
        --------
        Basic usage for binary classification:

        >>> from perpetual import PerpetualBooster
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, n_features=20)
        >>> model = PerpetualBooster(objective="LogLoss")
        >>> model.fit(X, y)
        >>> preds = model.predict(X[:5])

        Custom objective example:

        >>> def loss(y, pred, weight, group):
        ...     return (y - pred) ** 2
        >>> def gradient(y, pred, weight, group):
        ...     return (pred - y), None
        >>> def initial_value(y, weight, group):
        ...     return np.mean(y)
        >>> model = PerpetualBooster(objective=(loss, gradient, initial_value))
        >>> model.fit(X, y)
        """

        terminate_missing_features_ = (
            set() if terminate_missing_features is None else terminate_missing_features
        )
        monotone_constraints_ = (
            {} if monotone_constraints is None else monotone_constraints
        )

        if isinstance(objective, str):
            self.objective = objective
            self.loss = None
            self.grad = None
            self.init = None
        else:
            self.objective = None
            self.loss = objective[0]
            self.grad = objective[1]
            self.init = objective[2]
        self.budget = budget
        self.num_threads = num_threads
        self.monotone_constraints = monotone_constraints_
        self.force_children_to_bound_parent = force_children_to_bound_parent
        self.allow_missing_splits = allow_missing_splits
        self.missing = missing
        self.create_missing_branch = create_missing_branch
        self.terminate_missing_features = terminate_missing_features_
        self.missing_node_treatment = missing_node_treatment
        self.log_iterations = log_iterations
        self.feature_importance_method = feature_importance_method
        self.quantile = quantile
        self.reset = reset
        self.categorical_features = categorical_features
        self.timeout = timeout
        self.iteration_limit = iteration_limit
        self.memory_limit = memory_limit
        self.stopping_rounds = stopping_rounds
        self.max_bin = max_bin
        self.max_cat = max_cat

        booster = CratePerpetualBooster(
            objective=self.objective,
            budget=self.budget,
            max_bin=self.max_bin,
            num_threads=self.num_threads,
            monotone_constraints=dict(),
            force_children_to_bound_parent=self.force_children_to_bound_parent,
            missing=self.missing,
            allow_missing_splits=self.allow_missing_splits,
            create_missing_branch=self.create_missing_branch,
            terminate_missing_features=set(),
            missing_node_treatment=self.missing_node_treatment,
            log_iterations=self.log_iterations,
            quantile=self.quantile,
            reset=self.reset,
            categorical_features=set(),
            timeout=self.timeout,
            iteration_limit=self.iteration_limit,
            memory_limit=self.memory_limit,
            stopping_rounds=self.stopping_rounds,
            loss=self.loss,
            grad=self.grad,
            init=self.init,
        )
        self.booster = cast(BoosterType, booster)

    def fit(self, X, y, sample_weight=None, group=None) -> Self:
        """
        Fit the gradient booster on a provided dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be a Polars or Pandas DataFrame, or a 2D Numpy array.
            Polars DataFrames use a zero-copy columnar path for efficiency.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample. If None, all samples are weighted equally.
        group : array-like, optional
            Group labels for ranking objectives.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check if input is a Polars DataFrame for zero-copy columnar path
        is_polars = type_df(X) == "polars_df"

        if is_polars:
            # Use columnar path for Polars DataFrames (true zero-copy)
            (
                features_,
                columns,  # list of 1D arrays instead of flat_data
                masks,
                rows,
                cols,
                categorical_features_,
                cat_mapping,
            ) = convert_input_frame_columnar(X, self.categorical_features, self.max_cat)
        else:
            # Use existing flat path for pandas and numpy
            (
                features_,
                flat_data,
                rows,
                cols,
                categorical_features_,
                cat_mapping,
            ) = convert_input_frame(X, self.categorical_features, self.max_cat)

        self.n_features_ = cols
        self.cat_mapping = cat_mapping
        self.feature_names_in_ = features_

        y_, classes_ = convert_input_array(y, self.objective, is_target=True)
        self.classes_ = np.array(classes_).tolist()

        if sample_weight is None:
            sample_weight_ = None
        else:
            sample_weight_, _ = convert_input_array(sample_weight, self.objective)

        if group is None:
            group_ = None
        else:
            group_, _ = convert_input_array(group, self.objective, is_int=True)

        # Convert the monotone constraints into the form needed
        # by the rust code.
        crate_mc = self._standardize_monotonicity_map(X)
        crate_tmf = self._standardize_terminate_missing_features(X)

        if (len(classes_) <= 2) or (
            len(classes_) > 1 and self.objective == "SquaredLoss"
        ):
            booster = CratePerpetualBooster(
                objective=self.objective,
                budget=self.budget,
                max_bin=self.max_bin,
                num_threads=self.num_threads,
                monotone_constraints=crate_mc,
                force_children_to_bound_parent=self.force_children_to_bound_parent,
                missing=self.missing,
                allow_missing_splits=self.allow_missing_splits,
                create_missing_branch=self.create_missing_branch,
                terminate_missing_features=crate_tmf,
                missing_node_treatment=self.missing_node_treatment,
                log_iterations=self.log_iterations,
                quantile=self.quantile,
                reset=self.reset,
                categorical_features=categorical_features_,
                timeout=self.timeout,
                iteration_limit=self.iteration_limit,
                memory_limit=self.memory_limit,
                stopping_rounds=self.stopping_rounds,
                loss=self.loss,
                grad=self.grad,
                init=self.init,
            )
            self.booster = cast(BoosterType, booster)
        else:
            booster = CrateMultiOutputBooster(
                n_boosters=len(classes_),
                objective=self.objective,
                budget=self.budget,
                max_bin=self.max_bin,
                num_threads=self.num_threads,
                monotone_constraints=crate_mc,
                force_children_to_bound_parent=self.force_children_to_bound_parent,
                missing=self.missing,
                allow_missing_splits=self.allow_missing_splits,
                create_missing_branch=self.create_missing_branch,
                terminate_missing_features=crate_tmf,
                missing_node_treatment=self.missing_node_treatment,
                log_iterations=self.log_iterations,
                quantile=self.quantile,
                reset=self.reset,
                categorical_features=categorical_features_,
                timeout=self.timeout,
                iteration_limit=self.iteration_limit,
                memory_limit=self.memory_limit,
                stopping_rounds=self.stopping_rounds,
                loss=self.loss,
                grad=self.grad,
                init=self.init,
            )
            self.booster = cast(MultiOutputBoosterType, booster)

        self._set_metadata_attributes("n_features_", self.n_features_)
        self._set_metadata_attributes("cat_mapping", self.cat_mapping)
        self._set_metadata_attributes("feature_names_in_", self.feature_names_in_)
        self._set_metadata_attributes(
            "feature_importance_method", self.feature_importance_method
        )
        self._set_metadata_attributes("classes_", self.classes_)

        self.categorical_features = categorical_features_

        if is_polars:
            # Use columnar fit for Polars (zero-copy)
            self.booster.fit_columnar(
                columns=columns,
                masks=masks,
                rows=rows,
                y=y_,
                sample_weight=sample_weight_,  # type: ignore
                group=group_,
            )
        else:
            # Use standard fit for pandas/numpy
            self.booster.fit(
                flat_data=flat_data,
                rows=rows,
                cols=cols,
                y=y_,
                sample_weight=sample_weight_,  # type: ignore
                group=group_,
            )

        return self

    def prune(self, X, y, sample_weight=None, group=None) -> Self:
        """
        Prune the gradient booster on a provided dataset.

        This removes nodes that do not contribute to a reduction in loss on the provided
        validation set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Validation data.
        y : array-like of shape (n_samples,)
            Validation targets.
        sample_weight : array-like of shape (n_samples,), optional
            Weights for validation samples.
        group : array-like, optional
            Group labels for ranking objectives.

        Returns
        -------
        self : object
            Returns self.
        """

        _, flat_data, rows, cols = transform_input_frame(X, self.cat_mapping)

        y_, _ = convert_input_array(y, self.objective)

        if sample_weight is None:
            sample_weight_ = None
        else:
            sample_weight_, _ = convert_input_array(sample_weight, self.objective)

        if group is None:
            group_ = None
        else:
            group_, _ = convert_input_array(group, self.objective, is_int=True)

        self.booster.prune(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            sample_weight=sample_weight_,  # type: ignore
            group=group_,
        )

        return self

    def calibrate(
        self, X_train, y_train, X_cal, y_cal, alpha, sample_weight=None, group=None
    ) -> Self:
        """
        Calibrate the gradient booster for prediction intervals.

        Uses the provided training and calibration sets to compute scaling factors
        for intervals.

        Parameters
        ----------
        X_train : array-like
            Data used to train the base model.
        y_train : array-like
            Targets for training data.
        X_cal : array-like
            Independent calibration dataset.
        y_cal : array-like
            Targets for calibration data.
        alpha : float or array-like
            Significance level(s) for the intervals (1 - coverage).
        sample_weight : array-like, optional
            Sample weights.
        group : array-like, optional
            Group labels.

        Returns
        -------
        self : object
            Returns self.
        """

        is_polars = type_df(X_train) == "polars_df"
        if is_polars:
            features_train, cols_train, masks_train, rows_train, _ = (
                transform_input_frame_columnar(X_train, self.cat_mapping)
            )
            self._validate_features(features_train)
            features_cal, cols_cal, masks_cal, rows_cal, _ = (
                transform_input_frame_columnar(X_cal, self.cat_mapping)
            )
            # Use columnar calibration
            y_train_, _ = convert_input_array(y_train, self.objective)
            y_cal_, _ = convert_input_array(y_cal, self.objective)
            if sample_weight is None:
                sample_weight_ = None
            else:
                sample_weight_, _ = convert_input_array(sample_weight, self.objective)

            self.booster.calibrate_columnar(
                columns=cols_train,
                masks=masks_train,
                rows=rows_train,
                y=y_train_,
                columns_cal=cols_cal,
                masks_cal=masks_cal,
                rows_cal=rows_cal,
                y_cal=y_cal_,
                alpha=np.array(alpha),
                sample_weight=sample_weight_,  # type: ignore
                group=group,
            )
        else:
            _, flat_data_train, rows_train, cols_train = transform_input_frame(
                X_train, self.cat_mapping
            )

            y_train_, _ = convert_input_array(y_train, self.objective)

            _, flat_data_cal, rows_cal, cols_cal = transform_input_frame(
                X_cal, self.cat_mapping
            )

            y_cal_, _ = convert_input_array(y_cal, self.objective)

            if sample_weight is None:
                sample_weight_ = None
            else:
                sample_weight_, _ = convert_input_array(sample_weight, self.objective)

            self.booster.calibrate(
                flat_data=flat_data_train,
                rows=rows_train,
                cols=cols_train,
                y=y_train_,
                flat_data_cal=flat_data_cal,
                rows_cal=rows_cal,
                cols_cal=cols_cal,
                y_cal=y_cal_,
                alpha=np.array(alpha),
                sample_weight=sample_weight_,  # type: ignore
                group=group,
            )

        return self

    def _validate_features(self, features: List[str]):
        if len(features) > 0 and hasattr(self, "feature_names_in_"):
            if features[0] != "0" and self.feature_names_in_[0] != "0":
                if features != self.feature_names_in_:
                    raise ValueError(
                        f"Columns mismatch between data {features} passed, and data {self.feature_names_in_} used at fit."
                    )

    def predict_intervals(self, X, parallel: Union[bool, None] = None) -> dict:
        """
        Predict intervals with the fitted booster on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data for prediction.
        parallel : bool, optional
            Whether to run prediction in parallel. If None, uses class default.

        Returns
        -------
        intervals : dict
            A dictionary containing lower and upper bounds for the specified alpha levels.
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
            self._validate_features(features_)
            return self.booster.predict_intervals_columnar(
                columns=columns, masks=masks, rows=rows, parallel=parallel
            )

        features_, flat_data, rows, cols = transform_input_frame(X, self.cat_mapping)
        self._validate_features(features_)

        return self.booster.predict_intervals(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            parallel=parallel,
        )

    def predict(self, X, parallel: Union[bool, None] = None) -> np.ndarray:
        """
        Predict with the fitted booster on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        parallel : bool, optional
            Whether to run prediction in parallel.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            The predicted values (log-odds for classification, raw values for regression).
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
        else:
            features_, flat_data, rows, cols = transform_input_frame(
                X, self.cat_mapping
            )
        self._validate_features(features_)

        if len(self.classes_) == 0:
            if is_polars:
                return self.booster.predict_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            return self.booster.predict(
                flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
            )
        elif len(self.classes_) == 2:
            if is_polars:
                return np.rint(
                    self.booster.predict_proba_columnar(
                        columns=columns, masks=masks, rows=rows, parallel=parallel
                    )
                ).astype(int)
            return np.rint(
                self.booster.predict_proba(
                    flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
                )
            ).astype(int)
        else:
            if is_polars:
                preds = self.booster.predict_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            else:
                preds = self.booster.predict(
                    flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
                )
            preds_matrix = preds.reshape((-1, len(self.classes_)), order="F")
            indices = np.argmax(preds_matrix, axis=1)
            return np.array([self.classes_[i] for i in indices])

    def predict_proba(self, X, parallel: Union[bool, None] = None) -> np.ndarray:
        """
        Predict class probabilities with the fitted booster on new data.

        Only valid for classification tasks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        parallel : bool, optional
            Whether to run prediction in parallel.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            The class probabilities.
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
        else:
            features_, flat_data, rows, cols = transform_input_frame(
                X, self.cat_mapping
            )
        self._validate_features(features_)

        if len(self.classes_) > 2:
            if is_polars:
                probabilities = self.booster.predict_proba_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            else:
                probabilities = self.booster.predict_proba(
                    flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
                )
            return probabilities.reshape((-1, len(self.classes_)), order="C")
        elif len(self.classes_) == 2:
            if is_polars:
                probabilities = self.booster.predict_proba_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            else:
                probabilities = self.booster.predict_proba(
                    flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
                )
            return np.concatenate(
                [(1.0 - probabilities).reshape(-1, 1), probabilities.reshape(-1, 1)],
                axis=1,
            )
        else:
            warnings.warn(
                f"predict_proba not implemented for regression. n_classes = {len(self.classes_)}"
            )
            return np.ones((rows, 1))

    def predict_log_proba(self, X, parallel: Union[bool, None] = None) -> np.ndarray:
        """
        Predict class log-probabilities with the fitted booster on new data.

        Only valid for classification tasks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        parallel : bool, optional
            Whether to run prediction in parallel.

        Returns
        -------
        log_probabilities : ndarray of shape (n_samples, n_classes)
            The log-probabilities of each class.
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
        else:
            features_, flat_data, rows, cols = transform_input_frame(
                X, self.cat_mapping
            )
        self._validate_features(features_)

        if len(self.classes_) > 2:
            if is_polars:
                preds = self.booster.predict_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            else:
                preds = self.booster.predict(
                    flat_data=flat_data,
                    rows=rows,
                    cols=cols,
                    parallel=parallel,
                )
            return preds.reshape((-1, len(self.classes_)), order="F")
        elif len(self.classes_) == 2:
            if is_polars:
                return self.booster.predict_columnar(
                    columns=columns, masks=masks, rows=rows, parallel=parallel
                )
            return self.booster.predict(
                flat_data=flat_data,
                rows=rows,
                cols=cols,
                parallel=parallel,
            )
        else:
            warnings.warn("predict_log_proba not implemented for regression.")
            return np.ones((rows, 1))

    def predict_nodes(self, X, parallel: Union[bool, None] = None) -> List:
        """
        Predict leaf node indices with the fitted booster on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        parallel : bool, optional
            Whether to run prediction in parallel.

        Returns
        -------
        node_indices : list of ndarray
            A list where each element corresponds to a tree and contains node indices
            for each sample.
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
            self._validate_features(features_)
            return self.booster.predict_nodes_columnar(
                columns=columns, masks=masks, rows=rows, parallel=parallel
            )

        features_, flat_data, rows, cols = transform_input_frame(X, self.cat_mapping)
        self._validate_features(features_)

        return self.booster.predict_nodes(
            flat_data=flat_data, rows=rows, cols=cols, parallel=parallel
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        vals = self.calculate_feature_importance(
            method=self.feature_importance_method, normalize=True
        )
        if hasattr(self, "feature_names_in_"):
            vals = cast(Dict[str, float], vals)
            return np.array([vals.get(ft, 0.0) for ft in self.feature_names_in_])
        else:
            vals = cast(Dict[int, float], vals)
            return np.array([vals.get(ft, 0.0) for ft in range(self.n_features_)])

    def predict_contributions(
        self, X, method: str = "Average", parallel: Union[bool, None] = None
    ) -> np.ndarray:
        """
        Predict feature contributions (SHAP-like values) for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        method : str, default="Average"
            Method to calculate contributions. Options:

            - "Average": Internal node averages.
            - "Shapley": Exact tree SHAP values.
            - "Weight": Saabas-style leaf weights.
            - "BranchDifference": Difference between chosen and other branch.
            - "MidpointDifference": Weighted difference between branches.
            - "ModeDifference": Difference from the most frequent node.
            - "ProbabilityChange": Change in probability (LogLoss only).

        parallel : bool, optional
            Whether to run prediction in parallel.

        Returns
        -------
        contributions : ndarray of shape (n_samples, n_features + 1)
            The contribution of each feature to the prediction. The last column
            is the bias term.
        """
        is_polars = type_df(X) == "polars_df"
        if is_polars:
            features_, columns, masks, rows, cols = transform_input_frame_columnar(
                X, self.cat_mapping
            )
            self._validate_features(features_)
            contributions = self.booster.predict_contributions_columnar(
                columns=columns,
                masks=masks,
                rows=rows,
                method=CONTRIBUTION_METHODS.get(method, method),
                parallel=parallel,
            )
        else:
            features_, flat_data, rows, cols = transform_input_frame(
                X, self.cat_mapping
            )
            self._validate_features(features_)

            contributions = self.booster.predict_contributions(
                flat_data=flat_data,
                rows=rows,
                cols=cols,
                method=CONTRIBUTION_METHODS.get(method, method),
                parallel=parallel,
            )

        if len(self.classes_) > 2:
            return (
                np.reshape(contributions, (len(self.classes_), rows, cols + 1))
                .transpose(1, 0, 2)
                .reshape(rows, -1)
            )
        return np.reshape(contributions, (rows, cols + 1))

    def partial_dependence(
        self,
        X,
        feature: Union[str, int],
        samples: Optional[int] = 100,
        exclude_missing: bool = True,
        percentile_bounds: Tuple[float, float] = (0.2, 0.98),
    ) -> np.ndarray:
        """
        Calculate the partial dependence values of a feature.

        For each unique value of the feature, this gives the estimate of the predicted
        value for that feature, with the effects of all other features averaged out.

        Parameters
        ----------
        X : array-like
            Data used to calculate partial dependence. Should be the same format
            as passed to :meth:`fit`.
        feature : str or int
            The feature for which to calculate partial dependence.
        samples : int, optional, default=100
            Number of evenly spaced samples to select. If None, all unique values are used.
        exclude_missing : bool, optional, default=True
            Whether to exclude missing values from the calculation.
        percentile_bounds : tuple of float, optional, default=(0.2, 0.98)
            Lower and upper percentiles for sample selection.

        Returns
        -------
        pd_values : ndarray of shape (n_samples, 2)
            The first column contains the feature values, and the second column
            contains the partial dependence values.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pd_values = model.partial_dependence(X, feature="age")
        >>> plt.plot(pd_values[:, 0], pd_values[:, 1])
        """
        if isinstance(feature, str):
            is_polars = type_df(X) == "polars_df"
            if not (type_df(X) == "pandas_df" or is_polars):
                raise ValueError(
                    "If `feature` is a string, then the object passed as `X` must be a pandas or polars DataFrame."
                )
            if is_polars:
                values = X[feature].to_numpy()
            else:
                values = X.loc[:, feature].to_numpy()

            if hasattr(self, "feature_names_in_") and self.feature_names_in_[0] != "0":
                [feature_idx] = [
                    i for i, v in enumerate(self.feature_names_in_) if v == feature
                ]
            else:
                w_msg = (
                    "No feature names were provided at fit, but feature was a string, attempting to "
                    + "determine feature index from DataFrame Column, "
                    + "ensure columns are the same order as data passed when fit."
                )
                warnings.warn(w_msg)
                features = X.columns if is_polars else X.columns.to_list()
                [feature_idx] = [i for i, v in enumerate(features) if v == feature]
        elif isinstance(feature, int):
            feature_idx = feature
            if type_df(X) == "pandas_df":
                values = X.to_numpy()[:, feature]
            elif type_df(X) == "polars_df":
                values = X.to_numpy(allow_copy=False)[:, feature]
            else:
                values = X[:, feature]
        else:
            raise ValueError(
                f"The parameter `feature` must be a string, or an int, however an object of type {type(feature)} was passed."
            )
        min_p, max_p = percentile_bounds
        values = values[~(np.isnan(values) | (values == self.missing))]
        if samples is None:
            search_values = np.sort(np.unique(values))
        else:
            # Exclude missing from this calculation.
            search_values = np.quantile(values, np.linspace(min_p, max_p, num=samples))

        # Add missing back, if they wanted it...
        if not exclude_missing:
            search_values = np.append([self.missing], search_values)

        res = []
        for v in search_values:
            res.append(
                (v, self.booster.value_partial_dependence(feature=feature_idx, value=v))
            )
        return np.array(res)

    def calculate_feature_importance(
        self, method: str = "Gain", normalize: bool = True
    ) -> Union[Dict[int, float], Dict[str, float]]:
        """
        Calculate feature importance for the model.

        Parameters
        ----------
        method : str, optional, default="Gain"
            Importance method. Options:

            - "Weight": Number of times a feature is used in splits.
            - "Gain": Average improvement in loss brought by a feature.
            - "Cover": Average number of samples affected by splits on a feature.
            - "TotalGain": Total improvement in loss brought by a feature.
            - "TotalCover": Total number of samples affected by splits on a feature.

        normalize : bool, optional, default=True
            Whether to normalize importance scores to sum to 1.

        Returns
        -------
        importance : dict
            A dictionary mapping feature names (or indices) to importance scores.
        """
        importance_: Dict[int, float] = self.booster.calculate_feature_importance(
            method=method,
            normalize=normalize,
        )
        if hasattr(self, "feature_names_in_"):
            feature_map: Dict[int, str] = {
                i: f for i, f in enumerate(self.feature_names_in_)
            }
            return {feature_map[i]: v for i, v in importance_.items()}
        return importance_

    def text_dump(self) -> List[str]:
        """
        Return the booster model in a human-readable text format.

        Returns
        -------
        dump : list of str
            A list where each string represents a tree in the ensemble.
        """
        return self.booster.text_dump()

    def json_dump(self) -> str:
        """
        Return the booster model in JSON format.

        Returns
        -------
        dump : str
            The JSON representation of the model.
        """
        return self.booster.json_dump()

    @classmethod
    def load_booster(cls, path: str) -> Self:
        """
        Load a booster model from a file.

        Parameters
        ----------
        path : str
            Path to the saved booster (JSON format).

        Returns
        -------
        model : PerpetualBooster
            The loaded booster object.
        """
        try:
            booster = CratePerpetualBooster.load_booster(str(path))
        except ValueError:
            booster = CrateMultiOutputBooster.load_booster(str(path))

        params = booster.get_params()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cls(**params)
        c.booster = booster
        for m in c.metadata_attributes:
            try:
                m_ = c._get_metadata_attributes(m)
                setattr(c, m, m_)
                # If "feature_names_in_" is present, we know a
                # pandas dataframe was used for fitting, in this case
                # get back the original monotonicity map, with the
                # feature names as keys.
                if m == "feature_names_in_" and c.feature_names_in_[0] != "0":
                    if c.monotone_constraints is not None:
                        c.monotone_constraints = {
                            ft: c.monotone_constraints[i]
                            for i, ft in enumerate(c.feature_names_in_)
                        }
            except KeyError:
                pass
        return c

    def save_booster(self, path: str):
        """
        Save the booster model to a file.

        The model is saved in a JSON-based format.

        Parameters
        ----------
        path : str
            Path where the model will be saved.
        """
        self.booster.save_booster(str(path))

    def _standardize_monotonicity_map(
        self,
        X,
    ) -> Dict[int, Any]:
        if isinstance(X, np.ndarray):
            return self.monotone_constraints
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return {feature_map[f]: v for f, v in self.monotone_constraints.items()}

    def _standardize_terminate_missing_features(
        self,
        X,
    ) -> Set[int]:
        if isinstance(X, np.ndarray):
            return set(self.terminate_missing_features)
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return set(feature_map[f] for f in self.terminate_missing_features)

    def insert_metadata(self, key: str, value: str):
        """
        Insert metadata into the model.

        Metadata is saved alongside the model and can be retrieved later.

        Parameters
        ----------
        key : str
            The key for the metadata item.
        value : str
            The value for the metadata item.
        """
        self.booster.insert_metadata(key=key, value=value)

    def get_metadata(self, key: str) -> str:
        """
        Get metadata associated with a given key.

        Parameters
        ----------
        key : str
            The key to look up in the metadata.

        Returns
        -------
        value : str
            The value associated with the key.
        """
        v = self.booster.get_metadata(key=key)
        return v

    def _set_metadata_attributes(self, key: str, value: Any) -> None:
        value_ = self.metadata_attributes[key].serialize(value)
        self.insert_metadata(key=key, value=value_)

    def _get_metadata_attributes(self, key: str) -> Any:
        value = self.get_metadata(key)
        return self.metadata_attributes[key].deserialize(value)

    @property
    def base_score(self) -> Union[float, Iterable[float]]:
        """
        The base score(s) of the model.

        Returns
        -------
        score : float or iterable of float
            The initial prediction value(s) of the model.
        """
        return self.booster.base_score

    @property
    def number_of_trees(self) -> Union[int, Iterable[int]]:
        """
        The number of trees in the ensemble.

        Returns
        -------
        n_trees : int or iterable of int
            Total number of trees.
        """
        return self.booster.number_of_trees

    # Make picklable with getstate and setstate
    def __getstate__(self) -> Dict[Any, Any]:
        booster_json = self.json_dump()
        # Delete booster
        # Doing it like this, so it doesn't delete it globally.
        res = {k: v for k, v in self.__dict__.items() if k != "booster"}
        res["__booster_json_file__"] = booster_json
        return res

    def __setstate__(self, d: Dict[Any, Any]) -> None:
        # Load the booster object the pickled JSon string.
        try:
            booster_object = CratePerpetualBooster.from_json(d["__booster_json_file__"])
        except ValueError:
            booster_object = CrateMultiOutputBooster.from_json(
                d["__booster_json_file__"]
            )
        d["booster"] = booster_object
        # Are there any new parameters, that need to be added to the python object,
        # that would have been loaded in as defaults on the json object?
        # This makes sure that defaults set with a serde default function get
        # carried through to the python object.
        for p, v in booster_object.get_params().items():
            if p not in d:
                d[p] = v
        del d["__booster_json_file__"]
        self.__dict__ = d

    # Functions for scikit-learn compatibility, will feel out adding these manually,
    # and then if that feels too unwieldy will add scikit-learn as a dependency.
    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this booster.

        Parameters
        ----------
        deep : bool, default=True
            Currently ignored, exists for scikit-learn compatibility.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        args = inspect.getfullargspec(PerpetualBooster).kwonlyargs
        return {param: getattr(self, param) for param in args}

    def set_params(self, **params: Any) -> Self:
        """
        Set parameters for this booster.

        Parameters
        ----------
        **params : dict
            Booster parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        old_params = self.get_params()
        old_params.update(params)
        PerpetualBooster.__init__(self, **old_params)
        return self

    def get_node_lists(self, map_features_names: bool = True) -> List[List[Node]]:
        """
        Return tree structures as lists of node objects.

        Parameters
        ----------
        map_features_names : bool, default=True
            Whether to use feature names instead of indices.

        Returns
        -------
        trees : list of list of Node
            Each inner list represents a tree.
        """
        dump = json.loads(self.json_dump())
        if "trees" in dump:
            all_booster_trees = [dump["trees"]]
        else:
            # Multi-output
            all_booster_trees = [b["trees"] for b in dump["boosters"]]

        feature_map: Union[Dict[int, str], Dict[int, int]]
        leaf_split_feature: Union[str, int]
        if map_features_names and hasattr(self, "feature_names_in_"):
            feature_map = {i: ft for i, ft in enumerate(self.feature_names_in_)}
            leaf_split_feature = ""
        else:
            feature_map = {i: i for i in range(self.n_features_)}
            leaf_split_feature = -1

        trees = []
        for booster_trees in all_booster_trees:
            for t in booster_trees:
                nodes = []
                for node in t["nodes"].values():
                    if not node["is_leaf"]:
                        node["split_feature"] = feature_map[node["split_feature"]]
                    else:
                        node["split_feature"] = leaf_split_feature
                    nodes.append(Node(**node))
                trees.append(nodes)
        return trees

    def trees_to_dataframe(self) -> Any:
        """
        Return the tree structures as a DataFrame.

        Returns
        -------
        df : DataFrame
            A Polars or Pandas DataFrame containing tree information.
        """

        def node_to_row(
            n: Node,
            tree_n: int,
        ) -> Dict[str, Any]:
            def _id(i: int) -> str:
                return f"{tree_n}-{i}"

            return dict(
                Tree=tree_n,
                Node=n.num,
                ID=_id(n.num),
                Feature="Leaf" if n.is_leaf else str(n.split_feature),
                Split=None if n.is_leaf else n.split_value,
                Yes=None if n.is_leaf else _id(n.left_child),
                No=None if n.is_leaf else _id(n.right_child),
                Missing=None if n.is_leaf else _id(n.missing_node),
                Gain=n.weight_value if n.is_leaf else n.split_gain,
                Cover=n.hessian_sum,
                Left_Cats=n.left_cats,
                Right_Cats=n.right_cats,
            )

        # Flatten list of lists using list comprehension
        vals = [
            node_to_row(n, i)
            for i, tree in enumerate(self.get_node_lists())
            for n in tree
        ]

        try:
            import polars as pl

            return pl.from_records(vals).sort(
                ["Tree", "Node"], descending=[False, False]
            )
        except ImportError:
            import pandas as pd

            return pd.DataFrame.from_records(vals).sort_values(
                ["Tree", "Node"], ascending=[True, True]
            )

    def _to_xgboost_json(self) -> Dict[str, Any]:
        """Convert the Perpetual model to an XGBoost JSON model structure."""

        # Check if it's a multi-output model
        is_multi = len(self.classes_) > 2

        # Get raw dump
        raw_dump = json.loads(self.json_dump())

        # Initialize XGBoost structure
        xgb_json = {
            "learner": {
                "attributes": {},
                "feature_names": [],
                "feature_types": [],
                "gradient_booster": {
                    "model": {
                        "gbtree_model_param": {
                            "num_parallel_tree": "1",
                        },
                        "trees": [],
                        "tree_info": [],
                        "iteration_indptr": [],
                        "cats": {
                            "enc": [],
                            "feature_segments": [],
                            "sorted_idx": [],
                        },
                    },
                    "name": "gbtree",
                },
                "learner_model_param": {
                    "boost_from_average": "1",
                    "num_feature": str(self.n_features_),
                },
                "objective": {
                    "name": "binary:logistic",
                },
            },
            "version": [3, 1, 3],  # Use a reasonably recent version
        }

        # Fill feature names if available
        if hasattr(self, "feature_names_in_"):
            xgb_json["learner"]["feature_names"] = self.feature_names_in_
            xgb_json["learner"]["feature_types"] = ["float"] * self.n_features_
        else:
            xgb_json["learner"]["feature_names"] = [
                f"f{i}" for i in range(self.n_features_)
            ]
            xgb_json["learner"]["feature_types"] = ["float"] * self.n_features_

        # Objective and Base Score Handling
        if is_multi:
            # Multi-class
            n_classes = len(self.classes_)
            xgb_json["learner"]["objective"]["name"] = "multi:softprob"
            xgb_json["learner"]["objective"]["softmax_multiclass_param"] = {
                "num_class": str(n_classes)
            }
            xgb_json["learner"]["learner_model_param"]["num_class"] = str(n_classes)
            xgb_json["learner"]["learner_model_param"]["num_target"] = "1"

            # Base score vector [0.5, 0.5, ...]
            # 5.0E-1
            base_score_str = ",".join(["5.0E-1"] * n_classes)
            xgb_json["learner"]["learner_model_param"]["base_score"] = (
                f"[{base_score_str}]"
            )

            boosters = raw_dump["boosters"]

            trees = []
            tree_info = []
            # For multi-class, we need to interleave trees if we want to follow XGBoost structure perfectly?
            # Or can we just dump them? iteration_indptr depends on this.
            # XGBoost expects trees for iteration i to be contiguous.
            # Perpetual stores boosters separately.
            # Booster 0 has trees for class 0. Booster 1 for class 1.
            # We need to rearrange them: Round 0 (C0, C1, C2), Round 1 (C0, C1, C2)...

            # Assuming all boosters have same number of trees?
            num_trees_per_booster = [len(b["trees"]) for b in boosters]
            max_trees = max(num_trees_per_booster) if num_trees_per_booster else 0

            # Iteration pointers: 0, 3, 6...
            # But what if some booster has fewer trees? (Early stopping might cause this?)
            # Perpetual implementation usually stops all or none?
            # "MultiOutputBooster::fit" trains them sequentially but they might have different tree counts if EarlyStopping is per-booster.
            # But XGBoost expects consistent num_class trees per round (or use "multi:softprob"?)
            # If we just list them, XGBoost might get confused if we don't align them.

            # Let's try to align them by round.

            iteration_indptr = [0]
            current_ptr = 0

            for round_idx in range(max_trees):
                # For each class
                for group_id, booster_dump in enumerate(boosters):
                    booster_trees = booster_dump["trees"]
                    if round_idx < len(booster_trees):
                        tree = booster_trees[round_idx]
                        base_score = booster_dump["base_score"]

                        xgb_tree = self._convert_tree(tree, current_ptr)

                        if round_idx == 0:
                            self._adjust_tree_leaves(xgb_tree, base_score)

                        trees.append(xgb_tree)
                        tree_info.append(group_id)
                        current_ptr += 1
                    else:
                        # Missing tree for this class in this round?
                        # Should we insert a dummy tree (0 prediction)?
                        # For now, let's assume balanced trees or hope XGB handles it.
                        # If we skip, tree_info tracks class.
                        pass

                iteration_indptr.append(current_ptr)

            xgb_json["learner"]["gradient_booster"]["model"]["trees"] = trees
            xgb_json["learner"]["gradient_booster"]["model"]["tree_info"] = tree_info
            xgb_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ] = str(len(trees))
            xgb_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = (
                iteration_indptr
            )

        else:
            # Binary or Regression
            if self.objective == "LogLoss":
                xgb_json["learner"]["objective"]["name"] = "binary:logistic"
                xgb_json["learner"]["objective"]["reg_loss_param"] = {
                    "scale_pos_weight": "1"
                }
                xgb_json["learner"]["learner_model_param"]["num_class"] = "0"
                xgb_json["learner"]["learner_model_param"]["num_target"] = "1"

                # Base Score
                base_score_val = 1.0 / (1.0 + np.exp(-raw_dump["base_score"]))
                xgb_json["learner"]["learner_model_param"]["base_score"] = (
                    f"[{base_score_val:.6E}]"
                )

            elif self.objective == "SquaredLoss":
                xgb_json["learner"]["objective"]["name"] = "reg:squarederror"
                xgb_json["learner"]["objective"]["reg_loss_param"] = {}
                xgb_json["learner"]["learner_model_param"]["num_class"] = "0"
                xgb_json["learner"]["learner_model_param"]["num_target"] = "1"
                xgb_json["learner"]["learner_model_param"]["base_score"] = (
                    f"[{raw_dump['base_score']:.6E}]"
                )
            else:
                warnings.warn(
                    f"Objective {self.objective} not explicitly supported for XGBoost export. Defaulting to reg:squarederror."
                )
                xgb_json["learner"]["objective"]["name"] = "reg:squarederror"
                xgb_json["learner"]["objective"]["reg_loss_param"] = {}
                xgb_json["learner"]["learner_model_param"]["num_class"] = "0"
                xgb_json["learner"]["learner_model_param"]["num_target"] = "1"
                xgb_json["learner"]["learner_model_param"]["base_score"] = (
                    f"[{raw_dump['base_score']:.6E}]"
                )

            trees = []
            tree_info = []
            for tree_idx, tree in enumerate(raw_dump["trees"]):
                xgb_tree = self._convert_tree(tree, tree_idx)
                trees.append(xgb_tree)
                tree_info.append(0)

            xgb_json["learner"]["gradient_booster"]["model"]["trees"] = trees
            xgb_json["learner"]["gradient_booster"]["model"]["tree_info"] = tree_info
            xgb_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ] = str(len(trees))
            xgb_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = list(
                range(len(trees) + 1)
            )

        return xgb_json

    def _convert_tree(self, tree: Dict[str, Any], group_id: int) -> Dict[str, Any]:
        """Convert a single Perpetual tree to XGBoost dictionary format."""

        nodes_dict = tree["nodes"]
        # Convert keys to int and sort
        sorted_keys = sorted(nodes_dict.keys(), key=lambda x: int(x))

        # Mapping from Perpetual ID (int) to XGBoost Array Index (int)
        node_map = {int(k): i for i, k in enumerate(sorted_keys)}

        num_nodes = len(sorted_keys)
        # print(f"DEBUG: Converting tree group={group_id}. num_nodes={num_nodes}")

        left_children = [-1] * num_nodes
        right_children = [-1] * num_nodes
        parents = [2147483647] * num_nodes
        split_indices = [0] * num_nodes
        split_conditions = [0.0] * num_nodes
        split_type = [0] * num_nodes
        sum_hessian = [0.0] * num_nodes
        loss_changes = [0.0] * num_nodes
        base_weights = [0.0] * num_nodes
        default_left = [0] * num_nodes

        categories = []
        categories_nodes = []
        categories_segments = []
        categories_sizes = []

        for i, k in enumerate(sorted_keys):
            node = nodes_dict[k]
            nid = int(node["num"])
            idx = node_map[nid]

            # print(f"  DEBUG: Node {i} nid={nid} idx={idx}")

            sum_hessian[idx] = node["hessian_sum"]
            base_weights[idx] = node["weight_value"]
            loss_changes[idx] = node.get("split_gain", 0.0)

            if node["is_leaf"]:
                left_children[idx] = -1
                right_children[idx] = -1
                split_indices[idx] = 0
                split_conditions[idx] = node["weight_value"]
            else:
                left_id = node["left_child"]
                right_id = node["right_child"]

                left_idx = node_map[left_id]
                right_idx = node_map[right_id]

                left_children[idx] = left_idx
                right_children[idx] = right_idx
                parents[left_idx] = idx
                parents[right_idx] = idx

                split_indices[idx] = node["split_feature"]
                split_conditions[idx] = node["split_value"]

                # Missing handling
                # If missing_node goes left
                if node["missing_node"] == left_id:
                    default_left[idx] = 1
                else:
                    default_left[idx] = 0

                if (
                    "left_cats" in node
                    and node["left_cats"] is not None
                    and len(node["left_cats"]) > 0
                ):
                    # It's a categorical split
                    cats = node["left_cats"]
                    # XGBoost uses split_type=1 for categorical?
                    # Or just presence in categories_nodes?
                    # Docs say: split_type [default=0]: 0=numerical, 1=categorical
                    split_type[idx] = 1

                    # Update categorical arrays
                    categories_nodes.append(idx)
                    categories_sizes.append(len(cats))
                    # Segment is start index.
                    # If this is the first one, 0. Else prev_segment + prev_size?
                    # Actually valid XGBoost format usually has segments as exclusive scan.
                    # [0, len0, len0+len1, ...]
                    # Wait, segments length should be same as nodes?
                    # Let's check logic:
                    # segments[i] points to start of cats for node i (in categories_nodes)

                    next_segment = (
                        (categories_segments[-1] + categories_sizes[-2])
                        if categories_segments
                        else 0
                    )
                    categories_segments.append(next_segment)

                    categories.extend(sorted(cats))

                    # split_condition for categorical is usually NaN or special?
                    # XGBoost JSON parser might ignore it if type is categorical
                    # But often it is set to something.

        return {
            "base_weights": base_weights,
            "default_left": default_left,
            "id": group_id,
            "left_children": left_children,
            "loss_changes": loss_changes,
            "parents": parents,
            "right_children": right_children,
            "split_conditions": split_conditions,
            "split_indices": split_indices,
            "split_type": split_type,
            "sum_hessian": sum_hessian,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(self.n_features_),
                "num_nodes": str(num_nodes),
                "size_leaf_vector": "1",
            },
            "categories": categories,
            "categories_nodes": categories_nodes,
            "categories_segments": categories_segments,
            "categories_sizes": categories_sizes,
        }

    def _adjust_tree_leaves(self, xgb_tree: Dict[str, Any], adjustment: float):
        """Add adjustment value to all leaves in an XGBoost tree dict."""
        left_children = xgb_tree["left_children"]
        split_conditions = xgb_tree["split_conditions"]
        base_weights = xgb_tree["base_weights"]

        for i, left in enumerate(left_children):
            if left == -1:  # Leaf
                split_conditions[i] += adjustment
                base_weights[i] += adjustment

    def save_as_xgboost(self, path: str):
        """
        Save the model in XGBoost JSON format.

        Parameters
        ----------
        path : str
            The path where the XGBoost-compatible model will be saved.
        """
        xgboost_json = self._to_xgboost_json()
        with open(path, "w") as f:
            json.dump(xgboost_json, f, indent=2)

    def save_as_onnx(self, path: str, name: str = "perpetual_model"):
        """
        Save the model in ONNX format.

        Parameters
        ----------
        path : str
            The path where the ONNX model will be saved.
        name : str, optional, default="perpetual_model"
            The name of the graph in the exported model.
        """
        import json

        import onnx
        from onnx import TensorProto, helper

        raw_dump = json.loads(self.json_dump())
        is_classifier = len(self.classes_) >= 2
        is_multi = is_classifier and len(self.classes_) > 2
        n_classes = len(self.classes_) if is_classifier else 1

        if "trees" in raw_dump:
            booster_data = [{"trees": raw_dump["trees"]}]
        else:
            booster_data = raw_dump["boosters"]

        feature_map_inverse = (
            {v: k for k, v in enumerate(self.feature_names_in_)}
            if hasattr(self, "feature_names_in_")
            else None
        )

        nodes_treeids = []
        nodes_nodeids = []
        nodes_featureids = []
        nodes_values = []
        nodes_modes = []
        nodes_truenodeids = []
        nodes_falsenodeids = []
        nodes_missing_value_tracks_true = []

        target_treeids = []
        target_nodeids = []
        target_ids = []
        target_weights = []

        # Base score handling
        base_score = self.base_score
        if is_classifier:
            if is_multi:
                base_values = [float(b) for b in base_score]
            else:
                base_values = [float(base_score)]
        else:
            base_values = [float(base_score)]

        global_tree_idx = 0
        for b_idx, booster in enumerate(booster_data):
            for tree_data in booster["trees"]:
                nodes_dict = tree_data["nodes"]
                node_keys = sorted(nodes_dict.keys(), key=lambda x: int(x))

                node_id_to_idx = {}
                for i, k in enumerate(node_keys):
                    node_id_to_idx[int(k)] = i

                for k in node_keys:
                    node_dict = nodes_dict[k]
                    nid = int(node_dict["num"])
                    idx_for_onnx = node_id_to_idx[nid]

                    nodes_treeids.append(global_tree_idx)
                    nodes_nodeids.append(idx_for_onnx)

                    if node_dict["is_leaf"]:
                        nodes_modes.append("LEAF")
                        nodes_featureids.append(0)
                        nodes_values.append(0.0)
                        nodes_truenodeids.append(0)
                        nodes_falsenodeids.append(0)
                        nodes_missing_value_tracks_true.append(0)

                        target_treeids.append(global_tree_idx)
                        target_nodeids.append(idx_for_onnx)
                        target_ids.append(b_idx if is_multi else 0)
                        target_weights.append(float(node_dict["weight_value"]))
                    else:
                        nodes_modes.append("BRANCH_LT")
                        feat_val = node_dict["split_feature"]
                        f_idx = 0
                        if isinstance(feat_val, int):
                            f_idx = feat_val
                        elif feature_map_inverse and feat_val in feature_map_inverse:
                            f_idx = feature_map_inverse[feat_val]
                        elif isinstance(feat_val, str) and feat_val.isdigit():
                            f_idx = int(feat_val)

                        nodes_featureids.append(f_idx)
                        nodes_values.append(float(node_dict["split_value"]))

                        tracks_true = 0
                        if node_dict["missing_node"] == node_dict["left_child"]:
                            tracks_true = 1
                        nodes_missing_value_tracks_true.append(tracks_true)

                        nodes_truenodeids.append(
                            node_id_to_idx[int(node_dict["left_child"])]
                        )
                        nodes_falsenodeids.append(
                            node_id_to_idx[int(node_dict["right_child"])]
                        )

                global_tree_idx += 1

        input_name = "input"
        input_type = helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, [None, self.n_features_]
        )

        raw_scores_name = "raw_scores"
        reg_node = helper.make_node(
            "TreeEnsembleRegressor",
            inputs=[input_name],
            outputs=[raw_scores_name],
            domain="ai.onnx.ml",
            nodes_treeids=nodes_treeids,
            nodes_nodeids=nodes_nodeids,
            nodes_featureids=nodes_featureids,
            nodes_values=nodes_values,
            nodes_modes=nodes_modes,
            nodes_truenodeids=nodes_truenodeids,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            target_treeids=target_treeids,
            target_nodeids=target_nodeids,
            target_ids=target_ids,
            target_weights=target_weights,
            base_values=base_values,
            n_targets=n_classes if is_multi else 1,
            name="PerpetualTreeEnsemble",
        )

        ops = [reg_node]
        if is_classifier:
            # Prepare class labels mapping
            classes = self.classes_
            if all(isinstance(c, (int, np.integer)) for c in classes):
                tensor_type = TensorProto.INT64
                classes_array = np.array(classes, dtype=np.int64)
            elif all(isinstance(c, (float, np.floating)) for c in classes):
                tensor_type = TensorProto.FLOAT
                classes_array = np.array(classes, dtype=np.float32)
            else:
                tensor_type = TensorProto.STRING
                classes_array = np.array([str(c) for c in classes], dtype=object)

            classes_name = "class_labels"
            if tensor_type == TensorProto.STRING:
                classes_const_node = helper.make_node(
                    "Constant",
                    [],
                    [classes_name],
                    value=helper.make_tensor(
                        name="classes_tensor",
                        data_type=tensor_type,
                        dims=[len(classes)],
                        vals=[s.encode("utf-8") for s in classes_array],
                    ),
                )
            else:
                classes_const_node = helper.make_node(
                    "Constant",
                    [],
                    [classes_name],
                    value=helper.make_tensor(
                        name="classes_tensor",
                        data_type=tensor_type,
                        dims=[len(classes)],
                        vals=classes_array.flatten().tolist(),
                    ),
                )
            ops.append(classes_const_node)

            if is_multi:
                prob_name = "probabilities"
                softmax_node = helper.make_node(
                    "Softmax", [raw_scores_name], [prob_name], axis=1
                )
                label_idx_name = "label_idx"
                argmax_node = helper.make_node(
                    "ArgMax", [prob_name], [label_idx_name], axis=1, keepdims=0
                )
                label_name = "label"
                gather_node = helper.make_node(
                    "Gather", [classes_name, label_idx_name], [label_name], axis=0
                )
                ops.extend([softmax_node, argmax_node, gather_node])
                outputs = [
                    helper.make_tensor_value_info(label_name, tensor_type, [None]),
                    helper.make_tensor_value_info(
                        prob_name, TensorProto.FLOAT, [None, n_classes]
                    ),
                ]
            else:
                p_name = "p"
                sigmoid_node = helper.make_node("Sigmoid", [raw_scores_name], [p_name])
                one_name = "one"
                one_node = helper.make_node(
                    "Constant",
                    [],
                    [one_name],
                    value=helper.make_tensor("one_v", TensorProto.FLOAT, [1, 1], [1.0]),
                )
                one_minus_p_name = "one_minus_p"
                sub_node = helper.make_node(
                    "Sub", [one_name, p_name], [one_minus_p_name]
                )
                prob_name = "probabilities"
                concat_node = helper.make_node(
                    "Concat", [one_minus_p_name, p_name], [prob_name], axis=1
                )
                label_idx_name = "label_idx"
                argmax_node = helper.make_node(
                    "ArgMax", [prob_name], [label_idx_name], axis=1, keepdims=0
                )
                label_name = "label"
                gather_node = helper.make_node(
                    "Gather", [classes_name, label_idx_name], [label_name], axis=0
                )
                ops.extend(
                    [
                        sigmoid_node,
                        one_node,
                        sub_node,
                        concat_node,
                        argmax_node,
                        gather_node,
                    ]
                )
                outputs = [
                    helper.make_tensor_value_info(label_name, tensor_type, [None]),
                    helper.make_tensor_value_info(
                        prob_name, TensorProto.FLOAT, [None, 2]
                    ),
                ]
        else:
            prediction_name = "prediction"
            reg_node.output[0] = prediction_name
            outputs = [
                helper.make_tensor_value_info(
                    prediction_name, TensorProto.FLOAT, [None, 1]
                )
            ]

        graph_def = helper.make_graph(ops, name, [input_type], outputs)
        model_def = helper.make_model(
            graph_def,
            producer_name="perpetual",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("ai.onnx.ml", 2),
            ],
        )
        model_def.ir_version = 6
        onnx.save(model_def, path)
