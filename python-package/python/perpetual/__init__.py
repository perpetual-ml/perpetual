from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Union, cast

import numpy as np

from perpetual.perpetual import PerpetualBooster as CratePerpetualBooster  # type: ignore
from perpetual.serialize import BaseSerializer, ObjectSerializer


__all__ = ["PerpetualBooster"]


CONTRIBUTION_METHODS = {
    "weight": "Weight",
    "Weight": "Weight",
    "average": "Average",
    "Average": "Average",
    "branch-difference": "BranchDifference",
    "branchdifference": "BranchDifference",
    "BranchDifference": "BranchDifference",
    "midpoint-difference": "MidpointDifference",
    "midpointdifference": "MidpointDifference",
    "MidpointDifference": "MidpointDifference",
    "mode-difference": "ModeDifference",
    "modedifference": "ModeDifference",
    "ModeDifference": "ModeDifference",
    "ProbabilityChange": "ProbabilityChange",
    "probabilitychange": "ProbabilityChange",
    "probability-change": "ProbabilityChange",
}


@dataclass
class Node:
    """Dataclass representation of a node, this represents all of the fields present in a tree node."""

    num: int
    weight_value: float
    hessian_sum: float
    depth: int
    split_value: float
    split_feature: int | str
    split_gain: float
    missing_node: int
    left_child: int
    right_child: int
    is_leaf: bool
    node_type: str
    parent_node: int
    generalization: float | None
    left_categories: Iterable | None
    right_categories: Iterable | None


class BoosterType(Protocol):
    monotone_constraints: dict[int, int]
    terminate_missing_features: set[int]
    number_of_trees: int

    def fit(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        y: np.ndarray,
        budget: float,
        sample_weight: np.ndarray,
        parallel: bool = True,
    ):
        """Fit method"""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict method"""

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        """method"""

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        """pass"""

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> dict[int, float]:
        """pass"""

    def text_dump(self) -> list[str]:
        """pass"""

    @classmethod
    def load_booster(cls, path: str) -> BoosterType:
        """pass"""

    def save_booster(self, path: str):
        """pass"""

    @classmethod
    def from_json(cls, json_str: str) -> BoosterType:
        """pass"""

    def json_dump(self) -> str:
        """pass"""

    def get_params(self) -> dict[str, Any]:
        """pass"""

    def insert_metadata(self, key: str, value: str) -> None:
        """pass"""

    def get_metadata(self, key: str) -> str:
        """pass"""


def type_df(df):
    if type(df).__name__ == "DataFrame":
        if type(df).__module__.split(".")[0] == "pandas":
            return "pandas_df"
        elif type(df).__module__.split(".")[0] == "polars":
            return "polars_df"
    else:
        return ""


def type_series(y):
    if type(y).__name__ == "Series":
        if type(y).__module__.split(".")[0] == "pandas":
            return "pandas_series"
        elif type(y).__module__.split(".")[0] == "polars":
            return "polars_series"
    else:
        return ""


def convert_input_frame(
    X, categorical_features
) -> tuple[list[str], np.ndarray, int, int, Optional[Iterable[int]], Optional[dict]]:
    """Convert data to format needed by booster.

    Returns:
        tuple[list[str], np.ndarray, int, int, Optional[Iterable[int]], Optional[dict]]: Return column names, the flat data, number of rows, the number of columns, cat_index, cat_mapping
    """
    categorical_features_ = None
    if type_df(X) == "pandas_df":
        X_ = X.to_numpy()
        features_ = X.columns.to_list()
        if categorical_features == "auto":
            categorical_columns = X.select_dtypes(include=["category"]).columns.tolist()
            categorical_features_ = [
                features_.index(c) for c in categorical_columns
            ] or None
    elif type_df(X) == "polars_df":
        import polars.selectors as cs

        try:
            X_ = X.to_numpy(allow_copy=False)
        except RuntimeError:
            X_ = X.to_numpy(allow_copy=True)

        features_ = X.columns
        if categorical_features == "auto":
            categorical_columns = X.select(cs.categorical()).columns
            categorical_features_ = [
                features_.index(c) for c in categorical_columns
            ] or None
    else:
        # Assume it's a numpy array.
        X_ = X
        features_ = list(map(str, range(X_.shape[1])))

    if (
        categorical_features
        and all(isinstance(s, int) for s in categorical_features)
        and isinstance(categorical_features, list)
    ):
        categorical_features_ = categorical_features
    elif (
        categorical_features
        and all(isinstance(s, str) for s in categorical_features)
        and isinstance(categorical_features, list)
    ):
        categorical_features_ = [features_.index(c) for c in categorical_features]

    cat_mapping = {}  # key: feature_name, value: ordered category names
    if categorical_features_:
        for i in categorical_features_:
            categories = np.unique(X_[:, i].astype(dtype="str", copy=False))
            categories = [c for c in list(categories) if c != "nan"]
            categories.insert(0, "nan")
            cat_mapping[features_[i]] = categories

    if cat_mapping:
        print(f"Categorical features: {categorical_features_}")
        print(f"Mapping of categories: {cat_mapping}")
        for feature_name, categories in cat_mapping.items():
            feature_index = features_.index(feature_name)

            def f(x):
                try:
                    return (
                        np.nan
                        if str(x[feature_index]) == "nan"
                        else float(categories.index(str(x[feature_index])))
                    )
                except (ValueError, IndexError):
                    return np.nan

            X_[:, feature_index] = np.apply_along_axis(f, 1, X_)

    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape

    if isinstance(categorical_features_, list):
        categorical_features_ = np.array(categorical_features_, dtype=np.uint64)

    return features_, flat_data, rows, cols, categorical_features_, cat_mapping


def transform_input_frame(X, cat_mapping) -> tuple[list[str], np.ndarray, int, int]:
    """Convert data to format needed by booster.

    Returns:
        tuple[list[str], np.ndarray, int, int]: Return column names, the flat data, number of rows, the number of columns
    """
    if type_df(X) == "pandas_df":
        X_ = X.to_numpy()
        features_ = X.columns.to_list()
    elif type_df(X) == "polars_df":
        try:
            X_ = X.to_numpy(allow_copy=False)
        except RuntimeError:
            X_ = X.to_numpy(allow_copy=True)
        features_ = X.columns
    else:
        # Assume it's a numpy array.
        X_ = X
        features_ = list(map(str, range(X_.shape[1])))

    if cat_mapping:
        for feature_name, categories in cat_mapping.items():
            feature_index = features_.index(feature_name)

            def f(x):
                try:
                    return (
                        np.nan
                        if str(x[feature_index]) == "nan"
                        else float(categories.index(str(x[feature_index])))
                    )
                except (ValueError, IndexError):
                    return np.nan

            X_[:, feature_index] = np.apply_along_axis(f, 1, X_)

    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape

    return features_, flat_data, rows, cols


def _convert_input_array(x) -> np.ndarray:
    if type_series(x) == "pandas_series":
        x_ = x.to_numpy()
    elif type_series(x) == "polars_series":
        x_ = x.to_numpy(allow_copy=False)
    else:
        x_ = x
    if not np.issubdtype(x_.dtype, "float64"):
        x_ = x_.astype(dtype="float64", copy=False)
    return x_


def _convert_input_array_f32(x) -> np.ndarray:
    if type_series(x) == "pandas_series":
        x_ = x.to_numpy()
    elif type_series(x) == "polars_series":
        x_ = x.to_numpy(allow_copy=False)
    else:
        x_ = x
    if not np.issubdtype(x_.dtype, "float32"):
        x_ = x_.astype(dtype="float32", copy=False)
    return x_


class PerpetualBooster:
    # Define the metadata parameters
    # that are present on all instances of this class
    # this is useful for parameters that should be
    # attempted to be loaded in and set
    # as attributes on the booster after it is loaded.
    meta_data_attributes: dict[str, BaseSerializer] = {
        "feature_names_in_": ObjectSerializer(),
        "n_features_": ObjectSerializer(),
        "feature_importance_method": ObjectSerializer(),
        "cat_mapping": ObjectSerializer(),
    }

    def __init__(
        self,
        *,
        objective: str = "LogLoss",
        parallel: bool = True,
        missing: float = np.nan,
        allow_missing_splits: bool = True,
        create_missing_branch: bool = False,
        terminate_missing_features: Iterable[Any] | None = None,
        missing_node_treatment: str = "None",
        monotone_constraints: Union[dict[Any, int], None] = None,
        force_children_to_bound_parent: bool = False,
        log_iterations: int = 0,
        feature_importance_method: str = "Gain",
    ):
        """PerpetualBooster Class, used to generate gradient boosted decision tree ensembles.

        Args:
            objective (str, optional): The name of objective function used to optimize.
                Valid options include "LogLoss" to use logistic loss as the objective function
                (classification), or "SquaredLoss" to use Squared Error as the objective
                function (regression). Defaults to "LogLoss".
            parallel (bool, optional): Should multiple cores be used when training and predicting
                with this model? Defaults to `True`.
            missing (float, optional): Value to consider missing, when training and predicting
                with the booster. Defaults to `np.nan`.
            allow_missing_splits (bool, optional): Allow for splits to be made such that all missing values go
                down one branch, and all non-missing values go down the other, if this results
                in the greatest reduction of loss. If this is false, splits will only be made on non
                missing values. If `create_missing_branch` is set to `True` having this parameter be
                set to `True` will result in the missing branch further split, if this parameter
                is `False` then in that case the missing branch will always be a terminal node.
                Defaults to `True`.
            create_missing_branch (bool, optional): An experimental parameter, that if `True`, will
                create a separate branch for missing, creating a ternary tree, the missing node will be given the same
                weight value as the parent node. If this parameter is `False`, missing will be sent
                down either the left or right branch, creating a binary tree. Defaults to `False`.
            terminate_missing_features (set[Any], optional): An optional iterable of features
                (either strings, or integer values specifying the feature indices if numpy arrays are used for fitting),
                for which the missing node will always be terminated, even if `allow_missing_splits` is set to true.
                This value is only valid if `create_missing_branch` is also True.
            missing_node_treatment (str, optional): Method for selecting the `weight` for the missing node, if `create_missing_branch` is set to `True`. Defaults to "None". Valid options are:
                - "None": Calculate missing node weight values without any constraints.
                - "AssignToParent": Assign the weight of the missing node to that of the parent.
                - "AverageLeafWeight": After training each tree, starting from the bottom of the tree, assign the missing node weight to the weighted average of the left and right child nodes. Next assign the parent to the weighted average of the children nodes. This is performed recursively up through the entire tree. This is performed as a post processing step on each tree after it is built, and prior to updating the predictions for which to train the next tree.
                - "AverageNodeWeight": Set the missing node to be equal to the weighted average weight of the left and the right nodes.
            monotone_constraints (dict[Any, int], optional): Constraints that are used to enforce a
                specific relationship between the training features and the target variable. A dictionary
                should be provided where the keys are the feature index value if the model will be fit on
                a numpy array, or a feature name if it will be fit on a pandas Dataframe. The values of
                the dictionary should be an integer value of -1, 1, or 0 to specify the relationship
                that should be estimated between the respective feature and the target variable.
                Use a value of -1 to enforce a negative relationship, 1 a positive relationship,
                and 0 will enforce no specific relationship at all. Features not included in the
                mapping will not have any constraint applied. If `None` is passed no constraints
                will be enforced on any variable.  Defaults to `None`.
            force_children_to_bound_parent (bool, optional): Setting this parameter to `True` will restrict children nodes, so that they always contain the parent node inside of their range. Without setting this it's possible that both, the left and the right nodes could be greater, than or less than, the parent node. Defaults to `False`.
            log_iterations (bool, optional): Setting to a value (N) other than zero will result in information being logged about ever N iterations, info can be interacted with directly with the python [`logging`](https://docs.python.org/3/howto/logging.html) module. For an example of how to utilize the logging information see the example [here](/#logging-output).
            feature_importance_method (str, optional): The feature importance method type that will be used to calculate the `feature_importances_` attribute on the booster.

        Raises:
            TypeError: Raised if an invalid dtype is passed.

        Example:
            Once, the booster has been initialized, it can be fit on a provided dataset, and performance field. After fitting, the model can be used to predict on a dataset.
            In the case of this example, the predictions are the log odds of a given record being 1.

            ```python
            # Small example dataset
            from seaborn import load_dataset

            df = load_dataset("titanic")
            X = df.select_dtypes("number").drop(columns=["survived"])
            y = df["survived"]

            # Initialize a booster with defaults.
            from perpetual import PerpetualBooster
            model = PerpetualBooster(objective="LogLoss")
            model.fit(X, y)

            # Predict on data
            model.predict(X.head())
            # array([-1.94919663,  2.25863229,  0.32963671,  2.48732194, -3.00371813])

            # predict contributions
            model.predict_contributions(X.head())
            # array([[-0.63014213,  0.33880048, -0.16520798, -0.07798772, -0.85083578,
            #        -1.07720813],
            #       [ 1.05406709,  0.08825999,  0.21662544, -0.12083538,  0.35209258,
            #        -1.07720813],
            ```

        """

        terminate_missing_features_ = (
            set() if terminate_missing_features is None else terminate_missing_features
        )

        booster = CratePerpetualBooster(
            objective=objective,
            parallel=parallel,
            allow_missing_splits=allow_missing_splits,
            monotone_constraints={},
            missing=missing,
            create_missing_branch=create_missing_branch,
            terminate_missing_features=set(),
            missing_node_treatment=missing_node_treatment,
            force_children_to_bound_parent=force_children_to_bound_parent,
            log_iterations=log_iterations,
        )
        monotone_constraints_ = (
            {} if monotone_constraints is None else monotone_constraints
        )
        self.booster = cast(BoosterType, booster)
        self.objective = objective
        self.parallel = parallel
        self.allow_missing_splits = allow_missing_splits
        self.monotone_constraints = monotone_constraints_
        self.missing = missing
        self.create_missing_branch = create_missing_branch
        self.terminate_missing_features = terminate_missing_features_
        self.missing_node_treatment = missing_node_treatment
        self.force_children_to_bound_parent = force_children_to_bound_parent
        self.log_iterations = log_iterations
        self.feature_importance_method = feature_importance_method

        self._set_metadata_attributes(
            "feature_importance_method", feature_importance_method
        )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        budget: float = 1.0,
        reset: Union[bool, None] = None,
        categorical_features: Union[Iterable[int], Iterable[str], str, None] = "auto",
    ) -> PerpetualBooster:
        """Fit the gradient booster on a provided dataset.

        Args:
            X (FrameLike): Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
            y (ArrayLike): Either a Polars or Pandas Series, or a 1 dimensional Numpy array. If "LogLoss"
                was the objective type specified, then this should only contain 1 or 0 values,
                where 1 is the positive class being predicted. If "SquaredLoss" is the
                objective type, then any continuous variable can be provided.
            budget: a positive for fitting budget. Increasing this number will more
                likely result in increased accuracy.
            sample_weight (Union[ArrayLike, None], optional): Instance weights to use when
                training the model. If None is passed, a weight of 1 will be used for every record.
                Defaults to None.
            categorical_features: The names or indices for categorical features. auto for pandas categorical data type
        """

        features_, flat_data, rows, cols, categorical_features_, cat_mapping = (
            convert_input_frame(X, categorical_features)
        )
        self.n_features_ = cols
        self._set_metadata_attributes("n_features_", self.n_features_)
        self.cat_mapping = cat_mapping
        self._set_metadata_attributes("cat_mapping", self.cat_mapping)
        self.feature_names_in_ = features_
        self._set_metadata_attributes("feature_names_in_", self.feature_names_in_)

        y_ = _convert_input_array(y)

        if sample_weight is None:
            sample_weight_ = None
        else:
            sample_weight_ = _convert_input_array(sample_weight)

        # Convert the monotone constraints into the form needed
        # by the rust code.
        monotone_constraints_ = self._standardize_monotonicity_map(X)
        self.booster.monotone_constraints = monotone_constraints_
        self.booster.terminate_missing_features = (
            self._standardize_terminate_missing_features(X)
        )

        self.booster.fit(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            y=y_,
            budget=budget,
            sample_weight=sample_weight_,  # type: ignore
            reset=reset,
            categorical_features=categorical_features_,  # type: ignore
        )

        return self

    def _validate_features(self, features: list[str]):
        if len(features) > 0 and hasattr(self, "feature_names_in_"):
            if features[0] != "0" and self.feature_names_in_[0] != "0":
                if features != self.feature_names_in_:
                    raise ValueError(
                        f"Columns mismatch between data {features} passed, and data {self.feature_names_in_} used at fit."
                    )

    def predict(self, X, parallel: Union[bool, None] = None) -> np.ndarray:
        """Predict with the fitted booster on new data.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            parallel (Union[bool, None], optional): Optionally specify if the predict
                function should run in parallel on multiple threads. If `None` is
                passed, the `parallel` attribute of the booster will be used.
                Defaults to `None`.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        features_, flat_data, rows, cols = transform_input_frame(X, self.cat_mapping)
        self._validate_features(features_)
        parallel_ = self.parallel if parallel is None else parallel
        return self.booster.predict(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            parallel=parallel_,
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
        """Predict with the fitted booster on new data, returning the feature
        contribution matrix. The last column is the bias term.


        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
            method (str, optional): Method to calculate the contributions, available options are:

                - "Average": If this option is specified, the average internal node values are calculated.
                - "Shapley": Using this option will calculate contributions using the tree shap algorithm.
                - "Weight": This method will use the internal leaf weights, to calculate the contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
                - "BranchDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the other non-missing branch. This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "MidpointDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the mid-point between the right and left node weighted by the cover of each node. This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "ModeDifference": This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the node with the largest cover (the mode node). This method does not have the property where the contributions summed is equal to the final prediction of the model.
                - "ProbabilityChange": This method is only valid when the objective type is set to "LogLoss". This method will calculate contributions as the change in a records probability of being 1 moving from a parent node to a child node. The sum of the returned contributions matrix, will be equal to the probability a record will be 1. For example, given a model, `model.predict_contributions(X, method="ProbabilityChange") == 1 / (1 + np.exp(-model.predict(X)))`
            parallel (Union[bool, None], optional): Optionally specify if the predict
                function should run in parallel on multiple threads. If `None` is
                passed, the `parallel` attribute of the booster will be used.
                Defaults to `None`.

        Returns:
            np.ndarray: Returns a numpy array of the predictions.
        """
        features_, flat_data, rows, cols = transform_input_frame(X, self.cat_mapping)
        self._validate_features(features_)
        parallel_ = self.parallel if parallel is None else parallel

        contributions = self.booster.predict_contributions(
            flat_data=flat_data,
            rows=rows,
            cols=cols,
            method=CONTRIBUTION_METHODS.get(method, method),
            parallel=parallel_,
        )
        return np.reshape(contributions, (rows, cols + 1))

    def partial_dependence(
        self,
        X,
        feature: Union[str, int],
        samples: int | None = 100,
        exclude_missing: bool = True,
        percentile_bounds: tuple[float, float] = (0.2, 0.98),
    ) -> np.ndarray:
        """Calculate the partial dependence values of a feature. For each unique
        value of the feature, this gives the estimate of the predicted value for that
        feature, with the effects of all features averaged out. This information gives
        an estimate of how a given feature impacts the model.

        Args:
            X (FrameLike): Either a pandas DataFrame, or a 2 dimensional numpy array.
                This should be the same data passed into the models fit, or predict,
                with the columns in the same order.
            feature (Union[str, int]): The feature for which to calculate the partial
                dependence values. This can be the name of a column, if the provided
                X is a pandas DataFrame, or the index of the feature.
            samples (int | None, optional): Number of evenly spaced samples to select. If None
                is passed all unique values will be used. Defaults to 100.
            exclude_missing (bool, optional): Should missing excluded from the features? Defaults to True.
            percentile_bounds (tuple[float, float], optional): Upper and lower percentiles to start at
                when calculating the samples. Defaults to (0.2, 0.98) to cap the samples selected
                at the 5th and 95th percentiles respectively.

        Raises:
            ValueError: An error will be raised if the provided X parameter is not a
                pandas DataFrame, and a string is provided for the feature.

        Returns:
            np.ndarray: A 2 dimensional numpy array, where the first column is the
                sorted unique values of the feature, and then the second column
                is the partial dependence values for each feature value.

        Example:
            This information can be plotted to visualize how a feature is used in the model, like so.

            ```python
            from seaborn import lineplot
            import matplotlib.pyplot as plt

            pd_values = model.partial_dependence(X=X, feature="age", samples=None)

            fig = lineplot(x=pd_values[:,0], y=pd_values[:,1],)
            plt.title("Partial Dependence Plot")
            plt.xlabel("Age")
            plt.ylabel("Log Odds")
            ```
            <img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age.png">

            We can see how this is impacted if a model is created, where a specific constraint is applied to the feature using the `monotone_constraint` parameter.

            ```python
            model = GradientBooster(
                objective="LogLoss",
                monotone_constraints={"age": -1},
            )
            model.fit(X, y)

            pd_values = model.partial_dependence(X=X, feature="age")
            fig = lineplot(
                x=pd_values[:, 0],
                y=pd_values[:, 1],
            )
            plt.title("Partial Dependence Plot with Monotonicity")
            plt.xlabel("Age")
            plt.ylabel("Log Odds")
            ```
            <img  height="340" src="https://github.com/jinlow/forust/raw/main/resources/pdp_plot_age_mono.png">
        """
        if isinstance(feature, str):
            if not (type_df(X) == "pandas_df" or type_df(X) == "polars_df"):
                raise ValueError(
                    "If `feature` is a string, then the object passed as `X` must be a pandas DataFrame."
                )
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
                [feature_idx] = [i for i, v in enumerate(X.columns) if v == feature]
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
    ) -> dict[int, float] | dict[str, float]:
        """Feature importance values can be calculated with the `calculate_feature_importance` method. This function will return a dictionary of the features and their importance values. It should be noted that if a feature was never used for splitting it will not be returned in importance dictionary.

        Args:
            method (str, optional): Variable importance method. Defaults to "Gain". Valid options are:

                - "Weight": The number of times a feature is used to split the data across all trees.
                - "Gain": The average split gain across all splits the feature is used in.
                - "Cover": The average coverage across all splits the feature is used in.
                - "TotalGain": The total gain across all splits the feature is used in.
                - "TotalCover": The total coverage across all splits the feature is used in.
            normalize (bool, optional): Should the importance be normalized to sum to 1? Defaults to `True`.

        Returns:
            dict[str, float]: Variable importance values, for features present in the model.

        Example:
            ```python
            model.calculate_feature_importance("Gain")
            # {
            #   'parch': 0.0713072270154953,
            #   'age': 0.11609109491109848,
            #   'sibsp': 0.1486879289150238,
            #   'fare': 0.14309120178222656,
            #   'pclass': 0.5208225250244141
            # }
            ```
        """
        importance_: dict[int, float] = self.booster.calculate_feature_importance(
            method=method,
            normalize=normalize,
        )
        if hasattr(self, "feature_names_in_"):
            feature_map: dict[int, str] = {
                i: f for i, f in enumerate(self.feature_names_in_)
            }
            return {feature_map[i]: v for i, v in importance_.items()}
        return importance_

    def text_dump(self) -> list[str]:
        """Return all of the trees of the model in text form.

        Returns:
            list[str]: A list of strings, where each string is a text representation
                of the tree.
        Example:
            ```python
            model.text_dump()[0]
            # 0:[0 < 3] yes=1,no=2,missing=2,gain=91.50833,cover=209.388307
            #       1:[4 < 13.7917] yes=3,no=4,missing=4,gain=28.185467,cover=94.00148
            #             3:[1 < 18] yes=7,no=8,missing=8,gain=1.4576768,cover=22.090348
            #                   7:[1 < 17] yes=15,no=16,missing=16,gain=0.691266,cover=0.705011
            #                         15:leaf=-0.15120,cover=0.23500
            #                         16:leaf=0.154097,cover=0.470007
            ```
        """
        return self.booster.text_dump()

    def json_dump(self) -> str:
        """Return the booster object as a string.

        Returns:
            str: The booster dumped as a json object in string form.
        """
        return self.booster.json_dump()

    @classmethod
    def load_booster(cls, path: str) -> PerpetualBooster:
        """Load a booster object that was saved with the `save_booster` method.

        Args:
            path (str): Path to the saved booster file.

        Returns:
            PerpetualBooster: An initialized booster object.
        """
        booster = CratePerpetualBooster.load_booster(str(path))

        params = booster.get_params()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cls(**params)
        c.booster = booster
        for m in c.meta_data_attributes:
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
        """Save a booster object, the underlying representation is a json file.

        Args:
            path (str): Path to save the booster object.
        """
        self.booster.save_booster(str(path))

    def _standardize_monotonicity_map(
        self,
        X,
    ) -> dict[int, Any]:
        if isinstance(X, np.ndarray):
            return self.monotone_constraints
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return {feature_map[f]: v for f, v in self.monotone_constraints.items()}

    def _standardize_terminate_missing_features(
        self,
        X,
    ) -> set[int]:
        if isinstance(X, np.ndarray):
            return set(self.terminate_missing_features)
        else:
            feature_map = {f: i for i, f in enumerate(X.columns)}
            return set(feature_map[f] for f in self.terminate_missing_features)

    def insert_metadata(self, key: str, value: str):
        """Insert data into the models metadata, this will be saved on the booster object.

        Args:
            key (str): Key to give the inserted value in the metadata.
            value (str): String value to assign to the key.
        """  # noqa: E501
        self.booster.insert_metadata(key=key, value=value)

    def get_metadata(self, key: str) -> str:
        """Get the value associated with a given key, on the boosters metadata.

        Args:
            key (str): Key of item in metadata.

        Returns:
            str: Value associated with the provided key in the boosters metadata.
        """
        v = self.booster.get_metadata(key=key)
        return v

    def _set_metadata_attributes(self, key: str, value: Any) -> None:
        value_ = self.meta_data_attributes[key].serialize(value)
        self.insert_metadata(key=key, value=value_)

    def _get_metadata_attributes(self, key: str) -> Any:
        value = self.get_metadata(key)
        return self.meta_data_attributes[key].deserialize(value)

    @property
    def number_of_trees(self) -> int:
        """The number of trees in the model.

        Returns:
            int: The total number of trees in the model.
        """
        return self.booster.number_of_trees

    # Make picklable with getstate and setstate
    def __getstate__(self) -> dict[Any, Any]:
        booster_json = self.json_dump()
        # Delete booster
        # Doing it like this, so it doesn't delete it globally.
        res = {k: v for k, v in self.__dict__.items() if k != "booster"}
        res["__booster_json_file__"] = booster_json
        return res

    def __setstate__(self, d: dict[Any, Any]) -> None:
        # Load the booster object the pickled JSon string.
        booster_object = CratePerpetualBooster.from_json(d["__booster_json_file__"])
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
    def get_params(self, deep=True) -> dict[str, Any]:
        """Get all of the parameters for the booster.

        Args:
            deep (bool, optional): This argument does nothing, and is simply here for scikit-learn compatibility.. Defaults to True.

        Returns:
            dict[str, Any]: The parameters of the booster.
        """
        args = inspect.getfullargspec(PerpetualBooster).kwonlyargs
        return {param: getattr(self, param) for param in args}

    def set_params(self, **params: Any) -> PerpetualBooster:
        """Set the parameters of the booster, this has the same effect as reinstating the booster.

        Returns:
            PerpetualBooster: Booster with new parameters.
        """
        old_params = self.get_params()
        old_params.update(params)
        PerpetualBooster.__init__(self, **old_params)
        return self

    def get_node_lists(self, map_features_names: bool = True) -> list[list[Node]]:
        """Return the tree structures representation as a list of python objects.

        Args:
            map_features_names (bool, optional): Should the feature names tried to be mapped to a string, if a pandas dataframe was used. Defaults to True.

        Returns:
            list[list[Node]]: A list of lists where each sub list is a tree, with all of it's respective nodes.

        Example:
            This can be run directly to get the tree structure as python objects.

            ```python
            fmod = GradientBooster(max_depth=2)
            fmod.fit(X, y=y)

            fmod.get_node_lists()[0]

            # [Node(num=0, weight_value...,
            # Node(num=1, weight_value...,
            # Node(num=2, weight_value...,
            # Node(num=3, weight_value...,
            # Node(num=4, weight_value...,
            # Node(num=5, weight_value...,
            # Node(num=6, weight_value...,]
            ```
        """
        model = json.loads(self.json_dump())["trees"]
        feature_map: dict[int, str] | dict[int, int]
        leaf_split_feature: str | int
        if map_features_names and hasattr(self, "feature_names_in_"):
            feature_map = {i: ft for i, ft in enumerate(self.feature_names_in_)}
            leaf_split_feature = ""
        else:
            feature_map = {i: i for i in range(self.n_features_)}
            leaf_split_feature = -1

        trees = []
        for t in model:
            nodes = []
            for node in t["nodes"].values():
                if not node["is_leaf"]:
                    node["split_feature"] = feature_map[node["split_feature"]]
                else:
                    node["split_feature"] = leaf_split_feature
                nodes.append(Node(**node))
            trees.append(nodes)
        return trees

    def trees_to_dataframe(self):
        """Return the tree structure as a Polars or Pandas DataFrame object.

        Returns:
            DataFrame: Trees in a Polars or Pandas DataFrame.

        Example:
            This can be used directly to print out the tree structure as a dataframe. The Leaf values will have the "Gain" column replaced with the weight value.

            ```python
            model.trees_to_dataframe().head()
            ```

            |    |   Tree |   Node | ID   | Feature   |   Split | Yes   | No   | Missing   |    Gain |    Cover |
            |---:|-------:|-------:|:-----|:----------|--------:|:------|:-----|:----------|--------:|---------:|
            |  0 |      0 |      0 | 0-0  | pclass    |  3      | 0-1   | 0-2  | 0-2       | 91.5083 | 209.388  |
            |  1 |      0 |      1 | 0-1  | fare      | 13.7917 | 0-3   | 0-4  | 0-4       | 28.1855 |  94.0015 |
        """

        def node_to_row(
            n: Node,
            tree_n: int,
        ) -> dict[str, Any]:
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
                Left_Categories=n.left_categories,
                Right_Categories=n.right_categories,
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
