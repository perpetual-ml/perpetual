import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple


def type_df(df):
    library_name = type(df).__module__.split(".")[0]
    if type(df).__name__ == "DataFrame":
        if library_name == "pandas":
            return "pandas_df"
        elif library_name == "polars":
            return "polars_df"
    elif library_name == "numpy":
        return "numpy"
    else:
        return ""


def type_series(y):
    library_name = type(y).__module__.split(".")[0]
    if type(y).__name__ == "Series":
        if library_name == "pandas":
            return "pandas_series"
        elif library_name == "polars":
            return "polars_series"
    elif library_name == "numpy":
        return "numpy"
    else:
        return ""


def convert_input_array(x, objective) -> np.ndarray:
    classes_ = []

    if type(x).__module__.split(".")[0] == "numpy":
        if len(x.shape) == 2:
            classes_, x_, *_ = convert_input_frame(x, None)
        else:
            x_ = x
    elif type_series(x) == "pandas_series":
        x_ = x.to_numpy()
    elif type_series(x) == "polars_series":
        x_ = x.to_numpy(allow_copy=False)
    elif type_df(x) == "polars_df" or type_df(x) == "pandas_df":
        classes_, x_, *_ = convert_input_frame(x, None)
    else:
        x_ = x.to_numpy()

    if objective == "LogLoss" and len(x_.shape) == 1:
        classes_ = np.unique(x_)
        x_index = np.array([np.where(classes_ == i) for i in x_])
        if len(classes_) > 2:
            x_ = np.squeeze(np.eye(len(classes_))[x_index])

    if not np.issubdtype(x_.dtype, "float64"):
        x_ = x_.astype(dtype="float64", copy=False)

    if len(x_.shape) == 2:
        x_ = x_.ravel(order="F")

    return x_, classes_


def convert_input_frame(
    X, categorical_features
) -> Tuple[List[str], np.ndarray, int, int, Optional[Iterable[int]], Optional[Dict]]:
    """Convert data to format needed by booster.

    Returns:
        Tuple[List[str], np.ndarray, int, int, Optional[Iterable[int]], Optional[Dict]]: Return column names, the flat data, number of rows, the number of columns, cat_index, cat_mapping
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
        categorical_features_ = set(categorical_features_)

    return features_, flat_data, rows, cols, categorical_features_, cat_mapping


def transform_input_frame(X, cat_mapping) -> Tuple[List[str], np.ndarray, int, int]:
    """Convert data to format needed by booster.

    Returns:
        Tuple[List[str], np.ndarray, int, int]: Return column names, the flat data, number of rows, the number of columns
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
