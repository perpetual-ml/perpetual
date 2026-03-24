"""Internal utilities for data conversion and input validation."""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_AVG_CATEGORY_SUPPORT = 64.0
LOW_CARDINALITY_CATEGORY_SUPPORT = 48.0
LOW_CARDINALITY_CATEGORY_LIMIT = 16
MIN_ROWS_FOR_CATEGORY_SUPPORT_CHECK = 128


def _is_polars_string_like_dtype(dtype, pl_module) -> bool:
    base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype
    if base_type in {pl_module.Categorical, pl_module.String}:
        return True

    enum_dtype = getattr(pl_module, "Enum", None)
    return enum_dtype is not None and base_type == enum_dtype


CLASSIFICATION_OBJECTIVES = {
    "LogLoss",
    "BrierLoss",
    "HingeLoss",
    "CrossEntropyLoss",
    "CrossEntropyLambdaLoss",
}


def categorical_support_threshold(
    n_non_missing_categories: int, n_non_missing_rows: int
) -> float:
    if n_non_missing_rows < MIN_ROWS_FOR_CATEGORY_SUPPORT_CHECK:
        return 0.0
    if n_non_missing_categories <= LOW_CARDINALITY_CATEGORY_LIMIT:
        return LOW_CARDINALITY_CATEGORY_SUPPORT
    return MIN_AVG_CATEGORY_SUPPORT


def type_df(df):
    """Return a string tag for the DataFrame library (``'pandas_df'``, ``'polars_df'``, ``'numpy'``, or ``''``)."""
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
    """Return a string tag for the Series library (``'pandas_series'``, ``'polars_series'``, ``'numpy'``, or ``''``)."""
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


def convert_input_array(
    x, objective, is_target=False, is_int=False, is_classification=None
) -> np.ndarray:
    """Convert an array-like input to a flat ``float64`` (or ``uint64``) NumPy array.

    Returns
    -------
    tuple of (np.ndarray, list)
        The converted array and the detected class labels (empty list if not classification).
    """
    classes_ = []

    if type(x).__module__.split(".")[0] == "numpy":
        if len(x.shape) == 2:
            classes_, x_, *_ = convert_input_frame(x, None, 1000)
        else:
            x_ = x
    elif type_series(x) == "pandas_series":
        x_ = x.to_numpy()
    elif type_series(x) == "polars_series":
        x_ = x.to_numpy(allow_copy=False)
    elif type_df(x) == "polars_df" or type_df(x) == "pandas_df":
        classes_, x_, *_ = convert_input_frame(x, None, 1000)
    else:
        x_ = x.to_numpy()

    # Determine if we should perform class detection
    do_class_detection = False
    if is_target and len(x_.shape) == 1:
        if is_classification is True:
            do_class_detection = True
        elif is_classification is False:
            do_class_detection = False
        else:
            # Fallback to heuristic if not explicitly set
            do_class_detection = objective in CLASSIFICATION_OBJECTIVES

    if do_class_detection:
        classes_, x_index = np.unique(x_, return_inverse=True)

        if len(classes_) > 1:
            if len(classes_) > 2:
                x_ = np.eye(len(classes_))[x_index]
            else:
                x_ = x_index.astype("float64")

    if is_int and not np.issubdtype(x_.dtype, "uint64"):
        x_ = x_.astype(dtype="uint64", copy=False)

    if not is_int and not np.issubdtype(x_.dtype, "float64"):
        x_ = x_.astype(dtype="float64", copy=False)

    if len(x_.shape) == 2:
        x_ = x_.ravel(order="F")

    return x_, classes_


def convert_input_frame(
    X,
    categorical_features,
    max_cat,
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
            categorical_columns = X.select_dtypes(
                include=["category", "object", "string"]
            ).columns.tolist()
            categorical_features_ = [
                features_.index(c) for c in categorical_columns
            ] or None
    elif type_df(X) == "numpy":
        X_ = X
        features_ = list(map(str, range(X_.shape[1])))
    else:
        raise ValueError(f"Object type {type(X)} is not supported.")

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
    cat_to_num = []
    if categorical_features_:
        for i in categorical_features_:
            categories, inversed = np.unique(X_[:, i].astype(str), return_inverse=True)

            categories = list(categories)
            if "nan" in categories:
                categories.remove("nan")
            n_non_missing_categories = len(categories)
            n_non_missing_rows = np.count_nonzero(
                ~np.isnan(X_[:, i].astype(dtype="float64", copy=False))
                if np.issubdtype(X_[:, i].dtype, np.number)
                else X_[:, i].astype(str) != "nan"
            )
            avg_category_support = (
                n_non_missing_rows / n_non_missing_categories
                if n_non_missing_categories > 0
                else float("inf")
            )
            min_support_threshold = categorical_support_threshold(
                n_non_missing_categories, n_non_missing_rows
            )
            categories.insert(0, "nan")

            inversed = inversed + 1.0

            if n_non_missing_categories > max_cat:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {features_[i]} will be treated as numerical since the number of non-missing categories ({n_non_missing_categories}) exceeds max_cat ({max_cat}) threshold."
                )
            elif avg_category_support < min_support_threshold:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {features_[i]} will be treated as numerical since its average support per non-missing category ({avg_category_support:.1f}) is below the native categorical stability threshold ({min_support_threshold:.1f})."
                )

            feature_name = features_[i]
            cat_mapping[feature_name] = categories
            ind_nan = len(categories)
            inversed[inversed == ind_nan] = np.nan
            X_[:, i] = inversed

        categorical_features_ = [
            x for x in categorical_features_ if x not in cat_to_num
        ]

        logger.info(f"Categorical features: {categorical_features_}")
        logger.info(f"Mapping of categories: {cat_mapping}")

    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape

    if isinstance(categorical_features_, list):
        categorical_features_ = set(categorical_features_)

    return features_, flat_data, rows, cols, categorical_features_, cat_mapping


def convert_input_frame_columnar(
    X, categorical_features, max_cat
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[Optional[np.ndarray]],
    int,
    int,
    Optional[set],
    dict,
]:
    """Convert Polars DataFrame to columnar format for zero-copy transfer.

    Returns list of column arrays and list of validity masks.
    """
    import polars as pl

    features_ = list(X.columns)
    rows, cols = X.shape

    # Determine categorical features
    categorical_features_ = None
    if categorical_features == "auto":
        categorical_columns = [
            name
            for name, dtype in X.schema.items()
            if _is_polars_string_like_dtype(dtype, pl)
        ]
        categorical_features_ = [
            features_.index(c) for c in categorical_columns
        ] or None
    elif (
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

    cat_mapping = {}
    cat_to_num = []
    categorical_set = set(categorical_features_) if categorical_features_ else set()

    # Convert each column to numpy array
    columns = []
    masks = []
    for i, col_name in enumerate(features_):
        if i in categorical_set:
            # For categorical columns, cast to String first to bypass
            # Polars' global string cache which can make the Arrow
            # DictionaryArray dictionary unreliable (it may contain
            # categories from ALL categorical columns, not just this one).
            str_series = X[col_name].cast(pl.String)
            str_np = str_series.to_numpy(allow_copy=True)
            null_mask = str_series.is_null().to_numpy()

            # Get sorted unique non-null categories (consistent with Pandas np.unique path)
            valid_strs = str_np[~null_mask]
            if len(valid_strs) > 0:
                cats = sorted(set(valid_strs))
                cats = [c for c in cats if c != "nan"]
            else:
                cats = []
            n_non_missing_categories = len(cats)
            n_non_missing_rows = int((~null_mask).sum())
            avg_category_support = (
                n_non_missing_rows / n_non_missing_categories
                if n_non_missing_categories > 0
                else float("inf")
            )
            min_support_threshold = categorical_support_threshold(
                n_non_missing_categories, n_non_missing_rows
            )

            cats.insert(0, "nan")

            if n_non_missing_categories > max_cat:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {col_name} will be treated as numerical since the number of non-missing categories ({n_non_missing_categories}) exceeds max_cat ({max_cat}) threshold."
                )
            elif avg_category_support < min_support_threshold:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {col_name} will be treated as numerical since its average support per non-missing category ({avg_category_support:.1f}) is below the native categorical stability threshold ({min_support_threshold:.1f})."
                )

            cat_mapping[col_name] = cats

            # Encode: use searchsorted on sorted cats (without "nan" at 0) for vectorized mapping
            cats_no_nan = cats[1:]  # remove "nan" prefix
            cats_arr = np.array(cats_no_nan)
            out_values = np.full(rows, np.nan, dtype=np.float64)
            valid_indices = ~null_mask
            if valid_indices.any():
                positions = np.searchsorted(cats_arr, valid_strs)
                encoded = positions.astype(np.float64) + 1.0
                exact_match = np.zeros(len(valid_strs), dtype=bool)
                in_range = positions < len(cats_arr)
                if in_range.any():
                    exact_match[in_range] = (
                        cats_arr[positions[in_range]] == valid_strs[in_range]
                    )
                encoded[~exact_match] = np.nan
                out_values[valid_indices] = encoded

            columns.append(out_values)
            masks.append(None)  # Categorical encoding handles NaNs
        else:
            series = X[col_name].cast(pl.Float64)
            columns.append(
                series.to_numpy(allow_copy=True).astype(np.float64, copy=False)
            )
            masks.append(None)

    if categorical_features_:
        categorical_features_ = [
            x for x in categorical_features_ if x not in cat_to_num
        ]
        logger.info(f"Categorical features: {categorical_features_}")
        logger.info(f"Mapping of categories: {cat_mapping}")

    if isinstance(categorical_features_, list):
        categorical_features_ = set(categorical_features_)

    return features_, columns, masks, rows, cols, categorical_features_, cat_mapping


def transform_input_frame(X, cat_mapping) -> Tuple[List[str], np.ndarray, int, int]:
    """Convert data to format needed by booster.

    Returns:
        Tuple[List[str], np.ndarray, int, int]: Return column names, the flat data, number of rows, the number of columns
    """
    if type_df(X) == "pandas_df":
        X_ = X.to_numpy()
        features_ = X.columns.to_list()
    elif type_df(X) == "numpy":
        X_ = X
        features_ = list(map(str, range(X_.shape[1])))
    else:
        raise ValueError(f"Object type {type(X)} is not supported.")

    if cat_mapping:
        for feature_name, categories in cat_mapping.items():
            feature_index = features_.index(feature_name)
            cats = categories.copy()
            cats.remove("nan")
            values = X_[:, feature_index].astype(str)
            positions = np.searchsorted(cats, values)
            x_enc = positions.astype(np.float64) + 1.0
            exact_match = np.zeros(len(values), dtype=bool)
            in_range = positions < len(cats)
            if in_range.any():
                cat_array = np.asarray(cats, dtype=object)
                exact_match[in_range] = (
                    cat_array[positions[in_range]] == values[in_range]
                )
            x_enc[~exact_match] = np.nan
            X_[:, feature_index] = x_enc

    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape

    return features_, flat_data, rows, cols


def transform_input_frame_columnar(
    X, cat_mapping
) -> Tuple[List[str], List[np.ndarray], List[Optional[np.ndarray]], int, int]:
    """Convert Polars DataFrame to columnar format for zero-copy prediction.

    Returns list of column arrays and masks instead of flattened data, avoiding copies.
    """
    features_ = list(X.columns)
    rows, cols = X.shape

    columns = []
    masks = []
    import polars as pl

    for i, col_name in enumerate(features_):
        if cat_mapping and col_name in cat_mapping:
            # For categorical columns, encode using the existing cat_mapping.
            # Cast to String first to bypass Polars' global string cache
            # which can make Arrow DictionaryArray dictionaries unreliable.
            categories = cat_mapping[col_name]
            cats = categories.copy()
            cats.remove("nan")

            str_series = X[col_name].cast(pl.String)
            str_np = str_series.to_numpy(allow_copy=True)
            null_mask = str_series.is_null().to_numpy()

            x_enc = np.full(rows, np.nan, dtype=np.float64)
            valid_indices = ~null_mask
            if valid_indices.any():
                valid_strs = str_np[valid_indices]
                positions = np.searchsorted(cats, valid_strs)
                encoded = positions.astype(np.float64) + 1.0
                exact_match = np.zeros(len(valid_strs), dtype=bool)
                in_range = positions < len(cats)
                if in_range.any():
                    cat_array = np.asarray(cats, dtype=object)
                    exact_match[in_range] = (
                        cat_array[positions[in_range]] == valid_strs[in_range]
                    )
                encoded[~exact_match] = np.nan
                x_enc[valid_indices] = encoded

            columns.append(x_enc)
            masks.append(None)
        else:
            series = X[col_name].cast(pl.Float64)
            columns.append(
                series.to_numpy(allow_copy=True).astype(np.float64, copy=False)
            )
            masks.append(None)

    return features_, columns, masks, rows, cols


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
