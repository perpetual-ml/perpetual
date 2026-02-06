import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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


def convert_input_array(x, objective, is_target=False, is_int=False) -> np.ndarray:
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

    if is_target and objective == "LogLoss" and len(x_.shape) == 1:
        classes_, x_index = np.unique(x_, return_inverse=True)
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
            categorical_columns = X.select_dtypes(include=["category"]).columns.tolist()
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
            categories.insert(0, "nan")

            inversed = inversed + 1.0

            if len(categories) > max_cat:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {features_[i]} will be treated as numerical since the number of categories ({len(categories)}) exceeds max_cat ({max_cat}) threshold."
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
    import polars.selectors as cs

    features_ = list(X.columns)
    rows, cols = X.shape

    # Determine categorical features
    categorical_features_ = None
    if categorical_features == "auto":
        categorical_columns = X.select(cs.categorical()).columns
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
    import pyarrow as pa

    for i, col_name in enumerate(features_):
        if i in categorical_set:
            # For categorical columns, we need to encode them
            # Use Arrow to get codes and categories without forcing numpy object conversion
            arr = X[col_name].to_arrow()
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()

            if not isinstance(arr, pa.DictionaryArray):
                arr = arr.dictionary_encode()

            # Extract categories (dictionary)
            cats_orig = arr.dictionary.to_pylist()

            # Extract codes (indices)
            indices_raw = arr.indices.to_numpy(zero_copy_only=False)

            # Important: Polars Categorical columns might share a large global dictionary.
            # We must only include categories that are actually present in this column's data
            # to remain consistent with the Pandas/Numpy path (which uses np.unique).
            if arr.null_count < len(indices_raw):
                # Identify which dictionary indices are actually used
                valid_mask = np.ones(len(indices_raw), dtype=bool)
                if arr.null_count > 0 and arr.buffers()[0]:
                    valid_bits = np.frombuffer(arr.buffers()[0], dtype=np.uint8)
                    valid_mask = np.unpackbits(valid_bits, bitorder="little")[
                        : len(indices_raw)
                    ].astype(bool)

                used_indices = np.unique(indices_raw[valid_mask])
                cats_present = []
                for i in used_indices:
                    idx = int(i)
                    if 0 <= idx < len(cats_orig):
                        c = cats_orig[idx]
                        if c is not None and str(c) != "nan":
                            cats_present.append(str(c))
                cats = sorted(cats_present)
            else:
                cats = []

            # Create lookup table for remapping original indices to sorted indices
            # Note: We shift values by +1.0 because 0 in Perpetual is "nan"
            lookup = np.full(len(cats_orig), np.nan, dtype=np.float64)
            cat_to_idx = {cat: i for i, cat in enumerate(cats)}
            for i, cat in enumerate(cats_orig):
                if cat in cat_to_idx:
                    lookup[i] = float(cat_to_idx[cat]) + 1.0

            # Match indices to lookup table safely
            if np.issubdtype(indices_raw.dtype, np.floating):
                mask = ~np.isnan(indices_raw)
                out_values = np.full(len(indices_raw), np.nan, dtype=np.float64)
                out_values[mask] = lookup[indices_raw[mask].astype(np.int64)]
            else:
                out_values = lookup[indices_raw]

            # Handle Nulls (masked values in Arrow)
            # Set them to NaN
            if arr.null_count > 0:
                # Simpler: convert validity bitmap to byte array using numpy unpacking
                row_count = len(out_values)
                if arr.buffers()[0]:
                    valid_bits = np.frombuffer(arr.buffers()[0], dtype=np.uint8)
                    valid_mask = np.unpackbits(valid_bits, bitorder="little")[
                        :row_count
                    ].astype(bool)
                    # mask is 1 where valid, 0 where null.
                    # We want to set nulls (0) to NaN.
                    out_values[~valid_mask] = np.nan

            cats.insert(0, "nan")

            if len(cats) > max_cat:
                cat_to_num.append(i)
                logger.warning(
                    f"Feature {col_name} will be treated as numerical since the number of categories ({len(cats)}) exceeds max_cat ({max_cat}) threshold."
                )

            cat_mapping[col_name] = cats
            columns.append(out_values)
            masks.append(None)  # Categorical encoding handles NaNs
        else:
            # For non-categorical columns, use zero-copy via Arrow
            series = X[col_name]
            # Use Arrow to get validity bitmap and values zero-copy
            arr = series.to_arrow()
            if isinstance(arr, pa.ChunkedArray):
                if arr.num_chunks > 1:
                    arr = arr.combine_chunks()
                else:
                    arr = arr.chunk(0)

            # Check buffers
            buffers = arr.buffers()
            # buffers[0] is validity bitmap
            # buffers[1] is values
            if buffers[0] is None:
                masks.append(None)
            else:
                masks.append(np.frombuffer(buffers[0], dtype=np.uint8))

            # values
            col_array = np.frombuffer(buffers[1], dtype=np.float64)
            columns.append(col_array)

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
            x_enc = np.searchsorted(cats, X_[:, feature_index].astype(str))
            x_enc = x_enc + 1.0
            ind_nan = len(categories)
            x_enc[x_enc == ind_nan] = np.nan
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
    import pyarrow as pa

    for i, col_name in enumerate(features_):
        if cat_mapping and col_name in cat_mapping:
            # For categorical columns, we need to encode them using the existing cat_mapping
            categories = cat_mapping[col_name]

            # Use Arrow for zero-copy extraction
            arr = X[col_name].to_arrow()
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()
            if not isinstance(arr, pa.DictionaryArray):
                arr = arr.dictionary_encode()

            # Input categories
            new_cats = arr.dictionary.to_pylist()

            # Extract codes (indices)
            # We need integers for indexing `lookup`.
            # If indices has nulls, to_numpy() might return floats.
            # We fill nulls with 0 to ensure we get integers, then mask result later.
            filled_indices_arr = arr.indices.fill_null(0)
            new_indices = filled_indices_arr.to_numpy()

            # Build mapping from new_cats indices to old_cats indices
            # old_cats = categories. "nan" is at index 0.
            # We want to map new_cat_idx -> old_cat_idx.
            # If new_cat is "nan", map to 0?
            # Perpetual encoding: "nan" -> NaN (in float), Cat1 -> 1.0, Cat2 -> 2.0.
            # categories list has "nan" at 0.
            # So "A" is at index 1.

            # We need to map `new_indices` to `out_values`.

            # Create a lookup table (array)
            # lookup[new_code] = old_float_code

            lookup = np.full(len(new_cats), np.nan, dtype=np.float64)

            # optimization: map strings to indices for old categories
            old_cat_map = {c: i for i, c in enumerate(categories)}
            # categories[0] is "nan"

            for i, cat in enumerate(new_cats):
                if cat in old_cat_map:
                    idx = old_cat_map[cat]
                    # If idx is 0 ("nan"), we want result to be np.nan?
                    # Previous logic: `inversed[inversed == ind_nan] = np.nan`.
                    # Wait, `inversed` from `searchsorted` was 0-based index into `cats` (without "nan" inside `searchsorted` call?).
                    # Previous logic:
                    # `cats.remove("nan")`
                    # `searchsorted(cats, ...)` -> index into cats (0 to N-1).
                    # `+ 1.0`. So 1 to N.
                    # `categories` has "nan" inserted at 0.
                    # So index 1 corresponds to `categories[1]`.
                    # Logic holds.

                    if categories[idx] == "nan":
                        lookup[i] = np.nan
                    else:
                        lookup[i] = float(idx)
                        # Note: categories has "nan" at 0. "A" at 1.
                        # If `cat` is "A", `idx` is 1. We want 1.0. Correct.
                else:
                    # Unknown category -> NaN?
                    lookup[i] = np.nan

            # Apply lookup
            # Handle out of bounds indices just in case? Arrow indices should be valid.
            # `new_indices` are codes into `new_cats`.

            # Check for nulls in `new_indices` (masked)
            # If null, they map to NaN.

            # `take` style mapping
            # `lookup` has NaN for unknown/nan cats.
            x_enc = lookup[new_indices]

            # Handle array-level nulls
            if arr.null_count > 0:
                if arr.buffers()[0]:
                    valid_bits = np.frombuffer(arr.buffers()[0], dtype=np.uint8)
                    valid_mask = np.unpackbits(valid_bits, bitorder="little")[
                        : len(x_enc)
                    ].astype(bool)
                    x_enc[~valid_mask] = np.nan

            columns.append(x_enc)
            masks.append(None)
        else:
            series = X[col_name]
            arr = series.to_arrow()
            if isinstance(arr, pa.ChunkedArray):
                if arr.num_chunks > 1:
                    arr = arr.combine_chunks()  # Fallback for chunked
                else:
                    arr = arr.chunk(0)
            buffers = arr.buffers()
            if buffers[0] is None:
                masks.append(None)
            else:
                masks.append(np.frombuffer(buffers[0], dtype=np.uint8))
            columns.append(np.frombuffer(buffers[1], dtype=np.float64))

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
