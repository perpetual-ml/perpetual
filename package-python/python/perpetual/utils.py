"""Internal utilities for data conversion and input validation."""

import logging
import zlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_AVG_CATEGORY_SUPPORT = 64.0
LOW_CARDINALITY_CATEGORY_SUPPORT = 48.0
LOW_CARDINALITY_CATEGORY_LIMIT = 16
MIN_ROWS_FOR_CATEGORY_SUPPORT_CHECK = 128
MAX_POOLED_HEAD_CATEGORIES = 32
MAX_POOLED_CATEGORY_BUCKETS = 64
MIN_POOLED_TAIL_BUCKETS = 16
TARGET_ENCODING_FOLDS = 5
TARGET_ENCODING_PRIOR_FLOOR = 8.0
MIN_NATIVE_TARGET_MEAN_CATEGORIES = 10
MIN_NATIVE_TARGET_MEAN_ROWS = 2048
WIDE_BINARY_TARGET_MEAN_MIN_COLS = 128
MIN_WIDE_BINARY_TARGET_MEAN_FEATURES = 8
MAX_WIDE_BINARY_TARGET_MEAN_FEATURES = 16
MAX_PAIRWISE_CATEGORY_FEATURES = 3
MAX_PAIRWISE_SOURCE_FEATURES = 6
MIN_PAIRWISE_TARGET_ROWS = 2048
MAX_PAIRWISE_TOTAL_FEATURES = 64
PAIRWISE_CATEGORY_SEPARATOR = "\x1f"


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


def _stable_category_hash(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def _pooled_category_target_count(
    n_non_missing_categories: int,
    n_non_missing_rows: int,
    max_cat: int,
    min_support_threshold: float,
) -> int:
    target_count = min(max(max_cat, 1), max(n_non_missing_categories, 1))
    target_count = min(target_count, MAX_POOLED_CATEGORY_BUCKETS)
    if min_support_threshold > 0.0:
        support_limited_count = max(1, int(n_non_missing_rows // min_support_threshold))
        target_count = min(target_count, support_limited_count)
    return max(1, target_count)


def _pooled_tail_bucket_floor(pooled_category_count: int) -> int:
    if pooled_category_count <= 4:
        return 1
    if pooled_category_count <= 16:
        return max(1, pooled_category_count // 4)
    return min(MIN_POOLED_TAIL_BUCKETS, max(4, pooled_category_count // 4))


def _exact_category_positions(cats, values):
    if len(cats) == 0:
        return np.zeros(len(values), dtype=np.int64), np.zeros(len(values), dtype=bool)

    positions = np.searchsorted(cats, values)
    exact_match = np.zeros(len(values), dtype=bool)
    in_range = positions < len(cats)
    if in_range.any():
        exact_match[in_range] = cats[positions[in_range]] == values[in_range]
    return positions, exact_match


def _should_add_native_target_mean(
    n_non_missing_categories: int,
    n_non_missing_rows: int,
    avg_category_support: float,
    min_support_threshold: float,
    target_task: Optional[str],
) -> bool:
    return (
        target_task in {"binary", "regression"}
        and n_non_missing_rows >= MIN_NATIVE_TARGET_MEAN_ROWS
        and n_non_missing_categories >= MIN_NATIVE_TARGET_MEAN_CATEGORIES
        and avg_category_support >= min_support_threshold
    )


def _mapping_mode(mapping) -> str:
    if isinstance(mapping, dict):
        return mapping.get("mode", "native")
    return "native"


def _wide_binary_target_mean_feature_budget(total_feature_count: int) -> int:
    return int(
        np.clip(
            np.rint(np.sqrt(max(total_feature_count, 1))),
            MIN_WIDE_BINARY_TARGET_MEAN_FEATURES,
            MAX_WIDE_BINARY_TARGET_MEAN_FEATURES,
        )
    )


def _target_mean_candidate_rank(
    signal: float,
    mapping,
    n_non_missing_categories: int,
    n_non_missing_rows: int,
) -> Tuple[float, int, int, int]:
    return (
        float(signal),
        int(isinstance(mapping, dict) and mapping.get("mode") == "pooled"),
        int(n_non_missing_categories),
        int(n_non_missing_rows),
    )


def _finalize_target_mean_candidates(
    target_mean_candidates: List[dict],
    total_feature_count: int,
    target_task: Optional[str],
) -> Tuple[List[str], List[np.ndarray]]:
    if not target_mean_candidates:
        return [], []

    retained_sources = None
    if (
        target_task == "binary"
        and total_feature_count >= WIDE_BINARY_TARGET_MEAN_MIN_COLS
    ):
        budget = _wide_binary_target_mean_feature_budget(total_feature_count)
        if len(target_mean_candidates) > budget:
            ranked_candidates = sorted(
                target_mean_candidates,
                key=lambda candidate: (
                    -candidate["rank"][0],
                    -candidate["rank"][1],
                    -candidate["rank"][2],
                    -candidate["rank"][3],
                    candidate["source_feature"],
                ),
            )
            retained_sources = {
                candidate["source_feature"] for candidate in ranked_candidates[:budget]
            }

    derived_features = []
    derived_columns = []
    for candidate in target_mean_candidates:
        if (
            retained_sources is not None
            and candidate["source_feature"] not in retained_sources
        ):
            candidate["mapping"].pop("target_mean", None)
            continue

        derived_features.append(candidate["derived_name"])
        derived_columns.append(candidate["values"])

    return derived_features, derived_columns


def _target_signal_score(
    valid_values: np.ndarray,
    target_values: Optional[np.ndarray],
    target_task: Optional[str],
    min_support_threshold: float,
) -> float:
    if (
        target_values is None
        or target_task not in {"binary", "regression"}
        or len(valid_values) <= 1
    ):
        return 0.0

    unique_values, inverse, counts = np.unique(
        valid_values, return_inverse=True, return_counts=True
    )
    if len(unique_values) <= 1:
        return 0.0

    target_array = np.asarray(target_values, dtype=np.float64)
    if target_array.ndim != 1 or len(target_array) != len(valid_values):
        return 0.0

    global_mean = float(target_array.mean())
    prior_strength = max(float(min_support_threshold), TARGET_ENCODING_PRIOR_FLOOR)
    category_sums = np.bincount(
        inverse, weights=target_array, minlength=len(unique_values)
    )
    smoothed_mean = (category_sums + prior_strength * global_mean) / (
        counts + prior_strength
    )

    if target_task == "binary":
        baseline_scale = np.sqrt(max(global_mean * (1.0 - global_mean), 1e-3))
    else:
        baseline_scale = max(float(target_array.std()), 1e-3)

    standardized_lift = np.abs(smoothed_mean - global_mean) / baseline_scale
    confidence = np.sqrt(counts.astype(np.float64)) / np.sqrt(
        counts.astype(np.float64) + prior_strength
    )
    return float(
        np.average(standardized_lift * confidence, weights=counts.astype(np.float64))
    )


def _build_categorical_mapping(
    feature_name: str,
    valid_values: np.ndarray,
    max_cat: int,
    min_support_threshold: float,
    target_values: Optional[np.ndarray] = None,
    target_task: Optional[str] = None,
):
    unique_values, inverse, counts = np.unique(
        valid_values, return_inverse=True, return_counts=True
    )
    n_non_missing_categories = len(unique_values)
    n_non_missing_rows = len(valid_values)
    avg_category_support = (
        n_non_missing_rows / n_non_missing_categories
        if n_non_missing_categories > 0
        else float("inf")
    )

    if (
        n_non_missing_categories <= max_cat
        and avg_category_support >= min_support_threshold
    ):
        if _should_add_native_target_mean(
            n_non_missing_categories,
            n_non_missing_rows,
            avg_category_support,
            min_support_threshold,
            target_task,
        ):
            return {
                "mode": "native",
                "categories": ["nan", *unique_values.tolist()],
            }, None
        return ["nan", *unique_values.tolist()], None

    pooled_category_count = _pooled_category_target_count(
        n_non_missing_categories,
        n_non_missing_rows,
        max_cat,
        min_support_threshold,
    )
    target_scores = None
    if (
        target_values is not None
        and target_task in {"binary", "regression"}
        and len(target_values) == len(valid_values)
    ):
        target_array = np.asarray(target_values, dtype=np.float64)
        if target_array.ndim == 1:
            prior_strength = max(min_support_threshold, 8.0)
            global_mean = float(target_array.mean())
            category_sums = np.bincount(
                inverse, weights=target_array, minlength=n_non_missing_categories
            )
            smoothed_mean = (category_sums + prior_strength * global_mean) / (
                counts + prior_strength
            )

            if target_task == "binary":
                baseline_scale = np.sqrt(max(global_mean * (1.0 - global_mean), 1e-3))
            else:
                baseline_scale = max(float(target_array.std()), 1e-3)

            target_scores = (
                np.abs(smoothed_mean - global_mean) / baseline_scale
            ) * np.sqrt(counts.astype(np.float64)) + 0.15 * np.log1p(
                counts.astype(np.float64)
            )

    if target_scores is not None:
        ranked_categories = sorted(
            zip(unique_values.tolist(), counts.tolist(), target_scores.tolist()),
            key=lambda item: (-item[2], -item[1], item[0]),
        )
    else:
        ranked_categories = sorted(
            zip(unique_values.tolist(), counts.tolist()),
            key=lambda item: (-item[1], item[0]),
        )
    dedicated_limit = min(
        MAX_POOLED_HEAD_CATEGORIES,
        max(
            pooled_category_count - _pooled_tail_bucket_floor(pooled_category_count),
            1,
        ),
    )
    dedicated_categories = sorted(
        category for category, _count, *_ in ranked_categories[:dedicated_limit]
    )
    tail_bucket_count = max(1, pooled_category_count - len(dedicated_categories))
    mapping = {
        "mode": "pooled",
        "categories": ["nan", *dedicated_categories],
        "tail_bucket_count": tail_bucket_count,
    }

    pooling_reasons = []
    if n_non_missing_categories > max_cat:
        pooling_reasons.append(
            f"the number of non-missing categories ({n_non_missing_categories}) exceeds max_cat ({max_cat})"
        )
    if avg_category_support < min_support_threshold:
        pooling_reasons.append(
            "its average support per non-missing category "
            f"({avg_category_support:.1f}) is below the native categorical stability threshold ({min_support_threshold:.1f})"
        )

    log_message = (
        f"Feature {feature_name} will be pooled into {pooled_category_count} categorical buckets "
        f"({len(dedicated_categories)} dedicated + {tail_bucket_count} hashed tail) because "
        + " and ".join(pooling_reasons)
        + "."
    )
    return mapping, log_message


def _encode_categorical_values(
    values: np.ndarray, valid_mask: np.ndarray, mapping
) -> np.ndarray:
    encoded = np.full(len(values), np.nan, dtype=np.float64)
    if not valid_mask.any():
        return encoded

    valid_values = values[valid_mask]
    if isinstance(mapping, dict) and mapping.get("mode") == "pooled":
        dedicated_categories = np.asarray(mapping["categories"][1:], dtype=object)
        encoded_valid = np.empty(len(valid_values), dtype=np.float64)
        positions, exact_match = _exact_category_positions(
            dedicated_categories, valid_values
        )
        encoded_valid[exact_match] = positions[exact_match].astype(np.float64) + 1.0

        tail_mask = ~exact_match
        if tail_mask.any():
            tail_bucket_count = int(mapping["tail_bucket_count"])
            tail_bucket_offset = len(mapping["categories"])
            encoded_valid[tail_mask] = np.fromiter(
                (
                    tail_bucket_offset
                    + (_stable_category_hash(value) % tail_bucket_count)
                    for value in valid_values[tail_mask]
                ),
                dtype=np.float64,
                count=int(tail_mask.sum()),
            )
        encoded[valid_mask] = encoded_valid
        return encoded

    categories = np.asarray(
        mapping["categories"][1:] if isinstance(mapping, dict) else mapping[1:],
        dtype=object,
    )
    positions, exact_match = _exact_category_positions(categories, valid_values)
    encoded_valid = positions.astype(np.float64) + 1.0
    encoded_valid[~exact_match] = np.nan
    encoded[valid_mask] = encoded_valid
    return encoded


def _categorical_bucket_indices(
    values: np.ndarray, valid_mask: np.ndarray, mapping
) -> np.ndarray:
    bucket_indices = np.zeros(len(values), dtype=np.int64)
    if not valid_mask.any():
        return bucket_indices

    valid_values = values[valid_mask]
    if isinstance(mapping, dict) and mapping.get("mode") == "pooled":
        dedicated_categories = np.asarray(mapping["categories"][1:], dtype=object)
        encoded_valid = np.empty(len(valid_values), dtype=np.int64)
        positions, exact_match = _exact_category_positions(
            dedicated_categories, valid_values
        )
        encoded_valid[exact_match] = positions[exact_match] + 1

        tail_mask = ~exact_match
        if tail_mask.any():
            tail_bucket_count = int(mapping["tail_bucket_count"])
            tail_bucket_offset = len(mapping["categories"])
            encoded_valid[tail_mask] = np.fromiter(
                (
                    tail_bucket_offset
                    + (_stable_category_hash(value) % tail_bucket_count)
                    for value in valid_values[tail_mask]
                ),
                dtype=np.int64,
                count=int(tail_mask.sum()),
            )
        bucket_indices[valid_mask] = encoded_valid
        return bucket_indices

    categories = np.asarray(
        mapping["categories"][1:] if isinstance(mapping, dict) else mapping[1:],
        dtype=object,
    )
    positions, exact_match = _exact_category_positions(categories, valid_values)
    encoded_valid = np.zeros(len(valid_values), dtype=np.int64)
    encoded_valid[exact_match] = positions[exact_match] + 1
    bucket_indices[valid_mask] = encoded_valid
    return bucket_indices


def _pooled_target_mean_name(feature_name: str) -> str:
    return f"{feature_name}__target_mean"


def _pairwise_categorical_feature_name(
    left_feature_name: str, right_feature_name: str
) -> str:
    return f"{left_feature_name}__x__{right_feature_name}"


def _combine_categorical_pair_values(
    left_values: np.ndarray, right_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.full(len(left_values), "nan", dtype=object)
    valid_mask = (left_values != "nan") & (right_values != "nan")
    if valid_mask.any():
        combined[valid_mask] = np.fromiter(
            (
                f"{left_value}{PAIRWISE_CATEGORY_SEPARATOR}{right_value}"
                for left_value, right_value in zip(
                    left_values[valid_mask], right_values[valid_mask]
                )
            ),
            dtype=object,
            count=int(valid_mask.sum()),
        )
    return combined, valid_mask


def _build_pairwise_categorical_features(
    feature_payloads: List[dict],
    total_feature_count: int,
    max_cat: int,
    target_values: Optional[np.ndarray],
    target_task: Optional[str],
) -> List[dict]:
    if (
        target_values is None
        or target_task not in {"binary", "regression"}
        or len(target_values) < MIN_PAIRWISE_TARGET_ROWS
        or total_feature_count > MAX_PAIRWISE_TOTAL_FEATURES
        or len(feature_payloads) < 2
    ):
        return []

    eligible_payloads = [
        payload
        for payload in feature_payloads
        if payload["n_non_missing_rows"] >= MIN_PAIRWISE_TARGET_ROWS
        and payload["n_non_missing_categories"] >= 2
    ]
    if len(eligible_payloads) < 2:
        return []

    candidate_sources = sorted(
        eligible_payloads,
        key=lambda payload: (
            -payload["signal"],
            -payload["n_non_missing_categories"],
            payload["index"],
        ),
    )[:MAX_PAIRWISE_SOURCE_FEATURES]

    pair_candidates = []
    for left_idx in range(len(candidate_sources)):
        for right_idx in range(left_idx + 1, len(candidate_sources)):
            left_payload = candidate_sources[left_idx]
            right_payload = candidate_sources[right_idx]
            pair_priority = (left_payload["signal"] + 0.05) * (
                right_payload["signal"] + 0.05
            )
            if left_payload["mode"] == "pooled" or right_payload["mode"] == "pooled":
                pair_priority *= 1.15
            pair_priority *= np.log1p(
                min(
                    left_payload["n_non_missing_categories"],
                    right_payload["n_non_missing_categories"],
                )
            )
            pair_candidates.append((pair_priority, left_payload, right_payload))

    pair_features = []
    for _priority, left_payload, right_payload in sorted(
        pair_candidates,
        key=lambda candidate: (
            -candidate[0],
            candidate[1]["index"],
            candidate[2]["index"],
        ),
    )[:MAX_PAIRWISE_CATEGORY_FEATURES]:
        pair_name = _pairwise_categorical_feature_name(
            left_payload["feature_name"], right_payload["feature_name"]
        )
        pair_values, valid_mask = _combine_categorical_pair_values(
            left_payload["values"], right_payload["values"]
        )
        n_non_missing_rows = int(valid_mask.sum())
        if n_non_missing_rows < MIN_PAIRWISE_TARGET_ROWS:
            continue

        n_non_missing_categories = len(np.unique(pair_values[valid_mask]))
        if n_non_missing_categories < 2:
            continue

        min_support_threshold = categorical_support_threshold(
            n_non_missing_categories, n_non_missing_rows
        )
        pair_signal = _target_signal_score(
            pair_values[valid_mask],
            target_values[valid_mask],
            target_task,
            min_support_threshold,
        )
        mapping, log_message = _build_categorical_mapping(
            pair_name,
            pair_values[valid_mask],
            min(max_cat, MAX_POOLED_CATEGORY_BUCKETS),
            min_support_threshold,
            target_values=target_values[valid_mask],
            target_task=target_task,
        )
        if not isinstance(mapping, dict):
            mapping = {"mode": "native", "categories": mapping}
        mapping["source_features"] = [
            left_payload["feature_name"],
            right_payload["feature_name"],
        ]

        descriptor, target_mean_values = _build_target_mean_descriptor(
            pair_name,
            pair_values,
            valid_mask,
            mapping,
            target_values,
            target_task,
            min_support_threshold,
        )
        if descriptor is not None:
            mapping["target_mean"] = descriptor

        pair_features.append(
            {
                "feature_name": pair_name,
                "mapping": mapping,
                "encoded_values": _encode_categorical_values(
                    pair_values, valid_mask, mapping
                ),
                "target_mean_values": target_mean_values,
                "target_mean_name": (
                    None if descriptor is None else descriptor["derived_name"]
                ),
                "target_mean_descriptor": descriptor,
                "signal": pair_signal,
                "n_non_missing_rows": n_non_missing_rows,
                "n_non_missing_categories": n_non_missing_categories,
                "log_message": log_message,
            }
        )

    return pair_features


def _target_encoding_fold_ids(
    target_values: np.ndarray, target_task: str
) -> np.ndarray:
    n_rows = len(target_values)
    if n_rows <= 1:
        return np.zeros(n_rows, dtype=np.int64)

    n_folds = min(TARGET_ENCODING_FOLDS, n_rows)
    rng = np.random.default_rng(0)
    fold_ids = np.empty(n_rows, dtype=np.int64)

    if target_task == "binary":
        for target_class in np.unique(target_values):
            class_index = np.flatnonzero(target_values == target_class)
            shuffled = rng.permutation(class_index)
            fold_ids[shuffled] = np.arange(len(shuffled), dtype=np.int64) % n_folds
        return fold_ids

    shuffled = rng.permutation(n_rows)
    fold_ids[shuffled] = np.arange(n_rows, dtype=np.int64) % n_folds
    return fold_ids


def _build_target_mean_descriptor(
    feature_name: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    mapping,
    target_values: Optional[np.ndarray],
    target_task: Optional[str],
    min_support_threshold: float,
):
    if target_values is None or target_task not in {"binary", "regression"}:
        return None, None

    if not isinstance(mapping, dict):
        return None, None

    target_array = np.asarray(target_values, dtype=np.float64)
    if target_array.ndim != 1 or len(target_array) != len(values):
        return None, None

    mode = mapping.get("mode")
    if mode == "pooled":
        if (
            target_task == "binary"
            and int(mapping.get("tail_bucket_count", 0)) < MIN_POOLED_TAIL_BUCKETS
        ):
            return None, None
        bucket_count = len(mapping["categories"]) + int(mapping["tail_bucket_count"])
    elif mode == "native":
        bucket_count = len(mapping["categories"])
    else:
        return None, None

    bucket_indices = _categorical_bucket_indices(values, valid_mask, mapping)
    global_mean = float(target_array.mean())
    prior_strength = max(float(min_support_threshold), TARGET_ENCODING_PRIOR_FLOOR)

    full_sums = np.bincount(
        bucket_indices, weights=target_array, minlength=bucket_count
    )
    full_counts = np.bincount(bucket_indices, minlength=bucket_count)
    full_means = (full_sums + prior_strength * global_mean) / (
        full_counts + prior_strength
    )

    train_encoded = np.empty(len(target_array), dtype=np.float64)
    if len(target_array) <= 1:
        train_encoded[:] = full_means[bucket_indices]
    else:
        fold_ids = _target_encoding_fold_ids(target_array, target_task)
        n_folds = int(fold_ids.max()) + 1
        for fold_idx in range(n_folds):
            train_mask = fold_ids != fold_idx
            val_mask = ~train_mask
            fold_sums = np.bincount(
                bucket_indices[train_mask],
                weights=target_array[train_mask],
                minlength=bucket_count,
            )
            fold_counts = np.bincount(
                bucket_indices[train_mask], minlength=bucket_count
            )
            fold_means = (fold_sums + prior_strength * global_mean) / (
                fold_counts + prior_strength
            )
            train_encoded[val_mask] = fold_means[bucket_indices[val_mask]]

    descriptor = {
        "derived_name": _pooled_target_mean_name(feature_name),
        "bucket_means": full_means.tolist(),
        "global_mean": global_mean,
    }
    return descriptor, train_encoded


def _apply_target_mean_descriptor(
    values: np.ndarray, valid_mask: np.ndarray, mapping, descriptor: dict
) -> np.ndarray:
    bucket_indices = _categorical_bucket_indices(values, valid_mask, mapping)
    bucket_means = np.asarray(descriptor["bucket_means"], dtype=np.float64)
    clipped_indices = np.clip(bucket_indices, 0, len(bucket_means) - 1)
    encoded = bucket_means[clipped_indices]
    encoded[~np.isfinite(encoded)] = float(descriptor["global_mean"])
    return encoded


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
    target_values=None,
    target_task=None,
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

    cat_mapping = {}  # key: feature_name, value: ordered category names or pooled descriptor
    target_array = None if target_values is None else np.asarray(target_values)
    appended_categorical_features = []
    appended_categorical_columns = []
    target_mean_candidates = []
    feature_payloads = []
    if categorical_features_:
        for i in categorical_features_:
            values = X_[:, i].astype(str)
            n_non_missing_rows = np.count_nonzero(
                ~np.isnan(X_[:, i].astype(dtype="float64", copy=False))
                if np.issubdtype(X_[:, i].dtype, np.number)
                else values != "nan"
            )
            n_non_missing_categories = len(np.unique(values[values != "nan"]))
            min_support_threshold = categorical_support_threshold(
                n_non_missing_categories, n_non_missing_rows
            )
            valid_mask = values != "nan"
            mapping, log_message = _build_categorical_mapping(
                features_[i],
                values[valid_mask],
                max_cat,
                min_support_threshold,
                target_values=(
                    None if target_array is None else target_array[valid_mask]
                ),
                target_task=target_task,
            )

            feature_name = features_[i]
            feature_payloads.append(
                {
                    "feature_name": feature_name,
                    "index": i,
                    "values": values,
                    "mode": _mapping_mode(mapping),
                    "signal": _target_signal_score(
                        values[valid_mask],
                        None if target_array is None else target_array[valid_mask],
                        target_task,
                        min_support_threshold,
                    ),
                    "n_non_missing_rows": n_non_missing_rows,
                    "n_non_missing_categories": n_non_missing_categories,
                }
            )
            descriptor, target_mean_values = _build_target_mean_descriptor(
                feature_name,
                values,
                valid_mask,
                mapping,
                target_array,
                target_task,
                min_support_threshold,
            )
            if descriptor is not None:
                mapping["target_mean"] = descriptor
                target_mean_candidates.append(
                    {
                        "source_feature": feature_name,
                        "derived_name": descriptor["derived_name"],
                        "values": target_mean_values,
                        "mapping": mapping,
                        "rank": _target_mean_candidate_rank(
                            feature_payloads[-1]["signal"],
                            mapping,
                            n_non_missing_categories,
                            n_non_missing_rows,
                        ),
                    }
                )
            cat_mapping[feature_name] = mapping
            if log_message is not None:
                logger.warning(log_message)
            X_[:, i] = _encode_categorical_values(values, valid_mask, mapping)

        pair_feature_start = len(features_)
        for pair_feature in _build_pairwise_categorical_features(
            feature_payloads,
            len(features_),
            max_cat,
            target_array,
            target_task,
        ):
            feature_name = pair_feature["feature_name"]
            cat_mapping[feature_name] = pair_feature["mapping"]
            appended_categorical_features.append(feature_name)
            appended_categorical_columns.append(pair_feature["encoded_values"])
            if pair_feature["target_mean_name"] is not None:
                target_mean_candidates.append(
                    {
                        "source_feature": feature_name,
                        "derived_name": pair_feature["target_mean_name"],
                        "values": pair_feature["target_mean_values"],
                        "mapping": pair_feature["mapping"],
                        "rank": _target_mean_candidate_rank(
                            pair_feature["signal"],
                            pair_feature["mapping"],
                            pair_feature["n_non_missing_categories"],
                            pair_feature["n_non_missing_rows"],
                        ),
                    }
                )
            if pair_feature["log_message"] is not None:
                logger.warning(pair_feature["log_message"])

        total_feature_count = len(features_) + len(appended_categorical_features)
        derived_features, derived_columns = _finalize_target_mean_candidates(
            target_mean_candidates,
            total_feature_count,
            target_task,
        )

        if appended_categorical_columns:
            X_ = np.column_stack([X_, *appended_categorical_columns])
            features_.extend(appended_categorical_features)
            categorical_features_.extend(
                range(
                    pair_feature_start,
                    pair_feature_start + len(appended_categorical_features),
                )
            )

        logger.info(f"Categorical features: {categorical_features_}")
        logger.info(f"Mapping of categories: {cat_mapping}")
    else:
        derived_features, derived_columns = _finalize_target_mean_candidates(
            target_mean_candidates,
            len(features_),
            target_task,
        )

    if derived_columns:
        X_ = np.column_stack([X_, *derived_columns])
        features_.extend(derived_features)

    if not np.issubdtype(X_.dtype, "float64"):
        X_ = X_.astype(dtype="float64", copy=False)
    flat_data = X_.ravel(order="F")
    rows, cols = X_.shape

    if isinstance(categorical_features_, list):
        categorical_features_ = set(categorical_features_)

    return features_, flat_data, rows, cols, categorical_features_, cat_mapping


def convert_input_frame_columnar(
    X,
    categorical_features,
    max_cat,
    target_values=None,
    target_task=None,
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
    categorical_set = set(categorical_features_) if categorical_features_ else set()
    target_array = None if target_values is None else np.asarray(target_values)
    appended_categorical_features = []
    appended_categorical_columns = []
    target_mean_candidates = []
    feature_payloads = []

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
            n_non_missing_rows = int((~null_mask).sum())
            min_support_threshold = categorical_support_threshold(
                len(cats), n_non_missing_rows
            )

            valid_indices = ~null_mask & (str_np != "nan")
            mapping, log_message = _build_categorical_mapping(
                col_name,
                str_np[valid_indices],
                max_cat,
                min_support_threshold,
                target_values=(
                    None if target_array is None else target_array[valid_indices]
                ),
                target_task=target_task,
            )
            cat_mapping[col_name] = mapping
            feature_payloads.append(
                {
                    "feature_name": col_name,
                    "index": i,
                    "values": str_np,
                    "mode": _mapping_mode(mapping),
                    "signal": _target_signal_score(
                        str_np[valid_indices],
                        None if target_array is None else target_array[valid_indices],
                        target_task,
                        min_support_threshold,
                    ),
                    "n_non_missing_rows": n_non_missing_rows,
                    "n_non_missing_categories": len(cats),
                }
            )
            descriptor, target_mean_values = _build_target_mean_descriptor(
                col_name,
                str_np,
                valid_indices,
                mapping,
                target_array,
                target_task,
                min_support_threshold,
            )
            if descriptor is not None:
                mapping["target_mean"] = descriptor
                target_mean_candidates.append(
                    {
                        "source_feature": col_name,
                        "derived_name": descriptor["derived_name"],
                        "values": target_mean_values,
                        "mapping": mapping,
                        "rank": _target_mean_candidate_rank(
                            feature_payloads[-1]["signal"],
                            mapping,
                            len(cats),
                            n_non_missing_rows,
                        ),
                    }
                )
            if log_message is not None:
                logger.warning(log_message)

            out_values = _encode_categorical_values(str_np, valid_indices, mapping)

            columns.append(out_values)
            masks.append(None)  # Categorical encoding handles NaNs
        else:
            series = X[col_name].cast(pl.Float64)
            columns.append(
                series.to_numpy(allow_copy=True).astype(np.float64, copy=False)
            )
            masks.append(None)

    if categorical_features_:
        pair_feature_start = len(features_)
        for pair_feature in _build_pairwise_categorical_features(
            feature_payloads,
            len(features_),
            max_cat,
            target_array,
            target_task,
        ):
            feature_name = pair_feature["feature_name"]
            cat_mapping[feature_name] = pair_feature["mapping"]
            appended_categorical_features.append(feature_name)
            appended_categorical_columns.append(pair_feature["encoded_values"])
            if pair_feature["target_mean_name"] is not None:
                target_mean_candidates.append(
                    {
                        "source_feature": feature_name,
                        "derived_name": pair_feature["target_mean_name"],
                        "values": pair_feature["target_mean_values"],
                        "mapping": pair_feature["mapping"],
                        "rank": _target_mean_candidate_rank(
                            pair_feature["signal"],
                            pair_feature["mapping"],
                            pair_feature["n_non_missing_categories"],
                            pair_feature["n_non_missing_rows"],
                        ),
                    }
                )
            if pair_feature["log_message"] is not None:
                logger.warning(pair_feature["log_message"])

        total_feature_count = len(features_) + len(appended_categorical_features)
        derived_features, derived_columns = _finalize_target_mean_candidates(
            target_mean_candidates,
            total_feature_count,
            target_task,
        )

        if appended_categorical_columns:
            columns.extend(appended_categorical_columns)
            masks.extend([None] * len(appended_categorical_columns))
            features_.extend(appended_categorical_features)
            categorical_features_.extend(
                range(
                    pair_feature_start,
                    pair_feature_start + len(appended_categorical_features),
                )
            )

        logger.info(f"Categorical features: {categorical_features_}")
        logger.info(f"Mapping of categories: {cat_mapping}")
    else:
        derived_features, derived_columns = _finalize_target_mean_candidates(
            target_mean_candidates,
            len(features_),
            target_task,
        )

    if derived_columns:
        columns.extend(derived_columns)
        masks.extend([None] * len(derived_columns))
        features_.extend(derived_features)
        cols = len(columns)

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

    appended_categorical_features = []
    appended_categorical_columns = []
    derived_features = []
    derived_columns = []
    raw_feature_values = {
        feature_name: X_[:, idx].astype(str)
        for idx, feature_name in enumerate(features_)
    }
    if cat_mapping:
        for feature_name, mapping in cat_mapping.items():
            source_features = (
                mapping.get("source_features") if isinstance(mapping, dict) else None
            )
            if source_features:
                left_values = raw_feature_values[source_features[0]]
                right_values = raw_feature_values[source_features[1]]
                values, valid_mask = _combine_categorical_pair_values(
                    left_values, right_values
                )
                appended_categorical_features.append(feature_name)
                appended_categorical_columns.append(
                    _encode_categorical_values(values, valid_mask, mapping)
                )
            else:
                feature_index = features_.index(feature_name)
                values = raw_feature_values[feature_name]
                valid_mask = values != "nan"
                X_[:, feature_index] = _encode_categorical_values(
                    values, valid_mask, mapping
                )
            if isinstance(mapping, dict) and "target_mean" in mapping:
                descriptor = mapping["target_mean"]
                derived_features.append(descriptor["derived_name"])
                derived_columns.append(
                    _apply_target_mean_descriptor(
                        values, valid_mask, mapping, descriptor
                    )
                )

    if appended_categorical_columns or derived_columns:
        X_ = np.column_stack([X_, *appended_categorical_columns, *derived_columns])
        features_.extend(appended_categorical_features)
        features_.extend(derived_features)

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

    appended_categorical_features = []
    appended_categorical_columns = []
    derived_features = []
    derived_columns = []

    for i, col_name in enumerate(features_):
        if cat_mapping and col_name in cat_mapping:
            # For categorical columns, encode using the existing cat_mapping.
            # Cast to String first to bypass Polars' global string cache
            # which can make Arrow DictionaryArray dictionaries unreliable.
            str_series = X[col_name].cast(pl.String)
            str_np = str_series.to_numpy(allow_copy=True)
            null_mask = str_series.is_null().to_numpy()

            valid_indices = ~null_mask & (str_np != "nan")
            x_enc = _encode_categorical_values(
                str_np, valid_indices, cat_mapping[col_name]
            )

            columns.append(x_enc)
            masks.append(None)
            mapping = cat_mapping[col_name]
            if isinstance(mapping, dict) and "target_mean" in mapping:
                descriptor = mapping["target_mean"]
                derived_features.append(descriptor["derived_name"])
                derived_columns.append(
                    _apply_target_mean_descriptor(
                        str_np, valid_indices, mapping, descriptor
                    )
                )
        else:
            series = X[col_name].cast(pl.Float64)
            columns.append(
                series.to_numpy(allow_copy=True).astype(np.float64, copy=False)
            )
            masks.append(None)

    if cat_mapping:
        for feature_name, mapping in cat_mapping.items():
            source_features = (
                mapping.get("source_features") if isinstance(mapping, dict) else None
            )
            if not source_features:
                continue

            left_series = X[source_features[0]].cast(pl.String)
            right_series = X[source_features[1]].cast(pl.String)
            left_values = left_series.to_numpy(allow_copy=True)
            right_values = right_series.to_numpy(allow_copy=True)
            pair_values, valid_indices = _combine_categorical_pair_values(
                left_values, right_values
            )
            appended_categorical_features.append(feature_name)
            appended_categorical_columns.append(
                _encode_categorical_values(pair_values, valid_indices, mapping)
            )
            if isinstance(mapping, dict) and "target_mean" in mapping:
                descriptor = mapping["target_mean"]
                derived_features.append(descriptor["derived_name"])
                derived_columns.append(
                    _apply_target_mean_descriptor(
                        pair_values, valid_indices, mapping, descriptor
                    )
                )

    if appended_categorical_columns or derived_columns:
        columns.extend(appended_categorical_columns)
        masks.extend([None] * len(appended_categorical_columns))
        columns.extend(derived_columns)
        masks.extend([None] * len(derived_columns))
        features_.extend(appended_categorical_features)
        features_.extend(derived_features)
        cols = len(columns)

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
