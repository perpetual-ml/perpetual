import numpy as np
import pandas as pd
import pytest
from perpetual.utils import (
    convert_input_array,
    convert_input_frame,
    transform_input_frame,
    type_df,
    type_series,
)


def test_type_df_pandas():
    df = pd.DataFrame({"a": [1]})
    assert type_df(df) == "pandas_df"


def test_type_df_numpy():
    arr = np.array([[1]])
    assert type_df(arr) == "numpy"


def test_type_series_pandas():
    s = pd.Series([1])
    assert type_series(s) == "pandas_series"


def test_convert_input_array_numpy():
    x = np.array([1, 0, 1])
    x_conv, classes = convert_input_array(x, "LogLoss", is_target=True)
    assert len(classes) == 2
    assert x_conv.shape == (3,)


def test_convert_input_array_brierloss_with_string_classes():
    x = np.array(["yes", "no", "yes"], dtype=object)
    x_conv, classes = convert_input_array(x, "BrierLoss", is_target=True)
    assert list(classes) == ["no", "yes"]
    assert x_conv.tolist() == [1.0, 0.0, 1.0]


def test_convert_input_array_hingeloss_with_string_classes():
    x = np.array(["up", "down", "up"], dtype=object)
    x_conv, classes = convert_input_array(x, "HingeLoss", is_target=True)
    assert list(classes) == ["down", "up"]
    assert x_conv.tolist() == [1.0, 0.0, 1.0]


def test_convert_input_frame_numpy():
    X = np.random.randn(10, 2)
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        X, None, 1000
    )
    assert rows == 10
    assert cols == 2
    assert len(features) == 2


def test_convert_input_frame_invalid():
    with pytest.raises(ValueError, match="is not supported"):
        convert_input_frame([1, 2, 3], None, 1000)


def test_transform_input_frame_unsupported():
    with pytest.raises(ValueError, match="is not supported"):
        transform_input_frame([1, 2, 3], {})


def test_convert_input_array_with_classes():
    # Multi-class detection
    y = np.array([0, 1, 2, 1, 0])
    y_conv, classes = convert_input_array(
        y, "LogLoss", is_target=True, is_classification=True
    )
    assert len(classes) == 3
    # Multi-class returns one-hot flattened in Fortran order
    assert y_conv.shape == (15,)  # 5 samples * 3 classes


def test_convert_input_frame_auto_categorical():
    df = pd.DataFrame({"a": [1, 2, 3], "b": pd.Categorical(["cat1", "cat2", "cat1"])})
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, "auto", 1000
    )
    assert cat_features == {1}
    assert "b" in cat_mapping


def test_convert_input_frame_auto_object_categorical():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, "auto", 1000
    )
    assert cat_features == {1}
    assert "b" in cat_mapping


def test_convert_input_frame_auto_string_categorical():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": pd.Series(["cat1", "cat2", "cat1"], dtype="string"),
        }
    )
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, "auto", 1000
    )
    assert cat_features == {1}
    assert "b" in cat_mapping


def test_convert_input_frame_categorical_list_str():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, ["b"], 1000
    )
    assert cat_features == {1}


def test_convert_input_frame_keeps_low_cardinality_borderline_support_category():
    df = pd.DataFrame(
        {
            "a": np.arange(104),
            "b": ["cat0"] * 52 + ["cat1"] * 52,
        }
    )
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, ["b"], 1000
    )
    assert cat_features == {1}
    assert "b" in cat_mapping


def test_convert_input_frame_demotes_high_cardinality_low_support_category():
    df = pd.DataFrame(
        {
            "a": np.arange(320),
            "b": [f"cat{i % 20}" for i in range(320)],
        }
    )
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, ["b"], 1000
    )
    matrix = flat_data.reshape((rows, cols), order="F")

    assert cat_features == {1}
    assert cat_mapping["b"]["mode"] == "pooled"
    assert len(np.unique(matrix[:, 1][~np.isnan(matrix[:, 1])])) <= 5


def test_convert_input_frame_pools_high_cardinality_feature_to_max_cat():
    df = pd.DataFrame(
        {
            "a": np.arange(12),
            "b": [f"cat{i}" for i in range(12)],
        }
    )
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, ["b"], 4
    )
    matrix = flat_data.reshape((rows, cols), order="F")

    assert cat_features == {1}
    assert cat_mapping["b"]["mode"] == "pooled"
    assert len(np.unique(matrix[:, 1][~np.isnan(matrix[:, 1])])) <= 4


def test_convert_input_frame_target_aware_pooling_keeps_predictive_category():
    df = pd.DataFrame(
        {
            "a": np.arange(40),
            "b": ["common"] * 20
            + ["rare_signal"] * 4
            + [f"tail{i}" for i in range(16)],
        }
    )
    y = np.array([0] * 20 + [1] * 4 + [0] * 16, dtype=np.float64)
    _, _, _, _, cat_features, cat_mapping = convert_input_frame(
        df,
        ["b"],
        4,
        target_values=y,
        target_task="binary",
    )

    assert cat_features == {1}
    assert cat_mapping["b"]["mode"] == "pooled"
    assert "rare_signal" in cat_mapping["b"]["categories"]


def test_convert_input_frame_regression_pooled_target_mean_adds_derived_feature():
    df = pd.DataFrame(
        {
            "a": np.arange(40),
            "b": ["common"] * 20
            + ["rare_signal"] * 4
            + [f"tail{i}" for i in range(16)],
        }
    )
    y = np.array([0.1] * 20 + [1.5] * 4 + [0.2] * 16, dtype=np.float64)
    features, flat_data, rows, cols, _, cat_mapping = convert_input_frame(
        df,
        ["b"],
        4,
        target_values=y,
        target_task="regression",
    )
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cat_mapping["b"]["target_mean"]["derived_name"] == "b__target_mean"
    assert cols == 3
    assert matrix[20:24, target_mean_col].mean() > matrix[:20, target_mean_col].mean()


def test_convert_input_frame_binary_pooling_adds_target_mean_feature_for_high_cardinality_tail():
    tail_categories = [f"tail{i}" for i in range(200)]
    df = pd.DataFrame(
        {
            "a": np.arange(5000),
            "b": ["common"] * 4700 + ["rare_signal"] * 100 + tail_categories,
        }
    )
    y = np.array([0] * 4700 + [1] * 100 + [0] * len(tail_categories), dtype=np.float64)
    features, flat_data, rows, cols, _, cat_mapping = convert_input_frame(
        df,
        ["b"],
        64,
        target_values=y,
        target_task="binary",
    )
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cat_mapping["b"]["target_mean"]["derived_name"] == "b__target_mean"
    assert cols == 3
    assert (
        matrix[4700:4800, target_mean_col].mean()
        > matrix[:4700, target_mean_col].mean()
    )
    assert len(np.unique(matrix[:, 1][~np.isnan(matrix[:, 1])])) <= 64


def test_convert_input_frame_binary_pooling_skips_target_mean_for_small_tail_pool():
    train = pd.DataFrame(
        {
            "a": np.arange(40),
            "b": ["common"] * 20
            + ["rare_signal"] * 4
            + [f"tail{i}" for i in range(16)],
        }
    )
    y = np.array([0] * 20 + [1] * 4 + [0] * 16, dtype=np.float64)
    features, flat_data, rows, cols, _, cat_mapping = convert_input_frame(
        train,
        ["b"],
        4,
        target_values=y,
        target_task="binary",
    )

    matrix = flat_data.reshape((rows, cols), order="F")

    assert "b__target_mean" not in features
    assert "target_mean" not in cat_mapping["b"]
    assert cols == 2
    assert len(np.unique(matrix[:, 1][~np.isnan(matrix[:, 1])])) <= 4


def test_convert_input_frame_binary_native_high_card_adds_target_mean_feature():
    categories = [f"cat{i:02d}" for i in range(64)]
    values = np.repeat(categories, 64)
    df = pd.DataFrame({"a": np.arange(len(values)), "b": values})
    y = np.zeros(len(values), dtype=np.float64)
    for idx, category in enumerate(values):
        if category in {"cat00", "cat01", "cat02", "cat03"}:
            y[idx] = 1.0

    features, flat_data, rows, cols, _, cat_mapping = convert_input_frame(
        df,
        ["b"],
        1000,
        target_values=y,
        target_task="binary",
    )
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cat_mapping["b"]["mode"] == "native"
    assert cat_mapping["b"]["target_mean"]["derived_name"] == "b__target_mean"
    assert cols == 3
    signal_mask = df["b"].isin({"cat00", "cat01", "cat02", "cat03"}).to_numpy()
    assert (
        matrix[signal_mask, target_mean_col].mean()
        > matrix[~signal_mask, target_mean_col].mean()
    )


def test_transform_input_frame_recreates_binary_pooled_target_mean_feature():
    tail_categories = [f"tail{i}" for i in range(200)]
    train = pd.DataFrame(
        {
            "a": np.arange(5000),
            "b": ["common"] * 4700 + ["rare_signal"] * 100 + tail_categories,
        }
    )
    y = np.array([0] * 4700 + [1] * 100 + [0] * len(tail_categories), dtype=np.float64)
    _, _, _, _, _, cat_mapping = convert_input_frame(
        train,
        ["b"],
        64,
        target_values=y,
        target_task="binary",
    )

    test = pd.DataFrame({"a": [41, 42], "b": ["rare_signal", "unseen"]})
    features, flat_data, rows, cols = transform_input_frame(test, cat_mapping)
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cols == 3
    assert features[-1] == "b__target_mean"
    assert matrix[0, target_mean_col] > matrix[1, target_mean_col]


def test_transform_input_frame_recreates_regression_pooled_target_mean_feature():
    train = pd.DataFrame(
        {
            "a": np.arange(40),
            "b": ["common"] * 20
            + ["rare_signal"] * 4
            + [f"tail{i}" for i in range(16)],
        }
    )
    y = np.array([0.1] * 20 + [1.5] * 4 + [0.2] * 16, dtype=np.float64)
    _, _, _, _, _, cat_mapping = convert_input_frame(
        train,
        ["b"],
        4,
        target_values=y,
        target_task="regression",
    )

    test = pd.DataFrame({"a": [41, 42], "b": ["rare_signal", "unseen"]})
    features, flat_data, rows, cols = transform_input_frame(test, cat_mapping)
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cols == 3
    assert features[-1] == "b__target_mean"
    assert matrix[0, target_mean_col] > matrix[1, target_mean_col]


def test_transform_input_frame_recreates_binary_native_target_mean_feature():
    categories = [f"cat{i:02d}" for i in range(64)]
    values = np.repeat(categories, 64)
    train = pd.DataFrame({"a": np.arange(len(values)), "b": values})
    y = np.zeros(len(values), dtype=np.float64)
    for idx, category in enumerate(values):
        if category in {"cat00", "cat01", "cat02", "cat03"}:
            y[idx] = 1.0

    _, _, _, _, _, cat_mapping = convert_input_frame(
        train,
        ["b"],
        1000,
        target_values=y,
        target_task="binary",
    )

    test = pd.DataFrame({"a": [4097, 4098], "b": ["cat00", "unseen"]})
    features, flat_data, rows, cols = transform_input_frame(test, cat_mapping)
    matrix = flat_data.reshape((rows, cols), order="F")
    target_mean_col = features.index("b__target_mean")

    assert cols == 3
    assert features[-1] == "b__target_mean"
    assert matrix[0, target_mean_col] > matrix[1, target_mean_col]


def test_convert_input_frame_adds_pairwise_categorical_feature_for_binary_task():
    counts = {
        ("left", "red"): 1024,
        ("left", "blue"): 256,
        ("right", "red"): 256,
        ("right", "blue"): 1024,
    }
    rows = []
    targets = []
    for (left_value, right_value), count in counts.items():
        rows.extend([(left_value, right_value)] * count)
        targets.extend(
            [
                (
                    1.0
                    if (left_value, right_value) in {("left", "red"), ("right", "blue")}
                    else 0.0
                )
            ]
            * count
        )

    df = pd.DataFrame(rows, columns=["a", "b"])
    df.insert(0, "num", np.arange(len(df)))
    y = np.asarray(targets, dtype=np.float64)

    features, flat_data, row_count, cols, cat_features, cat_mapping = (
        convert_input_frame(
            df,
            ["a", "b"],
            1000,
            target_values=y,
            target_task="binary",
        )
    )
    matrix = flat_data.reshape((row_count, cols), order="F")
    pair_feature = "a__x__b"
    pair_target_mean = f"{pair_feature}__target_mean"

    assert pair_feature in features
    assert pair_feature in cat_mapping
    assert cat_mapping[pair_feature]["source_features"] == ["a", "b"]
    assert pair_target_mean in features
    assert cat_features == {1, 2, features.index(pair_feature)}
    assert (
        matrix[: counts[("left", "red")], features.index(pair_target_mean)].mean()
        > matrix[
            counts[("left", "red")] : counts[("left", "red")]
            + counts[("left", "blue")],
            features.index(pair_target_mean),
        ].mean()
    )


def test_transform_input_frame_recreates_pairwise_categorical_feature():
    counts = {
        ("left", "red"): 1024,
        ("left", "blue"): 256,
        ("right", "red"): 256,
        ("right", "blue"): 1024,
    }
    rows = []
    targets = []
    for (left_value, right_value), count in counts.items():
        rows.extend([(left_value, right_value)] * count)
        targets.extend(
            [
                (
                    1.0
                    if (left_value, right_value) in {("left", "red"), ("right", "blue")}
                    else 0.0
                )
            ]
            * count
        )

    train = pd.DataFrame(rows, columns=["a", "b"])
    train.insert(0, "num", np.arange(len(train)))
    y = np.asarray(targets, dtype=np.float64)
    _, _, _, _, _, cat_mapping = convert_input_frame(
        train,
        ["a", "b"],
        1000,
        target_values=y,
        target_task="binary",
    )

    test = pd.DataFrame(
        {
            "num": [len(train), len(train) + 1],
            "a": ["left", "left"],
            "b": ["red", "blue"],
        }
    )
    features, flat_data, rows, cols = transform_input_frame(test, cat_mapping)
    matrix = flat_data.reshape((rows, cols), order="F")
    pair_feature = "a__x__b"
    pair_target_mean = f"{pair_feature}__target_mean"

    assert pair_feature in features
    assert pair_target_mean in features
    assert (
        matrix[0, features.index(pair_feature)]
        != matrix[1, features.index(pair_feature)]
    )
    assert (
        matrix[0, features.index(pair_target_mean)]
        > matrix[1, features.index(pair_target_mean)]
    )


def test_transform_input_frame_unseen_category_maps_to_nan():
    train = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    _, _, _, _, _, cat_mapping = convert_input_frame(train, ["b"], 1000)

    test = pd.DataFrame({"a": [4, 5], "b": ["cat2", "cat3"]})
    _, flat_data, rows, cols = transform_input_frame(test, cat_mapping)

    matrix = flat_data.reshape((rows, cols), order="F")
    assert matrix[0, 1] == 2.0
    assert np.isnan(matrix[1, 1])


def test_transform_input_frame_pooled_unseen_category_uses_tail_bucket():
    train = pd.DataFrame(
        {
            "a": np.arange(6),
            "b": ["cat0", "cat1", "cat2", "cat3", "cat4", "cat5"],
        }
    )
    _, _, _, _, _, cat_mapping = convert_input_frame(train, ["b"], 4)

    test = pd.DataFrame({"a": [6, 7], "b": ["cat1", "unseen"]})
    _, flat_data, rows, cols = transform_input_frame(test, cat_mapping)

    matrix = flat_data.reshape((rows, cols), order="F")
    assert matrix[0, 1] == 2.0
    assert not np.isnan(matrix[1, 1])
    assert matrix[1, 1] > len(cat_mapping["b"]["categories"]) - 1


def test_convert_input_frame_wide_binary_task_limits_target_mean_companions():
    rows = 4096
    y = ((np.arange(rows) % 10) < 2).astype(np.float64)
    data = {f"num_{idx:03d}": np.linspace(idx, idx + 1, rows) for idx in range(120)}

    strong_pattern = np.array(
        [f"strong_{int(label)}_{idx % 10}" for idx, label in enumerate(y)]
    )
    medium_pattern = np.array(
        [
            f"medium_{int(((idx % 4) == 0) ^ bool(label))}_{idx % 10}"
            for idx, label in enumerate(y)
        ]
    )
    weak_pattern = np.array([f"weak_{idx % 10}" for idx in range(rows)])

    for idx in range(5):
        data[f"cat_strong_{idx:02d}"] = np.roll(strong_pattern, idx * 13)
    for idx in range(5):
        data[f"cat_medium_{idx:02d}"] = np.roll(medium_pattern, idx * 17)
    for idx in range(5):
        data[f"cat_weak_{idx:02d}"] = np.roll(weak_pattern, idx * 19)

    df = pd.DataFrame(data)
    categorical_columns = [name for name in df.columns if name.startswith("cat_")]
    features, _, _, _, _, cat_mapping = convert_input_frame(
        df,
        categorical_columns,
        1000,
        target_values=y,
        target_task="binary",
    )

    target_mean_features = [name for name in features if name.endswith("__target_mean")]
    retained_categorical_target_means = [
        feature_name
        for feature_name in categorical_columns
        if "target_mean" in cat_mapping[feature_name]
    ]
    dropped_categorical_target_means = [
        feature_name
        for feature_name in categorical_columns
        if "target_mean" not in cat_mapping[feature_name]
    ]

    assert len(target_mean_features) == 12
    assert len(retained_categorical_target_means) == 12
    assert len(dropped_categorical_target_means) == 3
    for idx in range(5):
        feature_name = f"cat_strong_{idx:02d}"
        assert f"{feature_name}__target_mean" in target_mean_features
        assert "target_mean" in cat_mapping[feature_name]


def test_convert_input_frame_narrow_binary_task_keeps_all_target_mean_companions():
    rows = 4096
    y = ((np.arange(rows) % 10) < 2).astype(np.float64)
    data = {f"num_{idx:03d}": np.linspace(idx, idx + 1, rows) for idx in range(40)}

    strong_pattern = np.array(
        [f"strong_{int(label)}_{idx % 10}" for idx, label in enumerate(y)]
    )
    medium_pattern = np.array(
        [
            f"medium_{int(((idx % 4) == 0) ^ bool(label))}_{idx % 10}"
            for idx, label in enumerate(y)
        ]
    )
    weak_pattern = np.array([f"weak_{idx % 10}" for idx in range(rows)])

    for idx in range(5):
        data[f"cat_strong_{idx:02d}"] = np.roll(strong_pattern, idx * 13)
    for idx in range(5):
        data[f"cat_medium_{idx:02d}"] = np.roll(medium_pattern, idx * 17)
    for idx in range(5):
        data[f"cat_weak_{idx:02d}"] = np.roll(weak_pattern, idx * 19)

    df = pd.DataFrame(data)
    categorical_columns = [name for name in df.columns if name.startswith("cat_")]
    features, _, _, _, _, cat_mapping = convert_input_frame(
        df,
        categorical_columns,
        1000,
        target_values=y,
        target_task="binary",
    )

    target_mean_features = [name for name in features if name.endswith("__target_mean")]
    base_target_mean_features = [
        name for name in target_mean_features if "__x__" not in name
    ]
    pair_target_mean_features = [
        name for name in target_mean_features if "__x__" in name
    ]

    assert len(base_target_mean_features) == 15
    assert len(pair_target_mean_features) == 3
    for feature_name in categorical_columns:
        assert f"{feature_name}__target_mean" in base_target_mean_features
        assert "target_mean" in cat_mapping[feature_name]
