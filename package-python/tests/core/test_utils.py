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
    assert cat_features == set()


def test_transform_input_frame_unseen_category_maps_to_nan():
    train = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    _, _, _, _, _, cat_mapping = convert_input_frame(train, ["b"], 1000)

    test = pd.DataFrame({"a": [4, 5], "b": ["cat2", "cat3"]})
    _, flat_data, rows, cols = transform_input_frame(test, cat_mapping)

    matrix = flat_data.reshape((rows, cols), order="F")
    assert matrix[0, 1] == 2.0
    assert np.isnan(matrix[1, 1])
