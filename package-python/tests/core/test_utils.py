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


def test_convert_input_frame_categorical_list_str():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        df, ["b"], 1000
    )
    assert cat_features == {1}


def test_scipy_sparse_support():
    import scipy.sparse as sp

    # Test CSR and CSC matrices
    csr = sp.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    csc = sp.csc_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    assert type_df(csr) == "scipy_sparse"
    assert type_df(csc) == "scipy_sparse"

    # convert_input_frame
    features, flat_data, rows, cols, cat_features, cat_mapping = convert_input_frame(
        csr, None, 1000
    )
    assert rows == 2
    assert cols == 3
    assert len(features) == 3
    assert np.allclose(flat_data, [1.0, 0.0, 0.0, 3.0, 2.0, 0.0])  # Fortran order

    # transform_input_frame
    features_t, flat_data_t, rows_t, cols_t = transform_input_frame(csc, {})
    assert rows_t == 2
    assert cols_t == 3
    assert np.allclose(flat_data_t, [1.0, 0.0, 0.0, 3.0, 2.0, 0.0])

    # convert_input_array
    y_sparse = sp.csr_matrix([[0], [1], [0]])
    y_conv, classes = convert_input_array(y_sparse, "LogLoss", is_target=True)
    assert len(classes) == 2
    assert y_conv.shape == (3,)
