import os

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
import xgboost as xgb
from perpetual import PerpetualBooster


@pytest.fixture(scope="module")
def data_binary():
    df = pd.read_csv("../resources/titanic.csv")
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"]
    return X, y


@pytest.fixture(scope="module")
def data_multiclass():
    X = pd.read_csv("../resources/cover_types_train.csv", index_col=False, nrows=2000)
    X = X.sample(n=200, random_state=0)
    y = np.array(X.pop("Cover_Type"))
    X = X.iloc[:, :10]
    return X, y


@pytest.fixture(scope="module")
def data_regression():
    X = pd.read_csv("../resources/cal_housing_train.csv", index_col=False, nrows=1000)
    X = X.sample(n=200, random_state=0)
    y = X.pop("MedHouseVal").to_numpy()
    X = X.iloc[:, :5]
    return X, y


def test_xgboost_export_binary(data_binary, tmp_path):
    X, y = data_binary
    model = PerpetualBooster(objective="LogLoss", iteration_limit=1)
    model.fit(X, y)

    xgb_path = tmp_path / "model_binary.json"
    model.save_as_xgboost(str(xgb_path))

    assert os.path.exists(xgb_path)

    bst = xgb.Booster()
    bst.load_model(str(xgb_path))

    dmat = xgb.DMatrix(X)
    xgb_preds = bst.predict(dmat)
    perp_proba = model.predict_proba(X)[:, 1]

    np.testing.assert_allclose(perp_proba, xgb_preds, rtol=1e-3, atol=1e-3)


def test_xgboost_export_multiclass(data_multiclass, tmp_path):
    X, y = data_multiclass
    model = PerpetualBooster(objective="LogLoss", iteration_limit=5)
    model.fit(X, y)

    xgb_path = tmp_path / "model_multi.json"
    model.save_as_xgboost(str(xgb_path))

    assert os.path.exists(xgb_path)

    bst = xgb.Booster()
    bst.load_model(str(xgb_path))

    dmat = xgb.DMatrix(X)
    xgb_preds = bst.predict(dmat)
    perp_proba = model.predict_proba(X)

    np.testing.assert_allclose(perp_proba, xgb_preds, rtol=1e-3, atol=1e-3)


def test_xgboost_export_regression(data_regression, tmp_path):
    X, y = data_regression
    model = PerpetualBooster(objective="SquaredLoss", iteration_limit=1)
    model.fit(X, y)

    xgb_path = tmp_path / "model_reg.json"
    model.save_as_xgboost(str(xgb_path))

    assert os.path.exists(xgb_path)

    bst = xgb.Booster()
    bst.load_model(str(xgb_path))

    dmat = xgb.DMatrix(X)
    xgb_preds = bst.predict(dmat)
    perp_preds = model.predict(X)

    np.testing.assert_allclose(perp_preds, xgb_preds, rtol=1e-3, atol=1e-3)


def test_onnx_export_binary(data_binary, tmp_path):
    X, y = data_binary
    model = PerpetualBooster(objective="LogLoss", iteration_limit=1)
    model.fit(X, y)

    onnx_path = tmp_path / "model_binary.onnx"
    model.save_as_onnx(str(onnx_path))

    assert os.path.exists(onnx_path)

    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    X_f32 = X.to_numpy().astype(np.float32)

    res = sess.run(None, {input_name: X_f32})
    onnx_proba = res[1]

    perp_proba = model.predict_proba(X)

    # ONNX uses float32 internally. Sum of probs should be 1.0.
    np.testing.assert_allclose(perp_proba, onnx_proba, rtol=5e-2, atol=5e-2)

    onnx_labels = res[0]
    perp_labels = model.predict(X)
    np.testing.assert_array_equal(perp_labels, onnx_labels)


def test_onnx_export_multiclass(data_multiclass, tmp_path):
    X, y = data_multiclass
    model = PerpetualBooster(objective="LogLoss", iteration_limit=1)
    model.fit(X, y)

    onnx_path = tmp_path / "model_multi.onnx"
    model.save_as_onnx(str(onnx_path))

    assert os.path.exists(onnx_path)

    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    X_f32 = X.to_numpy().astype(np.float32)

    res = sess.run(None, {input_name: X_f32})
    onnx_proba = res[1]

    perp_proba = model.predict_proba(X)

    np.testing.assert_allclose(perp_proba, onnx_proba, rtol=5e-2, atol=5e-2)

    onnx_labels = res[0]
    perp_labels = model.predict(X)
    np.testing.assert_array_equal(perp_labels, onnx_labels)


def test_onnx_export_regression(data_regression, tmp_path):
    X, y = data_regression
    model = PerpetualBooster(objective="SquaredLoss", iteration_limit=1)
    model.fit(X, y)

    onnx_path = tmp_path / "model_reg.onnx"
    model.save_as_onnx(str(onnx_path))

    assert os.path.exists(onnx_path)

    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    X_f32 = X.to_numpy().astype(np.float32)

    res = sess.run(None, {input_name: X_f32})
    onnx_preds = res[0].flatten()

    perp_preds = model.predict(X)

    np.testing.assert_allclose(perp_preds, onnx_preds, rtol=5e-2, atol=5e-2)
