import json
import os
import tempfile

import numpy as np
import pytest
from perpetual import PerpetualBooster
from perpetual.booster import compute_calibration_curve, expected_calibration_error
from perpetual.serialize import NumpySerializer, ObjectSerializer, ScalerSerializer
from perpetual.sklearn import PerpetualRanker


def test_serialization_module():
    # ScalerSerializer
    ss = ScalerSerializer()
    assert ss.serialize(1) == "1"
    assert ss.serialize("test") == "'test'"
    assert ss.deserialize("1") == 1
    assert ss.deserialize("'test'") == "test"

    # ObjectSerializer
    os_ser = ObjectSerializer()
    obj = {"a": [1, 2], "b": 3.0}
    ser = os_ser.serialize(obj)
    assert os_ser.deserialize(ser) == obj
    # Test numpy array handled in ObjectSerializer
    assert os_ser.serialize(np.array([1, 2])) == "[1, 2]"

    # NumpySerializer
    ns = NumpySerializer()
    a1 = np.array([1, 2, 3], dtype=np.float64)
    ser1 = ns.serialize(a1)
    assert np.allclose(ns.deserialize(ser1), a1)

    a2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ser2 = ns.serialize(a2)
    a2_deser = ns.deserialize(ser2)
    assert a2_deser.shape == (2, 2)
    assert np.array_equal(a2_deser, a2)


def test_perpetual_ranker():
    n = 100
    X = np.random.randn(n, 3)
    y = np.random.randn(n)
    # 5 groups of 20
    group = np.array([20] * 5)

    ranker = PerpetualRanker(budget=0.1)
    ranker.fit(X, y, group=group)
    preds = ranker.predict(X)
    assert preds.shape == (n,)
    assert isinstance(ranker.score(X, y), float)


def test_calibration_metrics():
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.uniform(0, 1, 100)

    # compute_calibration_curve
    pt, pp = compute_calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
    assert len(pt) <= 5
    pt_q, pp_q = compute_calibration_curve(
        y_true, y_prob, n_bins=5, strategy="quantile"
    )
    assert len(pt_q) <= 5

    with pytest.raises(ValueError, match="Invalid strategy"):
        compute_calibration_curve(y_true, y_prob, strategy="invalid")

    # expected_calibration_error
    ece = expected_calibration_error(y_true, y_prob, strategy="uniform")
    assert 0 <= ece <= 1
    ece_q = expected_calibration_error(y_true, y_prob, strategy="quantile")
    assert 0 <= ece_q <= 1

    with pytest.raises(ValueError, match="Invalid strategy"):
        expected_calibration_error(y_true, y_prob, strategy="invalid")


def test_exports_binary():
    X = np.random.randn(50, 2)
    y = np.random.randint(0, 2, 50).astype(float)
    model = PerpetualBooster(objective="LogLoss", budget=0.1)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # XGBoost
        xgb_path = os.path.join(tmp_dir, "model.xgb.json")
        model.save_as_xgboost(xgb_path)
        assert os.path.exists(xgb_path)
        with open(xgb_path, "r") as f:
            d = json.load(f)
            assert d["learner"]["objective"]["name"] == "binary:logistic"

        # ONNX
        onnx_path = os.path.join(tmp_dir, "model.onnx")
        model.save_as_onnx(onnx_path)
        assert os.path.exists(onnx_path)


def test_exports_regression():
    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    model = PerpetualBooster(objective="SquaredLoss", budget=0.1)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        xgb_path = os.path.join(tmp_dir, "model.xgb.json")
        model.save_as_xgboost(xgb_path)
        assert os.path.exists(xgb_path)

        onnx_path = os.path.join(tmp_dir, "model.onnx")
        model.save_as_onnx(onnx_path)
        assert os.path.exists(onnx_path)


def test_exports_multiclass():
    X = np.random.randn(60, 2)
    # Ensure exactly 3 classes
    y = np.array([0, 1, 2] * 20).astype(float)
    model = PerpetualBooster(objective="LogLoss", budget=0.1)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        xgb_path = os.path.join(tmp_dir, "model.xgb.json")
        model.save_as_xgboost(xgb_path)
        assert os.path.exists(xgb_path)
        with open(xgb_path, "r") as f:
            d = json.load(f)
            assert d["learner"]["objective"]["name"] == "multi:softprob"

        onnx_path = os.path.join(tmp_dir, "model.onnx")
        model.save_as_onnx(onnx_path)
        assert os.path.exists(onnx_path)


def test_booster_helpers():
    X = np.random.randn(20, 2)
    y = np.random.randn(20)

    # Regression model to test warning
    model_reg = PerpetualBooster(budget=0.1, objective="SquaredLoss")
    model_reg.fit(X, y)

    # Check if classes_ is indeed empty
    assert len(model_reg.classes_) == 0

    # Test predict_log_proba warning
    with pytest.warns(UserWarning):
        lp = model_reg.predict_log_proba(X)
        assert lp.shape == (20, 1)

    # Classification model for other helpers
    y_clf = np.random.randint(0, 2, 20).astype(float)
    model = PerpetualBooster(budget=0.1, objective="LogLoss")
    model.fit(X, y_clf)

    # predict_nodes (returns list of lists or arrays)
    nodes = model.predict_nodes(X)
    # number_of_trees is a property
    assert len(nodes) == model.number_of_trees
    assert len(nodes[0]) == 20

    # Partial dependence (returns array of shape (samples, 2))
    pd_vals = model.partial_dependence(X, feature=0)
    assert pd_vals.shape == (100, 2)

    # trees_to_dataframe
    df = model.trees_to_dataframe()
    assert len(df) > 0

    # get_node_lists (with names)
    model.feature_names_in_ = ["a", "b"]
    node_lists = model.get_node_lists(map_features_names=True)
    assert len(node_lists) == model.number_of_trees


def test_multiclass_extras():
    X = np.random.randn(30, 2)
    # Exactly 3 classes
    y = np.array([0, 1, 2] * 10).astype(float)
    model = PerpetualBooster(budget=0.1, objective="LogLoss")
    model.fit(X, y)

    # get_node_lists for multi-output returns sum of trees
    nl = model.get_node_lists()
    # number_of_trees for multi-output is an array
    n_trees_array = np.atleast_1d(model.number_of_trees)
    assert len(nl) == np.sum(n_trees_array)

    # set_params
    model.set_params(budget=0.2)
    assert model.budget == 0.2
