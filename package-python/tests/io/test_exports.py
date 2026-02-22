import json
import os
import tempfile

import numpy as np
from perpetual import PerpetualBooster


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
