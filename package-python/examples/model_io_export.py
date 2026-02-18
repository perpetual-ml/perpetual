import os
import pickle

import numpy as np
import pandas as pd
from perpetual import PerpetualBooster


def run_model_io_example():
    # 1. Prepare some data
    print("Preparing data...")
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # 2. Fit the model
    print("Fitting PerpetualBooster...")
    model = PerpetualBooster(objective="SquaredLoss", budget=0.1)
    model.fit(X_df, y)

    # Add some custom metadata
    model.insert_metadata("author", "Perpetual User")
    model.insert_metadata("dataset", "Synthetic Regression")

    original_preds = model.predict(X_df[:5])
    print(f"Original predictions: {original_preds}")

    # 3. Native Save/Load
    print("\nTesting Native Save/Load...")
    model_path = "model.json"
    model.save_booster(model_path)

    loaded_model = PerpetualBooster.load_booster(model_path)
    loaded_preds = loaded_model.predict(X_df[:5])
    print(f"Loaded predictions: {loaded_preds}")
    print(f"Metadata 'author': {loaded_model.get_metadata('author')}")

    assert np.allclose(original_preds, loaded_preds)
    os.remove(model_path)

    # 4. Pickle Save/Load
    print("\nTesting Pickle Save/Load...")
    pickle_path = "model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)

    with open(pickle_path, "rb") as f:
        pickled_model = pickle.load(f)

    pickled_preds = pickled_model.predict(X_df[:5])
    print(f"Pickled predictions: {pickled_preds}")
    assert np.allclose(original_preds, pickled_preds)
    os.remove(pickle_path)

    # 5. Export to XGBoost JSON
    print("\nExporting to XGBoost format...")
    xgb_path = "model_xgb.json"
    model.save_as_xgboost(xgb_path)
    print(f"XGBoost model saved to {xgb_path}")

    # If xgboost is installed, we could verify it here
    try:
        import xgboost as xgb

        bst = xgb.Booster()
        bst.load_model(xgb_path)
        dmat = xgb.DMatrix(X_df[:5])
        xgb_preds = bst.predict(dmat)
        print(f"XGBoost predictions: {xgb_preds}")
    except ImportError:
        print("XGBoost not installed, skipping verification.")

    if os.path.exists(xgb_path):
        os.remove(xgb_path)

    # 6. Export to ONNX
    print("\nExporting to ONNX format...")
    onnx_path = "model.onnx"
    model.save_as_onnx(onnx_path)
    print(f"ONNX model saved to {onnx_path}")

    # If onnxruntime is installed, we could verify it here
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        X_f32 = X_df[:5].to_numpy().astype(np.float32)
        onnx_preds = sess.run(None, {input_name: X_f32})[0].flatten()
        print(f"ONNX predictions: {onnx_preds}")
    except ImportError:
        print("onnxruntime not installed, skipping verification.")

    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_model_io_example()
