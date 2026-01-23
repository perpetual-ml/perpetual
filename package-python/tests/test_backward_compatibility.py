from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from perpetual import PerpetualBooster
from sklearn.model_selection import train_test_split


def test_load_v0_10_0_model():
    resource_dir = Path(__file__).parent.parent.parent / "resources"
    model_path = resource_dir / "model_v0.10.0.json"

    if not model_path.exists():
        pytest.skip(
            f"Model artifact not found at {model_path}. Run scripts/make_resources.py to generate it."
        )

    # Load the model
    model = PerpetualBooster.load_booster(str(model_path))

    # Replicate Titanic data loading to match generation script
    df = sns.load_dataset("titanic")
    X = df.drop(columns=["survived"])
    y = df["survived"]
    X["sex"] = pd.get_dummies(X["sex"], drop_first=True, dtype=float).to_numpy()
    X["adult_male"] = pd.get_dummies(
        X["adult_male"], drop_first=True, dtype=float
    ).to_numpy()
    X.drop(columns=["alive"], inplace=True)
    X["alone"] = pd.get_dummies(X["alone"], drop_first=True, dtype=float).to_numpy()
    cols = [
        "pclass",
        "sibsp",
        "parch",
        "embarked",
        "class",
        "who",
        "deck",
        "embark_town",
    ]
    X[cols] = X[cols].astype("category")

    data_train, data_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"Current perpetual version: {PerpetualBooster.__module__.split('.')[0]}"
    )  # fallback if __version__ not directly available top-level
    import perpetual

    if hasattr(perpetual, "__version__"):
        print(f"Current perpetual version: {perpetual.__version__}")

    # Verify predictions
    preds = model.predict(data_test)

    # Assert predictions are valid (not NaN, correct shape)
    assert preds.shape[0] == data_test.shape[0]
    assert not np.isnan(preds).any()

    # Check objective
    assert model.objective == "LogLoss"

    # Compare with saved predictions from v0.10.0
    preds_path = resource_dir / "model_v0.10.0_preds.csv"
    if preds_path.exists():
        expected_preds = (
            pd.read_csv(preds_path, header=None).squeeze("columns").to_numpy()
        )
        if model.objective == "LogLoss":
            np.testing.assert_array_equal(np.rint(preds), np.rint(expected_preds))

            # Compare probabilities
            probs_path = resource_dir / "model_v0.10.0_probs.csv"
            expected_probs = pd.read_csv(probs_path, header=None).to_numpy()
            preds_probs = model.predict_proba(data_test)
            np.testing.assert_allclose(preds_probs, expected_probs, rtol=1e-5)
        else:
            np.testing.assert_allclose(preds, expected_preds, rtol=1e-5)
    else:
        pytest.fail(f"Prediction artifact not found at {preds_path}")
