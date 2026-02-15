import numpy as np
import pytest
from perpetual import PerpetualBooster
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42
    )
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def test_calibration_grp(data):
    X_train, y_train, X_cal, y_cal, X_test, y_test = data

    model = PerpetualBooster(objective="SquaredLoss", save_node_stats=True)
    model.fit(X_train, y_train)

    # Calibrate using GRP
    model.calibrate(X_cal, y_cal, alpha=0.1, method="GRP")
    intervals = model.predict_intervals(X_test)

    lower = intervals["0.1"][:, 0]
    upper = intervals["0.1"][:, 1]

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)

    # Expected values based on Python logic in reproduce_issue.py
    # Coverage ~ 0.9014, Width ~ 1.3793
    print(f"GRP Coverage: {coverage}, Width: {width}")

    assert coverage >= 0.89  # Allow some fluctuation but should be close to 0.9
    # Prior to fix, width was likely > 3.0. With fix it is ~1.38.
    assert width < 1.5


def test_calibration_min_max(data):
    X_train, y_train, X_cal, y_cal, X_test, y_test = data

    model = PerpetualBooster(objective="SquaredLoss", save_node_stats=True)
    model.fit(X_train, y_train)

    # Calibrate using MinMax
    model.calibrate(X_cal, y_cal, alpha=0.1, method="MinMax")
    intervals = model.predict_intervals(X_test)

    lower = intervals["0.1"][:, 0]
    upper = intervals["0.1"][:, 1]

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)

    print(f"MinMax Coverage: {coverage}, Width: {width}")

    assert coverage >= 0.89
    # Without fix, MinMax width would also be inflated.
    # We expect verify tight intervals with MinMax usually, but checking upper bound to ensuring fix worked.
    assert width < 2.0
