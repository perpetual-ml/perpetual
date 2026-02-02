import numpy as np
import pandas as pd
import pytest
from perpetual.sklearn import PerpetualRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_sklearn_custom_objective(data):
    X_train, X_test, y_train, y_test = data

    def loss(y, pred, weight, group):
        return (y - pred) ** 2

    def gradient(y, pred, weight, group):
        return (pred - y), None

    def initial_value(y, weight, group):
        return np.mean(y)

    # Use the signature we just documented: (loss, grad, init)
    model_custom = PerpetualRegressor(
        objective=(loss, gradient, initial_value), budget=0.1
    )
    model_custom.fit(X_train, y_train)

    model_standard = PerpetualRegressor(objective="SquaredLoss", budget=0.1)
    model_standard.fit(X_train, y_train)

    preds_custom = model_custom.predict(X_test)
    preds_standard = model_standard.predict(X_test)

    assert np.allclose(preds_custom, preds_standard, atol=1e-5)
