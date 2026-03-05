import pytest
from perpetual import PerpetualBooster
from sklearn.datasets import make_regression


def test_objective_parameters_regression():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    # Test QuantileLoss
    model_q = PerpetualBooster(objective="QuantileLoss", quantile=0.7)
    assert model_q.objective_parameter == 0.7
    model_q.fit(X, y)

    # Test AdaptiveHuberLoss
    model_ah = PerpetualBooster(objective="AdaptiveHuberLoss", quantile=0.8)
    assert model_ah.objective_parameter == 0.8
    model_ah.fit(X, y)

    # Test HuberLoss
    model_h = PerpetualBooster(objective="HuberLoss", delta=1.5)
    assert model_h.objective_parameter == 1.5
    model_h.fit(X, y)

    # Test FairLoss
    model_f = PerpetualBooster(objective="FairLoss", c=2.0)
    assert model_f.objective_parameter == 2.0
    model_f.fit(X, y)

    # Test TweedieLoss
    # Tweedie targets should be strictly positive for p > 1
    y_pos = y - y.min() + 0.1
    model_t = PerpetualBooster(objective="TweedieLoss", p=1.5)
    assert model_t.objective_parameter == 1.5
    model_t.fit(X, y_pos)


def test_objective_missing_kwarg():
    # It should still initialize, however its objective_parameter defaults to None (which maps to Rust defaults)
    model = PerpetualBooster(objective="QuantileLoss")
    assert model.objective_parameter is None


def test_invalid_kwarg():
    with pytest.raises(ValueError, match="Unknown keyword arguments"):
        PerpetualBooster(objective="SquaredLoss", some_invalid_kwarg=123)

    with pytest.raises(ValueError, match="Unknown keyword arguments"):
        # Even if valid for another objective, it should raise
        PerpetualBooster(objective="SquaredLoss", quantile=0.5)
