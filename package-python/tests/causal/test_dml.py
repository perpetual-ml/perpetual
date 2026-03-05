import numpy as np
import pytest
from perpetual.dml import DMLEstimator


def test_dml_estimator_native():
    """Test that the native Rust DMLEstimator fits and predicts correctly."""
    # Synthetic data
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, size=(n, 5))

    # Treatment is continuous or binary, let's use binary with confounding
    # W ~ Bernoulli(sigma(X[:, 0]))
    logits = X[:, 0]
    p = 1 / (1 + np.exp(-logits))
    W = np.random.binomial(1, p)

    # Outcome Y: treatment effect is 2.0 (constant for simplicity natively)
    # y = \theta(X) W + g(X) + \epsilon
    # g(X) = 1.5 * X[:, 1]
    y = 2.0 * W + 1.5 * X[:, 1] + np.random.normal(0, 0.1, size=n)

    # Initialize DMLEstimator
    dml = DMLEstimator(budget=0.5, n_folds=3, seed=42)

    # Fit
    dml.fit(X, W, y)

    # Predict CATE
    cate_pred = dml.predict(X)

    # Check that predictions are close to true effect 2.0
    assert cate_pred.shape == (n,)
    assert np.mean(cate_pred) == pytest.approx(2.0, abs=0.2), (
        f"Mean CATE was {np.mean(cate_pred)}"
    )

    # Test ATE Inference
    inf = dml.ate_inference()

    assert "ate" in inf
    assert "std_err" in inf
    assert "ci_lower" in inf
    assert "ci_upper" in inf

    ate = inf["ate"]
    se = inf["std_err"]
    lower = inf["ci_lower"]
    upper = inf["ci_upper"]

    assert ate == pytest.approx(2.0, abs=0.2)
    assert se > 0
    assert lower < ate < upper

    # Test feature importances
    importances = dml.feature_importances_
    assert importances is not None
    assert len(importances) == 5
    assert isinstance(importances, np.ndarray)
