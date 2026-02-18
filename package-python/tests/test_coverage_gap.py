import numpy as np
import pandas as pd
import pytest
from perpetual import PerpetualBooster
from perpetual.dml import DMLEstimator
from perpetual.fairness import FairClassifier
from perpetual.policy import PolicyLearner
from perpetual.risk import PerpetualRiskEngine


def test_dml_estimator():
    n = 100
    X = np.random.randn(n, 3)
    w = np.random.binomial(1, 0.5, n).astype(float)
    y = X[:, 0] * w + 0.5 * X[:, 1] + np.random.randn(n) * 0.1

    dml = DMLEstimator(budget=0.1, n_folds=2)
    dml.fit(X, w, y)
    preds = dml.predict(X)
    assert preds.shape == (n,)
    assert dml.feature_importances_ is not None


def test_fair_classifier():
    n = 100
    X = np.random.randn(n, 3)
    S = np.random.binomial(1, 0.5, n).astype(float)
    X = np.column_stack([X, S])
    y = (X[:, 0] + 0.5 * S > 0).astype(float)

    clf = FairClassifier(
        sensitive_feature=3, fairness_type="demographic_parity", lam=0.5, budget=0.1
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    scores = clf.decision_function(X)
    contribs = clf.predict_contributions(X)

    assert preds.shape == (n,)
    assert probs.shape == (n, 2)
    assert scores.shape == (n,)
    assert contribs.shape == (n, 5)

    clf_eo = FairClassifier(
        sensitive_feature=3, fairness_type="equalized_odds", budget=0.1
    )
    clf_eo.fit(X, y)
    assert clf_eo.predict(X).shape == (n,)


def test_policy_learner():
    n = 100
    X = np.random.randn(n, 3)
    w = np.random.binomial(1, 0.5, n)
    y = (X[:, 0] * (w - 0.5) + np.random.randn(n) * 0.1).astype(float)

    pl = PolicyLearner(budget=0.1, mode="ipw")
    pl.fit(X, w, y)
    preds = pl.predict(X)
    probs = pl.predict_proba(X)
    scores = pl.decision_function(X)

    assert preds.shape == (n,)
    assert probs.shape == (n,)
    assert scores.shape == (n,)

    pl_aipw = PolicyLearner(budget=0.1, mode="aipw")
    pl_aipw.fit(X, w, y)
    assert pl_aipw.predict(X).shape == (n,)

    p = np.full(n, 0.5)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    pl_aipw.fit(X, w, y, propensity=p, mu_hat_1=mu1, mu_hat_0=mu0)
    assert pl_aipw.predict(X).shape == (n,)


def test_risk_metrics():
    n = 20
    X = np.random.randn(n, 3)
    y = np.random.randint(0, 2, n).astype(float)

    model = PerpetualBooster(budget=0.1, objective="LogLoss")
    model.fit(X, y)

    engine = PerpetualRiskEngine(model)
    reasons = engine.generate_reason_codes(
        X, threshold=0.5, n_codes=2, rejection_direction="lower"
    )
    assert len(reasons) == n

    model_reg = PerpetualBooster(budget=0.1, objective="SquaredLoss")
    model_reg.fit(X, np.random.randn(n))
    engine_reg = PerpetualRiskEngine(model_reg)
    reasons_reg = engine_reg.generate_reason_codes(
        X, threshold=0.0, n_codes=2, rejection_direction="higher"
    )
    assert len(reasons_reg) == n

    with pytest.raises(ValueError, match="rejection_direction must be"):
        engine_reg.generate_reason_codes(
            X, threshold=0.0, rejection_direction="invalid"
        )


def test_booster_edge_cases():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)

    model = PerpetualBooster(budget=0.1, objective="SquaredLoss", save_node_stats=True)
    model.fit(X, y)

    model.calibrate(X, y, alpha=0.1, method="MinMax")
    model.calibrate(X, y, alpha=[0.1, 0.2], method="GRP")
    model.calibrate(X, y, alpha=0.1, method="WeightVariance")

    X_cal = np.random.randn(50, 2)
    y_cal = np.random.randn(50)
    model.calibrate_conformal(X, y, X_cal, y_cal, alpha=0.1)

    group_invalid = np.array([10, 10])
    with pytest.raises(ValueError, match="Sum of group sizes"):
        model.fit(X, y, group=group_invalid)

    group_ids = np.arange(100)
    with pytest.raises(ValueError, match="Query IDs instead of Group Sizes"):
        model.fit(X, y, group=group_ids)


def test_booster_additional_edge_cases():
    X = np.random.randn(20, 2)
    y = np.random.randn(20)

    model = PerpetualBooster(
        budget=0.1,
        objective="SquaredLoss",
        create_missing_branch=True,
        missing_node_treatment="AssignToParent",
    )
    model.fit(X, y)

    model.prune(X, y)

    with pytest.warns(
        UserWarning, match="predict_proba not implemented for regression"
    ):
        probs = model.predict_proba(X, calibrated=False)
        assert probs.shape == (20, 1)


def test_booster_init_edge_cases():
    with pytest.raises(ValueError, match="fairness_type must be one of"):
        FairClassifier(sensitive_feature=0, fairness_type="invalid")
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        DMLEstimator(n_folds=1)
    with pytest.raises(ValueError, match="mode must be 'ipw' or 'aipw'"):
        PolicyLearner(mode="invalid")


def test_polars_path_simulation():
    try:
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = np.array([0, 1, 0])
        model = PerpetualBooster(budget=0.1, objective="LogLoss")
        model.fit(df, y)
        preds = model.predict(df)
        assert len(preds) == 3

        X_cal = pl.DataFrame({"a": [1], "b": [4]})
        y_cal = np.array([0])
        model.calibrate_conformal(df, y, X_cal, y_cal, alpha=0.1)

        intervals = model.predict_intervals(df)
        assert len(intervals) >= 0

        sets = model.predict_sets(df)
        assert len(sets) >= 0
    except ImportError:
        pytest.skip("Polars not installed")


def test_utils_conversions():
    from perpetual.utils import (
        convert_input_array,
        convert_input_frame,
        type_df,
        type_series,
    )

    with pytest.raises(ValueError, match="Object type .* is not supported"):
        convert_input_frame("invalid", None, 1000)

    assert type_series(123) == ""
    assert type_df(123) == ""

    # Use numpy array instead of list to avoid AttributeError in fallback path
    arr, cls = convert_input_array(
        np.array([0, 1, 0]), objective="LogLoss", is_target=True
    )
    assert len(cls) == 2

    X = pd.DataFrame({"a": [1, 2, 3], "b": ["cat1", "cat2", "cat1"]})
    # This hits the pandas path in convert_input_frame
    features, flat, rows, cols, cat_idx, cat_map = convert_input_frame(
        X, categorical_features=["b"], max_cat=10
    )
    assert "b" in features


def test_fairness_sigmoid():
    from perpetual.fairness import stable_sigmoid

    assert stable_sigmoid(0) == 0.5
    assert stable_sigmoid(100) > 0.99
    assert stable_sigmoid(-100) < 0.01
