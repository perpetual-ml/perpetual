import numpy as np
from perpetual.booster import PerpetualBooster
from perpetual.iv import BraidedBooster
from perpetual.meta_learners import DRLearner, SLearner, TLearner, XLearner
from perpetual.uplift import UpliftBooster


def test_braided_booster():
    n = 100
    X = np.random.randn(n, 2)
    Z = np.random.randn(n, 2)
    w = 0.5 * Z[:, 0] + 0.5 * X[:, 0] + np.random.randn(n) * 0.1
    y = 2.0 * w + 1.0 * X[:, 1] + np.random.randn(n) * 0.1

    iv = BraidedBooster(stage1_budget=0.1, stage2_budget=0.1)
    iv.fit(X, Z, y, w)
    preds = iv.predict(X, w_counterfactual=np.ones(n))
    assert preds.shape == (n,)

    js = iv.to_json()
    iv2 = BraidedBooster.from_json(js)
    # Restored budgets are floats, so check approximately
    assert np.isclose(iv2.stage1_budget, 0.1, atol=1e-5)


def test_uplift_booster():
    n = 100
    X = np.random.randn(n, 2)
    T = np.random.binomial(1, 0.5, n).astype(float)
    y = 0.5 * X[:, 0] + 1.0 * T * (X[:, 1] > 0) + np.random.randn(n) * 0.1

    ub = UpliftBooster(outcome_budget=0.1, propensity_budget=0.1, effect_budget=0.1)
    ub.fit(X, T, y)
    ite = ub.predict(X)
    assert ite.shape == (n,)

    js = ub.to_json()
    ub2 = UpliftBooster.from_json(js)
    assert ub2.booster is not None


def test_meta_learners():
    n = 100
    X = np.random.randn(n, 2)
    T = np.random.binomial(1, 0.5, n).astype(float)
    y = 0.5 * X[:, 0] + 1.0 * T + np.random.randn(n) * 0.1

    # Correct order: (X, w, y)
    tl = TLearner(budget=0.1)
    tl.fit(X, T, y)
    assert tl.predict(X).shape == (n,)
    assert tl.feature_importances_.shape == (2,)

    sl = SLearner(budget=0.1)
    sl.fit(X, T, y)
    assert sl.predict(X).shape == (n,)
    assert sl.feature_importances_.shape == (2,)

    xl = XLearner(budget=0.1)
    xl.fit(X, T, y)
    assert xl.predict(X).shape == (n,)
    assert xl.feature_importances_.shape == (2,)

    dl = DRLearner(budget=0.1)
    dl.fit(X, T, y)
    assert dl.predict(X).shape == (n,)
    assert dl.feature_importances_.shape == (2,)


def test_multi_output_booster_via_perpetual():
    n = 100
    X = np.random.randn(n, 2)
    y = np.random.randint(0, 3, n).astype(float)

    model = PerpetualBooster(budget=0.1, objective="LogLoss")
    model.fit(X, y)

    assert len(model.classes_) == 3
    probs = model.predict_proba(X)
    assert probs.shape == (n, 3)
    preds = model.predict(X)
    assert preds.shape == (n,)
    contribs = model.predict_contributions(X)
    assert contribs.shape == (n, (2 + 1) * 3)


def test_booster_serialization():
    X = np.random.randn(20, 2)
    y = np.random.randn(20)
    model = PerpetualBooster(budget=0.1)
    model.fit(X, y)

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_booster(tmp_path)
        model2 = PerpetualBooster.load_booster(tmp_path)
        assert np.allclose(model.predict(X), model2.predict(X))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_sklearn_wrapper():
    from perpetual.sklearn import PerpetualClassifier, PerpetualRegressor

    X = np.random.randn(50, 2)
    y_reg = np.random.randn(50)
    y_clf = np.random.randint(0, 2, 50)

    reg = PerpetualRegressor(budget=0.1)
    reg.fit(X, y_reg)
    assert reg.predict(X).shape == (50,)

    clf = PerpetualClassifier(budget=0.1)
    clf.fit(X, y_clf)
    assert clf.predict(X).shape == (50,)
    assert clf.predict_proba(X).shape == (50, 2)

    clf.set_params(budget=0.2)
    assert clf.budget == 0.2
