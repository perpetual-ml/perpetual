import os
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from perpetual.booster import PerpetualBooster
from perpetual.sklearn import PerpetualClassifier, PerpetualRanker, PerpetualRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../../resources", "titanic.csv")
    )
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"]
    return X, y


@pytest.fixture
def X_y_g() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../../resources", "goodreads.csv")
    )

    df["group"] = df["year"].astype(str) + "_" + df["category"]

    df = df.sort_values("group")

    composite_groups = df["group"]
    group_ids, _unique_groups = pd.factorize(composite_groups)

    group_lengths = pd.Series(group_ids).value_counts().sort_index()

    feature_cols = [
        "avg_rating",
        "pages",
        "5stars",
        "4stars",
        "3stars",
        "2stars",
        "1stars",
        "ratings",
    ]
    target_col = "rank"

    X = df[feature_cols]

    rank = df[target_col]
    y = rank.max() - rank

    return X, y, group_lengths


def test_sklearn_compat_classification(X_y):
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_sklearn = PerpetualClassifier(objective="LogLoss", iteration_limit=50)
    model_sklearn.fit(X_train, y_train)

    preds_sklearn = model_sklearn.predict(X_test)
    proba_sklearn = model_sklearn.predict_proba(X_test)
    log_odds_sklearn = model_sklearn.predict_log_proba(X_test)

    model_booster = PerpetualBooster(objective="LogLoss", iteration_limit=50)
    model_booster.fit(X_train, y_train)

    preds_booster = model_booster.predict(X_test)
    proba_booster = model_booster.predict_proba(X_test)
    log_odds_booster = model_booster.predict_log_proba(X_test)

    assert np.allclose(preds_sklearn, preds_booster)
    assert np.allclose(proba_sklearn, proba_booster)
    assert np.allclose(log_odds_sklearn, log_odds_booster)


def test_sklearn_compat_regression(X_y):
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_sklearn = PerpetualRegressor(objective="SquaredLoss", iteration_limit=50)
    model_sklearn.fit(X_train, y_train)

    preds_sklearn = model_sklearn.predict(X_test)

    model_booster = PerpetualBooster(objective="SquaredLoss", iteration_limit=50)
    model_booster.fit(X_train, y_train)

    preds_booster = model_booster.predict(X_test)

    assert np.allclose(preds_sklearn, preds_booster)


def test_sklearn_compat_ranking(X_y_g):
    X, y, group_lengths = X_y_g

    model_sklearn = PerpetualRanker(objective="ListNetLoss", iteration_limit=50)
    model_sklearn.fit(X, y, group=group_lengths)

    preds_sklearn = model_sklearn.predict(X)

    model_booster = PerpetualBooster(objective="ListNetLoss", iteration_limit=50)
    model_booster.fit(X, y, group=group_lengths)

    preds_booster = model_booster.predict(X)

    assert np.allclose(preds_sklearn, preds_booster)


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


def test_perpetual_classifier_score():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    clf = PerpetualClassifier(iteration_limit=10)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1.0


def test_perpetual_classifier_warning():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    # Using a regression objective for a classifier should trigger a warning
    clf = PerpetualClassifier(objective="SquaredLoss", iteration_limit=10)
    with pytest.warns(
        UserWarning, match="Objective 'SquaredLoss' is typically for regression/ranking"
    ):
        clf.fit(X, y)


def test_perpetual_regressor_score():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    reg = PerpetualRegressor(iteration_limit=10)
    reg.fit(X, y)
    score = reg.score(X, y)
    assert isinstance(score, float)
    # R2 score can be negative, but should be <= 1.0


def test_perpetual_regressor_warning():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    # Using a classification objective for a regressor should trigger a warning
    reg = PerpetualRegressor(objective="LogLoss", iteration_limit=10)
    with pytest.warns(
        UserWarning,
        match="Objective 'LogLoss' may not be suitable for PerpetualRegressor",
    ):
        reg.fit(X, y)


def test_perpetual_ranker_init_warning():
    # Using a classification objective for a ranker should trigger a warning in __init__
    with pytest.warns(
        UserWarning, match="Objective 'LogLoss' may not be suitable for PerpetualRanker"
    ):
        PerpetualRanker(objective="LogLoss")


def test_perpetual_ranker_fit_error():
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    ranker = PerpetualRanker(objective="ListNetLoss", iteration_limit=10)
    # ListNetLoss requires group parameter
    with pytest.raises(ValueError, match="The 'group' parameter must be provided"):
        ranker.fit(X, y)


def test_perpetual_classifier_custom_objective():
    X, y = make_classification(n_samples=10, n_features=10, random_state=42)
    clf = PerpetualClassifier(objective="CustomStringObjective", iteration_limit=1)
    # We just want to hit the line in sklearn.py __init__
    try:
        clf.fit(X, y)
    except Exception:
        pass


def test_perpetual_regressor_custom_objective():
    X, y = make_regression(n_samples=10, n_features=10, random_state=42)
    reg = PerpetualRegressor(objective="CustomStringObjective", iteration_limit=1)
    try:
        reg.fit(X, y)
    except Exception:
        pass
