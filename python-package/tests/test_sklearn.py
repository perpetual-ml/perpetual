from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from perpetual.booster import PerpetualBooster
from perpetual.sklearn import PerpetualClassifier, PerpetualRanker, PerpetualRegressor
from sklearn.model_selection import train_test_split


@pytest.fixture
def X_y() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../resources/titanic.csv")
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"]
    return X, y


@pytest.fixture
def X_y_g() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv("../resources/goodreads.csv")

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

    np.allclose(preds_sklearn, preds_booster)
    np.allclose(proba_sklearn, proba_booster)
    np.allclose(log_odds_sklearn, log_odds_booster)


def test_sklearn_compat_regression(X_y):

    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_sklearn = PerpetualRegressor(objective="SquaredLoss", iteration_limit=50)
    model_sklearn.fit(X_train, y_train)

    preds_sklearn = model_sklearn.predict(X_test)

    model_booster = PerpetualBooster(objective="SquaredLoss", iteration_limit=50)
    model_booster.fit(X_train, y_train)

    preds_booster = model_booster.predict(X_test)

    np.allclose(preds_sklearn, preds_booster)


def test_sklearn_compat_ranking(X_y_g):

    X, y, group_lengths = X_y_g

    model_sklearn = PerpetualRanker(objective="ListNetLoss", iteration_limit=50)
    model_sklearn.fit(X, y, group=group_lengths)

    preds_sklearn = model_sklearn.predict(X)

    model_booster = PerpetualBooster(objective="ListNetLoss", iteration_limit=50)
    model_booster.fit(X, y, group=group_lengths)

    preds_booster = model_booster.predict(X)

    np.allclose(preds_sklearn, preds_booster)
