from __future__ import annotations

import itertools
import json
import random
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from perpetual import PerpetualBooster
from sklearn.base import clone
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split


def loggodds_to_odds(v):
    return 1 / (1 + np.exp(-v))


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


def test_booster_max_cat():
    df = pd.read_csv("../resources/titanic.csv")
    X = df.drop(columns="survived").reset_index(drop=True)
    y = df["survived"]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    all_cols = X.columns.tolist()
    cat_cols = [x for x in all_cols if x not in num_cols]
    X[cat_cols] = X[cat_cols].astype("category")

    model = PerpetualBooster(objective="LogLoss", max_cat=4)
    model.fit(X, y)


def test_booster_no_variance(X_y):
    X, y = X_y
    X.iloc[:, 3] = 1
    X.iloc[:, 1] = np.nan

    model = PerpetualBooster(objective="LogLoss")
    model.fit(X, y)
    assert model.feature_importances_[1] == 0.0
    assert model.feature_importances_[3] == 0.0

    model.fit(X.iloc[:, [1]], y)
    assert len(np.unique(model.predict_log_proba(X.iloc[:, [1]]))) == 1

    model.fit(X.iloc[:, [3]], y)
    assert len(np.unique(model.predict_log_proba(X.iloc[:, [3]]))) == 1


def test_sklearn_clone(X_y):
    X, y = X_y
    model = PerpetualBooster(objective="LogLoss", num_threads=1)
    model_cloned = clone(model)
    model_cloned.fit(X, y)

    model.fit(X, y)

    # After it's fit, it can still be cloned.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_cloned_post_fit = clone(model)
    model_cloned_post_fit.fit(X, y)

    model_preds = model.predict_log_proba(X)
    model_cloned_preds = model_cloned.predict_log_proba(X)
    model_cloned_post_fit_preds = model_cloned_post_fit.predict_log_proba(X)

    assert np.allclose(model_preds, model_cloned_preds)
    assert np.allclose(model_preds, model_cloned_post_fit_preds)


def test_multiple_fit_calls(X_y):
    X, y = X_y
    model = PerpetualBooster(objective="LogLoss", num_threads=1)
    model.fit(X, y)
    preds = model.predict_log_proba(X)

    model.fit(X, y)
    fit_again_preds = model.predict_log_proba(X)

    assert np.allclose(preds, fit_again_preds)


def test_different_data_passed(X_y):
    X, y = X_y
    model = PerpetualBooster(objective="LogLoss", num_threads=1)
    model.fit(X, y)
    model.predict_log_proba(X)
    with pytest.raises(ValueError, match="Columns mismatch"):
        model.predict_log_proba(X.iloc[:, 0:4])

    with pytest.raises(ValueError, match="Columns mismatch"):
        X_ = X.rename(columns=lambda x: x + "_wrong" if x == "age" else x)
        model.predict_log_proba(X_)


def test_booster_from_numpy(X_y):
    X, y = X_y
    X = X.astype("float32").astype("float64")
    model1 = PerpetualBooster(objective="LogLoss")
    model1.fit(X, y)
    model1_preds = model1.predict_log_proba(X)

    model2 = PerpetualBooster(objective="LogLoss")
    model2.fit(X, y)
    model2_preds = model2.predict_log_proba(X.to_numpy())

    model3 = PerpetualBooster(objective="LogLoss")
    model3.fit(X.to_numpy().astype("float32"), y)
    model3_preds = model3.predict_log_proba(X)

    assert np.allclose(model1_preds, model2_preds)
    assert np.allclose(model2_preds, model3_preds)


def test_predict_proba(X_y):
    X, y = X_y
    model = PerpetualBooster(objective="LogLoss")
    model.fit(X, y)

    y_proba = model.predict_proba(X)

    assert np.allclose(y_proba.shape, (len(X), 2))


def test_predict_nodes(X_y):
    X, y = X_y

    model = PerpetualBooster(
        objective="LogLoss",
        budget=0.1,
        memory_limit=0.1,
        iteration_limit=10,
    )
    model.fit(X, y)

    prediction_nodes = model.predict_nodes(X)

    assert np.allclose(len(prediction_nodes), model.number_of_trees)
    assert np.allclose(len(prediction_nodes[0]), len(X))


def test_get_node_list(X_y):
    X, y = X_y

    model = PerpetualBooster(objective="LogLoss")
    model.fit(X, y)
    assert all(
        [
            isinstance(n.split_feature, int)
            for i, tree in enumerate(model.get_node_lists(map_features_names=False))
            for n in tree
        ]
    )
    assert all(
        [
            isinstance(n.split_feature, str)
            for i, tree in enumerate(model.get_node_lists(map_features_names=True))
            for n in tree
        ]
    )


test_args = itertools.product(
    [True, False], ["Weight", "Cover", "Gain", "TotalGain", "TotalCover"]
)


@pytest.mark.parametrize("is_numpy,importance_method", test_args)
def test_feature_importance_method_init(
    X_y: tuple[pd.DataFrame, pd.Series], is_numpy: bool, importance_method: str
) -> None:
    X, y = X_y
    model = PerpetualBooster(
        objective="LogLoss",
        feature_importance_method=importance_method,
    )
    if is_numpy:
        X_ = X.to_numpy()
    else:
        X_ = X
    model.fit(X_, y)

    imp = model.calculate_feature_importance(method=importance_method, normalize=True)

    # for ft, cf in zip(model.feature_names_in_, model.feature_importances_):
    #    print(imp.get(ft, 0.0), cf)
    #    print(imp.get(ft, 0.0) == cf)

    assert all(
        [
            imp.get(ft, 0.0) == cf
            for ft, cf in zip(model.feature_names_in_, model.feature_importances_)
        ]
    )


def test_booster_with_new_missing(X_y):
    X, y = X_y
    X = X
    model1 = PerpetualBooster(objective="LogLoss", num_threads=1)
    model1.fit(X, y=y)
    preds1 = model1.predict_log_proba(X)

    Xm = X.copy().fillna(-9999)
    model2 = PerpetualBooster(objective="LogLoss", missing=-9999, num_threads=1)
    model2.fit(Xm, y)
    preds2 = model2.predict_log_proba(Xm)
    assert np.allclose(preds1, preds2)


def test_monotone_constraints(X_y):
    X, y = X_y
    X = X
    mono_ = X.apply(lambda x: int(np.sign(x.corr(y)))).to_dict()
    model = PerpetualBooster(
        objective="SquaredLoss",
        monotone_constraints=mono_,
    )
    model.fit(X, y)
    for f, m in mono_.items():
        p_d = model.partial_dependence(X, feature=f)
        p_d = p_d[~np.isnan(p_d[:, 0])]
        if m < 0:
            assert np.all(p_d[0:-1, 1] >= p_d[1:, 1])
        else:
            assert np.all(p_d[0:-1, 1] <= p_d[1:, 1])

    for f, m in mono_.items():
        p_d = model.partial_dependence(X, feature=f, samples=None)
        p_d = p_d[~np.isnan(p_d[:, 0])]
        if m < 0:
            assert np.all(p_d[0:-1, 1] >= p_d[1:, 1])
        else:
            assert np.all(p_d[0:-1, 1] <= p_d[1:, 1])


def test_partial_dependence_errors(X_y):
    X, y = X_y
    model = PerpetualBooster(num_threads=1)
    model.fit(X, y)
    with pytest.raises(ValueError, match="If `feature` is a string, then"):
        model.partial_dependence(X.to_numpy(), "pclass")

    model = PerpetualBooster(num_threads=1)
    model.fit(X.to_numpy(), y)
    with pytest.warns(
        expected_warning=UserWarning, match="No feature names were provided at fit"
    ):
        res = model.partial_dependence(X, "pclass")

    # This is the same as if we used a dataframe at fit.
    model = PerpetualBooster(num_threads=1)
    model.fit(X, y)
    assert np.allclose(model.partial_dependence(X, "pclass"), res)

    model = PerpetualBooster(num_threads=1)
    model.fit(X, y)
    pclass_n = next(i for i, ft in enumerate(X.columns) if ft == "pclass")
    assert np.allclose(
        model.partial_dependence(X, "pclass"), model.partial_dependence(X, pclass_n)
    )
    assert np.allclose(
        model.partial_dependence(X, "pclass"),
        model.partial_dependence(X.to_numpy(), pclass_n),
    )

    with pytest.raises(
        ValueError, match="The parameter `feature` must be a string, or an int"
    ):
        model.partial_dependence(X, 1.0)


def test_partial_dependence_exclude_missing(X_y):
    X, y = X_y
    model = PerpetualBooster()
    model.fit(X, y)
    res1 = model.partial_dependence(X, "age", samples=None)
    res2 = model.partial_dependence(X, "age", samples=None, exclude_missing=False)
    assert (res1.shape[0] + 1) == res2.shape[0]
    assert np.allclose(res2[~np.isnan(res2[:, 0]), :], res1)


def test_booster_contributions_missing_branch_methods(X_y):
    X, y = X_y
    X = X
    model = PerpetualBooster(
        objective="LogLoss",
        create_missing_branch=True,
        allow_missing_splits=True,
        missing_node_treatment="AssignToParent",
    )
    model.fit(X, y)
    preds = model.predict_log_proba(X)
    contribs_average = model.predict_contributions(X)
    preds[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), preds)

    contribs_weight = model.predict_contributions(X, method="Weight")
    assert (contribs_weight[:, :-1][X.isna()] == 0).all()
    assert np.allclose(contribs_weight.sum(1), preds)
    assert not np.allclose(contribs_weight, contribs_average)

    contribs_branch_difference = model.predict_contributions(
        X, method="BranchDifference"
    )
    assert (contribs_branch_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_branch_difference.sum(1), preds)
    assert not np.allclose(contribs_branch_difference, contribs_average)

    contribs_midpoint_difference = model.predict_contributions(
        X, method="MidpointDifference"
    )
    assert (contribs_midpoint_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_midpoint_difference.sum(1), preds)
    assert not np.allclose(contribs_midpoint_difference, contribs_average)

    contribs_mode_difference = model.predict_contributions(X, method="ModeDifference")
    assert (contribs_mode_difference[:, :-1][X.isna()] == 0).all()
    assert not np.allclose(contribs_mode_difference.sum(1), preds)
    assert not np.allclose(contribs_mode_difference, contribs_average)


def test_booster_contributions(X_y):
    X, y = X_y
    X = X
    model = PerpetualBooster(objective="LogLoss")
    model.fit(X, y)
    preds = model.predict_log_proba(X)
    contribs_average = model.predict_contributions(X)
    preds[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), preds)

    contribs_weight = model.predict_contributions(X, method="Weight")
    assert np.allclose(contribs_weight.sum(1), preds)
    # assert not np.allclose(contribs_weight, contribs_average)

    contribs_difference = model.predict_contributions(X, method="BranchDifference")
    assert not np.allclose(contribs_difference.sum(1), preds)
    assert not np.allclose(contribs_difference, contribs_average)

    contribs_proba = model.predict_contributions(X, method="ProbabilityChange")
    assert np.allclose(contribs_proba.sum(1), loggodds_to_odds(preds))
    assert not np.allclose(contribs_proba, contribs_average)


def test_booster_contributions_shapley(X_y):
    X, y = X_y
    X = X.round(0)
    model = PerpetualBooster(objective="LogLoss")
    model.fit(X, y)
    preds = model.predict_log_proba(X)
    contribs_average = model.predict_contributions(X)
    preds[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    contribs_average.sum(1)[~np.isclose(contribs_average.sum(1), preds, rtol=5)]
    assert contribs_average.shape[1] == X.shape[1] + 1
    assert np.allclose(contribs_average.sum(1), preds)

    contribs_shapley = model.predict_contributions(X, method="Shapley")
    assert np.allclose(contribs_shapley.sum(1), preds)
    assert not np.allclose(contribs_shapley, contribs_average)


def test_missing_branch_with_contributions(X_y):
    X, y = X_y
    X = X
    model_miss_leaf = PerpetualBooster(
        objective="LogLoss", allow_missing_splits=False, create_missing_branch=True
    )
    model_miss_leaf.fit(X, y)
    model_miss_leaf_preds = model_miss_leaf.predict_log_proba(X)
    model_miss_leaf_conts = model_miss_leaf.predict_contributions(X)
    assert np.allclose(model_miss_leaf_conts.sum(1), model_miss_leaf_preds)

    model_miss_leaf_conts = model_miss_leaf.predict_contributions(X, method="weight")
    assert np.allclose(model_miss_leaf_conts.sum(1), model_miss_leaf_preds)

    model_miss_leaf_conts = model_miss_leaf.predict_contributions(
        X, method="probability-change"
    )
    assert np.allclose(
        model_miss_leaf_conts.sum(1), loggodds_to_odds(model_miss_leaf_preds)
    )

    model_miss_branch = PerpetualBooster(
        objective="LogLoss", allow_missing_splits=True, create_missing_branch=True
    )
    model_miss_branch.fit(X, y)
    model_miss_branch_preds = model_miss_branch.predict_log_proba(X)
    model_miss_branch_conts = model_miss_branch.predict_contributions(X)
    assert np.allclose(model_miss_branch_conts.sum(1), model_miss_branch_preds)
    assert not np.allclose(model_miss_branch_preds, model_miss_leaf_preds)

    model_miss_branch_conts = model_miss_branch.predict_contributions(
        X, method="weight"
    )
    assert np.allclose(model_miss_branch_conts.sum(1), model_miss_branch_preds)

    model_miss_branch_conts = model_miss_branch.predict_contributions(
        X, method="probability-change"
    )
    assert np.allclose(
        model_miss_branch_conts.sum(1), loggodds_to_odds(model_miss_branch_preds)
    )


def test_text_dump(X_y):
    model = PerpetualBooster()
    model.fit(*X_y)
    assert len(model.text_dump()) > 0


def test_booster_terminate_missing_features(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    model = PerpetualBooster(
        objective="LogLoss",
        allow_missing_splits=True,
        create_missing_branch=True,
        terminate_missing_features=["pclass", "fare"],
    )
    model.fit(X, y)
    [pclass_idx] = [i for i, f in enumerate(X.columns) if f == "pclass"]
    [fare_idx] = [i for i, f in enumerate(X.columns) if f == "fare"]

    def check_feature_split(feature: int, nodes: dict, n: int):
        node = nodes[str(n)]
        if node["is_leaf"]:
            return
        if node["split_feature"] == feature:
            if not nodes[str(node["missing_node"])]["is_leaf"]:
                raise ValueError("Node split more!")
        check_feature_split(feature, nodes, node["missing_node"])
        check_feature_split(feature, nodes, node["left_child"])
        check_feature_split(feature, nodes, node["right_child"])

    # Does pclass never get split out?
    for tree in json.loads(model.json_dump())["trees"]:
        nodes = tree["nodes"]
        check_feature_split(pclass_idx, nodes, 0)
        check_feature_split(fare_idx, nodes, 0)

    model = PerpetualBooster(
        objective="SquaredLoss",
        allow_missing_splits=True,
        create_missing_branch=True,
        # terminate_missing_features=["pclass"]
    )
    model.fit(X, y)

    # Does age never get split out?
    pclass_one_bombed = False
    for tree in json.loads(model.json_dump())["trees"]:
        try:
            check_feature_split(pclass_idx, tree["nodes"], 0)
        except ValueError:
            pclass_one_bombed = True
    assert pclass_one_bombed

    fare_one_bombed = False
    for tree in json.loads(model.json_dump())["trees"]:
        try:
            check_feature_split(fare_idx, tree["nodes"], 0)
        except ValueError:
            fare_one_bombed = True
    assert fare_one_bombed


def test_missing_treatment(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    model = PerpetualBooster(
        objective="LogLoss",
        allow_missing_splits=False,
        create_missing_branch=True,
        missing_node_treatment="AverageLeafWeight",
    )
    model.fit(X, y)
    preds = model.predict_log_proba(X)

    contribus_weight = model.predict_contributions(X, method="weight")
    assert contribus_weight.shape[1] == X.shape[1] + 1
    assert np.allclose(contribus_weight.sum(1), preds, atol=0.001)
    assert np.allclose(contribus_weight[:, :-1][X.isna()], 0, atol=0.001)
    assert (contribus_weight[:, :-1][X.isna()] == 0).all()
    assert (X.isna().sum().sum()) > 0

    contribus_average = model.predict_contributions(X, method="average")
    assert contribus_average.shape[1] == X.shape[1] + 1
    # Wont be exactly zero because of floating point error.
    assert np.allclose(contribus_average[:, :-1][X.isna()], 0, atol=0.001)
    assert np.allclose(contribus_weight.sum(1), preds)
    assert (X.isna().sum().sum()) > 0

    # Even the contributions should be approximately the same.
    assert np.allclose(contribus_average, contribus_weight, atol=0.001)


def test_missing_treatment_split_further(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    model = PerpetualBooster(
        objective="LogLoss",
        create_missing_branch=True,
        allow_missing_splits=True,
        missing_node_treatment="AverageLeafWeight",
        terminate_missing_features=["pclass", "fare"],
    )

    [pclass_idx] = [i for i, f in enumerate(X.columns) if f == "pclass"]
    [fare_idx] = [i for i, f in enumerate(X.columns) if f == "fare"]

    model.fit(X, y)
    preds = model.predict_log_proba(X)
    contribus_weight = model.predict_contributions(X, method="weight")

    # For the features that we terminate missing, these features should have a contribution
    # of zero.
    assert (contribus_weight[:, pclass_idx][X["pclass"].isna()] == 0).all()
    assert (contribus_weight[:, fare_idx][X["fare"].isna()] == 0).all()

    # For the others it might not be zero.
    all_others = [i for i in range(X.shape[1]) if i not in [fare_idx, pclass_idx]]
    assert (contribus_weight[:, all_others][X.iloc[:, all_others].isna()] != 0).any()
    assert (contribus_weight[:, all_others][X.iloc[:, all_others].isna()] != 0).any()

    contribus_average = model.predict_contributions(X, method="average")
    assert np.allclose(contribus_weight.sum(1), preds)
    assert np.allclose(contribus_average, contribus_weight, atol=0.001)


def test_AverageNodeWeight_missing_node_treatment(X_y):
    X, y = X_y
    X = X.copy()
    missing_mask = np.random.default_rng(0).uniform(0, 1, size=X.shape)
    X = X.mask(missing_mask < 0.3)
    model = PerpetualBooster(
        objective="LogLoss",
        allow_missing_splits=True,
        create_missing_branch=True,
        terminate_missing_features=["pclass", "fare"],
        missing_node_treatment="AverageNodeWeight",
    )
    model.fit(X, y)

    def check_missing_is_average(tree: dict, n: int):
        node = tree[str(n)]
        if node["is_leaf"]:
            return
        missing_weight = tree[str(node["missing_node"])]["weight_value"]
        left = tree[str(node["left_child"])]
        right = tree[str(node["right_child"])]

        weighted_weight = (
            (left["weight_value"] * left["hessian_sum"])
            + (right["weight_value"] * right["hessian_sum"])
        ) / (left["hessian_sum"] + right["hessian_sum"])

        assert np.isclose(missing_weight, weighted_weight)

        check_missing_is_average(tree, node["missing_node"])
        check_missing_is_average(tree, node["left_child"])
        check_missing_is_average(tree, node["right_child"])

    # Does pclass never get split out?
    for tree in json.loads(model.json_dump())["trees"]:
        check_missing_is_average(tree["nodes"], 0)

    model = PerpetualBooster(
        objective="SquaredLoss",
        allow_missing_splits=True,
        create_missing_branch=True,
        # terminate_missing_features=["pclass"]
    )
    model.fit(X, y)

    with pytest.raises(AssertionError):
        for tree in json.loads(model.json_dump())["trees"]:
            check_missing_is_average(tree["nodes"], 0)


def test_get_params(X_y):
    X, y = X_y
    nt = 2
    model = PerpetualBooster(num_threads=nt)
    assert model.get_params()["num_threads"] == nt
    model.fit(X, y)
    assert model.get_params()["num_threads"] == nt


def test_set_params(X_y):
    X, y = X_y
    model = PerpetualBooster(num_threads=2)
    assert model.get_params()["num_threads"] == 2
    assert model.set_params(num_threads=1)
    assert model.get_params()["num_threads"] == 1
    model.fit(X, y)


def test_categorical():
    X = pd.read_csv("../resources/titanic_test_df.csv", index_col=False)
    y = np.array(
        pd.read_csv(
            "../resources/titanic_test_y.csv", index_col=False, header=None
        ).squeeze("columns")
    )
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
    model = PerpetualBooster()
    model.fit(X, y)


def test_polars():
    import polars as pl

    X = pl.from_pandas(pd.read_csv("../resources/titanic_test_df.csv", index_col=False))
    y = np.array(
        pd.read_csv(
            "../resources/titanic_test_y.csv", index_col=False, header=None
        ).squeeze("columns")
    )
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
    X = X.with_columns(pl.col(cols).cast(pl.String).cast(pl.Categorical))
    model = PerpetualBooster(budget=0.1)
    model.fit(X, y)
    model.predict(X)
    model.trees_to_dataframe()


def test_calibration():
    X_train = pd.read_csv("../resources/cal_housing_train.csv", index_col=False)
    y_train = X_train.pop("MedHouseVal").to_numpy()
    X_cal = pd.read_csv("../resources/cal_housing_test.csv", index_col=False)
    y_cal = X_cal.pop("MedHouseVal").to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, random_state=42
    )

    model = PerpetualBooster(objective="SquaredLoss")
    model.fit(X_train, y_train)

    alpha = [0.1]
    model.calibrate(X_train, y_train, X_cal, y_cal, alpha)
    predicted_intervals = model.predict_intervals(X_test)

    for i, (alpha_, intervals) in enumerate(predicted_intervals.items()):
        lower_preds = intervals[0]
        upper_preds = intervals[1]
        assert float(alpha_) == alpha[i]
        target_coverage = 1.0 - alpha[i]
        coverage_count = 0
        for j in range(len(X_test)):
            if lower_preds[j] < y_test[j] < upper_preds[j]:
                coverage_count += 1
        actual_coverage = coverage_count / len(X_test)
        assert actual_coverage > target_coverage


def test_pruning():
    X_train = pd.read_csv("../resources/titanic_test_df.csv", index_col=False)
    y_train = np.array(
        pd.read_csv(
            "../resources/titanic_test_y.csv", index_col=False, header=None
        ).squeeze("columns")
    )
    X_cal = pd.read_csv("../resources/titanic_test_df.csv", index_col=False)
    y_cal = np.array(
        pd.read_csv(
            "../resources/titanic_test_y.csv", index_col=False, header=None
        ).squeeze("columns")
    )
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
    X_train[cols] = X_train[cols].astype("category")
    X_cal[cols] = X_cal[cols].astype("category")
    model = PerpetualBooster()
    model.fit(X_train, y_train)
    model.prune(X_cal, y_cal)


def test_ranking(X_y_g):
    X, y, group = X_y_g
    assert len(y) == sum(group)

    model = PerpetualBooster(objective="ListNetLoss")
    model.fit(X, y, group=group)

    random.seed(42)
    random_relevance = [random.random() for _ in range(len(y))]

    yhat = model.predict(X)

    assert np.isnan(yhat).sum() == 0
    assert np.isinf(yhat).sum() == 0

    start = 0
    end = 0
    model_ndcgs = []
    random_ndcgs = []
    for group_length in group:
        end += group_length
        real_y = y[start:end]
        model_y = yhat[start:end]
        random_y = random_relevance[start:end]

        k = 10

        model_ndcg = ndcg_score([real_y], [model_y], k=k)
        random_ndcg = ndcg_score([real_y], [random_y], k=k)

        model_ndcgs.append(model_ndcg)
        random_ndcgs.append(random_ndcg)

        start = end

    assert np.mean(model_ndcgs) > np.mean(random_ndcgs)


def test_custom_objective():

    X_train = pd.read_csv("../resources/cal_housing_train.csv", index_col=False)
    y_train = X_train.pop("MedHouseVal").to_numpy()

    X_test = pd.read_csv("../resources/cal_housing_test.csv", index_col=False)
    X_test.pop("MedHouseVal").to_numpy()

    model_regular_loss = PerpetualBooster(objective="SquaredLoss")
    model_regular_loss.fit(X_train, y_train)

    def loss(y, pred, weight, group):
        return (y - pred) ** 2

    def gradient(y, pred, weight, group):
        return (pred - y), None

    def initial_value(y, weight, group):
        return np.mean(y)

    model_custom_loss = PerpetualBooster(objective=(loss, gradient, initial_value))
    model_custom_loss.fit(X_train, y_train)

    pred_regular = model_regular_loss.predict(X_test)
    pred_custom = model_custom_loss.predict(X_test)

    assert np.allclose(pred_regular, pred_custom)
