import pickle
import numpy as np
import pandas as pd
import pytest
import perpetual
from perpetual import PerpetualBooster


def X_y_so():
    df = pd.read_csv("../resources/titanic.csv")
    X = df.select_dtypes("number").drop(columns="survived").reset_index(drop=True)
    y = df["survived"]
    return X, y


def X_y_mo():
    X = pd.read_csv("../resources/cover_types_train.csv", index_col=False)
    X = X.sample(n=1000, random_state=0)
    X.dropna(inplace=True)
    X = X.loc[:, (X != X.iloc[0]).any()]
    y = X.pop("Cover_Type")
    return X, y


def pickle_booster(model: PerpetualBooster, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(model, file)


def unpickle_booster(path: str) -> PerpetualBooster:
    with open(path, "rb") as file:
        return pickle.load(file)


def save_booster(model: PerpetualBooster, path: str) -> None:
    model.save_booster(path)


def load_booster(path: str) -> PerpetualBooster:
    return PerpetualBooster.load_booster(path)


@pytest.mark.parametrize("X_y", [X_y_mo, X_y_so])
@pytest.mark.parametrize(
    "load_func,save_func",
    [(unpickle_booster, pickle_booster), (load_booster, save_booster)],
)
class TestSaveLoadFunctions:
    def test_booster_metadata(self, X_y, tmp_path, load_func, save_func):
        f64_model_path = tmp_path / "modelf64_sl.json"
        X, y = X_y()
        model = PerpetualBooster(
            objective="SquaredLoss", iteration_limit=10, memory_limit=1.0
        )
        save_func(model, f64_model_path)
        model.json_dump()
        model.fit(X, y)
        preds = model.predict(X)
        save_func(model, f64_model_path)
        model.insert_metadata("test-info", "some-info")
        assert model.get_metadata("test-info") == "some-info"
        save_func(model, f64_model_path)

        loaded = load_func(f64_model_path)
        assert loaded.get_metadata("test-info") == "some-info"

        with pytest.raises(KeyError):
            loaded.get_metadata("No-key")

        loaded_dict = loaded.__dict__
        model_dict = model.__dict__

        assert sorted(loaded_dict.keys()) == sorted(model_dict.keys())
        for k, v in loaded_dict.items():
            c_v = model_dict[k]
            if isinstance(v, float):
                if np.isnan(v):
                    assert np.isnan(c_v)
                else:
                    assert np.allclose(v, c_v)
            elif isinstance(v, perpetual.booster.CratePerpetualBooster) or isinstance(
                v, perpetual.booster.CrateMultiOutputBooster
            ):
                assert isinstance(
                    c_v, perpetual.booster.CratePerpetualBooster
                ) or isinstance(v, perpetual.booster.CrateMultiOutputBooster)
            else:
                print("else_block:")
                print(k)
                print(v)
                print(c_v)
                assert v == c_v, k
        loaded_preds = loaded.predict(X)
        assert np.allclose(preds, loaded_preds)

    def test_booster_saving(self, X_y, tmp_path, load_func, save_func):
        # SquaredLoss
        f64_model_path = tmp_path / "modelf64_sl.json"
        X, y = X_y()
        X = X
        model = PerpetualBooster(
            objective="SquaredLoss", iteration_limit=10, memory_limit=1.0
        )
        model.fit(X, y)
        preds = model.predict(X)
        save_func(model, f64_model_path)
        model_loaded = load_func(f64_model_path)
        assert all(preds == model_loaded.predict(X))

        # LogLoss
        f64_model_path = tmp_path / "modelf64_ll.json"
        X, y = X_y()
        model = PerpetualBooster(
            objective="LogLoss", iteration_limit=10, memory_limit=1.0
        )
        model.fit(X, y)
        preds = model.predict(X)
        save_func(model, f64_model_path)
        model_loaded = load_func(f64_model_path)
        assert model_loaded.feature_names_in_ == model.feature_names_in_
        assert model_loaded.feature_names_in_ == X.columns.to_list()
        assert all(preds == model_loaded.predict(X))

    def test_booster_saving_with_monotone_constraints(
        self, X_y, tmp_path, load_func, save_func
    ):
        # squared loss
        f64_model_path = tmp_path / "modelf64_sl.json"
        X, y = X_y()

        def calculate_monotonicity(x, y):
            correlation = x.corr(y)
            if np.isnan(correlation):
                return 0  # Or another appropriate default value
            else:
                return int(np.sign(correlation))

        mono_ = X.apply(lambda x: calculate_monotonicity(x, y)).to_dict()

        model = PerpetualBooster(
            objective="SquaredLoss",
            monotone_constraints=mono_,
            iteration_limit=10,
            memory_limit=1.0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        save_func(model, f64_model_path)
        model_loaded = load_func(f64_model_path)
        assert model_loaded.feature_names_in_ == model.feature_names_in_
        assert model_loaded.feature_names_in_ == X.columns.to_list()
        assert all(preds == model_loaded.predict(X))
        assert all(
            [
                model.monotone_constraints[ft] == model_loaded.monotone_constraints[ft]
                for ft in model_loaded.feature_names_in_
            ]
        )
        assert all(
            [
                model.monotone_constraints[ft] == model_loaded.monotone_constraints[ft]
                for ft in model.feature_names_in_
            ]
        )
        assert all(
            [
                model.monotone_constraints[ft] == model_loaded.monotone_constraints[ft]
                for ft in mono_.keys()
            ]
        )

        # LogLoss
        f64_model_path = tmp_path / "modelf64_ll.json"
        X, y = X_y()
        X = X
        model = PerpetualBooster(
            objective="LogLoss",
            monotone_constraints=mono_,
            iteration_limit=10,
            memory_limit=1.0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        save_func(model, f64_model_path)
        model_loaded = load_func(f64_model_path)
        assert all(preds == model_loaded.predict(X))
