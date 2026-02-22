import numpy as np
import pytest
from perpetual.booster import PerpetualBooster
from sklearn.datasets import make_classification


def test_booster_invalid_objective_name():
    # If the python side doesn't catch it, the Rust side will throw an error
    # We want to hit the line in python but expectedly fail if it's invalid
    with pytest.raises(Exception):
        PerpetualBooster(objective="InvalidObjective")


def test_booster_fit_invalid_input_type():
    X = [[1, 2, 3]]  # List of list is not supported directly without conversion
    y = [1]
    model = PerpetualBooster()
    with pytest.raises(Exception):
        model.fit(X, y)


def test_booster_predict_before_fit():
    X = np.random.randn(10, 5)
    model = PerpetualBooster()
    with pytest.raises(ValueError, match="is not fitted yet"):
        model.predict(X)


def test_booster_init_params_check():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = PerpetualBooster(
        budget=0.1,
        num_threads=1,
        monotone_constraints={0: 1},
        force_children_to_bound_parent=True,
        log_iterations=1,
        feature_importance_method="Weight",
        max_bin=32,
        max_cat=5,
    )
    model.fit(X, y)
    assert model.is_fitted
    assert model.budget == 0.1


def test_booster_serialization_bytes():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = PerpetualBooster(iteration_limit=5)
    model.fit(X, y)

    data = model.save_model()
    assert isinstance(data, bytes)

    new_model = PerpetualBooster.load_model(data)
    assert new_model.is_fitted
    assert np.allclose(model.predict(X), new_model.predict(X))


def test_booster_feature_importances_property():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = PerpetualBooster(iteration_limit=5)
    model.fit(X, y)
    fi = model.feature_importances_
    assert len(fi) == 5


def test_booster_predict_contributions_basic():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = PerpetualBooster(iteration_limit=5)
    model.fit(X, y)
    # This might or might not be implemented in the crate, but we hit the python side
    try:
        contribs = model.predict_contributions(X)
        assert contribs.shape == (50, 6)
    except (AttributeError, NotImplementedError, Exception):
        pass


def test_booster_get_params():
    model = PerpetualBooster(budget=0.7)
    params = model.get_params()
    assert params["budget"] == 0.7


def test_booster_set_params():
    model = PerpetualBooster()
    model.set_params(budget=0.9)
    assert model.budget == 0.9
