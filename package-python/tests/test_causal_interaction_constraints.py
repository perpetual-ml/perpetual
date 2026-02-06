import numpy as np
from perpetual import BraidedBooster, UpliftBooster


def test_braided_interaction_constraints():
    X = np.random.rand(100, 5)
    Z = np.random.rand(100, 2)
    y = np.random.rand(100)
    w = np.random.rand(100)

    interaction_constraints = [[0, 1], [2, 3, 4]]

    model = BraidedBooster(
        stage1_budget=0.1,
        stage2_budget=0.1,
        interaction_constraints=interaction_constraints,
    )
    model.fit(X, Z, y, w)

    # Check if constraints are stored in the underlying crate objects
    # We can check via json_dump
    json_str = model.to_json()
    assert "interaction_constraints" in json_str


def test_uplift_interaction_constraints():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    # Treatment must be binary 0 or 1
    w = np.random.randint(0, 2, 100)

    interaction_constraints = [[0, 2], [1, 3, 4]]

    model = UpliftBooster(
        outcome_budget=0.1,
        propensity_budget=0.1,
        effect_budget=0.1,
        interaction_constraints=interaction_constraints,
    )
    model.fit(X, w, y)

    json_str = model.to_json()
    assert "interaction_constraints" in json_str


if __name__ == "__main__":
    test_braided_interaction_constraints()
    test_uplift_interaction_constraints()
