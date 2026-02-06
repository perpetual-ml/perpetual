import numpy as np
from perpetual import PerpetualBooster


def test_interaction_constraints():
    # Similar data to Rust test
    X = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # 3 features

    y = np.array([1.0, 0.0, 0.5, 0.5])

    # Constraints: [[0, 1], [2]]
    # 0 and 1 can interact. 2 is isolated.
    interaction_constraints = [[0, 1], [2]]

    booster = PerpetualBooster(
        objective="SquaredLoss",
        budget=1.0,
        interaction_constraints=interaction_constraints,
        max_bin=256,
        allow_missing_splits=True,
    )
    booster.fit(X, y)

    # Text dump to verify (soft check)
    trees = booster.text_dump()
    print("Trees dump:")
    for tree in trees:
        print(tree)

    json_dump = booster.json_dump()
    assert "interaction_constraints" in json_dump

    params = booster.get_params()
    assert "interaction_constraints" in params
    assert params["interaction_constraints"] == [[0, 1], [2]]

    print("Interaction constraints test passed!")


if __name__ == "__main__":
    test_interaction_constraints()
