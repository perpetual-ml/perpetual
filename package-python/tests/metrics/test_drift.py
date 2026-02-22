import numpy as np
from perpetual import PerpetualBooster


def test_drift_detection_single_booster():
    # 1. Prepare Data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

    # 2. Train Model - Regression (Single Booster)
    model = PerpetualBooster(objective="SquaredLoss", budget=1.0, save_node_stats=True)
    model.fit(X, y)

    # 3. Test on same data (should have low drift)
    drift_data = model.calculate_drift(X, drift_type="data", parallel=False)
    drift_concept = model.calculate_drift(X, drift_type="concept", parallel=False)

    print(
        f"Regression - Same data - Data Drift: {drift_data}, Concept Drift: {drift_concept}"
    )
    assert drift_data < 0.1
    assert drift_concept < 0.1

    # 4. Simulate Drift
    X_drifted = X.copy()
    X_drifted[:, 0] += 5.0  # Significant shift

    drift_data_shifted = model.calculate_drift(
        X_drifted, drift_type="data", parallel=False
    )
    print(f"Regression - Shifted data - Data Drift: {drift_data_shifted}")
    assert drift_data_shifted > drift_data


def test_drift_detection_multi_booster():
    # 1. Prepare Data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Binary classification with 0/1 targets triggers MultiOutputBooster if objective is not forced
    # but PerpetualBooster(objective="LogLoss") also uses MultiOutputBooster if classes > 2
    # For binary LogLoss it usually uses a single booster if n_classes=2.
    # To force MultiOutputBooster, we can use 3 classes.
    y = np.random.randint(0, 3, 100)

    # 2. Train Model - Multi-class (MultiOutputBooster)
    model = PerpetualBooster(budget=1.0, save_node_stats=True)
    model.fit(X, y)

    # 3. Test on same data
    drift_data = model.calculate_drift(X, drift_type="data", parallel=False)
    print(f"Multi-class - Same data - Data Drift: {drift_data}")
    assert drift_data < 0.1


def test_drift_no_stats():
    # If save_node_stats=False, it should return 0.0 (per current implementation)
    X = np.random.randn(100, 2)
    y = X[:, 0]
    model = PerpetualBooster(save_node_stats=False)
    model.fit(X, y)

    drift = model.calculate_drift(X)
    assert drift == 0.0


if __name__ == "__main__":
    test_drift_detection_single_booster()
    test_drift_detection_multi_booster()
    test_drift_no_stats()
    print("All tests passed!")
