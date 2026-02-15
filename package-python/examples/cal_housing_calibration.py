import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perpetual import PerpetualBooster
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def run_calibration_demo():
    # 1. Load data
    print("Loading California Housing data...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Split data: 60% train, 20% calibration, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42
    )

    print(
        f"Train size: {len(X_train)}, Calibration size: {len(X_cal)}, Test size: {len(X_test)}"
    )

    # 3. Define calibration methods to test
    methods = ["Conformal", "MinMax", "GRP", "WeightVariance"]
    alpha = 0.1  # 90% coverage target

    results = {}

    for method in methods:
        print(f"\n--- Method: {method} ---")

        # Initialize booster
        model = PerpetualBooster(
            objective="SquaredLoss", budget=1.0, save_node_stats=True
        )

        # Fit the model
        print("Fitting model...")
        model.fit(X_train, y_train)

        # Calibrate
        print("Calibrating...")
        if method == "Conformal":
            model.calibrate_conformal(X_train, y_train, X_cal, y_cal, alpha=[alpha])
        else:
            model.calibrate(X_cal, y_cal, alpha=[alpha], method=method)

        # Predict intervals on test set
        print("Predicting intervals...")
        intervals = model.predict_intervals(X_test)

        # Calculate coverage and average width
        lower = intervals[str(alpha)][:, 0]
        upper = intervals[str(alpha)][:, 1]

        covered = np.mean((y_test >= lower) & (y_test <= upper))
        avg_width = np.mean(upper - lower)

        print(f"Coverage: {covered:.2%}")
        print(f"Average Width: {avg_width:.4f}")

        results[method] = {
            "coverage": covered,
            "width": avg_width,
            "lower": lower,
            "upper": upper,
        }

    # 4. Simple visualization of first 50 test samples
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:50], "k.", label="Actual Values", alpha=0.6)

    colors = ["r", "g", "b", "orange"]
    for i, method in enumerate(methods):
        plt.plot(
            results[method]["lower"][:50], color=colors[i], linestyle="--", alpha=0.5
        )
        plt.plot(
            results[method]["upper"][:50],
            color=colors[i],
            linestyle="--",
            alpha=0.5,
            label=f"{method} bounds",
        )

    plt.title("Prediction Intervals (90% target) - First 50 Test Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("MedHouseVal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nSummary Results:")
    for method, metrics in results.items():
        print(
            f"{method:15}: Coverage={metrics['coverage']:.2%}, Avg Width={metrics['width']:.4f}"
        )


if __name__ == "__main__":
    run_calibration_demo()
