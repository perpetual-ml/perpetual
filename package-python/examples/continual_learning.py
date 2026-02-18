import time

import numpy as np
import pandas as pd
from perpetual import PerpetualBooster
from sklearn.datasets import fetch_california_housing


def continual_learning_example():
    # Load Data from sklearn
    print("Fetching California Housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    n_samples = len(X)
    n_features = X.shape[1]

    print(f"Total samples: {n_samples}, Features: {n_features}")

    # Configuration
    initial_batch_size = 2000
    batch_size = 1000

    strategies = ["Continual", "Retrain"]

    print(
        f"{'Strategy':<15} | {'Batch':<5} | {'Time(ms)':<10} | {'MSE':<8} | {'Dataset%'}"
    )
    print("-" * 60)

    for strategy in strategies:
        # Initial Batch
        initial_end = min(initial_batch_size, n_samples)
        X_initial = X.iloc[:initial_end]
        y_initial = y.iloc[:initial_end]

        # Initialize Model
        # Use string "SquaredLoss"
        reset = False if strategy == "Continual" else True
        model = PerpetualBooster(objective="SquaredLoss", budget=1.0, reset=reset)

        start_time = time.time()
        model.fit(X_initial, y_initial)
        initial_fit_time = (time.time() - start_time) * 1000

        # Initial eval
        preds = model.predict(X_initial)
        mse = np.mean((preds - y_initial) ** 2)

        print(
            f"{strategy:<15} | {0:<5} | {initial_fit_time:<10.2f} | {mse:<8.4f} | {initial_end / n_samples:.2%}"
        )

        current_idx = initial_end
        batch_idx = 1

        # Incremental Learning Loop
        while current_idx < n_samples:
            end_idx = min(current_idx + batch_size, n_samples)
            if current_idx >= end_idx:
                break

            X_batch = X.iloc[current_idx:end_idx]
            y_batch = y.iloc[current_idx:end_idx]

            # 1. Evaluate on next batch (before training/updating)
            preds = model.predict(X_batch)
            mse = np.mean((preds - y_batch) ** 2)

            # 2. Update / Retrain
            start_time = time.time()

            if strategy == "Retrain":
                # Retrain on all data seen so far
                X_cumulative = X.iloc[:end_idx]
                y_cumulative = y.iloc[:end_idx]

                # For "Retrain", a new model is created and fitted from scratch each time.
                model = PerpetualBooster(objective="SquaredLoss", budget=1.0)
                model.fit(X_cumulative, y_cumulative)

            else:  # Continual
                # Update with cumulative data but reuse model (warm start)
                X_cumulative = X.iloc[:end_idx]
                y_cumulative = y.iloc[:end_idx]

                # The model for "Continual" strategy is initialized once with reset=False
                # outside this while loop. Subsequent calls to fit on the same model
                # object will then continue training without resetting.
                model.fit(X_cumulative, y_cumulative)

            fit_time = (time.time() - start_time) * 1000

            print(
                f"{strategy:<15} | {batch_idx:<5} | {fit_time:<10.2f} | {mse:<8.4f} | {end_idx / n_samples:.2%}"
            )

            current_idx = end_idx
            batch_idx += 1


if __name__ == "__main__":
    continual_learning_example()
