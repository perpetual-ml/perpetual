"""
Compare PerpetualBooster and LightGBM on wind_onshore.csv time series (stationary, lag features).

uv run python package-python/examples/benchmark_wind_onshore.py
"""

import os
from time import process_time, time

import pandas as pd
from lightgbm import LGBMRegressor
from perpetual import PerpetualBooster
from sklearn.metrics import mean_squared_error


# --- Data Preparation ---
def load_wind_onshore_stationary(csv_path, test_size=0.2, lags=24):
    df = pd.read_csv(csv_path, sep=";")
    df["datetime"] = pd.to_datetime(df["Date/Time CET/CEST"], format="%d.%m.%Y/%H:%M")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["wind_onshore"] = (
        df["Wind Onshore [MWh]"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    # Stationary target: first difference
    df["wind_onshore_diff"] = df["wind_onshore"].diff()
    # Lag features from the stationary series
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["wind_onshore"].shift(lag)
    # Only stationary time features
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df = df.dropna().reset_index(drop=True)
    feature_cols = ["month", "dayofweek", "hour"] + [
        f"lag_{lag}" for lag in range(1, lags + 1)
    ]
    X = df[feature_cols]
    y = df["wind_onshore_diff"]
    n = len(df)
    n_test = int(n * test_size)

    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Try both possible resource paths
    possible_paths = [
        os.path.join("resources", "wind_onshore.csv"),
        os.path.join("..", "..", "resources", "wind_onshore.csv"),
        os.path.join("..", "resources", "wind_onshore.csv"),
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "wind_onshore.csv"
        ),
    ]
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    if csv_path is None:
        raise FileNotFoundError(
            "Could not find wind_onshore.csv in any expected location."
        )

    for lags in [0, 1, 2, 5, 10, 20, 50]:
        print(f"\n--- Benchmarking with {lags} lag features ---")
        X_train, X_test, y_train, y_test = load_wind_onshore_stationary(
            csv_path,
            test_size=0.2,
            lags=lags,
        )

        print(f"X_train.cols: {X_train.columns.tolist()}")

        # --- PerpetualBooster ---
        pb = PerpetualBooster(objective="SquaredLoss", budget=1.0)
        start = process_time()
        tick = time()
        pb.fit(X_train, y_train)
        pb_cpu = process_time() - start
        pb_wall = time() - tick
        y_pred_pb = pb.predict(X_test)
        mse_pb = mean_squared_error(y_test, y_pred_pb)
        print(f"PerpetualBooster.n_estimators: {pb.number_of_trees}")
        print(
            f"PerpetualBooster: mse={int(mse_pb)}, cpu={pb_cpu:.2f}s, wall={pb_wall:.2f}s"
        )

        # --- LightGBM ---
        lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        start = process_time()
        tick = time()
        lgbm.fit(X_train, y_train)
        lgbm_cpu = process_time() - start
        lgbm_wall = time() - tick
        y_pred_lgbm = lgbm.predict(X_test)
        mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
        print(
            f"LightGBM: mse={int(mse_lgbm)}, cpu={lgbm_cpu:.2f}s, wall={lgbm_wall:.2f}s"
        )
