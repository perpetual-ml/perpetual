"""
Benchmark PerpetualBooster and LightGBM on the AirPassengers time series dataset (monthly airline passengers 1949-1960).

This dataset is clean, well-known, and not noisy. It is available via statsmodels and many open sources.

Usage:
    uv run python package-python/examples/benchmark_air_passengers.py
"""

from time import process_time, time

import pandas as pd
from lightgbm import LGBMRegressor
from perpetual import PerpetualBooster
from sklearn.metrics import mean_squared_error
from statsmodels.datasets import get_rdataset


def load_air_passengers(test_size=0.2, lags=12):
    # Load dataset from statsmodels (monthly airline passengers 1949-1960)
    data = get_rdataset("AirPassengers").data
    df = pd.DataFrame(
        {
            "Month": pd.date_range(
                start="1949-01-01", periods=len(data["value"]), freq="MS"
            ),
            "Passengers": data["value"],
        }
    )
    df = df.sort_values("Month").reset_index(drop=True)
    # Stationary target: first difference
    df["Passengers_diff"] = df["Passengers"].diff()
    # Lag features
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["Passengers"].shift(lag)
    df["month"] = df["Month"].dt.month
    df = df.dropna().reset_index(drop=True)
    feature_cols = ["month"] + [f"lag_{lag}" for lag in range(1, lags + 1)]
    X = df[feature_cols]
    y = df["Passengers_diff"]
    n = len(df)
    n_test = int(n * test_size)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    for lags in [0, 1, 3, 6, 12]:
        print(f"\n--- Benchmarking with {lags} lag features ---")
        X_train, X_test, y_train, y_test = load_air_passengers(test_size=0.2, lags=lags)
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
            f"PerpetualBooster: mse={mse_pb:.2f}, cpu={pb_cpu:.2f}s, wall={pb_wall:.2f}s"
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
            f"LightGBM: mse={mse_lgbm:.2f}, cpu={lgbm_cpu:.2f}s, wall={lgbm_wall:.2f}s"
        )
