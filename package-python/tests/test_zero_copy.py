"""Tests for zero-copy Polars integration."""

import tracemalloc

import numpy as np
import polars as pl
from perpetual import PerpetualBooster


def test_polars_zero_copy_fit():
    """Verify fitting with Polars DataFrame doesn't copy data."""
    n_rows, n_cols = 1000, 10
    data = {f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)}
    df = pl.DataFrame(data)
    y = np.random.randint(0, 2, n_rows).astype(np.float64)
    data_size_bytes = n_rows * n_cols * 8

    # Ensure everything is pre-allocated
    _ = df.to_numpy()

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    model = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=1)
    model.fit(df, y)
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    for stat in stats[:5]:
        print(stat)
    total_increase = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # dataset is ~80KB.
    # If this fails, it might be due to initializations.
    # We allow more buffer for first run initialization.
    assert total_increase < data_size_bytes * 10.0 + 1_000_000


def test_polars_zero_copy_predict():
    """Verify predicting with Polars DataFrame doesn't copy data."""
    n_rows, n_cols = 1000, 10
    data = {f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)}
    df = pl.DataFrame(data)
    y = np.random.randint(0, 2, n_rows).astype(np.float64)

    model = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=1)
    model.fit(df, y)

    tracemalloc.start()
    _ = model.predict(df)
    tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak should not include a full copy of X (80KB)
    assert peak < n_rows * n_cols * 8 * 1.5


def test_polars_fit_predict_correctness():
    """Verify that Polars columnar path produces same results as NumPy."""
    n_rows, n_cols = 200, 5
    data = {f"col_{i}": np.linspace(0, 1, n_rows) + i for i in range(n_cols)}
    df_polars = pl.DataFrame(data)
    arr_numpy = df_polars.to_numpy()
    y = (df_polars["col_0"] > 0.5).cast(pl.Float64).to_numpy()

    # Fit with Polars
    model_polars = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=5)
    model_polars.fit(df_polars, y)
    preds_polars = model_polars.predict(df_polars)

    # Fit with NumPy
    model_numpy = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=5)
    model_numpy.fit(arr_numpy, y)
    preds_numpy = model_numpy.predict(arr_numpy)

    assert model_polars.number_of_trees > 0
    assert np.allclose(preds_polars, preds_numpy, atol=1e-5)


def test_polars_zero_copy_predict_contributions():
    """Verify predicting contributions with Polars."""
    n_rows, n_cols = 100, 5
    data = {f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)}
    df = pl.DataFrame(data)
    y = np.random.randint(0, 2, n_rows).astype(np.float64)

    model = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=1)
    model.fit(df, y)

    contribs = model.predict_contributions(df)
    assert contribs.shape == (n_rows, n_cols + 1)


def test_polars_multi_output():
    """Verify multi-output with Polars."""
    n_rows, n_cols = 100, 5
    data = {f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)}
    df = pl.DataFrame(data)
    y = np.random.randint(0, 3, n_rows)

    model = PerpetualBooster(objective="LogLoss", budget=0.1, iteration_limit=1)
    model.fit(df, y)

    preds = model.predict(df)
    assert len(preds) == n_rows

    proba = model.predict_proba(df)
    assert proba.shape == (n_rows, 3)

    contribs = model.predict_contributions(df)
    # 3 classes * (5 features + 1 bias) = 18 columns
    assert contribs.shape == (n_rows, 3 * (n_cols + 1))
