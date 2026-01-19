# Benchmark Results: Decisioning Branch vs v0.10.0 Baseline

Date: 2026-01-18

## Configuration

- **Hardware**: Local Machine
- **Benchmark**: `training_benchmark.rs`
- **Samples**: 40
- **Measurement Time**: 60s
- **Cover Types Optimization**: Top 10 features only (indices 0-9)

## Results

| Benchmark | Branch | Result (Time) | confidence interval | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `cal_housing` | `decisioning + opt 1 (hoist bin sums)` | **1.73s** | `[1.69s - 1.78s]` | ~16% faster than baseline |
| `cal_housing` | `decisioning + polars removal, cargo test fixes` | 2.26s | `[2.1828s - 2.3434s]` | Regression from decisioning |
| `cal_housing` | `decisioning` | 1.96s | `[1.8850s - 2.0485s]` | ~5% faster than baseline |
| `cal_housing` | `v0.10.0` | 2.06s | `[2.0157s - 2.1134s]` | Baseline |
| `cover_types` | `decisioning + opt 1 (hoist bin sums)` | **1.54s** | `[1.50s - 1.58s]` | ~18% faster than baseline |
| `cover_types` | `decisioning + polars removal, cargo test fixes` | 2.06s | `[1.9588s - 2.1870s]` | Improvement from decisioning |
| `cover_types` | `decisioning` | 2.43s | `[1.9510s - 3.0993s]` | ~29% slower than baseline |
| `cover_types` | `v0.10.0` | 1.88s | `[1.8389s - 1.9254s]` | Baseline |

## Summary

- **Optimization 1 (Hoist Bin Sums)**: Significant performance improvement observed. `cal_housing` reduced to **1.73s** and `cover_types` to **1.54s**. This validates the hypothesis that redundant bin sum calculations in the inner loop were a bottleneck.
- **Cal Housing**: The `decisioning` branch showed improvement (1.96s), but recent changes (polars removal, test fixes) introduced a regression (2.26s).
- **Cover Types**: The `decisioning` branch showed regression (2.43s), but recent changes improved it to 2.06s, though still slightly slower than the v0.10.0 baseline (1.88s).
