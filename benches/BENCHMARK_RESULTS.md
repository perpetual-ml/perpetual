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
| `cal_housing` | `decisioning` | **1.96s** | `[1.8850s - 2.0485s]` | ~5% faster than baseline |
| `cal_housing` | `v0.10.0` | 2.06s | `[2.0157s - 2.1134s]` | Baseline |
| `cover_types` | `decisioning` | 2.43s | `[1.9510s - 3.0993s]` | ~29% slower than baseline |
| `cover_types` | `v0.10.0` | **1.88s** | `[1.8389s - 1.9254s]` | Baseline |

## Summary

- **Cal Housing**: `decisioning` branch shows a slight performance improvement.
- **Cover Types**: `decisioning` branch shows a significant regression (~29%).
