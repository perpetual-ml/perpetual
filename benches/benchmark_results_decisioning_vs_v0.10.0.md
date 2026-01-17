# Benchmark Results: Decisioning vs v0.10.0

## Overview

We benchmarked the training performance of the `decisioning` branch against the `v0.10.0` release.
The benchmark includes two datasets:

1. **California Housing** (Regression, `SquaredLoss`)
2. **Cover Types** (Classification, `LogLoss`)

## Results

| Benchmark | v0.10.0 (Average) | Decisioning (Average) | Improvement |
| :--- | :--- | :--- | :--- |
| **Housing (Regression)** | 1.241 s | 1.045 s | **~15.8% Faster** |
| **Cover Types (Classification)** | 15.725 s | 13.701 s | **~12.9% Faster** |

## Methodology

- **Objective Functions**: Explicitly set to `SquaredLoss` for Housing and `LogLoss` for Cover Types to ensuring fair comparison.
- **Iterations**: 10 samples per benchmark.
- **Warm-up**: 5 seconds.
- **Measurement**: 20 seconds.
