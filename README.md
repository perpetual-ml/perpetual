<p align="center">
  <img  height="120" src="https://github.com/perpetual-ml/perpetual/raw/main/resources/perp_logo.png">
</p>

<div align="center">

[![Python Versions](https://img.shields.io/pypi/pyversions/perpetual.svg?logo=python&logoColor=white)](https://pypi.org/project/perpetual)
[![PyPI Version](https://img.shields.io/pypi/v/perpetual.svg?logo=pypi&logoColor=white)](https://pypi.org/project/perpetual)
[![Crates.io Version](https://img.shields.io/crates/v/perpetual?logo=rust&logoColor=white)](https://crates.io/crates/perpetual)

</div>

# Perpetual

## _A self-generalizing, hyperparameter-free gradient boosting machine_

PerpetualBooster is a gradient boosting machine (GBM) algorithm which doesn't have hyperparameters to be tuned so that you can use it without hyperparameter optimization packages unlike other GBM algorithms. Similar to AutoML libraries, it has a `budget` parameter. Increasing the `budget` parameter increases the predictive power of the algorithm and gives better results on unseen data. Start with a small budget and increase it once you are confident with your features. If you don't see any improvement with further increasing the `budget`, it means that you are already extracting the most predictive power out of your data.

## Benchmark

Hyperparameter optimization usually takes 100 iterations with plain GBM algorithms. PerpetualBooster achieves the same accuracy in the single run. Thus, it achieves around 100x speed-up at the same accuracy with different `budget` levels and with different datasets. The speed-up might be slightly lower or significantly higher than 100x depending on the dataset.

The following table summarizes the results for the [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset:

| Perpetual budget | LightGBM n_estimators | Perpetual mse | LightGBM mse | Perpetual cpu time | LightGBM cpu time | Speed-up |
| ---------------- | --------------------- | ------------- | ------------ | ------------------ | ----------------- | -------- |
| 1.1              | 100                   | 0.192         | 0.192        | 8.9                | 1003              | 113x     |
| 1.2              | 200                   | 0.190         | 0.191        | 11.0               | 2030              | 186x     |
| 1.5              | 300                   | 0.187         | 0.188        | 18.7               | 3272              | 179x     |

You can reproduce the results using the [performance_benchmark.ipynb](./python-package/examples/performance_benchmark.ipynb) notebook in the [examples](./python-package/examples) folder.

## Usage

You can use the algorithm like in the example below. Check examples folders for both Rust and Python.

```python
from perpetual import PerpetualBooster

model = PerpetualBooster(objective="SquaredLoss")
model.fit(X, y, budget=0.4)
```

## Documentation

Documentation for the Python API can be found [here](https://perpetual-ml.github.io/perpetual) and for the Rust API [here](https://docs.rs/perpetual/latest/perpetual/).

## Installation

The package can be installed directly from [pypi](https://pypi.org/project/perpetual).

```shell
pip install perpetual
```

To use in a rust project, add the following to your Cargo.toml file to get the package from [crates.io](https://crates.io/crates/perpetual).

```toml
perpetual = "0.1.0"
```

## Paper

PerpetualBooster prevents overfitting with a generalization algorithm. The paper is work-in-progress to explain how the algorithm works.
