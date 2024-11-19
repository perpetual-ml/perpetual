<p align="center">
  <img  height="120" src="https://github.com/perpetual-ml/perpetual/raw/main/resources/perp_logo.png">
</p>

<div align="center">

[![Python Versions](https://img.shields.io/pypi/pyversions/perpetual.svg?logo=python&logoColor=white)](https://pypi.org/project/perpetual)
[![PyPI Version](https://img.shields.io/pypi/v/perpetual.svg?logo=pypi&logoColor=white)](https://pypi.org/project/perpetual)
[![Crates.io Version](https://img.shields.io/crates/v/perpetual?logo=rust&logoColor=white)](https://crates.io/crates/perpetual)
[![Static Badge](https://img.shields.io/badge/join-discord-blue?logo=discord)](https://discord.gg/AyUK7rr6wy)

</div>

# Perpetual

PerpetualBooster is a gradient boosting machine (GBM) algorithm which doesn't need hyperparameter optimization unlike other GBM algorithms. Similar to AutoML libraries, it has a `budget` parameter. Increasing the `budget` parameter increases the predictive power of the algorithm and gives better results on unseen data. Start with a small budget (e.g. 1.0) and increase it (e.g. 2.0) once you are confident with your features. If you don't see any improvement with further increasing the `budget`, it means that you are already extracting the most predictive power out of your data.

## Benchmark

Hyperparameter optimization usually takes 100 iterations with plain GBM algorithms. PerpetualBooster achieves the same accuracy in a single run. Thus, it achieves up to 100x speed-up at the same accuracy with different `budget` levels and with different datasets.

The following table summarizes the results for the [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset (regression):

| Perpetual budget | LightGBM n_estimators | Perpetual mse | LightGBM mse | Speed-up wall time | Speed-up cpu time |
| ---------------- | --------------------- | ------------- | ------------ | ------------------ | ----------------- |
| 1.0              | 100                   | 0.192         | 0.192        | 54x                | 56x               |
| 1.5              | 300                   | 0.188         | 0.188        | 59x                | 58x               |
| 2.1              | 1000                  | 0.185         | 0.186        | 42x                | 41x               |

The following table summarizes the results for the [Cover Types](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) dataset (classification):

| Perpetual budget | LightGBM n_estimators | Perpetual log loss | LightGBM log loss | Speed-up wall time | Speed-up cpu time |
| ---------------- | --------------------- | ------------------ | ----------------- | ------------------ | ----------------- |
| 0.9              | 100                   | 0.091              | 0.084             | 72x                | 78x               |

You can reproduce the results using the scripts in the [examples](./python-package/examples) folder.

## Usage

You can use the algorithm like in the example below. Check examples folders for both Rust and Python.

```python
from perpetual import PerpetualBooster

model = PerpetualBooster(objective="SquaredLoss")
model.fit(X, y, budget=1.0)
```

## Documentation

Documentation for the Python API can be found [here](https://perpetual-ml.github.io/perpetual) and for the Rust API [here](https://docs.rs/perpetual/latest/perpetual/).

## Installation

The package can be installed directly from [pypi](https://pypi.org/project/perpetual):

```shell
pip install perpetual
```

Using [conda-forge](https://anaconda.org/conda-forge/perpetual):

```shell
conda install conda-forge::perpetual
```

To use in a Rust project and to get the package from [crates.io](https://crates.io/crates/perpetual):

```shell
cargo add perpetual
```

## Contribution

Contributions are welcome. Check CONTRIBUTING.md for the guideline.

## Paper

PerpetualBooster prevents overfitting with a generalization algorithm. The paper is work-in-progress to explain how the algorithm works. Check our [blog post](https://perpetual-ml.com/blog/how-perpetual-works) for a high level introduction to the algorithm.
