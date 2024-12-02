<p align="center">
  <img  height="120" src="https://github.com/perpetual-ml/perpetual/raw/main/resources/perp_logo.png">
</p>

<div align="center">

[![Python Versions](https://img.shields.io/pypi/pyversions/perpetual.svg?logo=python&logoColor=white)](https://pypi.org/project/perpetual)
[![PyPI Version](https://img.shields.io/pypi/v/perpetual.svg?logo=pypi&logoColor=white)](https://pypi.org/project/perpetual)
[![Crates.io Version](https://img.shields.io/crates/v/perpetual?logo=rust&logoColor=white)](https://crates.io/crates/perpetual)
[![Static Badge](https://img.shields.io/badge/join-discord-blue?logo=discord)](https://discord.gg/AyUK7rr6wy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/perpetual)

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

The results can be reproduced using the scripts in the [examples](./python-package/examples) folder.

PerpetualBooster is a GBM but behaves like AutoML so it is benchmarked also against AutoGluon (v1.2, best quality preset), the current leader in [AutoML benchmark](https://automlbenchmark.streamlit.app/cd_diagram). Top 10 datasets with the most number of rows are selected from [OpenML datasets](https://www.openml.org/). The results are summarized in the following table for regression tasks:

| OpenML Task                                  | Perpetual Training Duration | Perpetual Inference Duration                                      | Perpetual RMSE | AutoGluon Training Duration | AutoGluon Inference Duration                                      | AutoGluon RMSE |
| -------------------------------------------- | --------------------------- | ----------------------------------------------------------------- | -------------- | --------------------------- | ----------------------------------------------------------------- | -------------- |
| [Airlines_DepDelay_10M](openml.org/t/359929) | 518                         | 11.3                                                              | 29.0           | 520                         | 30.9 <td style="background-color:green;color:white;"> 28.8 </td>  |
| [bates_regr_100](openml.org/t/361940)        | 3421                        | 15.1 <td style="background-color:green;color:white;"> 1.084 </td> | OOM            | OOM                         | OOM                                                               |
| [BNG(libras_move)](openml.org/t/7327)        | 1956                        | 4.2 <td style="background-color:green;color:white;"> 2.51 </td>   | 1922           | 97.6                        | 2.53                                                              |
| [BNG(satellite_image)](openml.org/t/7326)    | 334                         | 1.6                                                               | 0.731          | 337                         | 10.0 <td style="background-color:green;color:white;"> 0.721 </td> |
| [COMET_MC](openml.org/t/14949)               | 44                          | 1.0 <td style="background-color:green;color:white;"> 0.0615 </td> | 47             | 5.0                         | 0.0662                                                            |
| [friedman1](openml.org/t/361939)             | 275                         | 4.2 <td style="background-color:green;color:white;"> 1.047 </td>  | 278            | 5.1                         | 1.487                                                             |
| [poker](openml.org/t/10102)                  | 38                          | 0.6 <td style="background-color:green;color:white;"> 0.256 </td>  | 41             | 1.2                         | 0.722                                                             |
| [subset_higgs](openml.org/t/361955)          | 868                         | 10.6 <td style="background-color:green;color:white;"> 0.420 </td> | 870            | 24.5                        | 0.421                                                             |
| [BNG(autoHorse)](openml.org/t/7319)          | 107                         | 1.1 <td style="background-color:green;color:white;"> 19.0 </td>   | 107            | 3.2                         | 20.5                                                              |
| [BNG(pbc)](openml.org/t/7318)                | 48                          | 0.6 <td style="background-color:green;color:white;"> 836.5 </td>  | 51             | 0.2                         | 957.1                                                             |
| average                                      | 465                         | 3.9                                                               | -              | 464                         | 19.7                                                              | -              |

PerpetualBooster outperformed AutoGluon on 8 out of 10 datasets, training equally fast and inferring 5x faster. The results can be reproduced using the automlbenchmark fork [here](https://github.com/deadsoul44/automlbenchmark).

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
