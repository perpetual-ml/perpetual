<!-- markdownlint-disable MD033 -->
# Perpetual

<p align="center">
  <img  height="120" src="https://github.com/perpetual-ml/perpetual/raw/main/resources/perp_logo.png" alt="Perpetual Logo">
</p>

<div align="center">

[![Python Versions](https://img.shields.io/pypi/pyversions/perpetual.svg?logo=python&logoColor=white)](https://pypi.org/project/perpetual)
[![PyPI Version](https://img.shields.io/pypi/v/perpetual.svg?logo=pypi&logoColor=white)](https://pypi.org/project/perpetual)
[![Crates.io Version](https://img.shields.io/crates/v/perpetual?logo=rust&logoColor=white)](https://crates.io/crates/perpetual)
[![R Version](https://img.shields.io/badge/R-1.0.0-blue?logo=r&logoColor=white)](https://github.com/perpetual-ml/perpetual/tree/main/package-r)
[![Static Badge](https://img.shields.io/badge/join-discord-blue?logo=discord)](https://discord.gg/AyUK7rr6wy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/perpetual)

</div>

PerpetualBooster is a gradient boosting machine (GBM) algorithm that doesn't need hyperparameter optimization unlike other GBMs. Similar to AutoML libraries, it has a `budget` parameter. Increasing the `budget` parameter increases the predictive power of the algorithm and gives better results on unseen data. Start with a small budget (e.g. 0.5) and increase it (e.g. 1.0) once you are confident with your features. If you don't see any improvement with further increasing the `budget`, it means that you are already extracting the most predictive power out of your data.

## Supported Languages

Perpetual is built in Rust and provides high-performance bindings for Python and R.

| Language | Installation | Documentation | Repository |
| :--- | :--- | :--- | :--- |
| **Python** | `pip install perpetual` | [Python API](https://perpetual-ml.github.io/perpetual) | [`package-python`](./package-python) |
| **Rust** | `cargo add perpetual` | [docs.rs](https://docs.rs/perpetual) | [`src`](./src) |
| **R** | `install.packages("perpetual")` | [pkgdown Site](https://perpetual-ml.github.io/perpetual/r) | [`package-r`](./package-r) |

## Usage

You can use the algorithm like in the example below. Check examples folders for both Rust and Python.

```python
from perpetual import PerpetualBooster

model = PerpetualBooster(objective="SquaredLoss", budget=0.5)
model.fit(X, y)
```

## Documentation

Comprehensive documentation for all supported languages is available:

- **Python**: [API Reference & Guides](https://perpetual-ml.github.io/perpetual)
- **Rust**: [docs.rs/perpetual](https://docs.rs/perpetual)
- **R**: [pkgdown Documentation](https://perpetual-ml.github.io/perpetual/r)

## Benchmark

### PerpetualBooster vs. Optuna + LightGBM

Hyperparameter optimization usually takes 100 iterations with plain GBM algorithms. PerpetualBooster achieves the same accuracy in a single run. Thus, it achieves up to 100x speed-up at the same accuracy with different `budget` levels and with different datasets.

The following table summarizes the results for the [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset (regression):

| Perpetual budget | LightGBM n_estimators | Perpetual mse | LightGBM mse | Speed-up wall time | Speed-up cpu time |
| :--------------- | :-------------------- | :------------ | :----------- | :----------------- | :---------------- |
| 1.0              | 100                   | 0.192         | 0.192        | 54x                | 56x               |
| 1.5              | 300                   | 0.188         | 0.188        | 59x                | 58x               |
| 2.1              | 1000                  | 0.185         | 0.186        | 42x                | 41x               |

The following table summarizes the results for the [Cover Types](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) dataset (classification):

| Perpetual budget | LightGBM n_estimators | Perpetual log loss | LightGBM log loss | Speed-up wall time | Speed-up cpu time |
| :--------------- | :-------------------- | :----------------- | :---------------- | :----------------- | :---------------- |
| 0.9              | 100                   | 0.091              | 0.084             | 72x                | 78x               |

The results can be reproduced using the scripts in the [examples](./package-python/examples) folder.

### PerpetualBooster vs. AutoGluon

PerpetualBooster is a GBM but behaves like AutoML so it is benchmarked also against AutoGluon (v1.2, best quality preset), the current leader in [AutoML benchmark](https://automlbenchmark.streamlit.app/cd_diagram). Top 10 datasets with the most number of rows are selected from [OpenML datasets](https://www.openml.org/) for both regression and classification tasks.

The results are summarized in the following table for regression tasks:

| OpenML Task                                              | Perpetual Training Duration | Perpetual Inference Duration | Perpetual RMSE      | AutoGluon Training Duration | AutoGluon Inference Duration | AutoGluon RMSE     |
| :------------------------------------------------------- | :-------------------------- | :--------------------------- | :------------------ | :-------------------------- | :--------------------------- | :----------------- |
| [Airlines_DepDelay_10M](https://www.openml.org/t/359929) | 518                         | 11.3                         | 29.0                | 520                         | 30.9                         | <ins> 28.8 </ins>  |
| [bates_regr_100](https://www.openml.org/t/361940)        | 3421                        | 15.1                         | <ins> 1.084 </ins>  | OOM                         | OOM                          | OOM                |
| [BNG(libras_move)](https://www.openml.org/t/7327)        | 1956                        | 4.2                          | <ins> 2.51 </ins>   | 1922                        | 97.6                         | 2.53               |
| [BNG(satellite_image)](https://www.openml.org/t/7326)    | 334                         | 1.6                          | 0.731               | 337                         | 10.0                         | <ins> 0.721 </ins> |
| [COMET_MC](https://www.openml.org/t/14949)               | 44                          | 1.0                          | <ins> 0.0615 </ins> | 47                          | 5.0                          | 0.0662             |
| [friedman1](https://www.openml.org/t/361939)             | 275                         | 4.2                          | <ins> 1.047 </ins>  | 278                         | 5.1                          | 1.487              |
| [poker](https://www.openml.org/t/10102)                  | 38                          | 0.6                          | <ins> 0.256 </ins>  | 41                          | 1.2                          | 0.722              |
| [subset_higgs](https://www.openml.org/t/361955)          | 868                         | 10.6                         | <ins> 0.420 </ins>  | 870                         | 24.5                         | 0.421              |
| [BNG(autoHorse)](https://www.openml.org/t/7319)          | 107                         | 1.1                          | <ins> 19.0 </ins>   | 107                         | 3.2                          | 20.5               |
| [BNG(pbc)](https://www.openml.org/t/7318)                | 48                          | 0.6                          | <ins> 836.5 </ins>  | 51                          | 0.2                          | 957.1              |
| average                                                  | 465                         | 3.9                          | -                   | 464                         | 19.7                         | -                  |

PerpetualBooster outperformed AutoGluon on 8 out of 10 regression tasks, training equally fast and inferring 5.1x faster.

The results are summarized in the following table for classification tasks:

| OpenML Task                                             | Perpetual Training Duration | Perpetual Inference Duration | Perpetual AUC      | AutoGluon Training Duration | AutoGluon Inference Duration | AutoGluon AUC |
| :------------------------------------------------------ | :-------------------------- | :--------------------------- | :----------------- | :-------------------------- | :--------------------------- | :------------ |
| [BNG(spambase)](https://www.openml.org/t/146163)        | 70.1                        | 2.1                          | <ins> 0.671 </ins> | 73.1                        | 3.7                          | 0.669         |
| [BNG(trains)](https://www.openml.org/t/208)             | 89.5                        | 1.7                          | <ins> 0.996 </ins> | 106.4                       | 2.4                          | 0.994         |
| [breast](https://www.openml.org/t/361942)               | 13699.3                     | 97.7                         | <ins> 0.991 </ins> | 13330.7                     | 79.7                         | 0.949         |
| [Click_prediction_small](https://www.openml.org/t/7291) | 89.1                        | 1.0                          | <ins> 0.749 </ins> | 101.0                       | 2.8                          | 0.703         |
| [colon](https://www.openml.org/t/361938)                | 12435.2                     | 126.7                        | <ins> 0.997 </ins> | 12356.2                     | 152.3                        | 0.997         |
| [Higgs](https://www.openml.org/t/362113)                | 3485.3                      | 40.9                         | <ins> 0.843 </ins> | 3501.4                      | 67.9                         | 0.816         |
| [SEA(50000)](https://www.openml.org/t/230)              | 21.9                        | 0.2                          | <ins> 0.936 </ins> | 25.6                        | 0.5                          | 0.935         |
| [sf-police-incidents](https://www.openml.org/t/359994)  | 85.8                        | 1.5                          | <ins> 0.687 </ins> | 99.4                        | 2.8                          | 0.659         |
| [bates_classif_100](https://www.openml.org/t/361941)    | 11152.8                     | 50.0                         | <ins> 0.864 </ins> | OOM                         | OOM                          | OOM           |
| [prostate](https://www.openml.org/t/361945)             | 13699.9                     | 79.8                         | <ins> 0.987 </ins> | OOM                         | OOM                          | OOM           |
| average                                                 | 3747.0                      | 34.0                         | -                  | 3699.2                      | 39.0                         | -             |

PerpetualBooster outperformed AutoGluon on 10 out of 10 classification tasks, training equally fast and inferring 1.1x faster.

PerpetualBooster demonstrates greater robustness compared to AutoGluon, successfully training on all 20 tasks, whereas AutoGluon encountered out-of-memory errors on 3 of those tasks.

The results can be reproduced using the [automlbenchmark fork](https://github.com/deadsoul44/automlbenchmark).

## Contribution

Contributions are welcome. Check [CONTRIBUTING.md](./CONTRIBUTING.md) for the guideline.

## Paper

PerpetualBooster prevents overfitting with a generalization algorithm. The paper is work-in-progress to explain how the algorithm works. Check our [blog post](https://perpetual-ml.com/blog/how-perpetual-works) for a high level introduction to the algorithm.

## Perpetual ML Suite

The **Perpetual ML Suite** is a comprehensive, batteries-included ML platform designed to deliver maximum predictive power with minimal effort. It allows you to track experiments, monitor metrics, and manage model drift through an intuitive interface.

For a fully managed, **serverless ML experience**, visit [app.perpetual-ml.com](https://app.perpetual-ml.com).

- **Serverless Marimo Notebooks**: Run interactive, reactive notebooks without managing any infrastructure.
- **Serverless ML Endpoints**: One-click deployment of models as production-ready endpoints for real-time inference.

Perpetual is also designed to live where your data lives. It is available as a native application on the [Snowflake Marketplace](https://app.snowflake.com/marketplace/listing/GZSYZX0EMJ/perpetual-ml-perpetual-ml-suite), with support for Databricks and other major data warehouses coming soon.
