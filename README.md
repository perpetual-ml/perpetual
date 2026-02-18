<!-- markdownlint-disable MD033 -->
# Perpetual

<p align="center">
  <img  height="120" src="https://github.com/perpetual-ml/perpetual/raw/main/resources/perp_logo.png" alt="Perpetual Logo">
</p>

<div align="center">

<a href="https://pypi.org/project/perpetual" target="_blank"><img src="https://img.shields.io/pypi/pyversions/perpetual.svg?logo=python&logoColor=white" alt="Python Versions"></a>
<a href="https://pypi.org/project/perpetual" target="_blank"><img src="https://img.shields.io/pypi/v/perpetual.svg?logo=pypi&logoColor=white" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/perpetual" target="_blank"><img src="https://img.shields.io/conda/v/conda-forge/perpetual?label=conda-forge&logo=anaconda&logoColor=white" alt="Conda Version"></a>
<a href="https://crates.io/crates/perpetual" target="_blank"><img src="https://img.shields.io/crates/v/perpetual?logo=rust&logoColor=white" alt="Crates.io Version"></a>
<a href="https://perpetual-ml.r-universe.dev/perpetual" target="_blank"><img src="https://img.shields.io/badge/dynamic/json?url=https://perpetual-ml.r-universe.dev/api/packages/perpetual&query=$.Version&label=r-universe&logo=R&logoColor=white&color=brightgreen" alt="R-Universe status"></a>
<a href="https://discord.gg/AyUK7rr6wy" target="_blank"><img src="https://img.shields.io/badge/join-discord-blue?logo=discord" alt="Static Badge"></a>
<a href="https://pypi.org/project/perpetual" target="_blank"><img src="https://img.shields.io/pypi/dm/perpetual?logo=pypi" alt="PyPI - Downloads"></a>
<a href="https://github.com/pre-commit/pre-commit" target="_blank"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
<a href="https://github.com/astral-sh/ruff" target="_blank"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
<a href="https://github.com/perpetual-ml/perpetual/actions/workflows/CI.yml" target="_blank"><img src="https://img.shields.io/badge/python_coverage-90%25%2B-brightgreen" alt="Python Coverage"></a>
<a href="https://github.com/perpetual-ml/perpetual/actions/workflows/CI.yml" target="_blank"><img src="https://img.shields.io/badge/rust_coverage-90%25%2B-brightgreen" alt="Rust Coverage"></a>
<a href="https://github.com/perpetual-ml/perpetual/actions/workflows/CI.yml" target="_blank"><img src="https://img.shields.io/badge/r_coverage-80%25%2B-brightgreen" alt="R Coverage"></a>
<a href="./LICENSE" target="_blank"><img src="https://img.shields.io/github/license/perpetual-ml/perpetual" alt="License"></a>

</div>

PerpetualBooster is a gradient boosting machine (GBM) that doesn't need hyperparameter optimization unlike other GBMs. Similar to AutoML libraries, it has a `budget` parameter. Increasing the `budget` parameter increases the predictive power of the algorithm and gives better results on unseen data. Start with a small budget (e.g. 0.5) and increase it (e.g. 1.0) once you are confident with your features. If you don't see any improvement with further increasing the `budget`, it means that you are already extracting the most predictive power out of your data.

## Supported Languages

Perpetual is built in Rust and provides high-performance bindings for Python and R.

<!-- markdownlint-disable MD060 -->
| Language   | Installation                                                            | Documentation                                                                       | Source                                                        | Package                                                                                                                                                             |
| :--------- | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------------- | :------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------- |
| **Python** | `pip install perpetual`<br><br>`conda install -c conda-forge perpetual` | <a href="https://perpetual-ml.github.io/perpetual" target="_blank">Python API</a>   | <a href="./package-python" target="_blank">`package-python`</a> | <a href="https://pypi.org/project/perpetual" target="_blank">PyPI</a><br><br><a href="https://anaconda.org/conda-forge/perpetual" target="_blank">Conda Forge</a>   |
| **Rust**   | `cargo add perpetual`                                                   | <a href="https://docs.rs/perpetual" target="_blank">docs.rs</a>                     | <a href="./src" target="_blank">`src`</a>                     | <a href="https://crates.io/crates/perpetual" target="_blank">crates.io</a>                                                          |
| **R**      | `install.packages("perpetual")`                                         | <a href="https://perpetual-ml.github.io/perpetual/r" target="_blank">pkgdown Site</a> | <a href="./package-r" target="_blank">`package-r`</a>         | <a href="https://perpetual-ml.r-universe.dev/perpetual" target="_blank">R-universe</a>                                              |

### Optional Dependencies

* `pandas`: Enables support for training directly on Pandas DataFrames.
* `polars`: Enables zero-copy training support for Polars DataFrames.
* `scikit-learn`: Provides a scikit-learn compatible wrapper interface.
* `xgboost`: Enables saving and loading models in XGBoost format for interoperability.
* `onnxruntime`: Enables exporting and loading models in ONNX standard format.

## Usage

You can use the algorithm like in the example below. Check examples folders for both Rust and Python.

```python
from perpetual import PerpetualBooster

model = PerpetualBooster(objective="SquaredLoss", budget=0.5)
model.fit(X, y)
```

## Benchmark

### PerpetualBooster vs. Optuna + LightGBM

Hyperparameter optimization usually takes 100 iterations with plain GBM algorithms. PerpetualBooster achieves the same accuracy in a single run. Thus, it achieves up to 100x speed-up at the same accuracy with different `budget` levels and with different datasets.

The following table summarizes the results for the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html" target="_blank">California Housing</a> dataset (regression):

| Perpetual budget | LightGBM n_estimators | Perpetual mse | LightGBM mse | Speed-up wall time | Speed-up cpu time |
| :--------------- | :-------------------- | :------------ | :----------- | :----------------- | :---------------- |
| 0.76             | 50                    | 0.201         | 0.201        | 72x                | 326x              |
| 0.85             | 100                   | 0.196         | 0.196        | 113x               | 613x              |
| 1.15             | 200                   | 0.190         | 0.190        | 405x               | 1985x             |

The following table summarizes the results for the <a href="https://www.openml.org/search?type=data&status=active&id=46951" target="_blank">Pumpkin Seeds</a> dataset (classification):

| Perpetual budget | LightGBM n_estimators | Perpetual auc      | LightGBM auc      | Speed-up wall time | Speed-up cpu time |
| :--------------- | :-------------------- | :----------------- | :---------------- | :----------------- | :---------------- |
| 1.0              | 100                   | 0.944              | 0.945             | 91x                | 184x              |

The results can be reproduced using the scripts in the <a href="./package-python/examples" target="_blank">examples</a> folder.

### PerpetualBooster vs. AutoGluon

PerpetualBooster is a GBM but behaves like AutoML so it is benchmarked also against AutoGluon (v1.2, best quality preset), the current leader in <a href="https://automlbenchmark.streamlit.app/cd_diagram" target="_blank">AutoML benchmark</a>. Top 10 datasets with the most number of rows are selected from <a href="https://www.openml.org/" target="_blank">OpenML datasets</a> for both regression and classification tasks.

The results are summarized in the following table for regression tasks:

| OpenML Task                                                                         | Perpetual Training Duration | Perpetual Inference Duration | Perpetual RMSE      | AutoGluon Training Duration | AutoGluon Inference Duration | AutoGluon RMSE     |
| :---------------------------------------------------------------------------------- | :-------------------------- | :--------------------------- | :------------------ | :-------------------------- | :--------------------------- | :----------------- |
| <a href="https://www.openml.org/t/359929" target="_blank">Airlines_DepDelay_10M</a> | 518                         | 11.3                         | 29.0                | 520                         | 30.9                         | <ins> 28.8 </ins>  |
| <a href="https://www.openml.org/t/361940" target="_blank">bates_regr_100</a>        | 3421                        | 15.1                         | <ins> 1.084 </ins>  | OOM                         | OOM                          | OOM                |
| <a href="https://www.openml.org/t/7327" target="_blank">BNG(libras_move)</a>        | 1956                        | 4.2                          | <ins> 2.51 </ins>   | 1922                        | 97.6                         | 2.53               |
| <a href="https://www.openml.org/t/7326" target="_blank">BNG(satellite_image)</a>    | 334                         | 1.6                          | 0.731               | 337                         | 10.0                         | <ins> 0.721 </ins> |
| <a href="https://www.openml.org/t/14949" target="_blank">COMET_MC</a>               | 44                          | 1.0                          | <ins> 0.0615 </ins> | 47                          | 5.0                          | 0.0662             |
| <a href="https://www.openml.org/t/361939" target="_blank">friedman1</a>             | 275                         | 4.2                          | <ins> 1.047 </ins>  | 278                         | 5.1                          | 1.487              |
| <a href="https://www.openml.org/t/10102" target="_blank">poker</a>                  | 38                          | 0.6                          | <ins> 0.256 </ins>  | 41                          | 1.2                          | 0.722              |
| <a href="https://www.openml.org/t/361955" target="_blank">subset_higgs</a>          | 868                         | 10.6                         | <ins> 0.420 </ins>  | 870                         | 24.5                         | 0.421              |
| <a href="https://www.openml.org/t/7319" target="_blank">BNG(autoHorse)</a>          | 107                         | 1.1                          | <ins> 19.0 </ins>   | 107                         | 3.2                          | 20.5               |
| <a href="https://www.openml.org/t/7318" target="_blank">BNG(pbc)</a>                | 48                          | 0.6                          | <ins> 836.5 </ins>  | 51                          | 0.2                          | 957.1              |
| average                                                                             | 465                         | 3.9                          | -                   | 464                         | 19.7                         | -                  |

PerpetualBooster outperformed AutoGluon on 8 out of 10 regression tasks, training equally fast and inferring 5.1x faster.

The results are summarized in the following table for classification tasks:

| OpenML Task                                                                        | Perpetual Training Duration | Perpetual Inference Duration | Perpetual AUC      | AutoGluon Training Duration | AutoGluon Inference Duration | AutoGluon AUC |
| :--------------------------------------------------------------------------------- | :-------------------------- | :--------------------------- | :----------------- | :-------------------------- | :--------------------------- | :------------ |
| <a href="https://www.openml.org/t/146163" target="_blank">BNG(spambase)</a>        | 70.1                        | 2.1                          | <ins> 0.671 </ins> | 73.1                        | 3.7                          | 0.669         |
| <a href="https://www.openml.org/t/208" target="_blank">BNG(trains)</a>             | 89.5                        | 1.7                          | <ins> 0.996 </ins> | 106.4                       | 2.4                          | 0.994         |
| <a href="https://www.openml.org/t/361942" target="_blank">breast</a>               | 13699.3                     | 97.7                         | <ins> 0.991 </ins> | 13330.7                     | 79.7                         | 0.949         |
| <a href="https://www.openml.org/t/7291" target="_blank">Click_prediction_small</a> | 89.1                        | 1.0                          | <ins> 0.749 </ins> | 101.0                       | 2.8                          | 0.703         |
| <a href="https://www.openml.org/t/361938" target="_blank">colon</a>                | 12435.2                     | 126.7                        | <ins> 0.997 </ins> | 12356.2                     | 152.3                        | 0.997         |
| <a href="https://www.openml.org/t/362113" target="_blank">Higgs</a>                | 3485.3                      | 40.9                         | <ins> 0.843 </ins> | 3501.4                      | 67.9                         | 0.816         |
| <a href="https://www.openml.org/t/230" target="_blank">SEA(50000)</a>              | 21.9                        | 0.2                          | <ins> 0.936 </ins> | 25.6                        | 0.5                          | 0.935         |
| <a href="https://www.openml.org/t/359994" target="_blank">sf-police-incidents</a>  | 85.8                        | 1.5                          | <ins> 0.687 </ins> | 99.4                        | 2.8                          | 0.659         |
| <a href="https://www.openml.org/t/361941" target="_blank">bates_classif_100</a>    | 11152.8                     | 50.0                         | <ins> 0.864 </ins> | OOM                         | OOM                          | OOM           |
| <a href="https://www.openml.org/t/361945" target="_blank">prostate</a>             | 13699.9                     | 79.8                         | <ins> 0.987 </ins> | OOM                         | OOM                          | OOM           |
| average                                                                            | 3747.0                      | 34.0                         | -                  | 3699.2                      | 39.0                         | -             |

PerpetualBooster outperformed AutoGluon on 10 out of 10 classification tasks, training equally fast and inferring 1.1x faster.

PerpetualBooster demonstrates greater robustness compared to AutoGluon, successfully training on all 20 tasks, whereas AutoGluon encountered out-of-memory errors on 3 of those tasks.

The results can be reproduced using the <a href="https://github.com/deadsoul44/automlbenchmark" target="_blank">automlbenchmark fork</a>.

## Contribution

Contributions are welcome. Check <a href="./CONTRIBUTING.md" target="_blank">CONTRIBUTING.md</a> for the guideline.

## Paper

PerpetualBooster prevents overfitting with a generalization algorithm. The paper is work-in-progress to explain how the algorithm works. Check our <a href="https://perpetual-ml.com/blog/how-perpetual-works" target="_blank">blog post</a> for a high level introduction to the algorithm.

## Perpetual ML Suite

The **Perpetual ML Suite** is a comprehensive, batteries-included ML platform designed to deliver maximum predictive power with minimal effort. It allows you to track experiments, monitor metrics, and manage model drift through an intuitive interface.

For a fully managed, **serverless ML experience**, visit <a href="https://app.perpetual-ml.com" target="_blank">app.perpetual-ml.com</a>.

* **Serverless Marimo Notebooks**: Run interactive, reactive notebooks without managing any infrastructure.
* **Serverless ML Endpoints**: One-click deployment of models as production-ready endpoints for real-time inference.

Perpetual is also designed to live where your data lives. It is available as a native application on the <a href="https://app.snowflake.com/marketplace/listing/GZSYZX0EMJ/perpetual-ml-perpetual-ml-suite" target="_blank">Snowflake Marketplace</a>, with support for Databricks and other major data warehouses coming soon.
