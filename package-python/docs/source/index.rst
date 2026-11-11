Perpetual Documentation
=======================

.. image:: _static/perp_logo.png
   :align: left
   :alt: Perpetual Logo
   :width: 100px
   :class: perp-logo

.. raw:: html

   <div style="clear: both;"></div>

**Perpetual** is a self-generalizing gradient boosting machine (GBM) implemented in Rust with native Python and R bindings. Unlike traditional gradient boosting libraries (such as LightGBM, XGBoost, and CatBoost) that require extensive hyperparameter optimization across learning rate, tree depth, and regularization parameters, ``PerpetualBooster`` automatically controls step size and node-split generalization via a single intuitive ``budget`` parameter.

Quick Links
-----------

* **Python Package**: `PyPI (perpetual) <https://pypi.org/project/perpetual/>`_ | :doc:`Installation <installation>` | :doc:`Quickstart <quickstart>` | :doc:`Python API Reference <api>`
* **R Package**: `R Documentation (pkgdown) <https://perpetual-ml.github.io/perpetual/r/>`_ | `R Source (package-r) <https://github.com/perpetual-ml/perpetual/tree/main/package-r>`_
* **Rust Crate**: `crates.io (perpetual) <https://crates.io/crates/perpetual>`_ | `Rust API Docs (docs.rs) <https://docs.rs/perpetual>`_
* **Source & Community**: `GitHub Repository <https://github.com/perpetual-ml/perpetual>`_ | `Discord Community <https://discord.gg/c23eezw379>`_ | `LinkedIn <https://www.linkedin.com/company/perpetual-ml/>`_ | Contact: `mutlusims3k@gmail.com <mailto:mutlusims3k@gmail.com>`_

Key Capabilities
----------------

* **Hyperparameter-Free Learning**: Achieves state-of-the-art accuracy in a single training run via a simple ``budget`` parameter, eliminating multi-trial hyperparameter tuning (see :doc:`blog/how_perpetual_works`).
* **High-Performance Rust Core**: Parallelized histogram construction, split finding, and inference powered by Rust and Rayon, with zero-copy support for Polars/Arrow columnar data and native Python & R bindings.
* **Comprehensive Objectives**: Supports Binary & Multi-class Classification, Regression (Squared, Absolute, Huber, Quantile, Poisson, Gamma, Tweedie, MAPE), and Learning-to-Rank (ListNet), as well as custom user-defined objectives.
* **Advanced Tree Features**: Natively handles categorical variables, learnable missing value branches, monotonic constraints, and feature interaction constraints.
* **Continual Learning**: Built-in warm-start continual learning (``reset=False``) that updates models incrementally as new batches arrive, reducing cumulative training complexity from quadratic :math:`O(n^2)` to linear :math:`O(n)` without sacrificing accuracy (see :doc:`continual_learning` and :doc:`blog/continual`).
* **Native Calibration & Prediction Intervals**: Predict calibrated probabilities and rigorous prediction intervals (Conformalized Quantile Regression, GRP, MinMax, WeightVariance) that achieve 35% narrower intervals than MAPIE at matching coverage (see :doc:`calibration` and :doc:`blog/confidence`).
* **Robust Drift Monitoring**: Monitor both data drift and concept drift directly from model tree topology without requiring ground-truth labels or retraining (see :doc:`drift_detection`).
* **Built-in Causal ML**: Out-of-the-box Uplift Modeling, Meta-Learners (S-, T-, X-, R-, DR-Learner), Double Machine Learning (DML), and Structural Policy Learning (see :doc:`causal_ml/index`).
* **Explainability & Interoperability**: Compute feature importances, partial dependence plots, and exact TreeSHAP values (:doc:`explainability`), and export models to industry-standard XGBoost or ONNX formats (:doc:`model_io_export`).

Benchmark Highlights
--------------------

AutoML Benchmark (104 Classification Datasets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tested on the top 104 classification datasets from the `OpenML AutoML Benchmark <https://openml.github.io/automlbenchmark/>`_ (`AutoML benchmark script <https://github.com/perpetual-ml/perpetual/blob/main/package-python/examples/automlbenchmark.py>`_), ``PerpetualBooster`` trains **3.1x faster** on average than ``AutoGluon`` (`best_quality` preset, single 5-fold model) at identical ROC AUC:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Model
     - Budget / Time Limit
     - Mean ROC AUC
     - Mean Fit Duration (s)
   * - AutoGluon
     - 900s limit
     - 0.903
     - 39.2
   * - **Perpetual**
     - **budget = 1.0**
     - **0.903**
     - **12.5 (3.1x faster)**

Initial Training Speed vs. Tuned LightGBM (100 Optuna Trials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By replacing 100 trials of hyperparameter optimization with a single ``budget`` run, ``PerpetualBooster`` achieves **76x–148x speed-ups** at matching test accuracy (see :doc:`blog/performance`):

.. list-table::
   :header-rows: 1
   :widths: 28 14 14 14 15 15

   * - Dataset (Task)
     - Perpetual Budget
     - Perpetual Loss
     - LightGBM Loss
     - Perpetual CPU Time
     - Speed-Up vs. LightGBM
   * - California Housing (Regression, MSE)
     - 0.30
     - 0.202
     - 0.203
     - 8.8s (vs. 982s)
     - **112x**
   * - California Housing (Regression, MSE)
     - 0.45
     - 0.198
     - 0.198
     - 19.1s (vs. 2,554s)
     - **134x**
   * - California Housing (Regression, MSE)
     - 0.60
     - 0.197
     - 0.197
     - 40.9s (vs. 6,057s)
     - **148x**
   * - Forest Cover Types (Classification, Log Loss)
     - 0.33
     - 0.0886
     - 0.0857
     - 32.2s (vs. 2,464s)
     - **76x**

Featured Blog Posts & Case Studies
----------------------------------

* :doc:`blog/how_perpetual_works` — How step size control and generalization control eliminate hyperparameter tuning.
* :doc:`blog/performance` — Achieving 100x faster training than Optuna-tuned LightGBM pipelines.
* :doc:`blog/confidence` — 35% tighter prediction intervals with Perpetual Conformalized Quantile Regression (CQR) vs. MAPIE.
* :doc:`blog/continual` — Reducing cumulative streaming training time from :math:`O(n^2)` to :math:`O(n)`.
* :doc:`blog/decisioning` — Maximizing lending profitability with calibrated probabilities on OpenML ``TVS_Loan_Default``.
* :doc:`blog/scoring` — Profit-driven lead scoring on the UCI Bank Marketing dataset.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   Causal ML <causal_ml/index>
   calibration
   drift_detection
   continual_learning
   explainability
   model_io_export
   tutorials/index
   architecture
   Blog <blog/index>
   case_studies
   parameters_tuning
   faq

API Reference & Indices
-----------------------

* :doc:`Python API Reference <api>`
* `R Package Documentation <https://perpetual-ml.github.io/perpetual/r/>`_
* `Rust Crate Documentation (docs.rs) <https://docs.rs/perpetual>`_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
