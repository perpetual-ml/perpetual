Perpetual Documentation
=======================

.. image:: _static/perp_logo.png
   :align: left
   :alt: Perpetual Logo
   :width: 100px
   :class: perp-logo

.. raw:: html

   <div style="clear: both;"></div>

Perpetual is a self-generalizing gradient boosting machine that doesn't need hyperparameter optimization. It is designed to be easy to use while providing state-of-the-art predictive performance.

Key Features
------------

* **Hyperparameter-Free Learning**: Achieves optimal accuracy in a single run via a simple ``budget`` parameter, eliminating the need for time-consuming hyperparameter optimization.
* **High-Performance Rust Core**: Blazing-fast training and inference with a native Rust core, zero-copy support for Polars/Arrow data, and robust Python & R bindings.
* **Comprehensive Objectives**: Fully supports Classification (Binary & Multi-class), Regression, and Ranking tasks.
* **Advanced Tree Features**: Natively handles categorical variables, learnable missing value splits, monotonic constraints, and feature interaction constraints.
* **Built-in Causal ML**: Out-of-the-box support for causal machine learning to estimate treatment effects.
* **Robust Drift Monitoring**: Built-in capabilities to monitor both data drift and concept drift without requiring ground truth labels or model retraining.
* **Continual Learning**: Built-in continual learning capabilities that significantly reduce computational time from O(n²) to O(n).
* **Native Calibration**: Built-in calibration features to predict fully calibrated distributions (marginal coverage) and conditional coverage without retraining.
* **Explainability**: Easily interpret model decisions using built-in feature importance, partial dependence plots, and Shapley (SHAP) values.
* **Production Ready & Interoperable**: Ready for production applications; seamlessly export models to industry-standard XGBoost or ONNX formats for straightforward deployment.

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
   parameters_tuning
   faq

API Reference
-------------

See the :doc:`api` for the detailed API reference.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
