Perpetual Documentation
=======================

.. image:: _static/perp_logo.png
   :align: center
   :alt: Perpetual Logo
   :width: 100px

Perpetual is a self-generalizing gradient boosting machine that doesn't need hyperparameter optimization. It is designed to be easy to use while providing state-of-the-art predictive performance.

Key Features
------------

* **Self-Generalizing**: No need for complex grid searches or Bayesian optimization.
* **Efficient**: Built in Rust with a zero-copy Python interface.
* **Versatile**: Supports regression, classification, and ranking.
* **Interpretable**: Built-in support for SHAP-like contributions and partial dependence plots.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   tutorials/index
   architecture
   parameters_tuning
   faq

API Reference
-------------

.. autosummary::
   :toctree: generated

   perpetual.PerpetualBooster

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
