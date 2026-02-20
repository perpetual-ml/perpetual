
Double Machine Learning (DML)
=============================

Double Machine Learning (DML) is a method for estimating causal effects when there are many confounding variables. 
It uses machine learning models to separately estimate the outcome and the treatment assignment, and then combines them using a Neyman-orthogonal score to obtain unbiased estimates of the treatment effect.

DMLEstimator
------------

The :class:`dml.DMLEstimator` allows estimating the Conditional Average Treatment Effect (CATE) for both discrete and continuous treatments using Gradient Boosting.

Example:

.. code-block:: python

   from perpetual.dml import DMLEstimator
   import numpy as np

   # X: covariates, w: treatment, y: outcome
   # DMLEstimator uses separate cross-fitted models for the outcome (y ~ X) and the treatment (w ~ X)
   model = DMLEstimator(
       budget=0.5,
       n_folds=2,
       objective="SquaredLoss"
   )

   model.fit(X, w, y)

   # Predict the Conditional Average Treatment Effect (CATE)
   cate_pred = model.predict(X_test)

Tutorials
---------

For a detailed walkthrough, see the :doc:`../tutorials/causal/dml_wage_gap`.
