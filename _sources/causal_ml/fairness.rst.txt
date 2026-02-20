
Fairness and Algorithmic Bias
=============================

When deploying machine learning models in sensitive domains (like lending, hiring, or criminal justice), it is crucial to ensure that the model's predictions are fair and unbiased with respect to protected attributes (e.g., race, gender, age).

FairClassifier
--------------

The :class:`fairness.FairClassifier` enforces fairness constraints natively during the boosting process. It supports definitions of fairness such as Demographic Parity or Equalized Odds, by regularizing the splits that would disproportionately impact the protected group.

Example:

.. code-block:: python

   from perpetual.fairness import FairClassifier
   import numpy as np

   # sensitive_feature is the column index of the protected attribute (0 or 1) in X
   # lam is the strength of the fairness penalty
   model = FairClassifier(
       sensitive_feature=3,
       fairness_type="demographic_parity",
       lam=2.0
   )

   model.fit(X, y)

   # Predictions will be constrained to satisfy the fairness metric
   y_pred = model.predict(X_test)

Tutorials
---------

For detailed walkthroughs, see:

* :doc:`../tutorials/causal/fair_classification`
* :doc:`../tutorials/causal/fairness_aware_modeling`
