
Instrumental Variables (BoostIV)
================================

Instrumental variables (IV) are used in causal inference to estimate causal effects when there is unobserved confounding between the treatment :math:`W` and the outcome :math:`Y`.

Unobserved confounding violates the consistency of standard estimators (like Ordinary Least Squares or standard Gradient Boosting). An **instrument** :math:`Z` is a variable that is correlated with the treatment but has no direct effect on the outcome except through the treatment.

Boosted Instrumental Variables
------------------------------

The :class:`iv.BraidedBooster` implements a **Control Function** approach using Gradient Boosting. This method avoids the biased "Forbidden Regression" by explicitly modeling the first-stage residuals to account for endogeneity.

1. **Stage 1 (Treatment Model)**: Model the treatment :math:`W` as a function of covariates :math:`X` and instruments :math:`Z`: :math:`\hat{W} = f(X, Z)`. Then compute residuals :math:`V = W - \hat{W}`.
2. **Stage 2 (Outcome Model)**: Model the outcome :math:`Y` as a function of covariates :math:`X`, predicted treatment :math:`\hat{W}`, and the residuals :math:`V`: :math:`\hat{Y} = g(X, \hat{W}, V)`.

Example:

.. code-block:: python

   from perpetual.iv import BraidedBooster

   # X: covariates, Z: instruments, y: outcome, w: treatment
   model = BraidedBooster(
       treatment_objective="SquaredLoss", 
       outcome_objective="SquaredLoss",
       stage1_budget=0.5,
       stage2_budget=0.5
   )
   
   model.fit(X, Z, y, w)

   # Predict outcome for a counterfactual treatment level
   y_pred = model.predict(X_test, w_counterfactual=np.ones(len(X_test)))

Tutorials
---------

For a detailed walkthrough using the Card (1995) education dataset, see the :doc:`../tutorials/causal/iv_causal_effect`.
