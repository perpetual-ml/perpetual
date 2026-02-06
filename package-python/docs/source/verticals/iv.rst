
Instrumental Variables (BoostIV)
================================

Instrumental variables (IV) are used in causal inference to estimate causal effects when there is unobserved confounding between the treatment $W$ and the outcome $Y$.

Unobserved confounding violates the consistency of standard estimators (like Ordinary Least Squares or standard Gradient Boosting). An **instrument** $Z$ is a variable that is correlated with the treatment but has no direct effect on the outcome except through the treatment.

Boosted Instrumental Variables
------------------------------

The :class:`iv.BraidedBooster` implements a **2-Stage Least Squares (2SLS)** approach using Gradient Boosting. This allows for capturing complex non-linear relationships in both the first stage (treatment assignment) and the second stage (outcome estimation).

1. **Stage 1**: Model the treatment $W$ as a function of covariates $X$ and instruments $Z$: $\hat{W} = f(X, Z)$.
2. **Stage 2**: Model the outcome $Y$ as a function of covariates $X$ and the predicted treatment from the first stage $\hat{W}$: $\hat{Y} = g(X, \hat{W})$.

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
