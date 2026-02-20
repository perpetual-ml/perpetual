
Policy Learning
===============

Policy learning is focused on finding an optimal treatment assignment policy that maximizes the overall outcome (e.g., maximizing revenue or health outcomes) across a population. 
Instead of just estimating the treatment effect, policy learning directly optimizes the decision rules determining who should receive treatment.

PolicyLearner
-------------

The :class:`policy.PolicyLearner` uses an Inverse Propensity Weighting (IPW) or an Augmented IPW (Doubly Robust) approach to evaluate and learn optimal policies. 
It takes observational data, models the outcome and propensity scores, and builds a decision tree that directly maximizes the estimated policy value.

Example:

.. code-block:: python

   from perpetual.policy import PolicyLearner

   # X: covariates, w: treatment (0 or 1), y: outcome
   model = PolicyLearner(
       budget=0.5,
       mode="aipw"
   )

   model.fit(X, w, y)

   # Predict the optimal treatment assignment (0 or 1)
   optimal_treatment = model.predict(X_test)

Tutorials
---------

For a detailed walkthrough, see the :doc:`../tutorials/causal/policy_learning`.
