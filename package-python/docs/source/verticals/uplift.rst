
Uplift Modeling
===============

Uplift modeling, also known as conditional average treatment effect (CATE) estimation, aims to predict the incremental impact of an action (the "treatment") on an individual's behavior.

Unlike standard predictive modeling which predicts the outcome $E[Y|X]$, uplift modeling predicts the difference:
$$\tau(x) = E[Y | X=x, W=1] - E[Y | X=x, W=0]$$
where $W$ is the treatment indicator.

Perpetual provides several ways to perform uplift modeling.

UpliftBooster (R-Learner)
--------------------------

The :class:`uplift.UpliftBooster` implements the **R-Learner** (Residual-on-Residual) meta-algorithm. This is a very powerful approach that can handle continuous outcomes and automatically accounts for selection bias if the propensity scores are modeled correctly.

The objective minimized is:
$$\hat{\tau} = \text{argmin}_{\tau} \sum_{i=1}^n \left( (Y_i - \hat{\mu}(X_i)) - \tau(X_i)(W_i - \hat{p}(X_i)) \right)^2$$

Where:
* $\hat{\mu}(X)$ is the estimated outcome (marginal).
* $\hat{p}(X)$ is the estimated propensity score.

Example:

.. code-block:: python

   from perpetual.uplift import UpliftBooster
   import numpy as np

   # X: features, w: treatment [0, 1], y: outcome
   model = UpliftBooster(outcome_budget=0.5, propensity_budget=0.5, effect_budget=0.5)
   model.fit(X, w, y)

   # Predict treatment effect (uplift)
   uplift_preds = model.predict(X_test)

Meta-Learners
-------------

For cases where you want more control or simpler algorithms, Perpetual offers standard Meta-Learners:

Tutorials
---------

For a detailed walkthrough using the Hillstrom marketing dataset, see the :doc:`../tutorials/verticals/uplift_marketing`.
* **S-Learner**: Uses a single model including the treatment as a feature.
* **T-Learner**: Uses two separate models, one for treatment and one for control.
* **X-Learner**: A multi-stage learner that is particularly effective when treatment groups are imbalanced.

.. code-block:: python

   from perpetual.meta_learners import XLearner

   model = XLearner(budget=0.5)
   model.fit(X, w, y)
   cate = model.predict(X_test)
