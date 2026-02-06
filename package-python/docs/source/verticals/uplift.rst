
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

DR-Learner (Doubly Robust)
--------------------------

The :class:`meta_learners.DRLearner` combines outcome modeling with inverse
propensity weighting to produce a *doubly robust* estimate of the CATE.  The
estimator is consistent when *either* the outcome models or the propensity
model is correctly specified.

The AIPW pseudo-outcome is:

$$\Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{W_i(Y_i - \hat{\mu}_1(X_i))}{\hat{p}(X_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{p}(X_i)}$$

Example:

.. code-block:: python

   from perpetual.meta_learners import DRLearner

   model = DRLearner(budget=0.5, clip=0.01)
   model.fit(X, w, y)
   cate = model.predict(X_test)

Meta-Learners
-------------

For cases where you want more control or simpler algorithms, Perpetual offers standard Meta-Learners:

* **S-Learner**: Uses a single model including the treatment as a feature.
* **T-Learner**: Uses two separate models, one for treatment and one for control.
* **X-Learner**: A multi-stage learner that is particularly effective when treatment groups are imbalanced.

.. code-block:: python

   from perpetual.meta_learners import XLearner

   model = XLearner(budget=0.5)
   model.fit(X, w, y)
   cate = model.predict(X_test)

Evaluation Metrics
------------------

Evaluating uplift models is challenging because individual-level treatment
effects are never directly observed.  Perpetual provides standard metrics that
exploit randomized treatment assignment:

* :func:`causal_metrics.cumulative_gain_curve` — the uplift gain curve.
* :func:`causal_metrics.auuc` — Area Under the Uplift Curve.
* :func:`causal_metrics.qini_curve` — the Qini curve.
* :func:`causal_metrics.qini_coefficient` — the Qini coefficient.

.. code-block:: python

   from perpetual.causal_metrics import auuc, qini_coefficient

   score = auuc(y_test, w_test, uplift_preds, normalize=True)
   qini  = qini_coefficient(y_test, w_test, uplift_preds)

Tutorials
---------

For a detailed walkthrough using the Hillstrom marketing dataset, see the :doc:`../tutorials/causal/uplift_marketing`.
