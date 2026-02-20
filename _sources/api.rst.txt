API Reference
=============

This page contains the detailed API reference for the Perpetual Python package.

PerpetualBooster
----------------

.. autoclass:: perpetual.PerpetualBooster
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Sklearn Interface
-----------------

.. autoclass:: perpetual.sklearn.PerpetualClassifier
   :members:
   :show-inheritance:

.. autoclass:: perpetual.sklearn.PerpetualRegressor
   :members:
   :show-inheritance:

.. autoclass:: perpetual.sklearn.PerpetualRanker
   :members:
   :show-inheritance:

Causal ML
---------

.. autoclass:: perpetual.iv.BraidedBooster
   :members:
   :show-inheritance:

.. autoclass:: perpetual.meta_learners.SLearner
   :members:
   :show-inheritance:

.. autoclass:: perpetual.meta_learners.TLearner
   :members:
   :show-inheritance:

.. autoclass:: perpetual.meta_learners.XLearner
   :members:
   :show-inheritance:

.. autoclass:: perpetual.meta_learners.DRLearner
   :members:
   :show-inheritance:

Double Machine Learning
-----------------------

.. autoclass:: perpetual.dml.DMLEstimator
   :members:
   :show-inheritance:

Uplift Modeling
---------------

.. autoclass:: perpetual.uplift.UpliftBooster
   :members:
   :show-inheritance:

Policy Learning
---------------

.. autoclass:: perpetual.policy.PolicyLearner
   :members:
   :show-inheritance:

Causal Metrics
--------------

.. autofunction:: perpetual.causal_metrics.cumulative_gain_curve

.. autofunction:: perpetual.causal_metrics.auuc

.. autofunction:: perpetual.causal_metrics.qini_curve

.. autofunction:: perpetual.causal_metrics.qini_coefficient

Fairness
--------

.. autoclass:: perpetual.fairness.FairClassifier
   :members:
   :show-inheritance:

Regulatory Risk
---------------

.. autoclass:: perpetual.risk.PerpetualRiskEngine
   :members:
   :show-inheritance:
