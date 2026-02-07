
Regulatory Risk & Interpretability
==================================

In many high-stakes industries like FinTech and Healthcare, it is not enough to have a performant model; the model must also be explainable and compliant with regulations (e.g., GDPR, ECOA).

Perpetual provides built-in tools to support these requirements.

Adverse Action Codes (Reason Codes)
------------------------------------

When a credit application is rejected, regulations often require providing the applicant with the main reasons for the rejection (Adverse Action Codes).

The :class:`risk.PerpetualRiskEngine` automates this process by analyzing the feature contributions of rejected samples relative to a decision threshold.

.. code-block:: python

   from perpetual import PerpetualBooster, PerpetualRiskEngine
   
   # model is a fitted PerpetualBooster
   engine = PerpetualRiskEngine(model)
   
   # threshold for approval (e.g., probability of default < 0.2)
   reasons = engine.generate_reason_codes(X_applicants, threshold=0.2)
   
   # reasons[i] contains top N negative contributors for rejected applicants

Monotonicity Constraints
------------------------

To ensure fairness and common-sense behavior (e.g., increasing income should not decrease the probability of loan approval), Perpetual supports strictly enforced monotonicity constraints.

.. code-block:: python

   model = PerpetualBooster(
       monotone_constraints={"income": 1} # 1 for increasing, -1 for decreasing
   )
   model.fit(X, y)

Tutorials
---------

For a detailed walkthrough using the German Credit dataset, see the :doc:`../tutorials/causal/risk_compliance`.
