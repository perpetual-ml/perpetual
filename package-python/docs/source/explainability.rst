Explainability
==============

Perpetual provides several methods to interpret the model and understand its predictions. These methods include Feature Importance, Partial Dependence, and Prediction Contributions (SHAP-like values).

Feature Importance
------------------

Feature importance scores indicate how useful each feature was for the construction of the boosted decision trees. Perpetual supports several importance metrics:

*   **Gain**: Average improvement in loss brought by a feature.
*   **Weight**: Number of times a feature is used in splits.
*   **Cover**: Average number of samples affected by splits on a feature.
*   **TotalGain**: Total improvement in loss brought by a feature.
*   **TotalCover**: Total number of samples affected by splits on a feature.

.. code-block:: python

    importance = model.calculate_feature_importance(method="Gain", normalize=True)
    print(importance)

Partial Dependence
------------------

Partial dependence plots (PDP) show the dependence between the target response and a set of input features, marginalizing over the values of all other features.

.. code-block:: python

    pd_values = model.partial_dependence(X, feature="feature_name", samples=100)
    # pd_values is an array where col 0 is feature value and col 1 is the predicted value

Prediction Contributions
------------------------

Perpetual can calculate the contribution of each feature to a specific prediction. This is often referred to as SHAP (SHapley Additive exPlanations) values. The sum of the contributions plus the bias term equals the model's raw prediction (e.g., log-odds for classification).

.. code-block:: python

    contributions = model.predict_contributions(X_sample, method="Average")
    # contributions[:, :-1] are the feature contributions
    # contributions[:, -1] is the bias (expected value)

Example
-------

Here is a complete example demonstrating these explainability methods:

.. literalinclude:: ../../examples/explainability.py
   :language: python
   :linenos:
