Drift Detection
===============

Perpetual provides built-in methods to detect **Data Drift** and **Concept Drift** using the internal structure of the trained model. 

How it Works
------------

Drift detection in Perpetual is based on comparing the distribution of samples across the decision tree nodes during training versus the distribution observed in new data.

1. **Data Drift (Multivariate)**: Calculates the average Chi-squared statistic across all internal nodes of the model. This detects if the feature distributions have shifted in a way that affects which paths samples take through the trees.
2. **Concept Drift**: Focuses on the nodes that are parents of leaves. This detects if the relationship between features and the target is likely shifting by monitoring changes in the final decision-level node distributions.

Usage
-----

To enable drift detection, you must initialize the model with ``save_node_stats=True``.

.. code-block:: python

    from perpetual import PerpetualBooster
    import numpy as np

    # 1. Train the model
    model = PerpetualBooster(save_node_stats=True)
    model.fit(X_train, y_train)

    # 2. Calculate drift on new data
    data_drift_score = model.calculate_drift(X_new, drift_type="data")
    concept_drift_score = model.calculate_drift(X_new, drift_type="concept")

    print(f"Data Drift: {data_drift_score}")
    print(f"Concept Drift: {concept_drift_score}")

Interpreting the Score
----------------------

The drift score is an average Chi-squared statistic. Larger values indicate more significant drift. 

* **Near 0**: The new data follows the same distribution as the training data.
* **Large values**: Suggest a significant shift in data distribution (Data Drift) or prediction patterns (Concept Drift).

Note: This method is unsupervised and does not require target values for the new data.

Examples
--------

For a detailed walkthrough, see the :doc:`Drift Detection Tutorial <tutorials/drift_tutorial.ipynb>`.
