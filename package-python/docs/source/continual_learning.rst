Continual Learning
==================

PerpetualBooster supports continual learning, allowing you to update an existing model with new data without retraining from scratch. This is particularly useful for streaming data or when you receive data in batches over time.

Mechanism
---------

In Perpetual, continual learning is implemented by maintaining the state of the model (including all previously trained trees) and adding new trees to correct the residual errors on the new data.

To enable continual learning, you must initialize the ``PerpetualBooster`` with ``reset=False``.

.. code-block:: python

    import pandas as pd

    # Initialize with reset=False to enable continual updates
    model = PerpetualBooster(objective="SquaredLoss", budget=1.0, reset=False)

    # First batch
    model.fit(X_batch1, y_batch1)

    # Second batch
    # IMPORTANT: Provide CUMULATIVE data to prevent catastrophic forgetting
    # The model uses existing trees for initial predictions and adds new trees for residuals.
    X_cumulative = pd.concat([X_batch1, X_batch2])
    y_cumulative = pd.concat([y_batch1, y_batch2])

    model.fit(X_cumulative, y_cumulative)


By default (``reset=True`` or default), calling ``fit`` will discard any previous training and start from scratch.

Warm Start vs Retraining
------------------------

- **Retraining (reset=True)**: The model is discarded and a new one is built from scratch using the provided data.
  
  - **Method**: Discard old model, train new model on all available data (cumulative).
  - **Total Complexity**: **O(n²)**, where *n* is the final dataset size.
  - **Why?**: You re-learn the same patterns repeatedly as the dataset grows.

- **Continual Learning (reset=False)**: The existing trees are kept, and new trees are added to reduce the error on the provided data.
  
  - **Method**: Keep existing model, update with all available data (cumulative).
  - **Total Complexity**: **O(n)**, where *n* is the final dataset size.
  - **Metrics**: **Maintains the same average metrics (e.g., MSE)** as retraining from scratch.
  - **Why?**: The model uses existing trees to explain the majority of the data and only adds new trees to correct residual errors on the new data or distributional shifts.

Performance Considerations
--------------------------

When performing continual learning, it is **critical** to provide **cumulative data** (all data seen so far) to the ``fit`` method, even when ``reset=False``. Perpetual uses the existing trees to make predictions on the cumulative data and then adds new trees to correct the errors. This prevents catastrophic forgetting and ensures the model remains accurate on the entire distribution.

If you only provide the *new* batch of data with ``reset=False``, the model will focus on minimizing error for that specific batch, which effectively forgets the patterns learned from previous batches.

Example
-------

A full example demonstrating continual learning and comparing it with retraining is available in the examples directory:

`examples/continual_learning.py <https://github.com/perpetual-ml/perpetual/blob/main/package-python/examples/continual_learning.py>`_

The example simulates a streaming scenario using the California Housing dataset (fetched from sklearn):

1. Trains an initial model on a subset of data.
2. Iteratively receives new batches of data.
3. Compares **Continual Learning** (warm start) vs **Retraining** at each step.

Results typically show that Continual Learning is significantly faster while maintaining competitive accuracy (MSE).
