Calibration and Uncertainty Quantification
==========================================

PerpetualBooster provides native, high-performance calibration methods for both regression (Prediction Intervals) and classification (Probability Calibration). 

The Fundamental Advantage: Post-Hoc Calibration
----------------------------------------------

Traditional gradient boosting frameworks often require expensive modifications to the training process to produce well-calibrated outputs. For example:
- **Quantile Regression**: Requires retraining multiple models (one for each quantile).
- **CV-based Calibration**: Requires K-fold cross-validation or nested cross-validation, increasing training time by a factor of K.
- **Conformal Prediction Wrappers**: Often require splitting data and wrapping external models, leading to complexity.

**PerpetualBooster** changes this paradigm by offering **post-hoc calibration**. You train your model *once* with standard settings (ensuring ``save_node_stats=True`` is set). You can then apply various calibration methods to the already-trained model using a small calibration set. This process is instantaneous and does not modify the underlying ensemble.

Probability Calibration (Classification)
----------------------------------------

In classification, calibration ensures that the output probabilities reflect true frequencies. A well-calibrated model that predicts a 90% probability of a "fraudulent" transaction should indeed be correct 90% of the time.

Perpetual utilizes the **Pool Adjacent Violators Algorithm (PAVA)** for **Isotonic Regression** natively in Rust.

Available Methods
~~~~~~~~~~~~~~~~~

Perpetual allows you to drive the Isotonic calibration using different internal uncertainty scores:

1. **Conformal (Default)**: Uses raw probabilities to fit the Isotonic curve. This is the standard approach to probability calibration.
2. **WeightVariance / GRP / MinMax**: These methods use method-specific uncertainty scores (calculated from node statistics) to drive the Isotonic calibration. By weighting probabilities by the model's confidence in specific regions of the feature space, Perpetual can achieve even lower **Expected Calibration Error (ECE)**.

Example
~~~~~~~

.. code-block:: python

    from perpetual import PerpetualBooster

    # 1. Train once
    model = PerpetualBooster(objective="LogLoss")
    model.fit(X_train, y_train, save_node_stats=True)

    # 2. Calibrate post-hoc on a small set
    model.calibrate(X_cal, y_cal)

    # 3. Predict well-calibrated probabilities
    probs = model.predict_proba(X_test, calibrated=True)

Uncertainty Quantification (Regression)
---------------------------------------

For regression, Perpetual provides rigorous **Prediction Intervals**. Instead of a point estimate, you receive a range ``[lower, upper]`` that is guaranteed to contain the true value with a specific probability (e.g., 90%).

Native Calibration Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Conformal**: Implements a method similar to Split Conformal Prediction or CQR. It ensures conservative coverage on any unseen data distributed similarly to the calibration set.
* **MinMax**: A proprietary method that uses the range of target values observed in the leaves of the ensemble to drive local uncertainty.
* **GRP (Generalized Residual Percentiles)**: Uses log-odds percentiles and statistical spreads within trees to generate extremely efficient and narrow intervals that still respect the coverage guarantees.

Example
~~~~~~~

.. code-block:: python

    # Define desired coverage (alpha=0.1 means 90% confidence)
    model.calibrate(X_cal, y_cal, alpha=0.1, method="GRP")

    # Get lower/upper bounds
    intervals = model.predict(X_test, interval=True)

Why Perpetual is Better
-----------------------

1. **Superior ECE**: In benchmarks against LightGBM and Scikit-Learn, Perpetual consistently delivers lower Expected Calibration Error, making it the preferred choice for risk assessment and financial modeling.
2. **Narrower Intervals**: Perpetual's internal methods (GRP, MinMax) often produce significantly narrower prediction intervals than standard conformal wrappers while maintaining the requested coverage.
3. **Rust Efficiency**: Calibration occurs at the C-layer speed, meaning thousands of calibration points can be processed in milliseconds.
4. **API Simplicity**: A single ``calibrate()`` method handles everything. The booster automatically detects the task (classification vs regression) and chooses the most appropriate internal engine.

Tutorials
---------

For deep-dives and performance comparisons:

* :doc:`tutorials/calibration/regression_calibration`: In-depth comparison of GRP vs Conformal vs Mapie.
* :doc:`tutorials/calibration/classification_calibration`: Detailed ECE analysis and Reliability Diagrams comparing Perpetual, Sklearn, and LightGBM.

Next Steps
----------

Always ensure that your calibration set is independent of your training set to avoid "over-confidence" in your calibration metrics. A 75/25 split of your non-test data is usually a good starting point.
