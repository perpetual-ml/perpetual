100x Faster Initial Training
============================

*Published: January 22, 2025 — A case study showing how much faster PerpetualBooster is at initial model training.*

Effective machine learning model training often requires meticulous hyperparameter tuning to prevent overfitting. This process typically involves employing hyperparameter optimization tools like Optuna, which iteratively trains the model with various hyperparameter combinations. A common practice is to perform 100 trials, essentially training the model 100 times with different hyperparameter sets to identify the optimal configuration.

However, what if you could bypass this time-consuming step altogether? Perpetual introduces **Perpetual Learning**, a novel approach that achieves precisely this feat. By eliminating the need for hyperparameter tuning, Perpetual Learning delivers a **100x speed-up** compared to traditional gradient boosting pipelines, while maintaining state-of-the-art accuracy on tabular data.

Perpetual Learning is applied to gradient boosting machines (GBM, GBDT, GBRT, GBT, or MART), which consistently produce outstanding results for tabular data tasks. This study presents a comparative analysis of ``PerpetualBooster`` and the widely used ``LightGBM`` package across both the `California Housing (regression) <https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset>`_ and `Forest Cover Types (classification) <https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes>`_ datasets.

California Housing (Regression)
-------------------------------

Table 1 shows the comparison results for the California Housing (regression) dataset. ``LightGBM`` employs the ``n_estimators`` parameter (alongside 100 Optuna hyperparameter tuning trials) to enhance model complexity and improve test performance. Similarly, ``PerpetualBooster`` utilizes a single ``budget`` parameter for performance optimization. We compared the models at equivalent error rates across three scenarios, and in each instance, ``PerpetualBooster`` demonstrated a clear advantage in speed, exceeding tuned ``LightGBM`` by over a factor of **100x**.

.. list-table:: Table 1: California Housing (Regression) Dataset Results
   :header-rows: 1
   :widths: 15 18 14 14 15 15 12

   * - Perpetual budget
     - LightGBM n_estimators
     - Perpetual MSE
     - LightGBM MSE
     - Perpetual CPU Time (s)
     - LightGBM CPU Time (s)
     - Speed-up
   * - 0.30
     - 100
     - 0.202
     - 0.203
     - 8.8
     - 982
     - **112x**
   * - 0.45
     - 200
     - 0.198
     - 0.198
     - 19.1
     - 2,554
     - **134x**
   * - 0.60
     - 500
     - 0.197
     - 0.197
     - 40.9
     - 6,057
     - **148x**

Forest Cover Types (Classification)
-----------------------------------

Table 2 presents a comparison of results on the Forest Cover Types dataset, a widely used benchmark for classification tasks. This comparison focuses on a single budget scenario due to the extended execution time associated with tuning ``LightGBM`` in this context. The results reveal that ``PerpetualBooster`` demonstrates superior efficiency, achieving **76 times faster** execution speed compared to ``LightGBM`` while maintaining equivalent log loss performance. This significant speed advantage makes ``PerpetualBooster`` a compelling option for real-world applications where rapid model training and deployment are crucial.

.. list-table:: Table 2: Forest Cover Types (Classification) Dataset Results
   :header-rows: 1
   :widths: 15 18 14 14 15 15 12

   * - Perpetual budget
     - LightGBM n_estimators
     - Perpetual Log Loss
     - LightGBM Log Loss
     - Perpetual CPU Time (s)
     - LightGBM CPU Time (s)
     - Speed-up
   * - 0.33
     - 100
     - 0.0886
     - 0.0857
     - 32.2
     - 2,464
     - **76x**

Conclusion
----------

Perpetual represents a significant advancement in machine learning by eliminating the need for hyperparameter tuning. This translates to dramatically faster training times while upholding or surpassing the accuracy of traditional approaches like ``LightGBM``.

