Better Prediction Intervals with Perpetual Learning
===================================================

*Published: January 23, 2025 — A case study showing how to achieve tighter, well-calibrated prediction intervals with Perpetual.*

This case study explores the advantages of Perpetual Learning for uncertainty quantification—significantly accelerating the learning process while delivering superior prediction intervals compared to traditional methods. Specifically, we compare Conformalized Quantile Regression (CQR) implemented with ``PerpetualBooster`` against the CQR implementation within the `MAPIE <https://mapie.readthedocs.io/>`_ open-source package.

Perpetual CQR vs. MAPIE CQR
---------------------------

Table 1 summarizes the comparison results for a target coverage rate of ``0.90`` using the California Housing dataset. Both MAPIE CQR and Perpetual CQR surpass the target coverage rate, demonstrating their empirical validity. Notably, **Perpetual CQR achieves a 35% reduction in the average width of prediction intervals**, indicating a substantial improvement in interval sharpness and informational value.

.. list-table:: Table 1: Perpetual CQR vs. MAPIE CQR (Target Coverage = 0.90)
   :header-rows: 1
   :widths: 30 20 20 30

   * - CQR Method
     - Target Coverage
     - Actual Coverage
     - Prediction Interval Average Width
   * - MAPIE CQR
     - 0.900
     - 0.905
     - 2.222
   * - **Perpetual CQR**
     - **0.900**
     - **0.907**
     - **1.450**

Conclusion
----------

These findings highlight the substantial benefits of Perpetual Learning for uncertainty quantification and conformal prediction. By enabling faster training and narrower prediction intervals—alongside built-in post-hoc calibration methods (``GRP``, ``MinMax``, ``WeightVariance``, and ``Conformal``)—``PerpetualBooster`` empowers practitioners to make more informed and reliable decisions based on their data.

See :doc:`../calibration` for complete API details and code examples for prediction intervals and probability calibration.

