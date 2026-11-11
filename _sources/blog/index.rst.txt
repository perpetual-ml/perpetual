Blog & Case Studies
===================

Engineering deep-dives and empirical case studies exploring how ``PerpetualBooster`` eliminates hyperparameter optimization, achieves 100x faster initial training, accelerates continual learning from :math:`O(n^2)` to :math:`O(n)`, sharpens conformal prediction intervals, and maximizes business profit through calibrated decision-making.

.. toctree::
   :maxdepth: 2

   how_perpetual_works
   performance
   confidence
   continual
   decisioning
   scoring

Articles Overview
-----------------

* :doc:`how_perpetual_works` *(July 19, 2025)* — A mathematical deep-dive into **Step Size Control** (forward-tracking tree search) and **Generalization Control** (in-split validation loss checks) that eliminate hyperparameter tuning.
* :doc:`performance` *(January 22, 2025)* — Benchmarking ``PerpetualBooster`` against Optuna-tuned ``LightGBM`` on California Housing (**112x–148x** faster) and Forest Cover Types (**76x** faster).
* :doc:`confidence` *(January 23, 2025)* — Comparing Conformalized Quantile Regression (CQR) in ``PerpetualBooster`` vs. ``MAPIE`` on California Housing (**35% narrower** prediction intervals at 90% coverage).
* :doc:`continual` *(January 24, 2025)* — Demonstrating how continual learning (``reset=False``) reduces cumulative training time from :math:`O(n^2)` to :math:`O(n)` (**2.6x** cumulative speed-up across 11 streaming updates at identical test MSE).
* :doc:`decisioning` *(October 27, 2025)* — Bridging probability calibration and expected utility on the OpenML ``TVS_Loan_Default`` dataset (**+$75,500 / +1.3%** higher lending profit than calibrated ``LightGBM``).
* :doc:`scoring` *(October 31, 2025)* — Profit-driven lead scoring on the UCI Bank Marketing dataset (**+1.0%** higher net campaign profit than ``LightGBM + CalibratedClassifierCV``).

