Frequently Asked Questions
==========================

How do I choose the best budget?
-------------------------------

Start with the default budget of 0.5. If you need more accuracy and have time for longer training, increase it to 1.0 or higher. For quick experiments, 0.1 or 0.2 may be enough.

Does Perpetual support categorical features?
-------------------------------------------

Yes, you can specify categorical features via the ``categorical_features`` parameter. Perpetual handles them internally without requiring one-hot encoding.

Is Perpetual faster than XGBoost?
--------------------------------

Perpetual is optimized for CPU performance using Rust. While speed depends on the dataset and hardware, Perpetual is designed to be highly competitive while eliminating the HPO phase, which often takes much more time than a single training run.
