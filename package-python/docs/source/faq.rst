Frequently Asked Questions
==========================

How do I choose the best budget?
--------------------------------

Start with the default budget of ``0.5``. If you need higher accuracy and can allocate more training time, increase ``budget`` to ``1.0`` or higher. For rapid prototyping, ``0.1`` to ``0.3`` is often sufficient. Because ``budget`` controls both step size (:math:`\alpha = 10^{-\text{budget}}`) and internal generalization thresholds, increasing ``budget`` monotonically increases model capacity until the internal generalization control automatically halts splitting.

How does Perpetual prevent overfitting without a separate validation set?
-------------------------------------------------------------------------

At every candidate node split, ``PerpetualBooster`` partitions node samples into internal training and validation folds and evaluates the second-order Taylor loss reduction on both. A split is only accepted when the validation loss reduction keeps pace with the training loss reduction (:math:`\text{generalization} \geq 1.0`). See :doc:`blog/how_perpetual_works` and :ref:`how-perpetual-works` in :doc:`architecture` for details.

Does Perpetual support categorical features and missing values?
---------------------------------------------------------------

Yes. You can specify categorical columns via the ``categorical_features`` parameter without one-hot encoding. Missing values (``NaN`` / ``None``) are also handled natively during split finding via learnable missing branches.

Is Perpetual faster than LightGBM, XGBoost, and AutoGluon?
----------------------------------------------------------

``PerpetualBooster`` is implemented in Rust with Rayon data parallelism. Because it eliminates multi-trial hyperparameter optimization (such as running 100 Optuna trials), end-to-end training is **76x–148x faster** than tuned LightGBM pipelines at matching accuracy (:doc:`blog/performance`) and **3.1x faster** on average than AutoGluon across the 104 OpenML AutoML benchmark classification datasets.

How do I update a trained model on new batches of data?
-------------------------------------------------------

Initialize ``PerpetualBooster(reset=False)`` and call ``fit`` with the cumulative dataset as new batches arrive. Perpetual reuses existing trees and only fits new trees to the residual errors, reducing cumulative training complexity from :math:`O(n^2)` to :math:`O(n)`. See :doc:`continual_learning` and :doc:`blog/continual`.

What languages are supported?
-----------------------------

* **Python**: `perpetual on PyPI <https://pypi.org/project/perpetual/>`_ (:doc:`Python API Reference <api>`)
* **R**: `perpetual R package <https://perpetual-ml.github.io/perpetual/r/>`_
* **Rust**: `perpetual crate on crates.io <https://crates.io/crates/perpetual>`_ (`docs.rs <https://docs.rs/perpetual>`_)

How can I get help, report issues, or contact the maintainer?
-------------------------------------------------------------

* **GitHub Issues & Discussions**: `perpetual-ml/perpetual <https://github.com/perpetual-ml/perpetual/issues>`_
* **Discord**: `Join the Perpetual ML Discord <https://discord.gg/c23eezw379>`_
* **LinkedIn**: `Perpetual ML on LinkedIn <https://www.linkedin.com/company/perpetual-ml/>`_
* **Email**: `mutlusims3k@gmail.com <mailto:mutlusims3k@gmail.com>`_
