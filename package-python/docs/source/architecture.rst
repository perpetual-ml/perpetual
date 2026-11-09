Architecture
============

Perpetual is designed as a high-performance gradient boosting machine with a clean separation between its compute-intensive core and its user-facing API.

System Overview
---------------

The system consists of two main layers:

1.  **Rust Core (`perpetual-rs`)**: A pure Rust crate that implements the gradient boosting algorithm, histogram construction, and tree learning logic.
2.  **Python Interface (`py-perpetual`)**: A Python extension module built with PyO3 that exposes the Rust core to Python users, handling data conversion and API bindings.

Rust Core
---------

The core logic resides in the `src/` directory of the repository. It is built for performance and memory safety.

Key Components
~~~~~~~~~~~~~~

*   **PerpetualBooster**: The central struct (`src/booster/core.rs`) that manages the ensemble of decision trees. It handles the training loop, including gradient calculation, tree growing, and prediction.
*   **Histogram-Based Learning**: Perpetual uses a histogram-based algorithm for finding optimal splits. Continuous features are discretized into bins (`src/binning.rs`), significantly reducing the computational complexity of finding splits.
*   **Parallelism**: The core heavily utilizes `rayon` for data parallelism. Operations like histogram building, partial dependence calculations, and predictions are multi-threaded.
*   **Generic Objectives**: The `Objective` trait (`src/objective/core.rs`) allows for a flexible implementation of loss functions. Perpetual supports standard objectives like LogLoss, SquaredLoss, and AbsoluteLoss, as well as complex ones like QuantileLoss, PoissonLoss, and custom user-defined objectives.

Python Interface
----------------

The Python package is a thin wrapper around the Rust core, ensuring that the heavy lifting is done in native code.

PyO3 Bindings
~~~~~~~~~~~~~

We use `PyO3` to generate the Python extension module. The `PerpetualBooster` class in Python (`package-python/python/perpetual/booster.py`) holds a reference to the Rust `PerpetualBooster` struct (`package-python/src/booster.rs`). Method calls in Python are directly forwarded to their Rust counterparts.

Zero-Copy Data Transfer
~~~~~~~~~~~~~~~~~~~~~~~

One of the key architectural features is the zero-copy interface for columnar data.

*   **Polars Integration**: When a Polars DataFrame is passed to `fit` or `predict`, Perpetual uses the `fit_columnar` path. This path reads the underlying memory buffers of the DataFrame directly from Rust without copying the data.
*   **Numpy/Pandas**: Standard Numpy arrays and Pandas DataFrames are handled via the contiguous array interface, which may involve copying if the data is not already in the expected memory layout (e.g., C-contiguous vs F-contiguous).

.. _how-perpetual-works:

PerpetualBooster Algorithm: How It Works
----------------------------------------

``PerpetualBooster`` bypasses the need for hyperparameter optimization through two complementary mechanisms: **step size control** and **generalization control**. Instead of tuning learning rate, tree depth, and regularization parameters individually, Perpetual links them to a single ``budget`` parameter.

Step Size Control
~~~~~~~~~~~~~~~~~

Step size control is inspired by `backtracking line search <https://en.wikipedia.org/wiki/Backtracking_line_search>`_, with three key differences.

In classical backtracking line search, the algorithm starts with a large step size and repeatedly shrinks it until the Armijo–Goldstein condition is satisfied—finding a large step size that achieves a sufficient decrease in the objective function:

.. math::

   f(x + \alpha p) \leq f(x) + \alpha c m

1. **Forward-Tracking Tree Search (Smallest Sufficient Step)**: Instead of starting large and shrinking, Perpetual grows a decision tree split by split and checks the loss decrease after every split. Once the loss decrease exceeds the target threshold, tree growth stops for that boosting round and the algorithm moves to the next round. In other words, Perpetual takes the *smallest* step that achieves a target loss decrease at each boosting round, where :math:`\alpha` (the learning rate ``eta``) is calculated directly from the user-defined ``budget`` parameter:

   .. math::

      \alpha = 10^{-\text{budget}}

2. **Constant Reference Slope**: Rather than recomputing :math:`m` at every boosting round, :math:`m` is kept constant and computed once prior to fitting using the initial base score (constant prediction :math:`\hat{y}`):

   .. math::

      m = \frac{1}{n} \sum_{i=1}^{n} \text{loss}(y_i, \hat{y})

3. **Budget-Derived Control Parameter**: The control parameter :math:`c` is derived from the ``budget`` parameter using a geometric series relation:

   .. math::

      r = \frac{10}{\text{budget}}

   .. math::

      \text{reciprocals of powers} = \frac{r}{r - 1}

   .. math::

      \text{truncated series sum} = \text{reciprocals of powers} - \left(1 + \frac{1}{r}\right)

   .. math::

      c = \frac{1}{r} - \text{truncated series sum}

See `src/booster/core.rs <https://github.com/perpetual-ml/perpetual/blob/main/src/booster/core.rs>`_ and `src/tree/core.rs <https://github.com/perpetual-ml/perpetual/blob/main/src/tree/core.rs>`_ for the implementation details.

Generalization Control
~~~~~~~~~~~~~~~~~~~~~~

Alongside step size control, Perpetual uses **generalization control** at every candidate node split. Before evaluating splits, the node's samples are partitioned into internal training and validation folds. For each candidate split, Perpetual evaluates both the training loss reduction and the validation loss reduction using the second-order Taylor approximation:

.. math::

   \text{loss}_{\text{train}} = G_{\text{train}} w + \frac{1}{2} H_{\text{train}} w^2

.. math::

   w = -\frac{G_{\text{train}}}{H_{\text{train}}}

.. math::

   \text{loss}_{\text{valid}} = G_{\text{valid}} w + \frac{1}{2} H_{\text{valid}} w^2

where :math:`G` is the sum of gradients, :math:`H` is the sum of hessians, and :math:`w` is the leaf weight derived from the training folds.

Using the parent loss and the post-split training and validation losses, Perpetual computes the generalization ratio:

.. math::

   \text{generalization} = \frac{\text{loss}_{\text{parent}} - \text{loss}_{\text{train}}}{\text{loss}_{\text{parent}} - \text{loss}_{\text{valid}}}

* **Accepted Split** (:math:`\text{generalization} \geq 1.0`): Both training and validation losses decrease relative to the parent node, indicating that the split generalizes to unseen data.
* **Rejected Split** (:math:`\text{generalization} < 1.0`): The validation loss does not sufficiently improve (or increases) compared to the training loss reduction, so splitting is stopped to prevent overfitting.

See `src/splitter.rs <https://github.com/perpetual-ml/perpetual/blob/main/src/splitter.rs>`_ for the detailed split evaluation implementation.

How Step Size and Generalization Control Work Together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finding the right balance between fitting the data and avoiding overfitting is achieved by combining both mechanisms across the boosting process:

* **Early Boosting Rounds**: Step size control dominates early on, because the initial splits significantly reduce the objective loss and reach the target step size threshold quickly.
* **Later Boosting Rounds**: Generalization control becomes increasingly important as boosting progresses. Trees naturally become shallower as less generalizable signal remains in the residuals.
* **Automatic Early Stopping**: If the algorithm encounters a very simple tree (a single split) with poor generalization (:math:`< 1.0`) three times, it automatically halts the boosting process.

For more details on configuring the ``budget`` and related parameters, please refer to :doc:`parameters_tuning` and :doc:`case_studies`.
