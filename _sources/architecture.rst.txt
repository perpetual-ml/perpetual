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
*   **Generic Objectives**: The `Objective` trait (`src/objective_functions/objective.rs`) allows for a flexible implementation of loss functions. Perpetual supports standard objectives like LogLoss and SquaredLoss, as well as complex ones like QuantileLoss and custom user-defined objectives.

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

PerpetualBooster Algorithm
--------------------------

The `PerpetualBooster` is a specialized implementation that removes the need for hyperparameter optimization.

Budget and Self-Generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of tuning learning rate, tree depth, and other regularization parameters individually, Perpetual links them to a single **budget** parameter.

*   **Learning Rate (`eta`)**: The learning rate is deterministically calculated from the budget: :math:`\eta = 10^{-\text{budget}}`.
*   **Stopping Criteria**: The algorithm monitors the generalization error of the trees during training. If the trees start to overfit (generalization capability drops below a threshold), the training stops early.

For a comprehensive explanation of the self-generalization algorithm, please refer to our blog post: `How Perpetual Works <https://perpetual-ml.com/blog/how-perpetual-works>`_.
