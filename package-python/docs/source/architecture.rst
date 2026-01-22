Architecture
============

Perpetual is built on top of a high-performance Rust core.

Rust Core
---------

The core algorithm is implemented in Rust to ensure memory safety and high performance. It uses thread-parallelism for histogram building and prediction.

Zero-Copy Interface
-------------------

The Python package communicates with the Rust core using PyO3. For Polars DataFrames, Perpetual uses a zero-copy columnar path, meaning data is not duplicated when passed from Python to Rust.

Self-Generalization
-------------------

The key innovation in Perpetual is its self-generalization capability. By linking the learning rate and stopping criteria to a single "budget" parameter, Perpetual eliminates the need for expensive hyperparameter optimization (HPO) loops.
