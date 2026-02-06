#![feature(array_ptr_get)]

//! # Perpetual
//!
//! `perpetual` is a high-performance, self-generalizing Gradient Boosting Machine (GBM) written in Rust.
//!
//! Unlike traditional GBMs (like XGBoost, LightGBM, CatBoost) that require extensive hyperparameter tuning
//! (learning rate, tree depth, number of leaves, L1/L2 regularization, etc.), Perpetual introduces a
//! novel **Budget**-based hyperparameter `budget`.
//!
//! The `budget` parameter controls the complexity of the model. A higher budget allows the model to
//! learn more complex patterns (more trees, deeper interactions), while a lower budget keeps the model
//! simple and robust. The algorithm automatically adjusts its internal learning rate and structure
//! based on this single parameter.
//!
//! ## Key Features
//!
//! * **Self-Generalizing**: Minimizes the need for hyperparameter tuning. The `budget` parameter is often sufficient.
//! * **High Performance**: Parallelized training and prediction using [Rayon](https://docs.rs/rayon).
//! * **Zero-Copy Data**: Efficiently handles data layouts with `ColumnarMatrix`, ideal for use with arrow or polars.
//! * **Versatile Objectives**: Supports Regression (Squared, Huber, Quantile), Classification (LogLoss), and Learning-to-Rank (ListNet).
//! * **Production Ready**: Handles missing values natively, supports monotonic constraints, and provides prediction intervals (conformal prediction).
//!
//! ## Quick Start
//!
//! ```rust
//! use perpetual::PerpetualBooster;
//! use perpetual::objective_functions::Objective;
//! use perpetual::Matrix;
//! use perpetual::booster::config::MissingNodeTreatment;
//! use std::collections::{HashSet, HashMap};
//!
//! // 1. Prepare Data
//! let data_vec = vec![
//!     1.0, 2.0, // Row 1
//!     3.0, 4.0, // Row 2
//!     5.0, 6.0  // Row 3
//! ];
//! let matrix = Matrix::new(&data_vec, 3, 2); // 3 rows, 2 columns
//! let target = vec![0.0, 1.0, 0.0];
//!
//! // 2. Configure and Initialize Booster
//! // Most parameters can be left to defaults/None, but 'budget' and 'objective' are key.
//! let budget = 1.0;
//! let objective = Objective::LogLoss;
//!
//! let mut model = PerpetualBooster::new(
//!     objective,
//!     budget,
//!     f64::NAN,   // base_score (NAN = auto-calculated)
//!     255,        // max_bin
//!     None,       // num_threads (None = all cores)
//!     None,       // monotone_constraints
//!     None,       // interaction_constraints
//!     false,      // force_children_to_bound_parent
//!     f64::NAN,   // missing value representation
//!     true,       // allow_missing_splits
//!     true,       // create_missing_branch
//!     HashSet::new(), // terminate_missing_features
//!     MissingNodeTreatment::AverageNodeWeight,
//!     10,         // log_iterations
//!     42,         // seed
//!     None,       // quantile (for QuantileLoss)
//!     None,       // reset
//!     None,       // categorical_features
//!     None,       // timeout
//!     None,       // iteration_limit
//!     None,       // memory_limit
//!     None,       // stopping_rounds
//! ).expect("Failed to initialize booster");
//!
//! // 3. Fit the Model
//! model.fit(&matrix, &target, None, None).expect("Training failed");
//!
//! // 4. Predict
//! let predictions = model.predict(&matrix, false);
//! println!("Predictions: {:?}", predictions);
//! ```

mod node;
mod partial_dependence;
mod shapley;

// Modules
pub mod bin;
pub mod binning;
pub mod booster;
pub mod causal;
pub mod conformal;
pub mod constants;
pub mod constraints;
pub mod data;
pub mod decision_tree;
pub mod errors;
pub mod grower;
pub mod histogram;
pub mod metrics;
pub mod objective_functions;
pub mod prune;
pub mod sampler;
pub mod splitter;
pub mod utils;

// Individual classes, and functions
pub use booster::core::PerpetualBooster;
pub use booster::multi_output::MultiOutputBooster;
pub use data::{ColumnarMatrix, Matrix};
