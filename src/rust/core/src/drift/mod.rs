//! Drift Detection Logic
//!
//! This module provides functions to calculate data drift and concept drift
//! using the tree structure of a trained `PerpetualBooster` model.

pub mod calculation;
pub mod stats;

pub use calculation::{calculate_drift, calculate_drift_columnar};
pub use stats::chi2_contingency_2x2;
