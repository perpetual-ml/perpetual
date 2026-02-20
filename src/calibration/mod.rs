//! Calibration Module
//!
//! This module provides various methods for calibrating model outputs to produce
//! reliable probabilities (for classification) or prediction intervals (for regression).
//!
//! # Submodules
//!
//! * `base`: Foundational traits and common logic for calibration.
//! * `classification`: Probability calibration methods (e.g., Platt scaling via Isotonic).
//! * `regression`: Methods for producing prediction intervals (MinMax, GRP, WeightVariance).
//! * `isotonic`: Implementation of Isotonic Regression for calibration.
//! * `cqr`: Conformalized Quantile Regression for robust prediction intervals.

pub mod base;
pub mod classification;
pub mod cqr;
pub mod isotonic;
pub mod regression;
#[cfg(test)]
mod tests;
