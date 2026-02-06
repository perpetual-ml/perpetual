//! Causal
//!
//! This module implements causal inference methods, such as Instrumental Variables (IV)
//! and Double Machine Learning, allowing for the estimation of causal effects.
pub mod fairness;
pub mod iv;
pub mod objective;
pub mod policy;
pub mod uplift;

mod tests;
