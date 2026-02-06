//! Causal
//!
//! This module implements causal inference methods, such as Instrumental Variables (IV),
//! Double Machine Learning (DML), uplift modelling (R-Learner), policy learning,
//! and fairness-aware objectives, allowing for the estimation of causal effects.
pub mod dml;
pub mod fairness;
pub mod iv;
pub mod objective;
pub mod policy;
pub mod uplift;

mod tests;
