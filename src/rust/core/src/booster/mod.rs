//! Booster
//!
//! This module contains the core GBM implementation, including configuration,
//! multi-output support, and prediction logic.
// public modules
pub mod config;
pub mod core;
pub mod multi_output;
pub mod predict;

// private modules
mod setters;
