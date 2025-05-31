//! Rust and C bindings for R
//!
//! ## Description
//! All base functionality are wrapped in pointers and can be passed between functions. The C-bindings
//! are found in 'perpetual.h' and the R bindings are in wrappers.c
//!
//! 
mod univariate;
pub use univariate::*;