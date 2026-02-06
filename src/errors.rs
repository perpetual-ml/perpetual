//! Errors
//!
//! Custom error types used throughout the `perpetual` crate.
use thiserror::Error;

/// Errors that can occur in the Perpetual Booster.
#[derive(Debug, Error)]
pub enum PerpetualError {
    /// No variance in a feature.
    #[error("Feature number {0} has no variance, when missing values are excluded.")]
    NoVariance(usize),
    /// Unable to write model to file.
    #[error("Unable to write model to file: {0}")]
    UnableToWrite(String),
    /// Unable to read model from file.
    #[error("Unable to read model from a file {0}")]
    UnableToRead(String),
    /// NaN value found in data when missing was expected.
    #[error("The value {0} is set to missing, but a NaN value was found in the data.")]
    NANVAlueFound(f64),
    /// Invalid value parsing.
    #[error("Invalid value {0} passed for {1}, expected one of {2}.")]
    ParseString(String, String, String),
    /// First value is the name of the parameter, second is expected, third is what was passed.
    #[error("Invalid parameter value passed for {0}, expected {1} but {2} provided.")]
    InvalidParameter(String, String, String),
}
