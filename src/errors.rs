use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForustError {
    #[error("Feature number {0} has no variance, when missing values are excluded.")]
    NoVariance(usize),
    #[error("Unable to write model to file: {0}")]
    UnableToWrite(String),
    #[error("Unable to read model from a file {0}")]
    UnableToRead(String),
    #[error("The value {0} is set to missing, but a NaN value was found in the data.")]
    NANVAlueFound(f64),
    #[error("Invalid value {0} passed for {1}, expected one of {2}.")]
    ParseString(String, String, String),
    /// First value is the name of the parameter, second is expected, third is what was passed.
    #[error("Invalid parameter value passed for {0}, expected {1} but {2} provided.")]
    InvalidParameter(String, String, String),
}
