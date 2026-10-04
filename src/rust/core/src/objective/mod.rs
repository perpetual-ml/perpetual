//! Objective Functions
//!
//! This module defines the various loss functions and their gradients/hessians
//! used by the GBM to optimize the model's performance.
//!
//! Custom objectives need only implement [`ObjectiveFunction::loss`] and
//! [`ObjectiveFunction::gradient`]; the remaining trait methods have sensible
//! defaults.

mod absolute_loss;
mod adaptive_huber_loss;
mod brier_loss;
mod cross_entropy;
mod cross_entropy_lambda;
mod fair_loss;
mod gamma;
mod hinge_loss;
mod huber_loss;
mod listnet_loss;
mod log_loss;
mod mape;
mod poisson;
mod quantile_loss;
mod squared_log_loss;
mod squared_loss;
mod tweedie;

pub use absolute_loss::AbsoluteLoss;
pub use adaptive_huber_loss::AdaptiveHuberLoss;
pub use brier_loss::BrierLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use cross_entropy_lambda::CrossEntropyLambdaLoss;
pub use fair_loss::FairLoss;
pub use gamma::GammaLoss;
pub use hinge_loss::HingeLoss;
pub use huber_loss::HuberLoss;
pub use listnet_loss::ListNetLoss;
pub use log_loss::LogLoss;
pub use mape::MapeLoss;
pub use poisson::PoissonLoss;
pub use quantile_loss::QuantileLoss;
pub use squared_log_loss::SquaredLogLoss;
pub use squared_loss::SquaredLoss;
pub use tweedie::TweedieLoss;

pub mod core;
pub use core::Objective;
pub use core::ObjectiveFunction;
