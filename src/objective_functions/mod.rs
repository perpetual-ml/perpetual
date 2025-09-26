// import modules
mod adaptive_huber_loss;
mod huber_loss;
mod listnet_loss;
mod log_loss;
mod quantile_loss;
mod squared_loss;

// make loss functions public
pub use adaptive_huber_loss::AdaptiveHuberLoss;
pub use huber_loss::HuberLoss;
pub use listnet_loss::ListNetLoss;
pub use log_loss::LogLoss;
pub use quantile_loss::QuantileLoss;
pub use squared_loss::SquaredLoss;

pub mod objective;

pub use objective::Objective;
pub use objective::ObjectiveFunction;
