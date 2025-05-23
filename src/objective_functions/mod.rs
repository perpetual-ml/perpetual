mod adaptive_huber_loss;
mod huber_loss;
mod log_loss;
mod quantile_loss;
mod squared_loss;

pub use adaptive_huber_loss::AdaptiveHuberLoss;
pub use huber_loss::HuberLoss;
pub use log_loss::LogLoss;
pub use quantile_loss::QuantileLoss;
pub use squared_loss::SquaredLoss;
use crate::metrics::Metric;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

pub trait ObjectiveFunction: Send + Sync {
    // The objective function
    // constrols the the flow downstream

    // Is the hessian const?
    fn hessian_is_constant(&self) -> bool;

    fn calc_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Vec<f32>;

    fn calc_grad_hess(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>);

    fn calc_init(
        &self,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> f64;

    fn default_metric(&self) -> Metric;
}

pub type ObjFn    = Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> (Vec<f32>, Option<Vec<f32>>) + Send + Sync + 'static>;
pub type LossFn   = Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> Vec<f32> + Send + Sync + 'static>;
pub type InitFn   = Arc<dyn Fn(&[f64], Option<&[f64]>) -> f64 + Send + Sync + 'static>;

pub fn loss_callables<T>(instance: T) -> LossFn
where T: ObjectiveFunction + 'static
{
    Arc::new(move |y, yhat, weight| instance.calc_loss(y, yhat, weight))
}

pub fn gradient_hessian_callables<T>(instance: T) -> ObjFn
where T: ObjectiveFunction + 'static
{
    Arc::new(move |y, yhat, weight| instance.calc_grad_hess(y, yhat, weight))
}


pub fn calc_init_callables<T>(instance: T) -> InitFn
where
    T: ObjectiveFunction + 'static,
{
    Arc::new(move |y, weight| {
        instance.calc_init(y, weight)
    })
}

// 1) Define your serde‐friendly enum of built-in objectives:
#[derive(Clone, Serialize, Deserialize)]
pub enum Objective {
    LogLoss,
    SquaredLoss,
    QuantileLoss { quantile: Option<f64> },
    HuberLoss { delta: Option<f64> },
    AdaptiveHuberLoss { quantile: Option<f64> },
}

impl Objective {
    /// Instantiate the concrete ObjectiveFunction for this variant
    pub fn instantiate(&self) -> Arc<dyn ObjectiveFunction> {
        match self {
            Objective::LogLoss => Arc::new(LogLoss::default()),
            Objective::SquaredLoss => Arc::new(SquaredLoss::default()),
            Objective::QuantileLoss { quantile} => Arc::new(QuantileLoss { quantile: *quantile }),
            Objective::HuberLoss { delta } => Arc::new(HuberLoss { delta: *delta }),
            Objective::AdaptiveHuberLoss { quantile } => Arc::new(AdaptiveHuberLoss { quantile: *quantile }),
        }
    }
}

/// Container for a fully‐custom objective,
/// if the user chooses to inject one at runtime.
#[derive(Clone)]
pub struct CustomObjective {
    pub grad_hess: ObjFn,
    pub loss:      LossFn,
    pub init:      InitFn,
    pub hessian_constant: bool,
    pub metric:    Metric,
}


impl<T> ObjectiveFunction for Arc<T>
where
    T: ObjectiveFunction + Send + Sync + ?Sized + 'static,
{
    fn hessian_is_constant(&self) -> bool {
        (**self).hessian_is_constant()
    }

    fn calc_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Vec<f32> {
        (**self).calc_loss(y, yhat, sample_weight)
    }

    fn calc_grad_hess(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        (**self).calc_grad_hess(y, yhat, sample_weight)
    }

    fn calc_init(
        &self,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> f64 {
        (**self).calc_init(y, sample_weight)
    }

    fn default_metric(&self) -> Metric {
        (**self).default_metric()
    }
}