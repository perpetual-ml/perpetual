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


pub trait ObjectiveFunction {
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

pub type ObjFn    = Box<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> (Vec<f32>, Option<Vec<f32>>) + 'static>;
pub type LossFn   = Box<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> Vec<f32> + 'static>;
pub type InitFn   = Box<dyn Fn(&[f64], Option<&[f64]>) -> f64 + 'static>;

pub fn loss_callables<T>(instance: T) -> LossFn
where T: ObjectiveFunction + 'static
{
    Box::new(move |y, yhat, weight| instance.calc_loss(y, yhat, weight))
}

pub fn gradient_hessian_callables<T>(instance: T) -> ObjFn
where T: ObjectiveFunction + 'static
{
    Box::new(move |y, yhat, weight| instance.calc_grad_hess(y, yhat, weight))
}


pub fn calc_init_callables<T>(instance: T) -> InitFn
where
    T: ObjectiveFunction + 'static,
{
    Box::new(move |y, weight| {
        instance.calc_init(y, weight)
    })
}


