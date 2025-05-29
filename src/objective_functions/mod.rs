//! Objective Functions
//! 
//! 
//! 
//! 

// import modules
mod adaptive_huber_loss;
mod huber_loss;
mod log_loss;
mod quantile_loss;
mod squared_loss;

// make loss functions public
pub use adaptive_huber_loss::AdaptiveHuberLoss;
pub use huber_loss::HuberLoss;
pub use log_loss::LogLoss;
pub use quantile_loss::QuantileLoss;
pub use squared_loss::SquaredLoss;

// crates
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::metrics::Metric;

// define types as smartpointers
// for thread safety
//
// NOTE: These can't be renamed without refactoring
// a few other modules - so don't do it unless you
// are ready to spend some time
pub type ObjFn    = Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> (Vec<f32>, Option<Vec<f32>>) + Send + Sync + 'static>;
pub type LossFn   = Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>) -> Vec<f32> + Send + Sync + 'static>;
pub type InitFn   = Arc<dyn Fn(&[f64], Option<&[f64]>) -> f64 + Send + Sync + 'static>;

// define traits for the objective
// functions.
//
// The ObjectiveFunction controls the
// logical flow downstream.
pub trait ObjectiveFunction: Send + Sync {

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

// define dispatches for each
// function in ObjectiveFunction
//
// These functions are called downstrean
// *after* instantiation
pub fn loss_callables<T>(instance: T) -> LossFn
where T: ObjectiveFunction + 'static
{
    Arc::new(
        move |y, yhat, weight| {
            instance.calc_loss(y, yhat, weight)
        }
    )
}

pub fn gradient_hessian_callables<T>(instance: T) -> ObjFn
where T: ObjectiveFunction + 'static
{
    Arc::new(
        move |y, yhat, weight| {
            instance.calc_grad_hess(y, yhat, weight)
        }
    )
}

pub fn calc_init_callables<T>(instance: T) -> InitFn
where
    T: ObjectiveFunction + 'static,
{
    Arc::new(
        move |y, weight| {
            instance.calc_init(y, weight)
        }
    )
}

// define Objective enum
// as before moving to smartpointers. The current
// implemnentation seems to be impossible without
// smartpointers.
// See commit: 581262534205b6bc8fd85694359a33c8983e8918
// for the old implementation.
//
// NOTE: *maybe* it is a good idea
// to pass values directly instead of 
// Option<f64>
//
// TODO: test at some point
#[derive(Clone, Serialize, Deserialize)]
pub enum Objective {
    LogLoss,
    SquaredLoss,
    QuantileLoss { quantile: Option<f64> },
    HuberLoss { delta: Option<f64> },
    AdaptiveHuberLoss { quantile: Option<f64> },

    /// Runtime-only variant: not serialized.
    #[serde(skip_serializing, skip_deserializing)]
    Custom(CustomObjective),
}


impl Objective {
    pub fn instantiate(&self) -> Arc<dyn ObjectiveFunction> {
        match self {
            Objective::LogLoss => Arc::new(LogLoss::default()),
            Objective::SquaredLoss => Arc::new(SquaredLoss::default()),
            Objective::QuantileLoss { quantile } => Arc::new(QuantileLoss { quantile: *quantile }),
            Objective::HuberLoss { delta } => Arc::new(HuberLoss { delta: *delta }),
            Objective::AdaptiveHuberLoss { quantile } => Arc::new(AdaptiveHuberLoss { quantile: *quantile }),
            Objective::Custom(c) => Arc::new(c.clone()),
        }
    }
}

// Blanket implementation for Arc-wrapped ObjectiveFunctions
impl<T> ObjectiveFunction for Arc<T>
where
    T: ObjectiveFunction + Send + Sync + ?Sized + 'static,
{
    fn hessian_is_constant(&self) -> bool { (**self).hessian_is_constant() }
    fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> Vec<f32> {
        (**self).calc_loss(y, yhat, sample_weight)
    }
    fn calc_grad_hess(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> (Vec<f32>, Option<Vec<f32>>) {
        (**self).calc_grad_hess(y, yhat, sample_weight)
    }
    fn calc_init(&self, y: &[f64], sample_weight: Option<&[f64]>) -> f64 {
        (**self).calc_init(y, sample_weight)
    }
    fn default_metric(&self) -> Metric { (**self).default_metric() }
}


/// A runtime-constructed objective that wraps closures or other `ObjectiveFunction` implementations.
#[derive(Clone)]
pub struct CustomObjective {
    pub grad_hess: ObjFn,
    pub loss: LossFn,
    pub init: InitFn,
    pub hessian_constant: bool,
    pub metric: Metric,
}

impl CustomObjective {
    /// Wrap an existing `ObjectiveFunction` implementation.
    pub fn from<T>(obj: T) -> Self
    where
        T: ObjectiveFunction + Clone + 'static,
    {
        CustomObjective {
            grad_hess: gradient_hessian_callables(obj.clone()),
            loss:      loss_callables(obj.clone()),
            init:      calc_init_callables(obj.clone()),
            hessian_constant: obj.hessian_is_constant(),
            metric:    obj.default_metric(),
        }
    }

    /// Construct by specifying each component closure manually.
    pub fn new(
        grad_hess: ObjFn,
        loss: LossFn,
        init: InitFn,
        hessian_constant: bool,
        metric: Metric,
    ) -> Self {
        CustomObjective { grad_hess, loss, init, hessian_constant, metric }
    }
}

impl ObjectiveFunction for CustomObjective {
    #[inline]
    fn hessian_is_constant(&self) -> bool {
        self.hessian_constant
    }

    #[inline]
    fn calc_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Vec<f32> {
        (self.loss)(y, yhat, sample_weight)
    }

    #[inline]
    fn calc_grad_hess(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        (self.grad_hess)(y, yhat, sample_weight)
    }

    #[inline]
    fn calc_init(
        &self,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> f64 {
        (self.init)(y, sample_weight)
    }

    #[inline]
    fn default_metric(&self) -> Metric {
        self.metric.clone()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::objective_functions::Objective;
    use crate::objective_functions::Arc;

    // Common data used
    // across tests
    static Y: &[f64] = &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    static YHAT1: &[f64] = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    static YHAT2: &[f64] = &[0.0, 0.0, -1.0, 1.0, 0.0, 1.0];

    // helper function for the 
    // tests
    fn sum_loss(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        obj.calc_loss(Y, yhat, None).iter().copied().sum()
    }

    fn sum_grad(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        let (g, _) = obj.calc_grad_hess(Y, yhat, None);
        g.iter().copied().sum()
    }
    
    // actual tests
    #[test]
    fn test_logloss_loss() {
        let objective_function = Objective::LogLoss.instantiate();
        assert!(sum_loss(&objective_function, YHAT1) < sum_loss(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_grad() {
        let objective_function = Objective::LogLoss.instantiate();
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_init() {
        let objective_function = Objective::LogLoss.instantiate();
        assert_eq!(objective_function.calc_init(Y, None), 0.0);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::LogLoss.instantiate().calc_init(&all_ones, None), f64::INFINITY);

        let all_zeros = vec![0.0; 6];
        assert_eq!(Objective::LogLoss.instantiate().calc_init(&all_zeros, None), f64::NEG_INFINITY);

        let mixed = &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let expected = f64::ln(2.0 / 4.0);
        assert_eq!(Objective::LogLoss.instantiate().calc_init(mixed, None), expected);
    }

    #[test]
    fn test_mse_init() {
        let objective_function = Objective::SquaredLoss.instantiate();
        assert_eq!(objective_function.calc_init(Y, None), 0.5);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::SquaredLoss.instantiate().calc_init(&all_ones, None), 1.0);

        let all_minus = vec![-1.0; 6];
        assert_eq!(Objective::SquaredLoss.instantiate().calc_init(&all_minus, None), -1.0);

        let mixed = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(Objective::SquaredLoss.instantiate().calc_init(mixed, None), 0.0);
    }

    #[test]
    fn test_quantile_init() {
        let weights = &[0.0, 0.5, 1.0, 0.3, 0.5];
        let y_vals = &[1.0, 2.0, 9.0, 3.2, 4.0];

        let objective_function_low = Objective::QuantileLoss { quantile: Some(0.1) }.instantiate();
        assert_eq!(objective_function_low.calc_init(y_vals, Some(weights)), 2.0);

        let objective_function_high = Objective::QuantileLoss { quantile: Some(0.9) }.instantiate();
        assert_eq!(objective_function_high.calc_init(y_vals, Some(weights)), 9.0);
    }

    #[test]
    fn test_adaptive_huberloss_loss_and_grad() {
        let objective_function = Objective::AdaptiveHuberLoss { quantile: Some(0.5) }.instantiate();
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_huberloss_loss_and_grad() {
        let objective_function = Objective::HuberLoss { delta: Some(1.0) }.instantiate();
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

}
