//! Objective Functions
//!
//! Some text
//!
//!
//!

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

// crates
use crate::metrics::Metric;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// define types as smartpointers for thread safety
pub type ObjectiveFn =
    Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>, Option<&[u64]>) -> (Vec<f32>, Option<Vec<f32>>) + Send + Sync + 'static>;
pub type LossFn = Arc<dyn Fn(&[f64], &[f64], Option<&[f64]>, Option<&[u64]>) -> Vec<f32> + Send + Sync + 'static>;
pub type InitialValueFn = Arc<dyn Fn(&[f64], Option<&[f64]>, Option<&[u64]>) -> f64 + Send + Sync + 'static>;

// define traits for the objective
// functions.
//
// The ObjectiveFunction controls the
// logical flow downstream.
/// Objective function traits
///
///
/// ## Traits:
///
/// * `fn loss`:
/// * `fn gradient`:
/// * `fn initial_value`:
///
/// ## Example:
///
///
///
pub trait ObjectiveFunction: Send + Sync {
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32>;
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>);
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64;
    fn default_metric(&self) -> Metric;
}

// define dispatches for each
// function in ObjectiveFunction
//
// These functions are called downstrean
// *after* instantiation
pub fn loss_callables<T>(instance: Arc<T>) -> LossFn
where
    T: ObjectiveFunction + ?Sized + 'static,
{
    Arc::new(move |y, yhat, w, g| instance.clone().loss(y, yhat, w, g))
}

pub fn gradient_callables<T>(instance: Arc<T>) -> ObjectiveFn
where
    T: ObjectiveFunction + ?Sized + 'static,
{
    Arc::new(move |y, yhat, w, g| instance.clone().gradient(y, yhat, w, g))
}

pub fn initial_value_callables<T>(instance: Arc<T>) -> InitialValueFn
where
    T: ObjectiveFunction + ?Sized + 'static,
{
    Arc::new(move |y, w, g| instance.clone().initial_value(y, w, g))
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
//
// NOTE: clone is only added for temporary
// compatibility downstream
#[derive(Serialize, Deserialize, Clone)]
pub enum Objective {
    LogLoss,
    SquaredLoss,
    QuantileLoss {
        quantile: Option<f64>,
    },
    HuberLoss {
        delta: Option<f64>,
    },
    AdaptiveHuberLoss {
        quantile: Option<f64>,
    },
    ListNetLoss,

    /// Custom objective function
    ///
    ///
    #[serde(skip_serializing, skip_deserializing)]
    Custom(Arc<dyn ObjectiveFunction>),
}

// NOTE: it is (most likely) not
// possible to pass parameters
// without foo::bar{} structure
impl Objective {
    // custom function instantiation
    pub fn function<T>(objective: T) -> Self
    where
        T: ObjectiveFunction + 'static,
    {
        Objective::Custom(Arc::new(objective))
    }

    pub fn as_function(&self) -> Arc<dyn ObjectiveFunction> {
        match self {
            Objective::LogLoss => Arc::new(LogLoss::default()),
            Objective::SquaredLoss => Arc::new(SquaredLoss::default()),
            Objective::QuantileLoss { quantile } => Arc::new(QuantileLoss { quantile: *quantile }),
            Objective::HuberLoss { delta } => Arc::new(HuberLoss { delta: *delta }),
            Objective::AdaptiveHuberLoss { quantile } => Arc::new(AdaptiveHuberLoss { quantile: *quantile }),
            Objective::ListNetLoss => Arc::new(ListNetLoss::default()),
            Objective::Custom(arc) => arc.clone(),
        }
    }
}

// This ensures that the objective
// functions can be called downstream
// - I am not sure exactly what it does
// but it works.
impl<T> ObjectiveFunction for Arc<T>
where
    T: ObjectiveFunction + Send + Sync + ?Sized + 'static,
{
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        (**self).loss(y, yhat, sample_weight, group)
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        (**self).gradient(y, yhat, sample_weight, group)
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64 {
        (**self).initial_value(y, sample_weight, group)
    }

    fn default_metric(&self) -> Metric {
        (**self).default_metric()
    }
}

// Custom objective
// struct (experimental)
#[derive(Clone)]
pub struct CustomObjective {
    pub grad_hess: ObjectiveFn,
    pub loss: LossFn,
    pub init: InitialValueFn,
    pub metric: Metric,
}

impl CustomObjective {
    pub fn new<T>(obj: T) -> Self
    where
        T: ObjectiveFunction + Clone + 'static,
    {
        let shared: Arc<T> = Arc::new(obj);
        CustomObjective {
            grad_hess: gradient_callables(shared.clone()),
            loss: loss_callables(shared.clone()),
            init: initial_value_callables(shared.clone()),
            metric: shared.default_metric(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objective_functions::Arc;
    use crate::objective_functions::Objective;

    // Common data used
    // across tests
    static Y: &[f64] = &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    static YHAT1: &[f64] = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    static YHAT2: &[f64] = &[0.0, 0.0, -1.0, 1.0, 0.0, 1.0];

    // helper function for the
    // tests
    fn sum_loss(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        obj.loss(Y, yhat, None, None).iter().copied().sum()
    }

    fn sum_grad(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y, yhat, None, None);
        g.iter().copied().sum()
    }

    // actual tests
    #[test]
    fn test_logloss_loss() {
        let objective_function = Objective::LogLoss.as_function();
        assert!(sum_loss(&objective_function, YHAT1) < sum_loss(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_grad() {
        let objective_function = Objective::LogLoss.as_function();
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_init() {
        let objective_function = Objective::LogLoss.as_function();
        assert_eq!(objective_function.initial_value(Y, None, None), 0.0);

        let all_ones = vec![1.0; 6];
        assert_eq!(
            Objective::LogLoss.as_function().initial_value(&all_ones, None, None),
            f64::INFINITY
        );

        let all_zeros = vec![0.0; 6];
        assert_eq!(
            Objective::LogLoss.as_function().initial_value(&all_zeros, None, None),
            f64::NEG_INFINITY
        );

        let mixed = &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let expected = f64::ln(2.0 / 4.0);
        assert_eq!(
            Objective::LogLoss.as_function().initial_value(mixed, None, None),
            expected
        );
    }

    #[test]
    fn test_mse_init() {
        let objective_function = Objective::SquaredLoss.as_function();
        assert_eq!(objective_function.initial_value(Y, None, None), 0.5);

        let all_ones = vec![1.0; 6];
        assert_eq!(
            Objective::SquaredLoss
                .as_function()
                .initial_value(&all_ones, None, None),
            1.0
        );

        let all_minus = vec![-1.0; 6];
        assert_eq!(
            Objective::SquaredLoss
                .as_function()
                .initial_value(&all_minus, None, None),
            -1.0
        );

        let mixed = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            Objective::SquaredLoss.as_function().initial_value(mixed, None, None),
            0.0
        );
    }

    #[test]
    fn test_quantile_init() {
        let weights = &[0.0, 0.5, 1.0, 0.3, 0.5];
        let y_vals = &[1.0, 2.0, 9.0, 3.2, 4.0];

        let objective_function_low = Objective::QuantileLoss { quantile: Some(0.1) }.as_function();
        assert_eq!(objective_function_low.initial_value(y_vals, Some(weights), None), 2.0);

        let objective_function_high = Objective::QuantileLoss { quantile: Some(0.9) }.as_function();
        assert_eq!(objective_function_high.initial_value(y_vals, Some(weights), None), 9.0);
    }

    #[test]
    fn test_adaptive_huberloss_loss_and_grad() {
        let objective_function = Objective::AdaptiveHuberLoss { quantile: Some(0.5) }.as_function();
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_huberloss_loss_and_grad() {
        let objective_function = Objective::HuberLoss { delta: Some(1.0) }.as_function();
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    static Y_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT1_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT2_RANK: &[f64] = &[3.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    static YHAT3_RANK: &[f64] = &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0]; // NOTE: should be the
                                                                 // same as YHAT1_RANK
    static GROUP: &[u64] = &[3, 3];

    fn sum_loss_rank(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        obj.loss(Y_RANK, yhat, None, Some(GROUP)).iter().copied().sum()
    }

    fn sum_grad_rank(obj: &Arc<dyn crate::objective_functions::ObjectiveFunction>, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y_RANK, yhat, None, Some(GROUP));
        g.iter().map(|x| x.abs()).sum()
    }

    #[test]
    fn test_listnet_loss_and_grad() {
        let objective_function = Objective::ListNetLoss.as_function();
        let good_loss_sum = sum_loss_rank(&objective_function, YHAT1_RANK);
        let bad_loss_sum = sum_loss_rank(&objective_function, YHAT2_RANK);
        let also_good_loss_sum = sum_loss_rank(&objective_function, YHAT3_RANK);

        let good_grad_sum = sum_grad_rank(&objective_function, YHAT1_RANK);
        let bad_grad_sum = sum_grad_rank(&objective_function, YHAT2_RANK);
        let also_good_grad_sum = sum_grad_rank(&objective_function, YHAT3_RANK);

        assert!(good_loss_sum < bad_loss_sum);
        assert!(good_grad_sum < bad_grad_sum);

        assert!(good_loss_sum == also_good_loss_sum);
        assert!(good_grad_sum == also_good_grad_sum);
    }
}
