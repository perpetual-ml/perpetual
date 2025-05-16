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
use serde::{Deserialize, Serialize};

type ObjFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f64>) -> (Vec<f32>, Option<Vec<f32>>);
type LossFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f64>) -> Vec<f32>;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Objective {
    LogLoss,
    SquaredLoss,
    QuantileLoss,
    AdaptiveHuberLoss,
    HuberLoss,
}

pub fn loss_callables(objective: &Objective) -> LossFn {
    match objective {
        Objective::LogLoss => LogLoss::calc_loss,
        Objective::SquaredLoss => SquaredLoss::calc_loss,
        Objective::QuantileLoss => QuantileLoss::calc_loss,
        Objective::AdaptiveHuberLoss => AdaptiveHuberLoss::calc_loss,
        Objective::HuberLoss => HuberLoss::calc_loss,
    }
}

pub fn gradient_hessian_callables(objective: &Objective) -> ObjFn {
    match objective {
        Objective::LogLoss => LogLoss::calc_grad_hess,
        Objective::SquaredLoss => SquaredLoss::calc_grad_hess,
        Objective::QuantileLoss => QuantileLoss::calc_grad_hess,
        Objective::AdaptiveHuberLoss => AdaptiveHuberLoss::calc_grad_hess,
        Objective::HuberLoss => HuberLoss::calc_grad_hess,
    }
}

pub fn calc_init_callables(objective: &Objective) -> fn(&[f64], Option<&[f64]>, Option<f64>) -> f64 {
    match objective {
        Objective::LogLoss => LogLoss::calc_init,
        Objective::SquaredLoss => SquaredLoss::calc_init,
        Objective::QuantileLoss => QuantileLoss::calc_init,
        Objective::AdaptiveHuberLoss => AdaptiveHuberLoss::calc_init,
        Objective::HuberLoss => HuberLoss::calc_init,
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>) -> Vec<f32>;
    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f64>,
    ) -> (Vec<f32>, Option<Vec<f32>>);
    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>) -> f64;
    fn default_metric() -> Metric;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_logloss_loss() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let l1 = LogLoss::calc_loss(&y, &yhat1, None, None);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = LogLoss::calc_loss(&y, &yhat2, None, None);
        assert!(l1.iter().sum::<f32>() < l2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let (g1, _) = LogLoss::calc_grad_hess(&y, &yhat1, None, None);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let (g2, _) = LogLoss::calc_grad_hess(&y, &yhat2, None, None);
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let l1 = LogLoss::calc_init(&y, None, None);
        assert!(l1 == 0.);

        let y = vec![1.0; 6];
        let l2 = LogLoss::calc_init(&y, None, None);
        assert!(l2 == f64::INFINITY);

        let y = vec![0.0; 6];
        let l3 = LogLoss::calc_init(&y, None, None);
        assert!(l3 == f64::NEG_INFINITY);

        let y = vec![0., 0., 0., 0., 1., 1.];
        let l4 = LogLoss::calc_init(&y, None, None);
        assert!(l4 == f64::ln(2. / 4.));
    }

    #[test]
    fn test_mse_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let l1 = SquaredLoss::calc_init(&y, None, None);
        assert!(l1 == 0.5);

        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let l2 = SquaredLoss::calc_init(&y, None, None);
        assert!(l2 == 1.);

        let y = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let l3 = SquaredLoss::calc_init(&y, None, None);
        assert!(l3 == -1.);

        let y = vec![-1.0, -1.0, -1.0, 1., 1., 1.];
        let l4 = SquaredLoss::calc_init(&y, None, None);
        assert!(l4 == 0.);
    }

    #[test]
    fn test_quantile_init() {
        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let w = vec![0.0, 0.5, 1.0, 0.3, 0.5];
        let l1 = QuantileLoss::calc_init(&y, Some(&w), Some(0.1));
        println!("{}", l1);
        assert!(l1 == 2.0);

        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let w = vec![0.0, 0.5, 1.0, 0.3, 0.5];
        let l2 = QuantileLoss::calc_init(&y, Some(&w), Some(0.9));
        println!("{}", l2);
        assert!(l2 == 9.0);
    }

    #[test]
    fn test_adaptive_huberloss_loss() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let l1 = AdaptiveHuberLoss::calc_loss(&y, &yhat1, None, Some(0.5));
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = AdaptiveHuberLoss::calc_loss(&y, &yhat2, None, Some(0.5));
        assert!(l1.iter().sum::<f32>() > l2.iter().sum::<f32>());
    }

    #[test]
    fn test_adaptive_huberloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let (g1, _) = AdaptiveHuberLoss::calc_grad_hess(&y, &yhat1, None, Some(0.5));
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let (g2, _) = AdaptiveHuberLoss::calc_grad_hess(&y, &yhat2, None, Some(0.5));
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }
}
