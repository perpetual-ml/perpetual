use crate::{
    metrics::evaluation::Metric,
    objective_functions::{AdaptiveHuberLoss, HuberLoss, ListNetLoss, LogLoss, QuantileLoss, SquaredLoss},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    #[serde(skip)]
    Custom(Arc<dyn ObjectiveFunction>),
}

impl Objective {
    pub fn new_custom<T>(objective: T) -> Self
    where
        T: ObjectiveFunction + 'static,
    {
        Objective::Custom(Arc::new(objective))
    }
}

impl ObjectiveFunction for Objective {
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        match self {
            Objective::LogLoss => LogLoss::default().loss(y, yhat, sample_weight, group),
            Objective::SquaredLoss => SquaredLoss::default().loss(y, yhat, sample_weight, group),
            Objective::QuantileLoss { quantile } => {
                QuantileLoss { quantile: *quantile }.loss(y, yhat, sample_weight, group)
            }
            Objective::HuberLoss { delta } => HuberLoss { delta: *delta }.loss(y, yhat, sample_weight, group),
            Objective::AdaptiveHuberLoss { quantile } => {
                AdaptiveHuberLoss { quantile: *quantile }.loss(y, yhat, sample_weight, group)
            }
            Objective::ListNetLoss => ListNetLoss::default().loss(y, yhat, sample_weight, group),
            Objective::Custom(arc) => arc.loss(y, yhat, sample_weight, group),
        }
    }

    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        match self {
            Objective::LogLoss => LogLoss::default().gradient(y, yhat, sample_weight, group),
            Objective::SquaredLoss => SquaredLoss::default().gradient(y, yhat, sample_weight, group),
            Objective::QuantileLoss { quantile } => {
                QuantileLoss { quantile: *quantile }.gradient(y, yhat, sample_weight, group)
            }
            Objective::HuberLoss { delta } => HuberLoss { delta: *delta }.gradient(y, yhat, sample_weight, group),
            Objective::AdaptiveHuberLoss { quantile } => {
                AdaptiveHuberLoss { quantile: *quantile }.gradient(y, yhat, sample_weight, group)
            }
            Objective::ListNetLoss => ListNetLoss::default().gradient(y, yhat, sample_weight, group),
            Objective::Custom(arc) => arc.gradient(y, yhat, sample_weight, group),
        }
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64 {
        match self {
            Objective::LogLoss => LogLoss::default().initial_value(y, sample_weight, group),
            Objective::SquaredLoss => SquaredLoss::default().initial_value(y, sample_weight, group),
            Objective::QuantileLoss { quantile } => {
                QuantileLoss { quantile: *quantile }.initial_value(y, sample_weight, group)
            }
            Objective::HuberLoss { delta } => HuberLoss { delta: *delta }.initial_value(y, sample_weight, group),
            Objective::AdaptiveHuberLoss { quantile } => {
                AdaptiveHuberLoss { quantile: *quantile }.initial_value(y, sample_weight, group)
            }
            Objective::ListNetLoss => ListNetLoss::default().initial_value(y, sample_weight, group),
            Objective::Custom(arc) => arc.initial_value(y, sample_weight, group),
        }
    }

    fn default_metric(&self) -> Metric {
        match self {
            Objective::LogLoss => LogLoss::default().default_metric(),
            Objective::SquaredLoss => SquaredLoss::default().default_metric(),
            Objective::QuantileLoss { quantile } => QuantileLoss { quantile: *quantile }.default_metric(),
            Objective::HuberLoss { delta } => HuberLoss { delta: *delta }.default_metric(),
            Objective::AdaptiveHuberLoss { quantile } => AdaptiveHuberLoss { quantile: *quantile }.default_metric(),
            Objective::ListNetLoss => ListNetLoss::default().default_metric(),
            Objective::Custom(arc) => arc.default_metric(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objective_functions::objective::Objective;

    // Common data used across tests
    static Y: &[f64] = &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    static YHAT1: &[f64] = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    static YHAT2: &[f64] = &[0.0, 0.0, -1.0, 1.0, 0.0, 1.0];

    // new helper function for the tests
    fn sum_loss(obj: &Objective, yhat: &[f64]) -> f32 {
        obj.loss(Y, yhat, None, None).iter().copied().sum()
    }

    fn sum_grad(obj: &Objective, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y, yhat, None, None);
        g.iter().copied().sum()
    }

    // actual tests
    #[test]
    fn test_logloss_loss() {
        let objective_function = Objective::LogLoss;
        assert!(sum_loss(&objective_function, YHAT1) < sum_loss(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_grad() {
        let objective_function = Objective::LogLoss;
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_logloss_init() {
        let objective_function = Objective::LogLoss;
        assert_eq!(objective_function.initial_value(Y, None, None), 0.0);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::LogLoss.initial_value(&all_ones, None, None), f64::INFINITY);

        let all_zeros = vec![0.0; 6];
        assert_eq!(
            Objective::LogLoss.initial_value(&all_zeros, None, None),
            f64::NEG_INFINITY
        );

        let mixed = &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let expected = f64::ln(2.0 / 4.0);
        assert_eq!(Objective::LogLoss.initial_value(mixed, None, None), expected);
    }

    #[test]
    fn test_mse_init() {
        let objective_function = Objective::SquaredLoss;
        assert_eq!(objective_function.initial_value(Y, None, None), 0.5);

        let all_ones = vec![1.0; 6];
        assert_eq!(Objective::SquaredLoss.initial_value(&all_ones, None, None), 1.0);

        let all_minus = vec![-1.0; 6];
        assert_eq!(Objective::SquaredLoss.initial_value(&all_minus, None, None), -1.0);

        let mixed = &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(Objective::SquaredLoss.initial_value(mixed, None, None), 0.0);
    }

    #[test]
    fn test_quantile_init() {
        let weights = &[0.0, 0.5, 1.0, 0.3, 0.5];
        let y_vals = &[1.0, 2.0, 9.0, 3.2, 4.0];

        let objective_function_low = Objective::QuantileLoss { quantile: Some(0.1) };
        assert_eq!(objective_function_low.initial_value(y_vals, Some(weights), None), 2.0);

        let objective_function_high = Objective::QuantileLoss { quantile: Some(0.9) };
        assert_eq!(objective_function_high.initial_value(y_vals, Some(weights), None), 9.0);
    }

    #[test]
    fn test_adaptive_huberloss_loss_and_grad() {
        let objective_function = Objective::AdaptiveHuberLoss { quantile: Some(0.5) };
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    #[test]
    fn test_huberloss_loss_and_grad() {
        let objective_function = Objective::HuberLoss { delta: Some(1.0) };
        assert!(sum_loss(&objective_function, YHAT1) > sum_loss(&objective_function, YHAT2));
        assert!(sum_grad(&objective_function, YHAT1) < sum_grad(&objective_function, YHAT2));
    }

    static Y_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT1_RANK: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    static YHAT2_RANK: &[f64] = &[3.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    static YHAT3_RANK: &[f64] = &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0]; // NOTE: should be the
                                                                 // same as YHAT1_RANK
    static GROUP: &[u64] = &[3, 3];

    fn sum_loss_rank(obj: &Objective, yhat: &[f64]) -> f32 {
        obj.loss(Y_RANK, yhat, None, Some(GROUP)).iter().copied().sum()
    }

    fn sum_grad_rank(obj: &Objective, yhat: &[f64]) -> f32 {
        let (g, _) = obj.gradient(Y_RANK, yhat, None, Some(GROUP));
        g.iter().map(|x| x.abs()).sum()
    }

    #[test]
    fn test_listnet_loss_and_grad() {
        let objective_function = Objective::ListNetLoss;
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
