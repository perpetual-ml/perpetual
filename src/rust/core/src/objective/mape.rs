//! Mape Loss objective for regression.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Mean Absolute Percentage Error (MAPE) Loss objective.
/// `|y - yhat| / max(|y|, epsilon)`.
pub struct MapeLoss {}

impl ObjectiveFunction for MapeLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let epsilon = 1e-4;
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let diff = (*yhat_ - *y_).abs();
                    let denom = y_.abs().max(epsilon);
                    (w_ * diff / denom) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let diff = (*yhat_ - *y_).abs();
                    let denom = y_.abs().max(epsilon);
                    (diff / denom) as f32
                })
                .collect(),
        }
    }

    #[inline]
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let epsilon = 1e-4_f32;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let diff = yhat_val - y_val;
                    let denom = y_val.abs().max(epsilon);

                    let sign = if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    g.push(w_val * sign / denom);
                }
                (g, None)
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let diff = yhat_val - y_val;
                    let denom = y_val.abs().max(epsilon);

                    let sign = if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    g.push(sign / denom);
                }
                (g, None)
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl MapeLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let epsilon = 1e-4;
        let diff = (yhat - y).abs();
        let denom = y.abs().max(epsilon);
        let l = diff / denom;
        match sample_weight {
            Some(w) => (l * w) as f32,
            None => l as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mape_loss_init() {
        let y = vec![1.0, 2.0, 3.0];
        let loss_fn = MapeLoss::default();
        // Default initial_value is mean = 2.0
        assert_eq!(loss_fn.initial_value(&y, None, None), 2.0);
    }

    #[test]
    fn test_mape_loss() {
        let y = vec![2.0, 0.5];
        let yhat = vec![1.0, 1.0];
        let loss_fn = MapeLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // Loss[0]: |1.0 - 2.0| / 2.0 = 0.5
        // Loss[1]: |1.0 - 0.5| / 0.5 = 1.0
        assert_eq!(l, vec![0.5, 1.0]);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(lw, vec![1.0, 0.5]);
    }

    #[test]
    fn test_mape_gradient() {
        let y = vec![2.0, 0.5];
        let yhat = vec![1.0, 1.0];
        let loss_fn = MapeLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        // grad[0]: sign(1.0-2.0) / 2.0 = -1.0 / 2.0 = -0.5
        // grad[1]: sign(1.0-0.5) / 0.5 = 1.0 / 0.5 = 2.0
        assert_eq!(g, vec![-0.5, 2.0]);
        assert!(h.is_none());

        // Weighted
        let w = vec![2.0, 0.5];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert_eq!(gw, vec![-1.0, 1.0]);
        assert!(hw.is_none());
    }

    #[test]
    fn test_mape_weighted() {
        let y = vec![2.0, 0.5];
        let yhat = vec![1.0, 1.0];
        let w = vec![2.0, 0.5];
        let loss_fn = MapeLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        assert_eq!(g, vec![-1.0, 1.0]);
        assert!(h.is_none());
        assert_eq!(l, vec![1.0, 0.5]);
    }

    #[test]
    fn test_mape_gradient_and_loss() {
        let y = vec![2.0, 0.5];
        let yhat = vec![1.0, 1.0];
        let loss_fn = MapeLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g, vec![-0.5, 2.0]);
        assert!(h.is_none());
        assert_eq!(l, vec![0.5, 1.0]);
    }

    #[test]
    fn test_mape_gradient_and_loss_into() {
        let y = vec![2.0, 0.5];
        let yhat = vec![1.0, 1.0];
        let loss_fn = MapeLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess: Option<Vec<f32>> = None;
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![-0.5, 2.0]);
        assert!(hess.is_none());
        assert_eq!(loss, vec![0.5, 1.0]);
    }

    #[test]
    fn test_mape_loss_single() {
        let loss_fn = MapeLoss::default();
        assert_eq!(loss_fn.loss_single(2.0, 1.0, None), 0.5);
        assert_eq!(loss_fn.loss_single(2.0, 1.0, Some(2.0)), 1.0);
    }
}
