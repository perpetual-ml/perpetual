//! Squared Loss function for regression.
use crate::{metrics::evaluation::Metric, objective::ObjectiveFunction};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Squared Error loss — minimizes `(y - ŷ)²`.
pub struct SquaredLoss {}

impl ObjectiveFunction for SquaredLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let s = *y_ - *yhat_;
                    (s * s * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let s = *y_ - *yhat_;
                    (s * s) as f32
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

        match sample_weight {
            Some(w) => {
                let mut h = Vec::with_capacity(len);
                for i in 0..len {
                    let diff = (yhat[i] - y[i]) as f32;
                    let w_val = w[i] as f32;
                    g.push(diff * w_val);
                    h.push(w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    g.push((yhat[i] - y[i]) as f32);
                }
                (g, None)
            }
        }
    }

    // `initial_value`: inherits the default weighted-mean implementation from the trait.

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredLogError
    }

    fn gradient_and_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut l = Vec::with_capacity(len);

        match sample_weight {
            Some(w) => {
                let mut h = Vec::with_capacity(len);
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    let w_val = w[i] as f32;
                    g.push(diff as f32 * w_val);
                    h.push(w_val);
                    l.push((diff * diff * w[i]) as f32);
                }
                (g, Some(h), l)
            }
            None => {
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    g.push(diff as f32);
                    l.push((diff * diff) as f32);
                }
                (g, None, l)
            }
        }
    }

    fn gradient_and_loss_into(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
        grad: &mut [f32],
        hess: &mut Option<Vec<f32>>,
        loss: &mut [f32],
    ) {
        let len = y.len();
        match sample_weight {
            Some(w) => {
                let h = hess.get_or_insert_with(|| vec![0.0; len]);
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    let w_val = w[i] as f32;
                    grad[i] = diff as f32 * w_val;
                    h[i] = w_val;
                    loss[i] = (diff * diff * w[i]) as f32;
                }
            }
            None => {
                *hess = None;
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    grad[i] = diff as f32;
                    loss[i] = (diff * diff) as f32;
                }
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl SquaredLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let s = y - yhat;
        let l = s * s;
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
    fn test_squared_loss() {
        let y = vec![1.0, 2.0];
        let yhat = vec![1.5, 1.0]; // diffs (yhat - y): 0.5, -1.0. s (y - yhat): -0.5, 1.0
        let loss_fn = SquaredLoss::default();

        // Test loss: s*s
        // s = -0.5 -> 0.25
        // s = 1.0 -> 1.0
        let l = loss_fn.loss(&y, &yhat, None, None);
        assert!((l[0] - 0.25).abs() < 1e-6);
        assert!((l[1] - 1.0).abs() < 1e-6);

        // Test gradient/hessian: g = yhat - y, h = 1.0
        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        assert_eq!(h, None); // SquaredLoss returns None for constant hessian when no weights
        assert!((g[0] - 0.5).abs() < 1e-6);
        assert!((g[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_squared_loss_weighted() {
        let y = vec![1.0];
        let yhat = vec![2.0]; // diff = 1.0
        let weights = vec![0.5];
        let loss_fn = SquaredLoss::default();

        let l = loss_fn.loss(&y, &yhat, Some(&weights), None);
        assert!((l[0] - 0.5).abs() < 1e-6);

        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&weights), None);
        let h = h.unwrap();
        assert!((g[0] - 0.5).abs() < 1e-6);
        assert!((h[0] - 0.5).abs() < 1e-6);
    }
}
