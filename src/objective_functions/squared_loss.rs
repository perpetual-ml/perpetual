//! Squared Loss function for regression.
use crate::{metrics::evaluation::Metric, objective_functions::objective::ObjectiveFunction};
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
}

/// Compute squared loss for a single sample without heap allocation.
#[inline]
#[allow(dead_code)]
pub fn squared_loss_single(y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
    let s = y - yhat;
    let l = s * s;
    match sample_weight {
        Some(w) => (l * w) as f32,
        None => l as f32,
    }
}
