//! Adaptive Huber Loss function — automatically adjusts `delta` from the
//! residual distribution.

use crate::{objective_functions::objective::ObjectiveFunction, utils::weighted_median};
use serde::{Deserialize, Serialize};

/// Adaptive Huber Loss — delta is derived from the `quantile` of the
/// absolute-error distribution at each evaluation.
#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct AdaptiveHuberLoss {
    pub quantile: Option<f64>,
}

/// Compute the adaptive delta: the `alpha`-quantile of |y - yhat|.
#[inline]
fn adaptive_delta(y: &[f64], yhat: &[f64], alpha: f64) -> f64 {
    let n = y.len();
    let mut abs_res: Vec<f64> = y.iter().zip(yhat).map(|(&yi, &yh)| (yi - yh).abs()).collect();
    abs_res.sort_by(|a, b| a.total_cmp(b));
    abs_res[((n as f64) * alpha).floor() as usize % n]
}

impl ObjectiveFunction for AdaptiveHuberLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let delta = adaptive_delta(y, yhat, self.quantile.unwrap_or(0.5));

        match sample_weight {
            Some(weights) => y
                .iter()
                .zip(yhat.iter())
                .enumerate()
                .map(|(i, (&yi, &yh))| {
                    let r = yi - yh;
                    let ar = r.abs();
                    let base = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    (base * weights[i]) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat.iter())
                .map(|(&yi, &yh)| {
                    let r = yi - yh;
                    let ar = r.abs();
                    let loss = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    loss as f32
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
        let delta = adaptive_delta(y, yhat, self.quantile.unwrap_or(0.5)) as f32;
        let n = y.len();
        let mut g = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);

        match sample_weight {
            Some(weights) => {
                for i in 0..n {
                    let diff = (yhat[i] - y[i]) as f32;
                    let w = weights[i] as f32;
                    if diff.abs() <= delta {
                        g.push(diff * w);
                        h.push(w);
                    } else {
                        g.push(delta * diff.signum() * w);
                        h.push(0.0);
                    }
                }
                (g, Some(h))
            }
            None => {
                for i in 0..n {
                    let diff = (yhat[i] - y[i]) as f32;
                    if diff.abs() <= delta {
                        g.push(diff);
                        h.push(1.0);
                    } else {
                        g.push(delta * diff.signum());
                        h.push(0.0);
                    }
                }
                (g, Some(h))
            }
        }
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        weighted_median(y, sample_weight)
    }

    // `default_metric`: inherits the trait default (`RootMeanSquaredError`).

    fn gradient_and_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        // Compute delta once to avoid sorting the residuals twice.
        let delta = adaptive_delta(y, yhat, self.quantile.unwrap_or(0.5));
        let delta32 = delta as f32;
        let n = y.len();
        let mut g = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut l = Vec::with_capacity(n);

        match sample_weight {
            Some(weights) => {
                for i in 0..n {
                    let r = yhat[i] - y[i];
                    let ar = r.abs();
                    let diff = r as f32;
                    let w = weights[i] as f32;
                    if ar <= delta {
                        g.push(diff * w);
                        h.push(w);
                        l.push((0.5 * r * r * weights[i]) as f32);
                    } else {
                        g.push(delta32 * diff.signum() * w);
                        h.push(0.0);
                        l.push((delta * (ar - 0.5 * delta) * weights[i]) as f32);
                    }
                }
            }
            None => {
                for i in 0..n {
                    let r = yhat[i] - y[i];
                    let ar = r.abs();
                    let diff = r as f32;
                    if ar <= delta {
                        g.push(diff);
                        h.push(1.0);
                        l.push((0.5 * r * r) as f32);
                    } else {
                        g.push(delta32 * diff.signum());
                        h.push(0.0);
                        l.push((delta * (ar - 0.5 * delta)) as f32);
                    }
                }
            }
        }
        (g, Some(h), l)
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl AdaptiveHuberLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        // We use a fixed delta of 1.0 as a heuristic during tree-growth
        // to ensure stability and performance. The true adaptive delta
        // is used during gradient and global loss calculation at the start of each iteration.
        let delta = 1.0;
        let r = y - yhat;
        let ar = r.abs();
        let l = if ar <= delta {
            0.5 * r * r
        } else {
            delta * (ar - 0.5 * delta)
        };
        match sample_weight {
            Some(w) => (l * w) as f32,
            None => l as f32,
        }
    }
}
