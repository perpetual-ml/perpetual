//! Huber Loss function for robust regression.

use crate::{objective_functions::objective::ObjectiveFunction, utils::weighted_median};
use serde::{Deserialize, Serialize};

/// Huber Loss — quadratic near zero, linear beyond `delta`.
#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct HuberLoss {
    pub delta: Option<f64>,
}

impl ObjectiveFunction for HuberLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let delta = self.delta.unwrap_or(1.0);
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
        let delta = self.delta.unwrap_or(1.0) as f32;
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);

        match sample_weight {
            Some(weights) => {
                for i in 0..len {
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
                for i in 0..len {
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

    #[inline]
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
        let delta = self.delta.unwrap_or(1.0);
        let delta32 = delta as f32;
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);
        let mut l = Vec::with_capacity(len);

        match sample_weight {
            Some(weights) => {
                for i in 0..len {
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
                for i in 0..len {
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
}
