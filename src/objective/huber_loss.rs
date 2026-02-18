//! Huber Loss function for robust regression.

use crate::{objective::ObjectiveFunction, utils::weighted_median};
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

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl HuberLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let delta = self.delta.unwrap_or(1.0);
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_loss() {
        let y = vec![1.0, 2.0];
        let yhat = vec![1.1, 3.0]; // diffs: 0.1, 1.0. Delta default is 1.0.
        let loss_fn = HuberLoss { delta: Some(1.0) };

        // Test loss
        let l = loss_fn.loss(&y, &yhat, None, None);
        assert_eq!(l.len(), 2);
        assert!((l[0] - 0.5 * 0.1 * 0.1).abs() < 1e-6); // quadratic
        assert!((l[1] - 0.5).abs() < 1e-6); // linear boundary (1.0 * (1.0 - 0.5))

        // Test gradient/hessian
        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // g = yhat - y for |diff| <= delta
        assert!((g[0] - 0.1).abs() < 1e-6);
        assert!((h[0] - 1.0).abs() < 1e-6);
        // g = delta * sign(yhat - y) for |diff| > delta
        // at diff=1.0, sign is 1.0, g=1.0*1.0=1.0
        assert!((g[1] - 1.0).abs() < 1e-6);
        assert!((h[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_weighted() {
        let y = vec![1.0];
        let yhat = vec![2.0]; // diff = 1.0
        let weights = vec![0.5];
        let loss_fn = HuberLoss { delta: Some(0.5) }; // delta < diff, so linear

        let l = loss_fn.loss(&y, &yhat, Some(&weights), None);
        // loss = delta * (diff - 0.5*delta) * weight = 0.5 * (1.0 - 0.25) * 0.5 = 0.5 * 0.75 * 0.5 = 0.1875
        assert!((l[0] - 0.1875).abs() < 1e-6);

        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&weights), None);
        let h = h.unwrap();
        // g = delta * sign(diff) * weight = 0.5 * 1.0 * 0.5 = 0.25
        assert!((g[0] - 0.25).abs() < 1e-6);
        assert!((h[0] - 0.0).abs() < 1e-6);
    }
}
