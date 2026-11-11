//! Adaptive Huber Loss function — automatically adjusts `delta` from the
//! residual distribution.

use crate::{objective::ObjectiveFunction, utils::weighted_median};
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_huber_loss() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.1, 2.2, 3.3, 4.4]; // diffs: 0.1, 0.2, 0.3, 0.4
        // n=4, alpha=0.5 -> index = floor(4 * 0.5) = 2. abs_res[2] = 0.3
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let delta = adaptive_delta(&y, &yhat, 0.5);
        assert!((delta - 0.3).abs() < 1e-6);

        let l = loss_fn.loss(&y, &yhat, None, None);
        assert_eq!(l.len(), 4);
        // diff 0.1 <= 0.3 -> quadratic: 0.5 * 0.1 * 0.1 = 0.005
        assert!((l[0] - 0.005).abs() < 1e-6);
        // diff 0.4 > 0.3 -> linear: 0.3 * (0.4 - 0.15) = 0.3 * 0.25 = 0.075
        assert!((l[3] - 0.075).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_delta() {
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let yhat = vec![0.1, 1.2, 2.3, 3.4, 4.5]; // abs_res: 0.1, 0.2, 0.3, 0.4, 0.5
        // n=5, alpha=0.5 -> index = floor(5 * 0.5) = 2. abs_res[2] = 0.3
        assert!((adaptive_delta(&y, &yhat, 0.5) - 0.3).abs() < 1e-6);
        // alpha=0.9 -> index = floor(5 * 0.9) = 4. abs_res[4] = 0.5
        assert!((adaptive_delta(&y, &yhat, 0.9) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_huber_loss_weighted() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.1, 2.2, 3.3, 4.4];
        let w = vec![2.0, 1.0, 1.0, 2.0];
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let l = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(l.len(), 4);
        // diff 0.1 <= delta=0.3 -> quadratic: 0.5*0.1*0.1*2 = 0.01
        assert!((l[0] - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_huber_gradient_weighted() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.1, 2.2, 3.3, 4.4];
        let w = vec![2.0, 1.0, 1.0, 2.0];
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        let h = h.unwrap();
        assert_eq!(g.len(), 4);
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn test_adaptive_huber_initial_value_weighted() {
        let y = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 3.0, 1.0];
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let init = loss_fn.initial_value(&y, Some(&w), None);
        assert!(init.is_finite());
    }

    #[test]
    fn test_adaptive_huber_gradient_and_loss() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.1, 2.2, 3.3, 4.4];
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g.len(), 4);
        assert!(h.is_some());
        assert_eq!(l.len(), 4);
    }

    #[test]
    fn test_adaptive_huber_gradient_and_loss_weighted() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.1, 2.2, 3.3, 4.4];
        let w = vec![2.0, 1.0, 1.0, 2.0];
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        assert_eq!(g.len(), 4);
        assert!(h.is_some());
        assert_eq!(l.len(), 4);
    }

    #[test]
    fn test_adaptive_huber_loss_single() {
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        // delta=1.0 (hardcoded in loss_single)
        let l1 = loss_fn.loss_single(1.0, 1.5, None); // r=-0.5, ar=0.5 <= 1 -> 0.5*0.25=0.125
        assert!((l1 - 0.125).abs() < 1e-5);
        let l2 = loss_fn.loss_single(1.0, 3.0, None); // r=-2, ar=2 > 1 -> 1*(2-0.5)=1.5
        assert!((l2 - 1.5).abs() < 1e-5);
        let l3 = loss_fn.loss_single(1.0, 1.5, Some(2.0));
        assert!((l3 - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_huber_requires_batch() {
        let loss_fn = AdaptiveHuberLoss { quantile: Some(0.5) };
        assert!(!loss_fn.requires_batch_evaluation());
    }
}
