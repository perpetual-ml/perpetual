//! Quantile Loss function for quantile regression.

use crate::{metrics::evaluation::Metric, objective::ObjectiveFunction};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Quantile Loss (pinball loss) — targets a specific quantile of the conditional distribution.
pub struct QuantileLoss {
    /// Target quantile in `(0, 1)`. For example, `0.5` for the median.
    pub quantile: Option<f64>,
}

impl ObjectiveFunction for QuantileLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let q = self.quantile.unwrap();
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let s = *y_ - *yhat_;
                    let l = if s >= 0.0 { q * s } else { (q - 1.0) * s };
                    (l * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let s = *y_ - *yhat_;
                    let l = if s >= 0.0 { q * s } else { (q - 1.0) * s };
                    l as f32
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
        let quantile = self.quantile.unwrap() as f32;
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);

        match sample_weight {
            Some(weights) => {
                for i in 0..len {
                    let diff = (yhat[i] - y[i]) as f32;
                    let w = weights[i] as f32;

                    if diff >= 0.0 {
                        g.push((1.0 - quantile) * w);
                    } else {
                        g.push(-quantile * w);
                    }
                    h.push(w);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let diff = (yhat[i] - y[i]) as f32;

                    if diff >= 0.0 {
                        g.push(1.0 - quantile);
                    } else {
                        g.push(-quantile);
                    }
                }
                (g, None)
            }
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        match sample_weight {
            Some(sample_weight) => {
                let mut indices = (0..y.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
                let w_tot: f64 = sample_weight.iter().sum();
                let w_target = w_tot * self.quantile.unwrap();
                let mut w_cum = 0.0_f64;
                let mut init_value = f64::NAN;
                for i in indices {
                    w_cum += sample_weight[i];
                    if w_cum >= w_target {
                        init_value = y[i];
                        break;
                    }
                }
                init_value
            }
            None => {
                let mut indices = (0..y.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
                let w_tot: f64 = y.len() as f64;
                let w_target = w_tot * self.quantile.unwrap();
                let mut w_cum = 0.0_f64;
                let mut init_value = f64::NAN;
                for i in indices {
                    w_cum += 1.0;
                    if w_cum >= w_target {
                        init_value = y[i];
                        break;
                    }
                }
                init_value
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::QuantileLoss
    }

    fn gradient_and_loss(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        let q = self.quantile.unwrap();
        let q32 = q as f32;
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut l = Vec::with_capacity(len);

        match sample_weight {
            Some(weights) => {
                let mut h = Vec::with_capacity(len);
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    let w = weights[i] as f32;
                    if diff >= 0.0 {
                        g.push((1.0 - q32) * w);
                        l.push(((1.0 - q) * diff * weights[i]) as f32);
                    } else {
                        g.push(-q32 * w);
                        l.push((-q * diff * weights[i]) as f32);
                    }
                    h.push(w);
                }
                (g, Some(h), l)
            }
            None => {
                for i in 0..len {
                    let diff = yhat[i] - y[i];
                    if diff >= 0.0 {
                        g.push(1.0 - q32);
                        l.push(((1.0 - q) * diff) as f32);
                    } else {
                        g.push(-q32);
                        l.push((-q * diff) as f32);
                    }
                }
                (g, None, l)
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl QuantileLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let q = self.quantile.unwrap();
        let s = y - yhat;
        let l = if s >= 0.0 { q * s } else { (q - 1.0) * s };
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
    fn test_quantile_loss() {
        let y = vec![1.0, 1.0];
        let yhat = vec![2.0, 0.5]; // diffs (yhat - y): 1.0, -0.5. s (y - yhat): -1.0, 0.5
        let loss_fn = QuantileLoss { quantile: Some(0.7) };

        // Test loss: if s >= 0: q*s else (q-1)*s
        // s = -1.0 < 0 -> (0.7-1)*(-1.0) = -0.3 * -1.0 = 0.3
        // s = 0.5 >= 0 -> 0.7 * 0.5 = 0.35
        let l = loss_fn.loss(&y, &yhat, None, None);
        assert!((l[0] - 0.3).abs() < 1e-6);
        assert!((l[1] - 0.35).abs() < 1e-6);

        // Test gradient: diff >= 0 (yhat >= y) -> (1-q), diff < 0 (yhat < y) -> -q
        // diff = 1.0 >= 0 -> 1.0 - 0.7 = 0.3
        // diff = -0.5 < 0 -> -0.7
        let (g, _) = loss_fn.gradient(&y, &yhat, None, None);
        assert!((g[0] - 0.3).abs() < 1e-6);
        assert!((g[1] - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_initial_value() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let loss_fn = QuantileLoss { quantile: Some(0.5) };
        // median of [1,2,3,4,5] is 3
        assert_eq!(loss_fn.initial_value(&y, None, None), 3.0);

        let loss_fn_q = QuantileLoss { quantile: Some(0.8) };
        // 0.8 quantile of 5 elements is the 4th or 5th. w_target = 5 * 0.8 = 4.0.
        // indices 0..5, w_cum reaching 4.0 at index 3 (value 4.0) or index 4?
        // Let's trace: 1(1.0), 2(2.0), 3(3.0), 4(4.0) -> w_cum=4.0 >= 4.0. So 4.0.
        assert_eq!(loss_fn_q.initial_value(&y, None, None), 4.0);
    }
}
