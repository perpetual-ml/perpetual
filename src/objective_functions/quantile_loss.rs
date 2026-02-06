//! Quantile Loss function for quantile regression.

use crate::{metrics::evaluation::Metric, objective_functions::objective::ObjectiveFunction};
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
}
