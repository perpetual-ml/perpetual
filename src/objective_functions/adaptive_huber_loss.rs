//! Adaptive Huber Loss function
//!
//!

use crate::{metrics::evaluation::Metric, objective_functions::objective::ObjectiveFunction};
use serde::{Deserialize, Serialize};

/// Adaptive Huber Loss
#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct AdaptiveHuberLoss {
    pub quantile: Option<f64>,
}

impl ObjectiveFunction for AdaptiveHuberLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // default alpha: 0.5
        // if not passed explicitly
        let alpha = self.quantile.unwrap_or(0.5);
        let n = y.len();

        let mut abs_res = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect::<Vec<_>>();
        abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((n as f64) * alpha).floor() as usize;
        let delta = abs_res[idx.min(n - 1)];

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
        // default alpha: 0.5
        // if not passed explicitly
        let alpha = self.quantile.unwrap_or(0.5);
        let n = y.len();

        let mut abs_res = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect::<Vec<_>>();
        abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((n as f64) * alpha).floor() as usize;
        let delta = abs_res[idx.min(n - 1)] as f32;

        // --- Vectorized Loop ---
        let mut g = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);

        match sample_weight {
            Some(weights) => {
                for i in 0..n {
                    let diff = (yhat[i] - y[i]) as f32;
                    let abs_diff = diff.abs();
                    let w = weights[i] as f32;

                    if abs_diff <= delta {
                        g.push(diff * w);
                        h.push(w);
                    } else {
                        let sign = diff.signum();
                        g.push(delta * sign * w);
                        h.push(0.0);
                    }
                }
                (g, Some(h))
            }
            None => {
                for i in 0..n {
                    let diff = (yhat[i] - y[i]) as f32;
                    let abs_diff = diff.abs();

                    if abs_diff <= delta {
                        g.push(diff);
                        h.push(1.0);
                    } else {
                        let sign = diff.signum();
                        g.push(delta * sign);
                        h.push(0.0);
                    }
                }
                (g, Some(h))
            }
        }
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        let mut idxs = (0..y.len()).collect::<Vec<_>>();
        idxs.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap());

        let total_w = sample_weight.map(|w| w.iter().sum::<f64>()).unwrap_or(y.len() as f64);
        let target = total_w * 0.5;

        // find weighted median via scan()
        let median = idxs
            .iter()
            .scan(0.0, |cum, &i| {
                *cum += sample_weight.map_or(1.0, |w| w[i]);
                Some((i, *cum))
            })
            .find(|&(_i, cum)| cum >= target)
            .map(|(i, _)| y[i])
            .unwrap_or(y[idxs[y.len() / 2]]);

        median
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }
}
