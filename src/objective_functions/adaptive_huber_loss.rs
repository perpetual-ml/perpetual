use super::ObjectiveFunction;

use crate::{data::FloatData, metrics::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

/// Adaptive Huber Loss
/// 
/// 
#[derive(Default)]
pub struct AdaptiveHuberLoss {}
impl ObjectiveFunction for AdaptiveHuberLoss {

    // calculate loss
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>, ) -> Vec<f32> {

        // default alpha: 0.5
        // if not passed explicitly
        let alpha = quantile.unwrap_or(0.5);
        let n = y.len();

        // absolute residuals
        let mut abs_res: Vec<f64> = y.iter()
            .zip(yhat)
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect();

        // calculate delta
        abs_res.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let idx = (n as f64 * alpha).floor() as usize;
        let idx = idx.min(n - 1);
        let delta = abs_res[idx];
        
        // sample-wise loss:
        let mut out = Vec::with_capacity(n);
        match sample_weight {
            Some(w) => {
                for i in 0..n {
                    let r = y[i] - yhat[i];
                    let ar = r.abs();
                    let base = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    out.push((base * w[i]) as f32);
                }
            }
            None => {
                for i in 0..n {
                    let r = y[i] - yhat[i];
                    let ar = r.abs();
                    let loss = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    out.push(loss as f32);
                }
            }
        }

        return out;
    }

    // calculate gradient and hessians
    fn calc_grad_hess(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>,) -> (Vec<f32>, Option<Vec<f32>>) {

        // default alpha: 0.5
        // if not passed explicitly
        let alpha = quantile.unwrap_or(0.5);
        let n = y.len();

        // absolute residuals
        let mut abs_res: Vec<f64> = y.iter()
            .zip(yhat)
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect();

        abs_res.sort_by(|a,b| a.partial_cmp(b).unwrap());

        // extract delta
        let idx = (n as f64 * alpha).floor() as usize;
        let idx = idx.min(n - 1);
        let delta = abs_res[idx];

        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);
        match sample_weight {
            Some(w) => {
                for i in 0..n {
                    let r = y[i] - yhat[i];
                    let ar = r.abs();
                    let g = if ar <= delta {
                        (yhat[i] - y[i]) * w[i]
                    } else {
                        delta * (yhat[i] - y[i]).signum() * w[i]
                    };
                    let h = if ar <= delta { w[i] } else { 0.0 };
                    grad.push(g as f32);
                    hess.push(h as f32);
                }
                (grad, Some(hess))
            }
            None => {
                for i in 0..n {
                    let r = y[i] - yhat[i];
                    let ar = r.abs();
                    let g = if ar <= delta {
                        (yhat[i] - y[i])
                    } else {
                        delta * (yhat[i] - y[i]).signum()
                    };
                    let h = if ar <= delta { 1.0 } else { 0.0 };
                    grad.push(g as f32);
                    hess.push(h as f32);
                }
                (grad, Some(hess))
            }
        }
    }

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>,) -> f64 {

        let mut idxs: Vec<usize> = (0..y.len()).collect();
        idxs.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap());
        let total_w = sample_weight
            .map(|w| w.iter().sum::<f64>())
            .unwrap_or(y.len() as f64);
        let target = total_w * 0.5;
        
        let mut cum = 0.0;
        for &i in &idxs {
            cum += sample_weight.map_or(1.0, |w| w[i]);
            if cum >= target {
                return y[i];
            }
        }

        y[idxs[y.len()/2]]

    }

    fn default_metric() -> Metric {
        Metric::RootMeanSquaredError
    }
}