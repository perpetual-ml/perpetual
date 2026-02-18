//! Log Loss (negative log-likelihood) for binary classification.
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Log Loss (binary cross-entropy) objective.
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let p = 1.0_f64 / (1.0_f64 + (-*yhat_).exp());
                    (-(*y_ * p.ln() + (1.0_f64 - *y_) * ((1.0_f64 - p).ln())) * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let p = 1.0_f64 / (1.0_f64 + (-*yhat_).exp());
                    (-(*y_ * p.ln() + (1.0_f64 - *y_) * ((1.0_f64 - p).ln()))) as f32
                })
                .collect(),
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        match sample_weight {
            Some(sample_weight) => {
                let mut ytot: f64 = 0.;
                let mut ntot: f64 = 0.;
                for i in 0..y.len() {
                    ytot += sample_weight[i] * y[i];
                    ntot += sample_weight[i];
                }
                f64::ln(ytot / (ntot - ytot))
            }
            None => {
                let ytot = fast_sum(y);
                let ntot = y.len() as f64;
                f64::ln(ytot / (ntot - ytot))
            }
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
        let mut h = Vec::with_capacity(len);

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    // Sigmoid in f32
                    let p = 1.0 / (1.0 + (-yhat_val).exp());

                    g.push((p - y_val) * w_val);
                    h.push(p * (1.0 - p) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    // Sigmoid in f32
                    let p = 1.0 / (1.0 + (-yhat_val).exp());

                    g.push(p - y_val);
                    h.push(p * (1.0 - p));
                }
                (g, Some(h))
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
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
        let mut h = Vec::with_capacity(len);
        let mut l = Vec::with_capacity(len);

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    g.push((p - y_val) * w_val);
                    h.push(p * (1.0 - p) * w_val);
                    // Loss uses f64 for precision
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push((-(y[i] * p64.ln() + (1.0 - y[i]) * (1.0 - p64).ln()) * w[i]) as f32);
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    g.push(p - y_val);
                    h.push(p * (1.0 - p));
                    // Loss uses f64 for precision
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push((-(y[i] * p64.ln() + (1.0 - y[i]) * (1.0 - p64).ln())) as f32);
                }
            }
        }
        (g, Some(h), l)
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
        let h = hess.get_or_insert_with(|| vec![0.0; len]);
        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    grad[i] = (p - y_val) * w_val;
                    h[i] = p * (1.0 - p) * w_val;
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = (-(y[i] * p64.ln() + (1.0 - y[i]) * (1.0 - p64).ln()) * w[i]) as f32;
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    grad[i] = p - y_val;
                    h[i] = p * (1.0 - p);
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = (-(y[i] * p64.ln() + (1.0 - y[i]) * (1.0 - p64).ln())) as f32;
                }
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl LogLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let p = 1.0_f64 / (1.0_f64 + (-yhat).exp());
        let l = -(y * p.ln() + (1.0_f64 - y) * (1.0_f64 - p).ln());
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
    fn test_log_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![1.0, -1.0]; // p approx 0.73, 0.27
        let loss_fn = LogLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        assert_eq!(l.len(), 2);
        // p = 1/(1+e^-1) = 0.73105857863
        // loss = -(1*ln(0.731) + 0*ln(1-0.731)) = 0.31326168751
        assert!((l[0] - 0.3132617).abs() < 1e-6);

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // g = p - y = 0.731 - 1.0 = -0.2689414
        // h = p * (1 - p) = 0.731 * 0.269 = 0.1966119
        assert!((g[0] - (-0.2689414)).abs() < 1e-6);
        assert!((h[0] - 0.1966119).abs() < 1e-6);
    }

    #[test]
    fn test_log_loss_init() {
        let y = vec![1.0, 1.0, 0.0]; // 2 ones, 1 zero
        let loss_fn = LogLoss::default();
        // log(2/1) = ln(2) = 0.693147
        assert!((loss_fn.initial_value(&y, None, None) - 0.693147).abs() < 1e-6);
    }
}
