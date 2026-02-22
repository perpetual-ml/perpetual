//! Gamma Loss for Gamma regression.
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Gamma Loss objective.
/// Minimizes `y * exp(-yhat) + yhat`.
/// Target `y` should be strictly positive.
pub struct GammaLoss {}

impl ObjectiveFunction for GammaLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| (w_ * (y_ * (-*yhat_).exp() + *yhat_)) as f32)
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| (y_ * (-*yhat_).exp() + *yhat_) as f32)
                .collect(),
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        let mean_y = match sample_weight {
            Some(w) => {
                let mut ytot: f64 = 0.;
                let mut ntot: f64 = 0.;
                for i in 0..y.len() {
                    ytot += w[i] * y[i];
                    ntot += w[i];
                }
                ytot / ntot
            }
            None => {
                let ytot = fast_sum(y);
                let ntot = y.len() as f64;
                ytot / ntot
            }
        };
        if mean_y <= 0.0 { 0.0 } else { mean_y.ln() }
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

                    let exp_neg_yhat = (-yhat_val).exp();
                    g.push((1.0 - y_val * exp_neg_yhat) * w_val);
                    h.push(y_val * exp_neg_yhat * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let exp_neg_yhat = (-yhat_val).exp();
                    g.push(1.0 - y_val * exp_neg_yhat);
                    h.push(y_val * exp_neg_yhat);
                }
                (g, Some(h))
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
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
                    let exp_neg_yhat = (-yhat_val).exp();

                    g.push((1.0 - y_val * exp_neg_yhat) * w_val);
                    h.push(y_val * exp_neg_yhat * w_val);

                    l.push((w[i] * (y[i] * (-yhat[i]).exp() + yhat[i])) as f32);
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let exp_neg_yhat = (-yhat_val).exp();

                    g.push(1.0 - y_val * exp_neg_yhat);
                    h.push(y_val * exp_neg_yhat);

                    l.push((y[i] * (-yhat[i]).exp() + yhat[i]) as f32);
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
                    let exp_neg_yhat = (-yhat_val).exp();

                    grad[i] = (1.0 - y_val * exp_neg_yhat) * w_val;
                    h[i] = y_val * exp_neg_yhat * w_val;
                    loss[i] = (w[i] * (y[i] * (-yhat[i]).exp() + yhat[i])) as f32;
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let exp_neg_yhat = (-yhat_val).exp();

                    grad[i] = 1.0 - y_val * exp_neg_yhat;
                    h[i] = y_val * exp_neg_yhat;
                    loss[i] = (y[i] * (-yhat[i]).exp() + yhat[i]) as f32;
                }
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl GammaLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let l = y * (-yhat).exp() + yhat;
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
    fn test_gamma_loss_init() {
        let y = vec![1.0, 2.0, 3.0];
        let loss_fn = GammaLoss::default();
        // Mean y = 2.0. ln(2.0) approx 0.693147
        assert!((loss_fn.initial_value(&y, None, None) - 2.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_loss() {
        let y = vec![2.0, 1.0];
        let yhat = vec![0.0, 0.0]; // p = exp(0) = 1.0
        let loss_fn = GammaLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // Loss = y * exp(-yhat) + yhat
        // Loss[0] = 2 * exp(0) + 0 = 2.0
        // Loss[1] = 1 * exp(0) + 0 = 1.0
        assert_eq!(l, vec![2.0, 1.0]);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(lw, vec![4.0, 0.5]);
    }

    #[test]
    fn test_gamma_gradient() {
        let y = vec![2.0, 1.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = GammaLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // Grad = 1 - y * exp(-yhat) = 1 - y * 1 = 1 - y
        // Grad[0] = 1 - 2 = -1
        // Grad[1] = 1 - 1 = 0
        assert_eq!(g, vec![-1.0, 0.0]);
        // Hess = y * exp(-yhat) = y
        assert_eq!(h, vec![2.0, 1.0]);

        // Weighted
        let w = vec![2.0, 0.5];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert_eq!(gw, vec![-2.0, 0.0]);
        assert_eq!(hw.unwrap(), vec![4.0, 0.5]);
    }

    #[test]
    fn test_gamma_gradient_and_loss() {
        let y = vec![2.0, 1.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = GammaLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g, vec![-1.0, 0.0]);
        assert_eq!(h.unwrap(), vec![2.0, 1.0]);
        assert_eq!(l, vec![2.0, 1.0]);

        let w = vec![2.0, 0.5];
        let (gw, hw, lw) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        assert_eq!(gw, vec![-2.0, 0.0]);
        assert_eq!(hw.unwrap(), vec![4.0, 0.5]);
        assert_eq!(lw, vec![4.0, 0.5]);
    }

    #[test]
    fn test_gamma_gradient_and_loss_into() {
        let y = vec![2.0, 1.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = GammaLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![-1.0, 0.0]);
        assert_eq!(hess.unwrap(), vec![2.0, 1.0]);
        assert_eq!(loss, vec![2.0, 1.0]);

        let w = vec![2.0, 0.5];
        let mut gradw = vec![0.0; 2];
        let mut hessw = Some(vec![0.0; 2]);
        let mut lossw = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, Some(&w), None, &mut gradw, &mut hessw, &mut lossw);
        assert_eq!(gradw, vec![-2.0, 0.0]);
        assert_eq!(hessw.unwrap(), vec![4.0, 0.5]);
        assert_eq!(lossw, vec![4.0, 0.5]);
    }

    #[test]
    fn test_gamma_init_weighted() {
        let y = vec![1.0, 3.0];
        let w = vec![1.0, 3.0];
        let loss_fn = GammaLoss::default();
        // Weighted mean = (1*1 + 3*3) / 4 = 10 / 4 = 2.5
        assert!((loss_fn.initial_value(&y, Some(&w), None) - 2.5_f64.ln()).abs() < 1e-6);

        let y_zero = vec![0.0, 0.0];
        assert_eq!(loss_fn.initial_value(&y_zero, None, None), 0.0);
    }
}
