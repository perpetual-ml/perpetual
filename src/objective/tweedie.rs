//! Tweedie Loss for Tweedie regression.
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
/// Tweedie Loss objective.
/// Minimizes negative log likelihood of Tweedie distribution.
/// Target `y` should be non-negative.
pub struct TweedieLoss {
    /// The variance power `p`. Usually between 1 and 2.
    pub p: Option<f64>,
}

impl Default for TweedieLoss {
    fn default() -> Self {
        Self { p: Some(1.5) }
    }
}

impl ObjectiveFunction for TweedieLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let p = self.p.unwrap_or(1.5);
        let term1 = 1.0 - p;
        let term2 = 2.0 - p;

        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let pt1 = -*y_ * (term1 * *yhat_).exp() / term1;
                    let pt2 = (term2 * *yhat_).exp() / term2;
                    (w_ * (pt1 + pt2)) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let pt1 = -*y_ * (term1 * *yhat_).exp() / term1;
                    let pt2 = (term2 * *yhat_).exp() / term2;
                    (pt1 + pt2) as f32
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
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);
        let p = self.p.unwrap_or(1.5) as f32;
        let term1 = 1.0 - p;
        let term2 = 2.0 - p;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let exp_term1 = (term1 * yhat_val).exp();
                    let exp_term2 = (term2 * yhat_val).exp();

                    g.push((exp_term2 - y_val * exp_term1) * w_val);
                    h.push((term2 * exp_term2 - term1 * y_val * exp_term1) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let exp_term1 = (term1 * yhat_val).exp();
                    let exp_term2 = (term2 * yhat_val).exp();

                    g.push(exp_term2 - y_val * exp_term1);
                    h.push(term2 * exp_term2 - term1 * y_val * exp_term1);
                }
                (g, Some(h))
            }
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

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl TweedieLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let p = self.p.unwrap_or(1.5);
        let term1 = 1.0 - p;
        let term2 = 2.0 - p;

        let pt1 = -y * (term1 * yhat).exp() / term1;
        let pt2 = (term2 * yhat).exp() / term2;
        let l = pt1 + pt2;

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
    fn test_tweedie_loss_init() {
        let y = vec![1.0, 2.0, 3.0];
        let loss_fn = TweedieLoss::default();
        // Mean y = 2.0. Init = ln(2.0)
        assert!((loss_fn.initial_value(&y, None, None) - 2.0_f64.ln()).abs() < 1e-6);

        // Weighted init
        let w = vec![2.0, 1.0, 1.0];
        // ytot = 2*1 + 1*2 + 1*3 = 7. ntot = 4. mean = 1.75
        assert!((loss_fn.initial_value(&y, Some(&w), None) - 1.75_f64.ln()).abs() < 1e-6);

        // Edge case: mean_y <= 0
        assert_eq!(loss_fn.initial_value(&[0.0], None, None), 0.0);
    }

    #[test]
    fn test_tweedie_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let w = vec![2.0, 0.5];
        let loss_fn = TweedieLoss { p: Some(1.5) };

        let l = loss_fn.loss(&y, &yhat, Some(&w), None);
        // Loss[0] = 4.0 * 2.0 = 8.0
        // Loss[1] = 2.0 * 0.5 = 1.0
        assert_eq!(l, vec![8.0, 1.0]);

        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        let h = h.unwrap();
        // Grad[0] = 0.0 * 2.0 = 0.0, Grad[1] = 1.0 * 0.5 = 0.5
        assert_eq!(g, vec![0.0, 0.5]);
        // Hess[0] = 1.0 * 2.0 = 2.0, Hess[1] = 0.5 * 0.5 = 0.25
        assert_eq!(h, vec![2.0, 0.25]);

        let (g2, h2, l2) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        assert_eq!(g2, g);
        assert_eq!(h2.unwrap(), h);
        assert_eq!(l2, l);
    }

    #[test]
    fn test_tweedie_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = TweedieLoss { p: Some(1.5) };

        let l = loss_fn.loss(&y, &yhat, None, None);
        // p = 1.5, t1 = -0.5, t2 = 0.5
        // pt1 = -y * exp(-0.5*0) / -0.5 = 2.0 * y
        // pt2 = exp(0.5*0) / 0.5 = 2.0
        // Loss[0] = 2.0 * 1.0 + 2.0 = 4.0
        // Loss[1] = 2.0 * 0.0 + 2.0 = 2.0
        assert_eq!(l, vec![4.0, 2.0]);
    }

    #[test]
    fn test_tweedie_gradient() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = TweedieLoss { p: Some(1.5) };

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // Grad = exp(0.5*yhat) - y*exp(-0.5*yhat) = 1 - y
        assert_eq!(g, vec![0.0, 1.0]);
        // Hess = 0.5*exp(0.5*yhat) - (-0.5)*y*exp(-0.5*yhat) = 0.5 + 0.5*y
        assert_eq!(h, vec![1.0, 0.5]);
    }

    #[test]
    fn test_tweedie_gradient_and_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = TweedieLoss { p: Some(1.5) };
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g, vec![0.0, 1.0]);
        assert_eq!(h.unwrap(), vec![1.0, 0.5]);
        assert_eq!(l, vec![4.0, 2.0]);
    }

    #[test]
    fn test_tweedie_gradient_and_loss_into() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = TweedieLoss { p: Some(1.5) };
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![0.0, 1.0]);
        assert_eq!(hess.unwrap(), vec![1.0, 0.5]);
        assert_eq!(loss, vec![4.0, 2.0]);
    }

    #[test]
    fn test_tweedie_loss_single() {
        let loss_fn = TweedieLoss { p: Some(1.5) };
        assert_eq!(loss_fn.loss_single(1.0, 0.0, None), 4.0);
        assert_eq!(loss_fn.loss_single(1.0, 0.0, Some(2.0)), 8.0);
    }
}
