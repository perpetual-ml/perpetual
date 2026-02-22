//! Hinge Loss objective for binary classification.
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Hinge Loss objective.
/// `max(0, 1 - y_binary * yhat)`, where y_binary maps {0, 1} targets to {-1, 1}.
pub struct HingeLoss {}

impl ObjectiveFunction for HingeLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let y_bin = if *y_ > 0.0 { 1.0 } else { -1.0 };
                    let diff = 1.0 - y_bin * yhat_;
                    let loss = if diff > 0.0 { diff } else { 0.0 };
                    (loss * w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let y_bin = if *y_ > 0.0 { 1.0 } else { -1.0 };
                    let diff = 1.0 - y_bin * yhat_;
                    let loss = if diff > 0.0 { diff } else { 0.0 };
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
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);

        // Hinge loss is piece-wise linear:
        // gradient is -y_bin if y_bin * yhat < 1, else 0.
        // hessian is 0 everywhere except undefined at 1. We use a smoothed approximation or a tiny constant.
        let h_val = 1e-6_f32;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let y_bin = if y_val > 0.0 { 1.0 } else { -1.0 };
                    let grad = if y_bin * yhat_val < 1.0 { -y_bin } else { 0.0 };

                    g.push(grad * w_val);
                    h.push(h_val * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let y_bin = if y_val > 0.0 { 1.0 } else { -1.0 };
                    let grad = if y_bin * yhat_val < 1.0 { -y_bin } else { 0.0 };

                    g.push(grad);
                    h.push(h_val);
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
                    ytot += w[i] * if y[i] > 0.0 { 1.0 } else { 0.0 };
                    ntot += w[i];
                }
                ytot / ntot
            }
            None => {
                let ytot = fast_sum(
                    &y.iter()
                        .map(|&yi| if yi > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f64>>(),
                );
                let ntot = y.len() as f64;
                ytot / ntot
            }
        };
        if mean_y > 0.5 {
            1.0 // Favor class 1
        } else if mean_y < 0.5 {
            -1.0 // Favor class -1
        } else {
            0.0
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl HingeLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let y_bin = if y > 0.0 { 1.0 } else { -1.0 };
        let diff = 1.0 - y_bin * yhat;
        let l = if diff > 0.0 { diff } else { 0.0 };
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
    fn test_hinge_loss_init() {
        let y = vec![1.0, 1.0, 0.0]; // Mean y = 2/3 > 0.5
        let loss_fn = HingeLoss::default();
        assert_eq!(loss_fn.initial_value(&y, None, None), 1.0);

        let y2 = vec![1.0, 0.0, 0.0]; // Mean y = 1/3 < 0.5
        assert_eq!(loss_fn.initial_value(&y2, None, None), -1.0);
    }

    #[test]
    fn test_hinge_loss() {
        let y = vec![1.0, 0.0]; // binary: 1, -1
        let yhat = vec![0.5, 0.5];
        let loss_fn = HingeLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // Loss[0]: max(0, 1 - 1*0.5) = 0.5
        // Loss[1]: max(0, 1 - (-1)*0.5) = 1.5
        assert_eq!(l, vec![0.5, 1.5]);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(lw, vec![1.0, 0.75]);
    }

    #[test]
    fn test_hinge_gradient() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.5, 2.0];
        let loss_fn = HingeLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        // grad[0]: y_bin*yhat = 0.5 < 1 -> -y_bin = -1.0
        // grad[1]: y_bin*yhat = -1 * 2 = -2 < 1 -> -y_bin = 1.0
        assert_eq!(g, vec![-1.0, 1.0]);
        assert_eq!(h.unwrap(), vec![1e-6, 1e-6]);

        // Weighted
        let w = vec![2.0, 0.5];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert_eq!(gw, vec![-2.0, 0.5]);
        assert_eq!(hw.unwrap(), vec![2e-6, 0.5e-6]);

        // Initial value weighted
        assert_eq!(loss_fn.initial_value(&vec![1.0, 0.0], Some(&vec![3.0, 1.0]), None), 1.0);
        assert_eq!(
            loss_fn.initial_value(&vec![1.0, 0.0], Some(&vec![1.0, 3.0]), None),
            -1.0
        );
        assert_eq!(loss_fn.initial_value(&vec![1.0, 0.0], Some(&vec![1.0, 1.0]), None), 0.0);
    }

    #[test]
    fn test_hinge_gradient_and_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.5, 2.0];
        let loss_fn = HingeLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g, vec![-1.0, 1.0]);
        assert_eq!(h.unwrap(), vec![1e-6, 1e-6]);
        assert_eq!(l, vec![0.5, 3.0]);
    }

    #[test]
    fn test_hinge_gradient_and_loss_into() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.5, 2.0];
        let loss_fn = HingeLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![-1.0, 1.0]);
        assert_eq!(hess.unwrap(), vec![1e-6, 1e-6]);
        assert_eq!(loss, vec![0.5, 3.0]);
    }

    #[test]
    fn test_hinge_loss_single() {
        let loss_fn = HingeLoss::default();
        assert_eq!(loss_fn.loss_single(1.0, 0.5, None), 0.5);
        assert_eq!(loss_fn.loss_single(1.0, 0.5, Some(2.0)), 1.0);
    }
}
