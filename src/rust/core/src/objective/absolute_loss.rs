//! Absolute Loss for L1 regression.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Absolute Error Loss objective.
/// `|y - yhat|`.
pub struct AbsoluteLoss {}

impl ObjectiveFunction for AbsoluteLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| (w_ * (*yhat_ - *y_).abs()) as f32)
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| (*yhat_ - *y_).abs() as f32)
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

        // Pseudo-Hessian trick: L1 loss has 0 hessian everywhere except non-differentiable at 0.
        // GBMs often use a small constant for hessian to allow trees to build, or a pseudo-huber approach.
        // XGBoost and LightGBM handle L1 by setting hessian to essentially 1.0 or weights.
        // Or leaf refreshing after tree built (for XGBoost). We'll set H to a small constant.
        let h_val = 1.0_f32;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let diff = yhat_val - y_val;
                    let sign = if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };

                    g.push(sign * w_val);
                    h.push(h_val * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let diff = yhat_val - y_val;
                    let sign = if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };

                    g.push(sign);
                    h.push(h_val);
                }
                (g, Some(h))
            }
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        // Initial value for L1 is the median, but for broad compatibility we construct a weighted median
        let mut cloned1: Vec<(f64, f64)> = match sample_weight {
            Some(w) => y.iter().zip(w.iter()).map(|(yi, wi)| (*yi, *wi)).collect(),
            None => y.iter().map(|yi| (*yi, 1.0)).collect(),
        };

        cloned1.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let total_w: f64 = cloned1.iter().map(|(_, w)| w).sum();
        let half_w = total_w / 2.0;

        let mut cum_w = 0.0;
        for (yi, wi) in cloned1.iter() {
            cum_w += wi;
            if cum_w >= half_w {
                return *yi;
            }
        }

        y[0]
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl AbsoluteLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let l = (yhat - y).abs();
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
    fn test_abs_loss_init() {
        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let loss_fn = AbsoluteLoss::default();
        // Median of [1, 2, 3.2, 4, 9] is 3.2
        assert_eq!(loss_fn.initial_value(&y, None, None), 3.2);
    }

    #[test]
    fn test_abs_loss_init_weighted() {
        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let w = vec![0.0, 0.5, 1.0, 0.3, 0.5];
        // Sorted: (1.0, 0.0), (2.0, 0.5), (3.2, 0.3), (4.0, 0.5), (9.0, 1.0)
        // Total weight = 2.3. Half = 1.15.
        // Cum weights: 0.0, 0.5, 0.8, 1.3 -> 4.0 is the weighted median
        let loss_fn = AbsoluteLoss::default();
        assert_eq!(loss_fn.initial_value(&y, Some(&w), None), 4.0);
    }

    #[test]
    fn test_abs_loss() {
        let y = vec![2.0, 0.0];
        let yhat = vec![1.0, -1.0];
        let loss_fn = AbsoluteLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // |1 - 2| = 1, |-1 - 0| = 1
        assert_eq!(l, vec![1.0, 1.0]);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(lw, vec![2.0, 0.5]);
    }

    #[test]
    fn test_abs_gradient() {
        let y = vec![2.0, 0.0, 1.0];
        let yhat = vec![1.0, 1.0, 1.0];
        let loss_fn = AbsoluteLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // yhat - y: -1, 1, 0
        // sign: -1, 1, 0
        assert_eq!(g, vec![-1.0, 1.0, 0.0]);
        assert_eq!(h, vec![1.0, 1.0, 1.0]);

        let w = vec![2.0, 0.5, 1.0];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        let hw = hw.unwrap();
        assert_eq!(gw, vec![-2.0, 0.5, 0.0]);
        assert_eq!(hw, vec![2.0, 0.5, 1.0]);
    }

    #[test]
    fn test_abs_loss_single() {
        let loss_fn = AbsoluteLoss::default();
        assert_eq!(loss_fn.loss_single(2.0, 1.0, None), 1.0);
        assert_eq!(loss_fn.loss_single(2.0, 1.0, Some(2.0)), 2.0);
    }

    #[test]
    fn test_abs_default_metric() {
        let loss_fn = AbsoluteLoss::default();
        assert!(matches!(loss_fn.default_metric(), Metric::RootMeanSquaredError));
    }
}
