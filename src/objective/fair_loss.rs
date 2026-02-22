//! Fair Loss objective for robust regression.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
/// Fair Loss objective.
/// `c * |y - yhat| - c^2 * ln(|y - yhat| / c + 1)`.
pub struct FairLoss {
    /// The threshold constant c.
    pub c: Option<f64>,
}

impl Default for FairLoss {
    fn default() -> Self {
        Self { c: Some(1.0) }
    }
}

impl ObjectiveFunction for FairLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let c = self.c.unwrap_or(1.0);
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let diff = (*yhat_ - *y_).abs();
                    (w_ * (c * diff - c * c * (diff / c + 1.0).ln())) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let diff = (*yhat_ - *y_).abs();
                    (c * diff - c * c * (diff / c + 1.0).ln()) as f32
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
        let c = self.c.unwrap_or(1.0) as f32;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let diff = yhat_val - y_val;
                    let abs_diff = diff.abs();
                    let denominator = abs_diff + c;

                    g.push(c * diff / denominator * w_val);
                    h.push(c * c / (denominator * denominator) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let diff = yhat_val - y_val;
                    let abs_diff = diff.abs();
                    let denominator = abs_diff + c;

                    g.push(c * diff / denominator);
                    h.push(c * c / (denominator * denominator));
                }
                (g, Some(h))
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl FairLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let c = self.c.unwrap_or(1.0);
        let diff = (yhat - y).abs();
        let l = c * diff - c * c * (diff / c + 1.0).ln();
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
    fn test_fair_loss_init() {
        let y = vec![1.0, 2.0, 3.0];
        let loss_fn = FairLoss::default();
        // Default initial_value is mean = 2.0
        assert_eq!(loss_fn.initial_value(&y, None, None), 2.0);
    }

    #[test]
    fn test_fair_loss() {
        let y = vec![2.0, 0.0];
        let yhat = vec![1.0, -1.0];
        let loss_fn = FairLoss::default(); // c = 1.0

        let l = loss_fn.loss(&y, &yhat, None, None);
        // diff = 1.0. Loss = 1*1 - 1*1*ln(1/1 + 1) = 1 - ln(2) approx 1 - 0.693147 = 0.306853
        assert!((l[0] - 0.30685282).abs() < 1e-6);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert!((lw[0] - 0.30685282 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fair_gradient() {
        let y = vec![2.0, 0.0];
        let yhat = vec![1.0, 1.0];
        let loss_fn = FairLoss::default(); // c = 1.0

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // diff = -1.0, abs_diff = 1.0, denom = 2.0
        // g = 1 * (-1) / 2 = -0.5
        // h = 1 * 1 / (2^2) = 0.25
        assert_eq!(g, vec![-0.5, 0.5]);
        assert_eq!(h, vec![0.25, 0.25]);

        let w = vec![2.0, 0.5];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert_eq!(gw, vec![-1.0, 0.25]);
        assert_eq!(hw.unwrap(), vec![0.5, 0.125]);
    }

    #[test]
    fn test_fair_loss_single() {
        let loss_fn = FairLoss::default();
        assert!((loss_fn.loss_single(2.0, 1.0, None) - 0.30685282).abs() < 1e-6);
        assert!((loss_fn.loss_single(2.0, 1.0, Some(2.0)) - 0.30685282 * 2.0).abs() < 1e-6);
    }
}
