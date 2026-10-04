//! Squared Log Loss objective for regression.
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Squared Log Loss objective.
/// Minimizes `0.5 * (ln(y + 1) - ln(yhat + 1))^2`.
/// Assumes `y > -1`.
pub struct SquaredLogLoss {}

impl ObjectiveFunction for SquaredLogLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        let epsilon = 1e-6; // to avoid log(-0.0) issues slightly below -1
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let y_log = (y_.max(-1.0 + epsilon) + 1.0).ln();
                    let yhat_log = (yhat_.max(-1.0 + epsilon) + 1.0).ln();
                    (w_ * 0.5 * (y_log - yhat_log).powi(2)) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let y_log = (y_.max(-1.0 + epsilon) + 1.0).ln();
                    let yhat_log = (yhat_.max(-1.0 + epsilon) + 1.0).ln();
                    (0.5 * (y_log - yhat_log).powi(2)) as f32
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
        let epsilon = 1e-6_f32;

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    let y_plus_1 = y_val.max(-1.0 + epsilon) + 1.0;
                    let yhat_plus_1 = yhat_val.max(-1.0 + epsilon) + 1.0;

                    let y_log = y_plus_1.ln();
                    let yhat_log = yhat_plus_1.ln();

                    let diff = yhat_log - y_log;

                    g.push(diff / yhat_plus_1 * w_val);
                    h.push((1.0 - diff) / (yhat_plus_1 * yhat_plus_1) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    let y_plus_1 = y_val.max(-1.0 + epsilon) + 1.0;
                    let yhat_plus_1 = yhat_val.max(-1.0 + epsilon) + 1.0;

                    let y_log = y_plus_1.ln();
                    let yhat_log = yhat_plus_1.ln();

                    let diff = yhat_log - y_log;

                    g.push(diff / yhat_plus_1);
                    h.push((1.0 - diff) / (yhat_plus_1 * yhat_plus_1));
                }
                (g, Some(h))
            }
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        // Init predict with weighted average of logged target,
        // then exponentiate back as our objective operates on predictions
        let epsilon = 1e-6;
        let log_mean = match sample_weight {
            Some(w) => {
                let mut ytot: f64 = 0.;
                let mut ntot: f64 = 0.;
                for i in 0..y.len() {
                    let y_log = (y[i].max(-1.0 + epsilon) + 1.0).ln();
                    ytot += w[i] * y_log;
                    ntot += w[i];
                }
                ytot / ntot
            }
            None => {
                let mut ytot: f64 = 0.;
                for yi in y {
                    ytot += (yi.max(-1.0 + epsilon) + 1.0).ln();
                }
                let ntot = y.len() as f64;
                ytot / ntot
            }
        };
        log_mean.exp() - 1.0
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredLogError
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl SquaredLogLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let epsilon = 1e-6;
        let y_log = (y.max(-1.0 + epsilon) + 1.0).ln();
        let yhat_log = (yhat.max(-1.0 + epsilon) + 1.0).ln();
        let l = 0.5 * (y_log - yhat_log).powi(2);
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
    fn test_sll_loss_init() {
        let y = vec![0.0, 1.0, 2.0];
        let loss_fn = SquaredLogLoss::default();
        // ln(1)=0, ln(2)=0.693, ln(3)=1.098
        // mean = (0 + 0.693 + 1.098)/3 = 0.597
        // exp(0.597) - 1 = 1.81 - 1 = 0.81
        let init = loss_fn.initial_value(&y, None, None);
        assert!((init - (0.597253_f64.exp() - 1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_sll_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![1.0, 0.0];
        let loss_fn = SquaredLogLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        assert_eq!(l, vec![0.0, 0.0]);

        let yhat2 = vec![0.0, 1.0];
        let l2 = loss_fn.loss(&y, &yhat2, None, None);
        // Loss = 0.5 * (ln(2) - ln(1))^2 = 0.5 * (ln(2))^2 approx 0.240226
        assert!((l2[0] - 0.2402265).abs() < 1e-6);
    }

    #[test]
    fn test_sll_gradient() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = SquaredLogLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // y_log = ln(2), yhat_log = ln(1) = 0
        // diff = yhat_log - y_log = -ln(2)
        // yhat_plus_1 = 1.0
        // g = diff / 1.0 = -ln(2) approx -0.693147
        assert!((g[0] as f64 - (-std::f64::consts::LN_2)).abs() < 1e-6);
        // h = (1 - diff) / 1.0 = 1 + ln(2) approx 1.693147
        assert!((h[0] - 1.6931472).abs() < 1e-6);

        // Weighted
        let w = vec![2.0, 0.5];
        let (gw, hw) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert!((gw[0] as f64 - (-std::f64::consts::LN_2 * 2.0)).abs() < 1e-6);
        assert!((hw.unwrap()[0] - 1.6931472 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sll_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let w = vec![2.0, 0.5];
        let loss_fn = SquaredLogLoss::default();

        let l = loss_fn.loss(&y, &yhat, Some(&w), None);
        // Loss[0] approx 0.240226 * 2.0 = 0.480453
        assert!((l[0] - 0.480453).abs() < 1e-6);
    }

    #[test]
    fn test_sll_gradient_and_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = SquaredLogLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert!((g[0] as f64 - (-std::f64::consts::LN_2)).abs() < 1e-6);
        assert!((h.unwrap()[0] - 1.6931472).abs() < 1e-6);
        assert!((l[0] - 0.2402265).abs() < 1e-6);
    }

    #[test]
    fn test_sll_gradient_and_loss_into() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = SquaredLogLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert!((grad[0] as f64 - (-std::f64::consts::LN_2)).abs() < 1e-6);
        assert!((hess.unwrap()[0] - 1.6931472).abs() < 1e-6);
        assert!((loss[0] - 0.2402265).abs() < 1e-6);
    }

    #[test]
    fn test_sll_init_weighted() {
        let y = vec![0.0, 1.0];
        let w = vec![1.0, 3.0];
        let loss_fn = SquaredLogLoss::default();
        // y_log = [0, ln(2)] = [0, 0.693]
        // mean = (0*1 + 0.693*3)/4 = 2.079/4 = 0.51975
        // exp(0.51975) - 1 = 1.6816 - 1 = 0.6816
        let init = loss_fn.initial_value(&y, Some(&w), None);
        assert!((init - (0.51986_f64.exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_sll_loss_single() {
        let loss_fn = SquaredLogLoss::default();
        assert!((loss_fn.loss_single(1.0, 0.0, None) - 0.2402265).abs() < 1e-6);
        assert!((loss_fn.loss_single(1.0, 0.0, Some(2.0)) - 0.480453).abs() < 1e-6);
    }
}
