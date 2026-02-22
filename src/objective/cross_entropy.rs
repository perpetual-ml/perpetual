//! Cross Entropy Loss objective for regression/classification in [0, 1].
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Cross Entropy Loss objective.
/// Conceptually identical to LogLoss (binary classification), but allows target `y` in `[0, 1]`.
pub struct CrossEntropyLoss {}

impl ObjectiveFunction for CrossEntropyLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let p = 1.0 / (1.0 + (-*yhat_).exp());
                    let l = -(*y_ * p.ln() + (1.0 - *y_) * (1.0 - p).ln());
                    (l * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let p = 1.0 / (1.0 + (-*yhat_).exp());
                    -(*y_ * p.ln() + (1.0 - *y_) * (1.0 - p).ln()) as f32
                })
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
        // Log-odds of the mean
        if mean_y <= 0.0 {
            f64::NEG_INFINITY
        } else if mean_y >= 1.0 {
            f64::INFINITY
        } else {
            (mean_y / (1.0 - mean_y)).ln()
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

                    let y64 = y[i];
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push((-(y64 * p64.ln() + (1.0 - y64) * (1.0 - p64).ln()) * w[i]) as f32);
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());

                    g.push(p - y_val);
                    h.push(p * (1.0 - p));

                    let y64 = y[i];
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push(-(y64 * p64.ln() + (1.0 - y64) * (1.0 - p64).ln()) as f32);
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

                    let y64 = y[i];
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = (-(y64 * p64.ln() + (1.0 - y64) * (1.0 - p64).ln()) * w[i]) as f32;
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());

                    grad[i] = p - y_val;
                    h[i] = p * (1.0 - p);

                    let y64 = y[i];
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = -(y64 * p64.ln() + (1.0 - y64) * (1.0 - p64).ln()) as f32;
                }
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl CrossEntropyLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let p = 1.0 / (1.0 + (-yhat).exp());
        let l = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
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
    fn test_ce_loss_init() {
        let y = vec![0.5, 0.5, 0.0];
        let loss_fn = CrossEntropyLoss::default();
        // Mean y = 1.0 / 3.0. Logit(1/3) = ln(1/3 / 2/3) = ln(1/2) = -ln(2)
        assert!((loss_fn.initial_value(&y, None, None) - (-2.0_f64.ln())).abs() < 1e-6);

        assert_eq!(loss_fn.initial_value(&vec![1.0], None, None), f64::INFINITY);
        assert_eq!(loss_fn.initial_value(&vec![0.0], None, None), f64::NEG_INFINITY);
    }

    #[test]
    fn test_ce_loss_init_weighted() {
        let y = vec![1.0, 0.0];
        let w = vec![2.0, 1.0];
        let loss_fn = CrossEntropyLoss::default();
        // mean = 2/3. Logit = ln(2)
        assert!((loss_fn.initial_value(&y, Some(&w), None) - 2.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_ce_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5
        let loss_fn = CrossEntropyLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // - (1 * ln(0.5) + 0 * ln(0.5)) = -ln(0.5) = ln(2) approx 0.693147
        assert!((l[0] - 0.69314718).abs() < 1e-6);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert!((lw[0] - 0.69314718 * 2.0).abs() < 1e-6);

        // Edge case init weighted
        assert_eq!(loss_fn.initial_value(&vec![1.0], Some(&vec![2.0]), None), f64::INFINITY);
        assert_eq!(
            loss_fn.initial_value(&vec![0.0], Some(&vec![2.0]), None),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_ce_gradient() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5
        let loss_fn = CrossEntropyLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // p - y = 0.5 - 1.0 = -0.5
        // p * (1-p) = 0.25
        assert_eq!(g, vec![-0.5, 0.5]);
        assert_eq!(h, vec![0.25, 0.25]);
    }

    #[test]
    fn test_ce_gradient_and_loss_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let w = vec![2.0, 0.5];
        let loss_fn = CrossEntropyLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        // Grad: -0.5 * 2 = -1.0, 0.5 * 0.5 = 0.25
        assert_eq!(g, vec![-1.0, 0.25]);
        // Hess: 0.25 * 2 = 0.5, 0.25 * 0.5 = 0.125
        assert_eq!(h.unwrap(), vec![0.5, 0.125]);
        // Loss: ln(2) * 2, ln(2) * 0.5
        assert!((l[0] - 0.69314718 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_ce_gradient_and_loss_into() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = CrossEntropyLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![-0.5, 0.5]);
        assert_eq!(hess.unwrap(), vec![0.25, 0.25]);
        assert!((loss[0] - 0.69314718).abs() < 1e-6);

        let w = vec![2.0, 0.5];
        let mut gradw = vec![0.0; 2];
        let mut hessw = Some(vec![0.0; 2]);
        let mut lossw = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, Some(&w), None, &mut gradw, &mut hessw, &mut lossw);
        assert_eq!(gradw, vec![-1.0, 0.25]);
        assert_eq!(hessw.unwrap(), vec![0.5, 0.125]);
        assert!((lossw[0] - 0.69314718 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_ce_loss_single() {
        let loss_fn = CrossEntropyLoss::default();
        assert!((loss_fn.loss_single(1.0, 0.0, None) - 0.69314718).abs() < 1e-6);
    }
}
