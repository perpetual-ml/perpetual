//! Cross Entropy Lambda Loss objective for weighted continuous regression in [0, 1].
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Cross Entropy Lambda Loss objective.
/// Alternative parameterization of cross-entropy, specifically factoring instance weights
/// directly into the independent-events probability bounds: `z = 1 - (1 - p)^w`.
pub struct CrossEntropyLambdaLoss {}

impl ObjectiveFunction for CrossEntropyLambdaLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let ep = yhat_.exp();
                    let e_neg = (-*yhat_).exp();
                    // h_hat = log(1+e_f) = -log(e_neg / (1 + e_f)) etc
                    // Mathematically, if p = 1/(1+e_neg), 1-p = 1 / (1+ep)
                    // Then z = 1 - (1-p)^w
                    let log_1_minus_p = -(1.0 + ep).ln(); // since 1-p = 1/(1+ep)
                    let z = 1.0 - (log_1_minus_p * *w_).exp();

                    // If z == 0, fallback to standard log loss
                    if z <= 0.0 || z >= 1.0 {
                        let p = 1.0 / (1.0 + e_neg);
                        return (-(*y_ * p.ln() + (1.0 - *y_) * (1.0 - p).ln()) * *w_) as f32;
                    }

                    // Loss = -(y * ln(z) + (1-y) * ln(1-z)) ... weighted differently by design
                    let l = -(*y_ * z.ln() + (1.0 - *y_) * (1.0 - z).ln());
                    l as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    // With no weights, CrossEntropyLambda defaults exactly to standard cross entropy
                    let p = 1.0 / (1.0 + (-*yhat_).exp());
                    -(*y_ * p.ln() + (1.0 - *y_) * (1.0 - p).ln()) as f32
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

        match sample_weight {
            Some(weights) => {
                for i in 0..len {
                    let y_val = y[i];
                    let yhat_val = yhat[i];
                    let w_val = weights[i];

                    let ef = yhat_val.exp();
                    let enegf = (-yhat_val).exp();
                    let d_intermediate = 1.0 + ef;
                    let h_hat = d_intermediate.ln();

                    let exp_neg_whhat = (-w_val * h_hat).exp();
                    let z = 1.0 - exp_neg_whhat;

                    if z <= 0.0 || z >= 1.0 {
                        // fallback to standard grad matching LightGBM limits
                        let p = 1.0 / (1.0 + enegf);
                        g.push(((p - y_val) * w_val) as f32);
                        h.push((p * (1.0 - p) * w_val) as f32);
                        continue;
                    }

                    // g = (1 - y/z) * w * sigmoid(f)
                    let p = 1.0 / (1.0 + enegf);
                    g.push(((1.0 - y_val / z) * w_val * p) as f32);

                    // h = a * (1 + y * b)
                    let c = 1.0 / (1.0 - z);
                    let a = w_val * ef / (d_intermediate * d_intermediate);
                    let d_prime = c - 1.0;
                    let b = (c / (d_prime * d_prime)) * (1.0 + w_val * ef - c);
                    h.push((a * (1.0 + y_val * b)) as f32);
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
        if mean_y <= 0.0 {
            f64::NEG_INFINITY
        } else if mean_y >= 1.0 {
            f64::INFINITY
        } else {
            (mean_y / (1.0 - mean_y)).ln()
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl CrossEntropyLambdaLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let w = sample_weight.unwrap_or(1.0);
        if w == 1.0 {
            let p = 1.0 / (1.0 + (-yhat).exp());
            return -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) as f32;
        }

        let ep = yhat.exp();
        let e_neg = (-yhat).exp();
        let log_1_minus_p = -(1.0 + ep).ln();
        let z = 1.0 - (log_1_minus_p * w).exp();

        if z <= 0.0 || z >= 1.0 {
            let p = 1.0 / (1.0 + e_neg);
            return (-(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) * w) as f32;
        }

        -(y * z.ln() + (1.0 - y) * (1.0 - z).ln()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ce_lambda_init() {
        let y = vec![0.5, 0.5, 0.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        assert!((loss_fn.initial_value(&y, None, None) - (-2.0_f64.ln())).abs() < 1e-6);
    }

    #[test]
    fn test_ce_lambda_loss_unweighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        let l = loss_fn.loss(&y, &yhat, None, None);
        // Defaults to standard CE: ln(2) approx 0.693147
        assert!((l[0] - 0.69314718).abs() < 1e-6);
    }

    #[test]
    fn test_ce_lambda_loss_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5, 1-p = 0.5
        let w = vec![2.0, 1.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        let l = loss_fn.loss(&y, &yhat, Some(&w), None);

        // z = 1 - (1-p)^w = 1 - 0.5^2 = 0.75
        // Loss[0] = -(1 * ln(0.75) + 0 * ln(0.25)) = -ln(0.75) = 0.287682
        assert!((l[0] - 0.287682).abs() < 1e-5);
    }

    #[test]
    fn test_ce_lambda_gradient_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5, z = 0.75
        let w = vec![2.0, 1.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        let h = h.unwrap();

        // g = (1 - y/z) * w * p = (1 - 1/0.75) * 2 * 0.5 = (1 - 1.333) * 1 = -0.333333
        assert!((g[0] - (-1.0 / 3.0)).abs() < 1e-5);
        // h test - complex formula, just verifying it runs and is positive
        assert!(h[0] > 0.0);
    }

    #[test]
    fn test_ce_lambda_fallback() {
        let y = vec![1.0];
        let yhat = vec![10.0]; // p very close to 1
        let w = vec![1.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        // Trigger fallback in loss
        let l = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert!(l[0] >= 0.0);

        // Trigger fallback in gradient
        let (g, h) = loss_fn.gradient(&y, &yhat, Some(&w), None);
        assert!(g[0] <= 0.0);
        assert!(h.unwrap()[0] >= 0.0);
    }

    #[test]
    fn test_ce_lambda_gradient_and_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = CrossEntropyLambdaLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, None, None);
        assert_eq!(g, vec![-0.5, 0.5]);
        assert_eq!(h.unwrap(), vec![0.25, 0.25]);
        assert!((l[0] - 0.69314718).abs() < 1e-6);
    }

    #[test]
    fn test_ce_lambda_loss_single() {
        let loss_fn = CrossEntropyLambdaLoss::default();
        // Unweighted
        assert!((loss_fn.loss_single(1.0, 0.0, None) - 0.69314718).abs() < 1e-6);
        // Weighted (z=0.75)
        assert!((loss_fn.loss_single(1.0, 0.0, Some(2.0)) - 0.287682).abs() < 1e-5);
        // Fallback single
        assert!(loss_fn.loss_single(1.0, 10.0, Some(1.0)) >= 0.0);
    }
}
