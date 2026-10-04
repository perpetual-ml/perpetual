//! Brier Score objective for probabilistic classification.
use crate::objective::ObjectiveFunction;
use crate::{metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
/// Brier Score objective.
/// Defined as the Mean Squared Error of predicted probabilities: `(y - p)^2`.
/// Target `y` should be 0 or 1.
pub struct BrierLoss {}

impl ObjectiveFunction for BrierLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let p = 1.0_f64 / (1.0_f64 + (-*yhat_).exp());
                    ((*y_ - p).powi(2) * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let p = 1.0_f64 / (1.0_f64 + (-*yhat_).exp());
                    (*y_ - p).powi(2) as f32
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
                let mean_y = ytot / ntot;
                if mean_y <= 0.0 || mean_y >= 1.0 {
                    0.0 // Brier score is 0 if confidently predicting
                } else {
                    f64::ln(mean_y / (1.0 - mean_y))
                }
            }
            None => {
                let ytot = fast_sum(y);
                let ntot = y.len() as f64;
                let mean_y = ytot / ntot;
                if mean_y <= 0.0 || mean_y >= 1.0 {
                    0.0
                } else {
                    f64::ln(mean_y / (1.0 - mean_y))
                }
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
                    // Grad Brier Score: d/dyhat (y - p)^2 = 2(p - y) * p * (1 - p)
                    // We can drop the constant 2, learning rate easily compensates,
                    // But for exact metric match, let's keep it.
                    // Let's drop it to match usual semantics of loss = g, but actual implementations
                    // of custom objectives often keep constants. We'll use the exact derivative.
                    g.push(2.0 * (p - y_val) * p * (1.0 - p) * w_val);
                    // Hessian Brier Score
                    // H = d/dyhat [ 2(p - y) * p * (1 - p) ]
                    //   = d/dyhat [ 2p^2 - 2p^3 - 2yp + 2yp^2 ]
                    //   = 2 * p * (1 - p) * (2p - 3p^2 - y + 2yp)  (there is a better factorization)
                    //   = 2 * p * (1 - p) * [ (p - y)*(1 - 2p) + p*(1 - p) ]
                    h.push(2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p)) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    // Sigmoid in f32
                    let p = 1.0 / (1.0 + (-yhat_val).exp());

                    g.push(2.0 * (p - y_val) * p * (1.0 - p));
                    h.push(2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p)));
                }
                (g, Some(h))
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::BrierLoss
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
                    g.push(2.0 * (p - y_val) * p * (1.0 - p) * w_val);
                    h.push(2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p)) * w_val);

                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push(((y[i] - p64).powi(2) * w[i]) as f32);
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    g.push(2.0 * (p - y_val) * p * (1.0 - p));
                    h.push(2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p)));

                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    l.push((y[i] - p64).powi(2) as f32);
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
                    grad[i] = 2.0 * (p - y_val) * p * (1.0 - p) * w_val;
                    h[i] = 2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p)) * w_val;
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = ((y[i] - p64).powi(2) * w[i]) as f32;
                }
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let p = 1.0_f32 / (1.0 + (-yhat_val).exp());
                    grad[i] = 2.0 * (p - y_val) * p * (1.0 - p);
                    h[i] = 2.0 * p * (1.0 - p) * ((p - y_val) * (1.0 - 2.0 * p) + p * (1.0 - p));
                    let p64 = 1.0_f64 / (1.0 + (-yhat[i]).exp());
                    loss[i] = (y[i] - p64).powi(2) as f32;
                }
            }
        }
    }

    fn requires_batch_evaluation(&self) -> bool {
        false
    }
}

impl BrierLoss {
    #[inline]
    pub fn loss_single(&self, y: f64, yhat: f64, sample_weight: Option<f64>) -> f32 {
        let p = 1.0_f64 / (1.0_f64 + (-yhat).exp());
        let l = (y - p).powi(2);
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
    fn test_brier_loss_init() {
        let y = vec![1.0, 1.0, 0.0];
        let loss_fn = BrierLoss::default();
        // Mean y = 2/3. Logit(2/3) = ln(2/1) = ln(2)
        assert!((loss_fn.initial_value(&y, None, None) - 2.0_f64.ln()).abs() < 1e-6);

        // Edge cases
        assert_eq!(loss_fn.initial_value(&[1.0], None, None), 0.0);
        assert_eq!(loss_fn.initial_value(&[0.0], None, None), 0.0);
    }

    #[test]
    fn test_brier_loss_init_weighted() {
        let y = vec![1.0, 0.0];
        let w = vec![2.0, 1.0];
        let loss_fn = BrierLoss::default();
        // ntot = 3, ytot = 2. mean = 2/3. Init = ln(2)
        assert!((loss_fn.initial_value(&y, Some(&w), None) - 2.0_f64.ln()).abs() < 1e-6);

        // Edge case mean=0 or mean=1 weighted
        assert_eq!(loss_fn.initial_value(&[1.0], Some(&[2.0]), None), 0.0);
        assert_eq!(loss_fn.initial_value(&[0.0], Some(&[2.0]), None), 0.0);
    }

    #[test]
    fn test_brier_loss() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5
        let loss_fn = BrierLoss::default();

        let l = loss_fn.loss(&y, &yhat, None, None);
        // (1 - 0.5)^2 = 0.25, (0 - 0.5)^2 = 0.25
        assert_eq!(l, vec![0.25, 0.25]);

        let w = vec![2.0, 0.5];
        let lw = loss_fn.loss(&y, &yhat, Some(&w), None);
        assert_eq!(lw, vec![0.5, 0.125]);
    }

    #[test]
    fn test_brier_gradient() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5
        let loss_fn = BrierLoss::default();

        let (g, h) = loss_fn.gradient(&y, &yhat, None, None);
        let h = h.unwrap();
        // Grad: 2(p-y)p(1-p) = 2(0.5-1.0)*0.5*0.5 = 2*(-0.5)*0.25 = -0.25
        // Grad2: 2(0.5-0.0)*0.5*0.5 = 2*0.5*0.25 = 0.25
        assert_eq!(g, vec![-0.25, 0.25]);
        // Hess: 2p(1-p)[(p-y)(1-2p) + p(1-p)]
        // Since 1-2p = 0, Hess = 2p(1-p)[0 + p(1-p)] = 2 * (0.25) * (0.25) = 0.125
        assert_eq!(h, vec![0.125, 0.125]);
    }

    #[test]
    fn test_brier_gradient_and_loss_weighted() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let w = vec![2.0, 0.5];
        let loss_fn = BrierLoss::default();
        let (g, h, l) = loss_fn.gradient_and_loss(&y, &yhat, Some(&w), None);
        // Grad: -0.25 * 2 = -0.5, 0.25 * 0.5 = 0.125
        assert_eq!(g, vec![-0.5, 0.125]);
        // Hess: 0.125 * 2 = 0.25, 0.125 * 0.5 = 0.0625
        assert_eq!(h.unwrap(), vec![0.25, 0.0625]);
        // Loss: 0.25 * 2 = 0.5, 0.25 * 0.5 = 0.125
        assert_eq!(l, vec![0.5, 0.125]);
    }

    #[test]
    fn test_brier_gradient_and_loss_into() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let loss_fn = BrierLoss::default();
        let mut grad = vec![0.0; 2];
        let mut hess = Some(vec![0.0; 2]);
        let mut loss = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, None, None, &mut grad, &mut hess, &mut loss);
        assert_eq!(grad, vec![-0.25, 0.25]);
        assert_eq!(hess.unwrap(), vec![0.125, 0.125]);
        assert_eq!(loss, vec![0.25, 0.25]);

        let w = vec![2.0, 0.5];
        let mut gradw = vec![0.0; 2];
        let mut hessw = Some(vec![0.0; 2]);
        let mut lossw = vec![0.0; 2];
        loss_fn.gradient_and_loss_into(&y, &yhat, Some(&w), None, &mut gradw, &mut hessw, &mut lossw);
        assert_eq!(gradw, vec![-0.5, 0.125]);
        assert_eq!(hessw.unwrap(), vec![0.25, 0.0625]);
        assert_eq!(lossw, vec![0.5, 0.125]);
    }

    #[test]
    fn test_brier_loss_single() {
        let loss_fn = BrierLoss::default();
        assert_eq!(loss_fn.loss_single(1.0, 0.0, None), 0.25);
        assert_eq!(loss_fn.loss_single(1.0, 0.0, Some(2.0)), 0.5);
    }
}
