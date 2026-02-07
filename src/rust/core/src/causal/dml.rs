//! Double / Debiased Machine Learning (DML) Objective
//!
//! Implements the Chernozhukov et al. (2018) partial-linear model objective
//! for the *effect model* stage in a DML pipeline.
//!
//! # Partial-Linear Model
//!
//! $$Y = \theta(X) \cdot W + g(X) + \epsilon$$
//!
//! where $\theta(X)$ is the heterogeneous treatment effect being learned.
//!
//! The DML approach orthogonalizes both the treatment and outcome with
//! respect to the nuisance function $g(X)$ and $m(X) = E[W|X]$:
//!
//! $$\tilde{Y} = Y - \hat{g}(X), \quad \tilde{W} = W - \hat{m}(X)$$
//!
//! The effect model then minimizes:
//! $$L = \bigl(\tilde{Y} - \theta(X) \cdot \tilde{W}\bigr)^2$$
//!
//! This is mathematically equivalent to the R-Learner objective, but the
//! naming and framing follow the DML literature.  The key difference from
//! [`RLearnerObjective`](super::objective::RLearnerObjective) is conceptual:
//! DML emphasizes cross-fitting for valid inference, while R-Learner is
//! framed as a meta-algorithm.  In practice, both can use this objective.
//!
//! ## Cross-fitting
//!
//! DML requires that nuisance parameters $\hat{g}(X)$ and $\hat{m}(X)$ be
//! estimated on a *separate* fold from the one used to fit $\theta(X)$.
//! This objective **does not** enforce cross-fitting internally — users
//! should provide cross-fitted residuals or use the Python-side `DRLearner`
//! which handles it automatically.

use crate::metrics::evaluation::Metric;
use crate::objective_functions::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};

/// Minimum hessian floor for numerical stability.
const HESSIAN_FLOOR: f64 = 1e-6;

/// Double Machine Learning (DML) objective for heterogeneous treatment effects.
///
/// Fits $\theta(X)$ by minimizing the orthogonalized squared loss:
/// $$L = \bigl(\tilde{Y}_i - \theta(X_i) \cdot \tilde{W}_i\bigr)^2$$
///
/// where:
/// * $\tilde{Y}_i = Y_i - \hat{g}(X_i)$ — outcome residual,
/// * $\tilde{W}_i = W_i - \hat{m}(X_i)$ — treatment residual.
///
/// # Usage
///
/// Supply pre-computed residuals from cross-fitted nuisance models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DMLObjective {
    /// Outcome residuals: $\tilde{Y}_i = Y_i - \hat{g}(X_i)$.
    pub y_residual: Vec<f64>,
    /// Treatment residuals: $\tilde{W}_i = W_i - \hat{m}(X_i)$.
    pub w_residual: Vec<f64>,
}

impl DMLObjective {
    /// Create a new `DMLObjective`.
    ///
    /// # Arguments
    ///
    /// * `y_residual` - Outcome residuals (cross-fitted).
    /// * `w_residual` - Treatment residuals (cross-fitted).
    ///
    /// # Panics
    ///
    /// Panics if the two vectors have different lengths.
    pub fn new(y_residual: Vec<f64>, w_residual: Vec<f64>) -> Self {
        assert_eq!(
            y_residual.len(),
            w_residual.len(),
            "y_residual and w_residual must have the same length"
        );
        Self { y_residual, w_residual }
    }
}

impl ObjectiveFunction for DMLObjective {
    fn loss(&self, _y: &[f64], yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        // L_i = (y_res_i - theta_i * w_res_i)^2
        // Here `yhat` is theta(X) from the booster.
        yhat.iter()
            .zip(self.y_residual.iter())
            .zip(self.w_residual.iter())
            .map(|((theta, yr), wr)| {
                let diff = yr - theta * wr;
                (diff * diff) as f32
            })
            .collect()
    }

    fn gradient(
        &self,
        _y: &[f64],
        yhat: &[f64],
        _sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        // d/dtheta of 0.5 * (y_res - theta * w_res)^2
        // g = -w_res * (y_res - theta * w_res) = -w_res * y_res + theta * w_res^2
        // h = w_res^2

        let n = yhat.len();
        let mut grad = Vec::with_capacity(n);
        let mut hess = Vec::with_capacity(n);

        for (i, &theta) in yhat.iter().enumerate().take(n) {
            let yr = self.y_residual[i];
            let wr = self.w_residual[i];

            let g = -wr * yr + theta * wr * wr;
            let h = (wr * wr).max(HESSIAN_FLOOR);

            grad.push(g as f32);
            hess.push(h as f32);
        }

        (grad, Some(hess))
    }

    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        // OLS-style initial estimate: sum(y_res * w_res) / sum(w_res^2)
        let num: f64 = self
            .y_residual
            .iter()
            .zip(self.w_residual.iter())
            .map(|(yr, wr)| yr * wr)
            .sum();
        let den: f64 = self.w_residual.iter().map(|wr| wr * wr).sum();
        if den.abs() < HESSIAN_FLOOR {
            0.0
        } else {
            num / den
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }
}
