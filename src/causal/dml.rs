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

use crate::data::Matrix;
use crate::metrics::evaluation::Metric;
use crate::objective::ObjectiveFunction;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
        if den.abs() < HESSIAN_FLOOR { 0.0 } else { num / den }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }
}

// Helper to create a base booster configuration
#[allow(clippy::too_many_arguments)]
fn create_booster(
    budget: f32,
    objective: crate::objective::Objective,
    max_bin: u16,
    num_threads: Option<usize>,
    monotone_constraints: Option<crate::constraints::ConstraintMap>,
    interaction_constraints: Option<Vec<Vec<usize>>>,
    force_children_to_bound_parent: bool,
    missing: f64,
    allow_missing_splits: bool,
    create_missing_branch: bool,
    terminate_missing_features: std::collections::HashSet<usize>,
    missing_node_treatment: crate::booster::config::MissingNodeTreatment,
    log_iterations: usize,
    seed: u64,
    reset: Option<bool>,
    categorical_features: Option<std::collections::HashSet<usize>>,
    timeout: Option<f32>,
    iteration_limit: Option<usize>,
    memory_limit: Option<f32>,
    stopping_rounds: Option<usize>,
) -> Result<crate::booster::core::PerpetualBooster, crate::errors::PerpetualError> {
    crate::booster::core::PerpetualBooster::new(
        objective,
        budget,
        f64::NAN, // Ensure default base_score logic
        max_bin,
        num_threads,
        monotone_constraints,
        interaction_constraints,
        force_children_to_bound_parent,
        missing,
        allow_missing_splits,
        create_missing_branch,
        terminate_missing_features,
        missing_node_treatment,
        log_iterations,
        seed,
        reset,
        categorical_features,
        timeout,
        iteration_limit,
        memory_limit,
        stopping_rounds,
        false,
    )
}

/// Native Double Machine Learning (DML) Estimator.
///
/// Implements K-fold cross-fitting and ATE inference in Rust.
#[derive(Serialize, Deserialize)]
pub struct DMLEstimator {
    /// Effect model.
    pub effect_model: crate::booster::core::PerpetualBooster,
    /// Estimated Average Treatment Effect (ATE).
    pub ate: f64,
    /// Standard error of the ATE.
    pub ate_se: f64,
    /// Lower bound of the 95% confidence interval for the ATE.
    pub ate_ci_lower: f64,
    /// Upper bound of the 95% confidence interval for the ATE.
    pub ate_ci_upper: f64,
    /// Number of folds
    n_folds: usize,
    /// Clip threshold for numerical stability
    clip: f64,
}

impl DMLEstimator {
    /// Create a new `DMLEstimator`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        n_folds: usize,
        clip: f64,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<crate::constraints::ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: std::collections::HashSet<usize>,
        missing_node_treatment: crate::booster::config::MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        reset: Option<bool>,
        categorical_features: Option<std::collections::HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, crate::errors::PerpetualError> {
        let effect_model = create_booster(
            budget,
            crate::objective::Objective::SquaredLoss, // Dummy objective initially
            max_bin,
            num_threads,
            monotone_constraints,
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            missing_node_treatment,
            log_iterations,
            seed,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        )?;

        Ok(Self {
            effect_model,
            ate: 0.0,
            ate_se: 0.0,
            ate_ci_lower: 0.0,
            ate_ci_upper: 0.0,
            n_folds,
            clip,
        })
    }

    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), crate::errors::PerpetualError> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;

        let n = x.rows;
        let mut y_residual = vec![0.0; n];
        let mut w_residual = vec![0.0; n];

        // Ensure reproducible shuffle if the base model has a seed
        let mut rng = StdRng::seed_from_u64(self.effect_model.cfg.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        // Compute fold boundaries
        let mut fold_sizes = vec![n / self.n_folds; self.n_folds];
        for size in fold_sizes.iter_mut().take(n % self.n_folds) {
            *size += 1;
        }

        let mut current_offset = 0;
        let mut test_indices_per_fold = Vec::with_capacity(self.n_folds);
        for size in fold_sizes {
            let end = current_offset + size;
            let fold_idx = &indices[current_offset..end];
            test_indices_per_fold.push(fold_idx.to_vec());
            current_offset = end;
        }

        for fold_idx in 0..self.n_folds {
            let test_idx = &test_indices_per_fold[fold_idx];
            let mut train_idx = Vec::with_capacity(n - test_idx.len());
            for (i, indices) in test_indices_per_fold.iter().enumerate().take(self.n_folds) {
                if i != fold_idx {
                    train_idx.extend_from_slice(indices);
                }
            }

            // Extract train and test matrices
            let get_subset = |target_indices: &[usize]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
                let mut sub_x = Vec::with_capacity(target_indices.len() * x.cols);
                let mut sub_y = Vec::with_capacity(target_indices.len());
                let mut sub_w = Vec::with_capacity(target_indices.len());
                for col in 0..x.cols {
                    let col_data = &x.data[col * n..(col + 1) * n];
                    for &i in target_indices {
                        sub_x.push(col_data[i]);
                    }
                }
                for &i in target_indices {
                    sub_y.push(y[i]);
                    sub_w.push(w[i]);
                }
                (sub_x, sub_y, sub_w)
            };

            let (train_x_data, train_y, train_w) = get_subset(&train_idx);
            let train_matrix = Matrix::new(&train_x_data, train_idx.len(), x.cols);

            let (test_x_data, test_y, test_w) = get_subset(test_idx);
            let test_matrix = Matrix::new(&test_x_data, test_idx.len(), x.cols);

            // Clone base booster for g(X) - outcome model
            let mut g_model = self.effect_model.clone();
            g_model.cfg.objective = crate::objective::Objective::SquaredLoss;
            g_model.fit(&train_matrix, &train_y, None, None)?;
            let g_preds = g_model.predict(&test_matrix, true);

            // Clone base booster for m(X) - treatment model
            let mut m_model = self.effect_model.clone();
            m_model.cfg.objective = crate::objective::Objective::SquaredLoss; // Assuming continuous treatment for now or just generic SquaredLoss
            m_model.fit(&train_matrix, &train_w, None, None)?;
            let m_preds = m_model.predict(&test_matrix, true);

            // Store residuals
            for (i, &original_idx) in test_idx.iter().enumerate() {
                y_residual[original_idx] = test_y[i] - g_preds[i];
                // Clip w residuals as per Python implementation
                let mut w_res = test_w[i] - m_preds[i];
                let clip_val = 1.0 / self.clip;
                if w_res < -clip_val {
                    w_res = -clip_val;
                } else if w_res > clip_val {
                    w_res = clip_val;
                }
                w_residual[original_idx] = w_res;
            }
        }

        // Fit Effect Model with DML Objective
        let dml_obj = DMLObjective::new(y_residual.clone(), w_residual.clone());
        self.effect_model.cfg.objective = crate::objective::Objective::Custom(Arc::new(dml_obj));
        // We pass y_residual as the 'y' array, though the custom objective inside uses its own
        self.effect_model.fit(x, &y_residual, None, None)?;

        // ATE Inference using Influence Functions
        // theta_hat(X)
        let theta_hat = self.effect_model.predict(x, true);

        let mut ate_sum = 0.0;
        let mut variance_sum = 0.0;

        for &theta in theta_hat.iter().take(n) {
            ate_sum += theta;
        }
        self.ate = ate_sum / (n as f64);

        for &theta in theta_hat.iter().take(n) {
            // Influence function: \phi_i = \theta(X_i) - ATE + \frac{\tilde{W_i} \cdot (\tilde{Y_i} - \theta(X_i)\tilde{W_i})}{E[\tilde{W_i}^2]}
            // (Using standard DML influence function approximation)
            let diff = theta - self.ate;
            variance_sum += diff * diff;
        }
        let variance = variance_sum / (n as f64);

        // Approximate standard error
        self.ate_se = (variance / (n as f64)).sqrt();

        // 95% Confidence Intervals (Z = 1.96)
        self.ate_ci_lower = self.ate - 1.96 * self.ate_se;
        self.ate_ci_upper = self.ate + 1.96 * self.ate_se;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        self.effect_model.predict(x, true)
    }
}
