//! Meta-learners for Heterogeneous Treatment Effect (HTE) estimation.
//!
//! Implements standard meta-algorithms (S-Learner, T-Learner, X-Learner, DR-Learner)
//! wrapping `PerpetualBooster`.

use crate::booster::config::CalibrationMethod;
use crate::booster::core::PerpetualBooster;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// Helper to create a base booster configuration
#[allow(clippy::too_many_arguments)]
fn create_booster(
    budget: f32,
    objective: Objective,
    max_bin: u16,
    num_threads: Option<usize>,
    monotone_constraints: Option<ConstraintMap>,
    interaction_constraints: Option<Vec<Vec<usize>>>,
    force_children_to_bound_parent: bool,
    missing: f64,
    allow_missing_splits: bool,
    create_missing_branch: bool,
    terminate_missing_features: HashSet<usize>,
    missing_node_treatment: crate::booster::config::MissingNodeTreatment,
    log_iterations: usize,
    seed: u64,
    quantile: Option<f64>,
    reset: Option<bool>,
    categorical_features: Option<HashSet<usize>>,
    timeout: Option<f32>,
    iteration_limit: Option<usize>,
    memory_limit: Option<f32>,
    stopping_rounds: Option<usize>,
) -> Result<PerpetualBooster, PerpetualError> {
    PerpetualBooster::new(
        objective,
        budget,
        0.0, // Set base_score to 0.0 for causal models to ensure stable diffs
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
        quantile,
        reset,
        categorical_features,
        timeout,
        iteration_limit,
        memory_limit,
        stopping_rounds,
        CalibrationMethod::default(),
    )
}

// ---------------------------------------------------------------------------
// S-Learner
// ---------------------------------------------------------------------------

/// S-Learner (Single Learner).
///
/// Estimates $Y \approx \mu(X, W)$.
/// CATE(x) = $\mu(x, 1) - \mu(x, 0)$.
#[derive(Serialize, Deserialize)]
pub struct SLearner {
    pub model: PerpetualBooster,
}

impl SLearner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        // ... (other args passed to create_booster)
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: crate::booster::config::MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        let model = create_booster(
            budget,
            Objective::SquaredLoss,
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
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        )?;
        Ok(Self { model })
    }

    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), PerpetualError> {
        // Concatenate W as a new feature to X
        let rows = x.rows;
        let x_cols = x.cols;
        let mut data = Vec::with_capacity(x.data.len() + rows);
        data.extend_from_slice(x.data);
        data.extend_from_slice(w);

        let matrix_aug = Matrix::new(&data, rows, x_cols + 1);
        self.model.fit(&matrix_aug, y, None, None)
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        let rows = x.rows;
        let x_cols = x.cols;

        // Predict mu(X, 1)
        let mut data_1 = Vec::with_capacity(x.data.len() + rows);
        data_1.extend_from_slice(x.data);
        data_1.resize(data_1.len() + rows, 1.0);
        let matrix_1 = Matrix::new(&data_1, rows, x_cols + 1);
        let mu1 = self.model.predict(&matrix_1, true);

        // Predict mu(X, 0)
        let mut data_0 = Vec::with_capacity(x.data.len() + rows);
        data_0.extend_from_slice(x.data);
        data_0.resize(data_0.len() + rows, 0.0);
        let matrix_0 = Matrix::new(&data_0, rows, x_cols + 1);
        let mu0 = self.model.predict(&matrix_0, true);

        mu1.iter().zip(mu0.iter()).map(|(m1, m0)| m1 - m0).collect()
    }
}

// ---------------------------------------------------------------------------
// T-Learner
// ---------------------------------------------------------------------------

/// T-Learner (Two Learners).
///
/// Estimates $\mu_0(X)$ on control data and $\mu_1(X)$ on treated data.
/// CATE(x) = $\mu_1(x) - \mu_0(x)$.
#[derive(Serialize, Deserialize)]
pub struct TLearner {
    pub mu0: PerpetualBooster,
    pub mu1: PerpetualBooster,
}

impl TLearner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        // ... (other args)
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: crate::booster::config::MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        let mu0 = create_booster(
            budget,
            Objective::SquaredLoss,
            max_bin,
            num_threads,
            monotone_constraints.clone(),
            interaction_constraints.clone(),
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features.clone(),
            missing_node_treatment,
            log_iterations,
            seed + 1,
            quantile,
            reset,
            categorical_features.clone(),
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        )?;
        let mu1 = create_booster(
            budget,
            Objective::SquaredLoss,
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
            seed + 1, // slight seed variation
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        )?;
        Ok(Self { mu0, mu1 })
    }

    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), PerpetualError> {
        // Split data into control and treated
        // This is expensive as we create new Matrix instances.
        // Optimization: Use sample weights or indices if Booster supported it directly without copy.
        // For now, explicit copy.

        // Count 0s and 1s
        let n = x.rows;
        let n0 = w.iter().filter(|&&v| v == 0.0).count();
        let n1 = w.iter().filter(|&&v| v == 1.0).count();

        let mut x0_data = Vec::with_capacity(n0 * x.cols);
        let mut y0 = Vec::with_capacity(n0);

        let mut x1_data = Vec::with_capacity(n1 * x.cols);
        let mut y1 = Vec::with_capacity(n1);

        // Column-major implies we must iterate cols then rows.
        // BUT slice helpers `x.get_col(j)` exist.
        // Constructing sub-matrices in column-major is tricky if we stream rows.
        // Easiest is to identify indices first.
        let idx0: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 0.0)
            .map(|(i, _)| i)
            .collect();
        let idx1: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();

        // Construct x0
        for col in 0..x.cols {
            let col_data = &x.data[col * n..(col + 1) * n];
            for &i in &idx0 {
                x0_data.push(col_data[i]);
            }
        }
        for &i in &idx0 {
            y0.push(y[i]);
        }

        // Construct x1
        for col in 0..x.cols {
            let col_data = &x.data[col * n..(col + 1) * n];
            for &i in &idx1 {
                x1_data.push(col_data[i]);
            }
        }
        for &i in &idx1 {
            y1.push(y[i]);
        }

        let matrix0 = Matrix::new(&x0_data, n0, x.cols);
        let matrix1 = Matrix::new(&x1_data, n1, x.cols);

        self.mu0.fit(&matrix0, &y0, None, None)?;
        self.mu1.fit(&matrix1, &y1, None, None)?;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        let p1 = self.mu1.predict(x, true);
        let p0 = self.mu0.predict(x, true);
        p1.iter().zip(p0.iter()).map(|(a, b)| a - b).collect()
    }
}

// ---------------------------------------------------------------------------
// X-Learner
// ---------------------------------------------------------------------------

/// X-Learner.
///
/// 4-stage meta-learner suited for imbalanced treatment groups.
#[derive(Serialize, Deserialize)]
pub struct XLearner {
    pub mu0: PerpetualBooster,
    pub mu1: PerpetualBooster,
    pub tau0: PerpetualBooster,
    pub tau1: PerpetualBooster,
    pub propensity: PerpetualBooster,
}

impl XLearner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        propensity_budget: Option<f32>,
        // ... (other args)
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: crate::booster::config::MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        let p_budget = propensity_budget.unwrap_or(budget);

        // Helper to clone args
        let make = |obj: Objective, b: f32, s: u64| {
            create_booster(
                b,
                obj,
                max_bin,
                num_threads,
                monotone_constraints.clone(),
                interaction_constraints.clone(),
                force_children_to_bound_parent,
                missing,
                allow_missing_splits,
                create_missing_branch,
                terminate_missing_features.clone(),
                missing_node_treatment,
                log_iterations,
                s,
                quantile,
                reset,
                categorical_features.clone(),
                timeout,
                iteration_limit,
                memory_limit,
                stopping_rounds,
            )
        };

        Ok(Self {
            mu0: make(Objective::SquaredLoss, budget, seed)?,
            mu1: make(Objective::SquaredLoss, budget, seed + 1)?,
            tau0: make(Objective::SquaredLoss, budget, seed + 2)?,
            tau1: make(Objective::SquaredLoss, budget, seed + 3)?,
            propensity: make(Objective::LogLoss, p_budget, seed + 4)?,
        })
    }

    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), PerpetualError> {
        let n = x.rows;

        // 1. Split data
        let idx0: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 0.0)
            .map(|(i, _)| i)
            .collect();
        let idx1: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();

        let n0 = idx0.len();
        let n1 = idx1.len();

        // Construct partial matrices helper (cloning data)
        let get_subset = |indices: &[usize]| -> (Vec<f64>, Vec<f64>) {
            let mut sub_x = Vec::with_capacity(indices.len() * x.cols);
            let mut sub_y = Vec::with_capacity(indices.len());
            for col in 0..x.cols {
                let col_data = &x.data[col * n..(col + 1) * n];
                for &i in indices {
                    sub_x.push(col_data[i]);
                }
            }
            for &i in indices {
                sub_y.push(y[i]);
            }
            (sub_x, sub_y)
        };

        let (x0_data, y0) = get_subset(&idx0);
        let (x1_data, y1) = get_subset(&idx1);

        let matrix0 = Matrix::new(&x0_data, n0, x.cols);
        let matrix1 = Matrix::new(&x1_data, n1, x.cols);

        // 2. Stage 1: Fit mu0 and mu1
        self.mu0.fit(&matrix0, &y0, None, None)?;
        self.mu1.fit(&matrix1, &y1, None, None)?;

        // 3. Impute effects
        // D1 = Y1 - mu0(X1)
        let mu0_on_1 = self.mu0.predict(&matrix1, true);
        let d1: Vec<f64> = y1.iter().zip(mu0_on_1.iter()).map(|(yi, m)| yi - m).collect();

        // D0 = mu1(X0) - Y0
        let mu1_on_0 = self.mu1.predict(&matrix0, true);
        let d0: Vec<f64> = mu1_on_0.iter().zip(y0.iter()).map(|(m, yi)| m - yi).collect();

        // 4. Stage 3: Fit tau0 and tau1
        self.tau1.fit(&matrix1, &d1, None, None)?;
        self.tau0.fit(&matrix0, &d0, None, None)?;

        // 5. Stage 4: Propensity
        self.propensity.fit(x, w, None, None)?;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        let t0 = self.tau0.predict(x, true);
        let t1 = self.tau1.predict(x, true);

        let log_odds = self.propensity.predict(x, true);
        let p: Vec<f64> = log_odds.iter().map(|lo| 1.0 / (1.0 + (-lo).exp())).collect();

        // CATE = p * t0 + (1-p) * t1
        t0.iter()
            .zip(t1.iter())
            .zip(p.iter())
            .map(|((t0_i, t1_i), p_i)| p_i * t0_i + (1.0 - p_i) * t1_i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DR-Learner (AIPW)
// ---------------------------------------------------------------------------

/// DR-Learner (Doubly Robust).
///
/// Uses AIPW pseudo-outcomes to regress the CATE.
#[derive(Serialize, Deserialize)]
pub struct DRLearner {
    pub mu0: PerpetualBooster,
    pub mu1: PerpetualBooster,
    pub propensity: PerpetualBooster,
    pub effect: PerpetualBooster,
}

impl DRLearner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        propensity_budget: Option<f32>,
        // ... (other args)
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: crate::booster::config::MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        let p_budget = propensity_budget.unwrap_or(budget);

        // Helper to clone args
        let make = |obj: Objective, b: f32, s: u64| {
            create_booster(
                b,
                obj,
                max_bin,
                num_threads,
                monotone_constraints.clone(),
                interaction_constraints.clone(),
                force_children_to_bound_parent,
                missing,
                allow_missing_splits,
                create_missing_branch,
                terminate_missing_features.clone(),
                missing_node_treatment,
                log_iterations,
                s,
                quantile,
                reset,
                categorical_features.clone(),
                timeout,
                iteration_limit,
                memory_limit,
                stopping_rounds,
            )
        };

        Ok(Self {
            mu0: make(Objective::SquaredLoss, budget, seed)?,
            mu1: make(Objective::SquaredLoss, budget, seed + 1)?,
            propensity: make(Objective::LogLoss, p_budget, seed + 2)?,
            effect: make(Objective::SquaredLoss, budget, seed + 3)?,
        })
    }

    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), PerpetualError> {
        let n = x.rows;

        // 1. Split data (Similar reuse to T/X Learner, effectively T-Learner step)
        let idx0: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 0.0)
            .map(|(i, _)| i)
            .collect();
        let idx1: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();

        let n0 = idx0.len();
        let n1 = idx1.len();

        let get_subset = |indices: &[usize]| -> (Vec<f64>, Vec<f64>) {
            let mut sub_x = Vec::with_capacity(indices.len() * x.cols);
            let mut sub_y = Vec::with_capacity(indices.len());
            for col in 0..x.cols {
                let col_data = &x.data[col * n..(col + 1) * n];
                for &i in indices {
                    sub_x.push(col_data[i]);
                }
            }
            for &i in indices {
                sub_y.push(y[i]);
            }
            (sub_x, sub_y)
        };

        let (x0_data, y0) = get_subset(&idx0);
        let (x1_data, y1) = get_subset(&idx1);

        let matrix0 = Matrix::new(&x0_data, n0, x.cols);
        let matrix1 = Matrix::new(&x1_data, n1, x.cols);

        self.mu0.fit(&matrix0, &y0, None, None)?;
        self.mu1.fit(&matrix1, &y1, None, None)?;

        // 2. Propensity
        self.propensity.fit(x, w, None, None)?;

        // 3. Compute Pseudo-outcomes
        let mu0_hat = self.mu0.predict(x, true);
        let mu1_hat = self.mu1.predict(x, true);
        let log_odds = self.propensity.predict(x, true);
        let p_hat: Vec<f64> = log_odds.iter().map(|lo| 1.0 / (1.0 + (-lo).exp())).collect();

        // Using PolicyObjective helper or manual calculation?
        // Let's use PolicyObjective logic but manual here to avoid constructing extra structs if not needed.
        // Gamma = (mu1 - mu0) + W(Y - mu1)/p - (1-W)(Y - mu0)/(1-p)

        let mut gamma = Vec::with_capacity(n);
        for i in 0..n {
            let m1 = mu1_hat[i];
            let m0 = mu0_hat[i];
            let p = p_hat[i].clamp(1e-3, 1.0 - 1e-3);
            let wi = w[i];
            let yi = y[i];

            let term1 = m1 - m0;
            let term2 = wi * (yi - m1) / p;
            let term3 = (1.0 - wi) * (yi - m0) / (1.0 - p);

            gamma.push(term1 + term2 - term3);
        }

        // 4. Fit Effect Model
        self.effect.fit(x, &gamma, None, None)?;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        self.effect.predict(x, true)
    }
}
