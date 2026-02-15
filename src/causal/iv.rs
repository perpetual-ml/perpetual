//! Instrumental Variable (BoostIV) Estimator
//!
//! Implements a two-stage least-squares (2SLS) approach using gradient boosting
//! to estimate causal effects in the presence of endogeneity.
use crate::booster::config::{CalibrationMethod, MissingNodeTreatment};
use crate::booster::core::PerpetualBooster;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Boosted Instrumental Variable (BoostIV) Estimator.
///
/// Implements a 2-Stage Least Squares (2SLS) approach using Gradient Boosting.
///
/// # Model Structure
///
/// * **Stage 1 (Treatment Model)**: Estimates the treatment intensity based on instruments and covariates.
///   $$ \hat{W} = f_1(X, Z) + \epsilon_1 $$
///   - Objective: `SquaredLoss` (Regression) or `LogLoss` (Classification/Propensity).
///
/// * **Stage 2 (Outcome Model)**: Estimates the outcome based on the *predicted* treatment and covariates.
///   $$ Y = f_2(X, \hat{W}) + \epsilon_2 $$
///   - The causal effect is derived from the dependence of $Y$ on $\hat{W}$.
///
#[derive(Serialize, Deserialize)]
pub struct IVBooster {
    /// Stage 1 model that predicts the treatment from instruments + covariates.
    pub treatment_model: PerpetualBooster,
    /// Stage 2 model that predicts the outcome from predicted treatment + covariates.
    pub outcome_model: PerpetualBooster,
    /// Budget allocated to the stage 1 (treatment) model.
    pub stage1_budget: f32,
    /// Budget allocated to the stage 2 (outcome) model.
    pub stage2_budget: f32,
}

impl IVBooster {
    /// Create a new `IVBooster` instance for Instrumental Variable estimation.
    ///
    /// # Arguments
    ///
    /// * `treatment_objective` - Objective for the stage 1 (treatment) model (e.g., `LogLoss` or `SquaredLoss`).
    /// * `outcome_objective` - Objective for the stage 2 (outcome) model (e.g., `SquaredLoss`).
    /// * `stage1_budget` - Learning budget for the treatment model.
    /// * `stage2_budget` - Learning budget for the outcome model.
    /// * `max_bin` - Maximum number of bins for feature discretization.
    /// * `num_threads` - Optional number of threads.
    /// * `monotone_constraints` - Optional monotonicity constraints.
    /// * `interaction_constraints` - Optional interaction constraints.
    /// * `force_children_to_bound_parent` - Whether to bound child predictions by parent's range.
    /// * `missing` - Missing value representation.
    /// * `allow_missing_splits` - Whether to allow splits isolating missing values.
    /// * `create_missing_branch` - Whether to create explicit missing branches (ternary trees).
    /// * `terminate_missing_features` - Features where missing branches should terminate early.
    /// * `missing_node_treatment` - Strategy for handling missing value splits.
    /// * `log_iterations` - Logging frequency.
    /// * `seed` - Random seed.
    /// * `quantile` - Target quantile for quantile regression.
    /// * `reset` - Whether to reset or continue training on fit.
    /// * `categorical_features` - Features to treat as categorical.
    /// * `timeout` - Hard limit for fitting time in seconds.
    /// * `iteration_limit` - Hard limit for number of iterations.
    /// * `memory_limit` - Memory limit in GB.
    /// * `stopping_rounds` - Number of rounds for early stopping.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        treatment_objective: Objective,
        outcome_objective: Objective,
        stage1_budget: f32,
        stage2_budget: f32,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<std::collections::HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        // Initialize Stage 1 Booster
        let treatment_model = PerpetualBooster::new(
            treatment_objective,
            stage1_budget,
            f64::NAN, // Auto-calc base score
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
            seed,
            quantile,
            reset,
            categorical_features.clone(),
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
            CalibrationMethod::default(),
        )?;

        // Initialize Stage 2 Booster
        let outcome_model = PerpetualBooster::new(
            outcome_objective,
            stage2_budget,
            f64::NAN,
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
        )?;

        Ok(IVBooster {
            treatment_model,
            outcome_model,
            stage1_budget,
            stage2_budget,
        })
    }

    /// Fit the IV Model (Control Function Approach).
    ///
    /// 1. Stage 1: Estimate $\hat{W} = E[W|X, Z]$.
    /// 2. Compute residuals $V = W - \hat{W}$.
    /// 3. Stage 2: Estimate $Y = E[Y | X, \hat{W}, V]$.
    ///    - $V$ acts as a control for endogeneity.
    pub fn fit(&mut self, x: &Matrix<f64>, z: &Matrix<f64>, y: &[f64], w: &[f64]) -> Result<(), PerpetualError> {
        // --- Stage 1: Treatment Model ---
        // Predict W using X and Z.
        let rows = x.rows;
        let x_cols = x.cols;
        let z_cols = z.cols;
        let total_cols_s1 = x_cols + z_cols;

        let mut stage1_data = Vec::with_capacity(x.data.len() + z.data.len());
        stage1_data.extend_from_slice(x.data);
        stage1_data.extend_from_slice(z.data);

        let matrix_stage1 = Matrix::new(&stage1_data, rows, total_cols_s1);

        self.treatment_model.fit(&matrix_stage1, w, None, None)?;

        // Predict W_hat and compute residuals V
        let w_hat = self.treatment_model.predict(&matrix_stage1, true);
        let v_res: Vec<f64> = w.iter().zip(w_hat.iter()).map(|(wi, what)| wi - what).collect();

        // --- Stage 2: Outcome Model (Control Function) ---
        // Predict Y using X, W_hat, and V.
        // Stage 2 Features: [X, W_hat, V]
        let mut stage2_data = Vec::with_capacity(x.data.len() + w_hat.len() + v_res.len());
        stage2_data.extend_from_slice(x.data);
        stage2_data.extend_from_slice(&w_hat);
        stage2_data.extend_from_slice(&v_res);

        let matrix_stage2 = Matrix::new(&stage2_data, rows, x_cols + 2); // X + W_hat + V

        // Fit Stage 2
        self.outcome_model.fit(&matrix_stage2, y, None, None)?;

        Ok(())
    }

    /// Predict Outcome given X and a Counterfactual Treatment W.
    ///
    /// Note: This implementation assumes the "Control Function" term $V$ is zero
    /// for counterfactual prediction (effectively estimating E[Y|do(W)] assuming
    /// the shift in W is exogenous or we are intervening).
    ///
    /// * `X` - Covariates
    /// * `w_counterfactual` - Treatment value to simulate.
    pub fn predict(&self, x: &Matrix<f64>, w_counterfactual: &[f64]) -> Vec<f64> {
        let rows = x.rows;
        let x_cols = x.cols;

        // Verify dimensions
        if w_counterfactual.len() != 1 && w_counterfactual.len() != rows {
            panic!("w_counterfactual must satisfy len == 1 or len == x.rows");
        }

        // Feature Construction: [X, W_cf, V=0]
        // We set V=0 because we are estimating the structural expectation E[Y|X, do(W)].
        // The control function term beta*V captures the bias from endogeneity in the observed
        // data. By setting V=0, we "remove" this bias term for prediction.

        let mut stage2_data = Vec::with_capacity(x.data.len() + rows * 2);
        stage2_data.extend_from_slice(x.data);

        // W_counterfactual column
        if w_counterfactual.len() == 1 {
            stage2_data.resize(stage2_data.len() + rows, w_counterfactual[0]);
        } else {
            stage2_data.extend_from_slice(w_counterfactual);
        }

        // V column (Zeros)
        stage2_data.resize(stage2_data.len() + rows, 0.0);

        let matrix_stage2 = Matrix::new(&stage2_data, rows, x_cols + 2);

        // Predict using Outcome Model
        self.outcome_model.predict(&matrix_stage2, true)
    }
}
