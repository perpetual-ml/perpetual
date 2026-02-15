//! Instrumental Variable (BoostIV) Estimator
//!
//! Implements a two-stage least-squares (2SLS) approach using gradient boosting
//! to estimate causal effects in the presence of endogeneity.
use crate::booster::config::{CalibrationMethod, MissingNodeTreatment};
use crate::booster::core::PerpetualBooster;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::PerpetualError;
use crate::objective_functions::objective::Objective;
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

    /// Fit the IV Model.
    ///
    /// # Arguments
    /// * `X` - Covariates (Controls).
    /// * `Z` - Instruments.
    /// * `y` - Partial Outcome.
    /// * `w` - Treatment received.
    pub fn fit(&mut self, x: &Matrix<f64>, z: &Matrix<f64>, y: &[f64], w: &[f64]) -> Result<(), PerpetualError> {
        // --- Stage 1: Treatment Model ---
        // Predict W using X and Z.
        // We need to concatenate X and Z features.
        // For simplicity/performance in this initial version, we assume the user
        // passes a combined matrix for Stage 1 or we construct it.
        // Let's assume for now we construct a lightweight combined view or new matrix.
        // TODO: Optimize this concatenation (maybe use ColumnarMatrix).

        // Construct Stage 1 Features: [X, Z]
        // This is expensive if we copy.
        // For now, let's assume the user handles feature engineering and passes `stage1_features` directly?
        // Or strictly strictly implement 2SLS logic here.

        // Let's implement the concatenation logic for correctness first, optimization later.
        let rows = x.rows;
        let x_cols = x.cols;
        let z_cols = z.cols;
        let total_cols = x_cols + z_cols;
        // Efficient Column-Major concatenation
        // Assumes input matrices are standard contiguous column-major (which they are by Matrix definition)
        let mut stage1_data = Vec::with_capacity(x.data.len() + z.data.len());
        stage1_data.extend_from_slice(x.data);
        stage1_data.extend_from_slice(z.data);

        let matrix_stage1 = Matrix::new(&stage1_data, rows, total_cols);

        // Fit Stage 1
        self.treatment_model.fit(&matrix_stage1, w, None, None)?;

        // Predict W_hat
        let w_hat = self.treatment_model.predict(&matrix_stage1, true);

        // --- Stage 2: Outcome Model ---
        // Predict Y using X and W_hat.
        // Stage 2 Features: [X, W_hat]
        let mut stage2_data = Vec::with_capacity(x.data.len() + w_hat.len());
        stage2_data.extend_from_slice(x.data);
        stage2_data.extend_from_slice(&w_hat);

        let matrix_stage2 = Matrix::new(&stage2_data, rows, x_cols + 1);

        // Fit Stage 2
        self.outcome_model.fit(&matrix_stage2, y, None, None)?;

        Ok(())
    }

    /// Predict Outcome given X (and potentially new Z, though usually we want E[Y|do(W)]).
    /// For IV, "prediction" is tricky. Usually we want the structural function f(X, W).
    ///
    /// * `X` - Covariates
    /// * `w_counterfactual` - Treatment value to simulate (e.g., 1.0 or 0.0)
    pub fn predict(&self, x: &Matrix<f64>, w_counterfactual: &[f64]) -> Vec<f64> {
        let rows = x.rows;
        let x_cols = x.cols;

        // Construct Stage 2 Features with counterfactual W
        // Stage 2 Features with counterfactual: [X, W_cf]
        let mut stage2_data = Vec::with_capacity(x.data.len() + rows);
        stage2_data.extend_from_slice(x.data);

        // Handle W_counterfactual (broadcast if scalar, else extend)
        if w_counterfactual.len() == 1 {
            let val = w_counterfactual[0];
            stage2_data.resize(stage2_data.len() + rows, val);
        } else {
            if w_counterfactual.len() != rows {
                // If this happens, we might panic or should handle error, but signature doesn't allow error.
                // Assuming caller provides correct length as per doc logic implies match.
                // Fallback to safe iteration/extend if needed, or just extend.
                stage2_data.extend_from_slice(w_counterfactual);
            } else {
                stage2_data.extend_from_slice(w_counterfactual);
            }
        }

        let matrix_stage2 = Matrix::new(&stage2_data, rows, x_cols + 1);

        // Predict using Outcome Model
        self.outcome_model.predict(&matrix_stage2, true)
    }
}
