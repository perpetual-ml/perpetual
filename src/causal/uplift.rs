//! Uplift Booster (R-Learner)
//!
//! Estimates the Conditional Average Treatment Effect (CATE) using the
//! R-Learner meta-algorithm backed by gradient boosting.
use crate::booster::config::MissingNodeTreatment;
use crate::booster::core::PerpetualBooster;
use crate::causal::objective::RLearnerObjective;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::PerpetualError;
use crate::objective_functions::objective::Objective;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Uplift Booster using the R-Learner Meta-Algorithm.
///
/// Estimates the Conditional Average Treatment Effect (CATE): $\tau(x) = E[Y | X, W=1] - E[Y | X, W=0]$.
///
/// # R-Learner Algorithm
/// 1. **Outcome Model**: $\mu(x) = E[Y|X]$.
/// 2. **Propensity Model**: $p(x) = E[W|X] = P(W=1|X)$.
/// 3. **Effect Model**: Minimizes R-Loss to find $\tau(x)$.
///    $$ L = ((Y - \mu(x)) - \tau(x)(W - p(x)))^2 $$
#[derive(Serialize, Deserialize)]
pub struct UpliftBooster {
    /// Nuisance outcome model $\mu(x) = E[Y|X]$.
    pub outcome_model: PerpetualBooster,
    /// Nuisance propensity model $p(x) = P(W=1|X)$.
    pub propensity_model: PerpetualBooster,
    /// Effect model that learns $\tau(x)$ by minimizing R-Loss.
    pub effect_model: PerpetualBooster,
}

impl UpliftBooster {
    /// Create a new `UpliftBooster` instance using the R-Learner meta-algorithm.
    ///
    /// # Arguments
    ///
    /// * `outcome_budget` - Budget for the outcome model mu(x).
    /// * `propensity_budget` - Budget for the propensity model p(x).
    /// * `effect_budget` - Budget for the effect model tau(x).
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
        outcome_budget: f32,
        propensity_budget: f32,
        effect_budget: f32,
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
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> Result<Self, PerpetualError> {
        // Base configurations
        let base_params = |budget: f32, objective: Objective| -> Result<PerpetualBooster, PerpetualError> {
            PerpetualBooster::new(
                objective,
                budget,
                f64::NAN,
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
            )
        };

        let outcome_model = base_params(outcome_budget, Objective::SquaredLoss)?; // Assuming continuous Y for now
        let propensity_model = base_params(propensity_budget, Objective::LogLoss)?; // Treatment is binary

        // The effect model uses the custom R-Learner objective, but we can't initialize it
        // without data (the residuals).
        // So we initialize it with a dummy Objective::SquaredLoss and swap it during fit?
        // OR, since Objective is an enum, we can't easily swap to a Custom one dynamically without wrapping.
        // The `PerpetualBooster` takes an `Objective` in its config.

        // Actually, we can initialize it with SquaredLoss as a placeholder,
        // and then in `fit`, we construct the `RLearnerObjective` and use it.
        // However, `PerpetualBooster`'s config is public.
        let effect_model = base_params(effect_budget, Objective::SquaredLoss)?;

        Ok(UpliftBooster {
            outcome_model,
            propensity_model,
            effect_model,
        })
    }

    /// Fit the Uplift Model.
    ///
    /// * `X`: Covariates
    /// * `w`: Treatment indicator (0 or 1)
    /// * `y`: Outcome
    pub fn fit(&mut self, x: &Matrix<f64>, w: &[f64], y: &[f64]) -> Result<(), PerpetualError> {
        // 1. Fit Outcome Model: Y ~ X
        // R-Learner typically fits mu(x) on all data (ignoring W, or including W?).
        // Nie & Wager suggest cross-fitting, but for a simple implementation:
        // We can just fit Y ~ X. (This is a simplified "nuisance" parameter estimation).
        self.outcome_model.fit(x, y, None, None)?;
        let mu_hat = self.outcome_model.predict(x, true);

        // 2. Fit Propensity Model: W ~ X
        self.propensity_model.fit(x, w, None, None)?;
        // Helper to get probabilities from LogLoss prediction (which are log-odds)
        let log_odds = self.propensity_model.predict(x, true);
        let p_hat: Vec<f64> = log_odds.iter().map(|lo| 1.0 / (1.0 + (-lo).exp())).collect();

        // 3. Prepare R-Learner Objective
        // We need to pass the treatment and the *predicted* nuisance parameters to the objective.
        let r_obj_fn = RLearnerObjective::new(w.to_vec(), mu_hat, p_hat);

        // 4. Update Effect Model's Objective
        // This is the key "Deep Integration" step.
        // We swap the objective of the effect_model to our custom one.
        self.effect_model.cfg.objective = Objective::new_custom(r_obj_fn);

        // 5. Fit Effect Model: Tau ~ X
        // The target 'y' passed to fit is actually used in the loss function context.
        // Our custom loss function uses the 'y' passed here as the actual 'y' from the dataset.
        // It internalizes mu_hat and p_hat.
        self.effect_model.fit(x, y, None, None)?;

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Vec<f64> {
        self.effect_model.predict(x, true)
    }
}
