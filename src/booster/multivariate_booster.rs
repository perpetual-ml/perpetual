//! Multivariate Booster
//! 
//! 
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::{Matrix, UnivariateBooster};
use crate::booster::config::*;
use crate::booster::config::MissingNodeTreatment;
use crate::objective_functions::{ObjectiveFunction, Objective, CustomObjective};
use std::sync::Arc;

/// Perpetual Booster object
#[derive(Clone, Serialize, Deserialize)]
pub struct MultivariateBooster {
    pub n_boosters: usize,
    pub cfg: BoosterConfig,
    pub boosters: Vec<UnivariateBooster>,
    metadata: HashMap<String, String>,
}

impl Default for MultivariateBooster {
    fn default() -> Self {
        let cfg = BoosterConfig::default();
        let n_boosters = 1;
        let boosters = vec![{
            let mut p = UnivariateBooster::default();
            p.cfg = cfg.clone();
            p
        }];

        Self {
            n_boosters,
            cfg,
            boosters,
            metadata: HashMap::new(),
        }
    }
}


impl MultivariateBooster {
    /// Multi Output Booster object
    ///
    /// * `objective` - The name of objective function used to optimize. Valid options are:
    ///      "LogLoss" to use logistic loss as the objective function,
    ///      "SquaredLoss" to use Squared Error as the objective function,
    ///      "QuantileLoss" for quantile regression.
    /// * `budget` - budget to fit the model.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///     a smaller number, will result in faster training time, while potentially sacrificing
    ///     accuracy. If there are more bins, than unique values in a column, all unique values
    ///     will be used.
    /// * `num_threads` - Number of threads to be used during training
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///     between the training features and the target variable.
    /// * `force_children_to_bound_parent` - force_children_to_bound_parent.
    /// * `missing` - Value to consider missing.
    /// * `allow_missing_splits` - Should the algorithm allow splits that completed seperate out missing
    ///     and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    ///     is true, setting this to true will result in the missin branch being further split.
    /// * `create_missing_branch` - Should missing be split out it's own separate branch?
    /// * `missing_node_treatment` - specify how missing nodes should be handled during training.
    /// * `log_iterations` - Setting to a value (N) other than zero will result in information being logged about ever N iterations.
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `quantile` - used only in quantile regression.
    /// * `reset` - Reset the model or continue training.
    /// * `categorical_features` - categorical features.
    /// * `timeout` - fit timeout limit in seconds.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    /// * `memory_limit` - optional limit for memory allocation.
    /// * `stopping_rounds` - optional limit for auto stopping rounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_boosters: usize,
        objective: Objective,
        budget: f32,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
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
        custom_objective: Option<CustomObjective>,
    ) -> Result<Self, PerpetualError> {
        // Build the common configuration object.
        let cfg = BoosterConfig {
            objective: objective.clone(),
            budget,
            max_bin,
            num_threads,
            monotone_constraints: monotone_constraints.clone(),
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features: terminate_missing_features.clone(),
            missing_node_treatment,
            log_iterations,
            seed,
            quantile,
            reset,
            categorical_features: categorical_features.clone(),
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
            custom_objective
        };

        // Base booster template that child boosters will clone.
        let template_booster = {
            let mut b = UnivariateBooster::default();
            b.cfg = cfg.clone();
            b
        };
        template_booster.validate_parameters()?;

        // Assemble the wrapper with `n_boosters` copies.
        let boosters = vec![template_booster; n_boosters.max(1)];

        Ok(MultivariateBooster {
            n_boosters: n_boosters.max(1),
            cfg,
            boosters,
            metadata: HashMap::new(),
        })
        
    }

    pub fn reset(&mut self) {
        for b in &mut self.boosters {
            b.reset();
        }
    }

    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            let _ = self.boosters[i].fit(data, y.get_col(i), sample_weight);
        }
        Ok(())
    }

    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            let _ = self.boosters[i].prune(data, y.get_col(i), sample_weight);
        }
        Ok(())
    }

    /// Get the boosters
    pub fn get_boosters(&self) -> &[UnivariateBooster] {
        &self.boosters
    }

    // Set methods for paramters

    /// Set n_boosters on the booster. This will also initialize the boosters by cloning the first one.
    /// * `n_boosters` - The number of boosters.
    pub fn set_n_boosters(mut self, n_boosters: usize) -> Self {
        self.n_boosters = n_boosters;
        self.boosters = (0..n_boosters).map(|_| self.boosters[0].clone()).collect();
        self
    }

    /// Set the objective on the booster.
    /// * `objective` - The objective type of the booster.
    pub fn set_objective(mut self, objective: Objective) -> Self {
        self.cfg.objective = objective.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_objective(objective.clone()))
            .collect();
        self
    }

    /// Set the budget on the booster.
    /// * `budget` - Budget to fit the booster.
    pub fn set_budget(mut self, budget: f32) -> Self {
        self.cfg.budget = budget;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_budget(budget)).collect();
        self
    }

    /// Set the number of bins on the booster.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_max_bin(mut self, max_bin: u16) -> Self {
        self.cfg.max_bin = max_bin;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_max_bin(max_bin)).collect();
        self
    }

    /// Set the number of threads on the booster.
    /// * `num_threads` - Set the number of threads to be used during training.
    pub fn set_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.cfg.num_threads = num_threads;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_num_threads(num_threads))
            .collect();
        self
    }

    /// Set the monotone_constraints on the booster.
    /// * `monotone_constraints` - The monotone constraints of the booster.
    pub fn set_monotone_constraints(mut self, monotone_constraints: Option<ConstraintMap>) -> Self {
        self.cfg.monotone_constraints = monotone_constraints.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_monotone_constraints(monotone_constraints.clone()))
            .collect();
        self
    }

    /// Set the force_children_to_bound_parent on the booster.
    /// * `force_children_to_bound_parent` - Set force children to bound parent.
    pub fn set_force_children_to_bound_parent(mut self, force_children_to_bound_parent: bool) -> Self {
        self.cfg.force_children_to_bound_parent = force_children_to_bound_parent;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| {
                b.clone()
                    .set_force_children_to_bound_parent(force_children_to_bound_parent)
            })
            .collect();
        self
    }

    /// Set missing value of the booster
    /// * `missing` - Float value to consider as missing.
    pub fn set_missing(mut self, missing: f64) -> Self {
        self.cfg.missing = missing;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_missing(missing)).collect();
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.cfg.create_missing_branch = allow_missing_splits;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_allow_missing_splits(allow_missing_splits))
            .collect();
        self
    }

    /// Set create missing value of the booster
    /// * `create_missing_branch` - Bool specifying if missing should get it's own
    /// branch.
    pub fn set_create_missing_branch(mut self, create_missing_branch: bool) -> Self {
        self.cfg.create_missing_branch = create_missing_branch;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_create_missing_branch(create_missing_branch))
            .collect();
        self
    }

    /// Set the features where whose missing nodes should
    /// always be terminated.
    /// * `terminate_missing_features` - Hashset of the feature indices for the
    /// features that should always terminate the missing node, if create_missing_branch
    /// is true.
    pub fn set_terminate_missing_features(mut self, terminate_missing_features: HashSet<usize>) -> Self {
        self.cfg.terminate_missing_features = terminate_missing_features.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| {
                b.clone()
                    .set_terminate_missing_features(terminate_missing_features.clone())
            })
            .collect();
        self
    }

    /// Set the missing_node_treatment on the booster.
    /// * `missing_node_treatment` - The missing node treatment of the booster.
    pub fn set_missing_node_treatment(mut self, missing_node_treatment: MissingNodeTreatment) -> Self {
        self.cfg.missing_node_treatment = missing_node_treatment;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_missing_node_treatment(missing_node_treatment.clone()))
            .collect();
        self
    }

    /// Set the log iterations on the booster.
    /// * `log_iterations` - The number of log iterations of the booster.
    pub fn set_log_iterations(mut self, log_iterations: usize) -> Self {
        self.cfg.log_iterations = log_iterations;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_log_iterations(log_iterations))
            .collect();
        self
    }

    /// Set the seed on the booster.
    /// * `seed` - Integer value used to see any randomness used in the algorithm.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_seed(seed)).collect();
        self
    }

    /// Set the quantile on the booster.
    /// * `quantile` - used only in quantile regression.
    pub fn set_quantile(mut self, quantile: Option<f64>) -> Self {
        self.cfg.quantile = quantile;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_quantile(quantile)).collect();
        self
    }

    /// Set the reset on the booster.
    /// * `reset` - Reset the model or continue training.
    pub fn set_reset(mut self, reset: Option<bool>) -> Self {
        self.cfg.reset = reset;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_reset(reset)).collect();
        self
    }

    /// Set the categorical features on the booster.
    /// * `categorical_features` - categorical features.
    pub fn set_categorical_features(mut self, categorical_features: Option<HashSet<usize>>) -> Self {
        self.cfg.categorical_features = categorical_features.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_categorical_features(categorical_features.clone()))
            .collect();
        self
    }

    /// Set the timeout on the booster.
    /// * `timeout` - fit timeout limit in seconds.
    pub fn set_timeout(mut self, timeout: Option<f32>) -> Self {
        self.cfg.timeout = timeout;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_timeout(timeout)).collect();
        self
    }

    /// Set the iteration limit on the booster.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    pub fn set_iteration_limit(mut self, iteration_limit: Option<usize>) -> Self {
        self.cfg.iteration_limit = iteration_limit;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_iteration_limit(iteration_limit))
            .collect();
        self
    }

    /// Set the memory limit on the booster.
    /// * `memory_limit` - optional limit for memory allocation.
    pub fn set_memory_limit(mut self, memory_limit: Option<f32>) -> Self {
        self.cfg.memory_limit = memory_limit;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_memory_limit(memory_limit))
            .collect();
        self
    }

    /// Set the stopping rounds on the booster.
    /// * `stopping_rounds` - optional limit for auto stopping rounds.
    pub fn set_stopping_rounds(mut self, stopping_rounds: Option<usize>) -> Self {
        self.cfg.stopping_rounds = stopping_rounds;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_stopping_rounds(stopping_rounds))
            .collect();
        self
    }

    /// Insert metadata
    /// * `key` - String value for the metadata key.
    /// * `value` - value to assign to the metadata key.
    pub fn insert_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get Metadata
    /// * `key` - Get the associated value for the metadata key.
    pub fn get_metadata(&self, key: &String) -> Option<String> {
        self.metadata.get(key).cloned()
    }
}