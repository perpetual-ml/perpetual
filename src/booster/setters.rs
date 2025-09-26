use crate::booster::config::MissingNodeTreatment;
use crate::objective_functions::objective::Objective;
use crate::{constraints::ConstraintMap, PerpetualBooster};
use std::collections::HashSet;

impl PerpetualBooster {
    // Set methods for paramters

    /// Set the objective on the booster.
    /// * `objective` - The objective type of the booster.
    pub fn set_objective(mut self, objective: Objective) -> Self {
        self.cfg.objective = objective;
        self
    }

    /// Set the budget on the booster.
    /// * `budget` - Budget to fit the booster.
    pub fn set_budget(mut self, budget: f32) -> Self {
        self.cfg.budget = budget;
        self
    }

    /// Set the base_score on the booster.
    /// * `base_score` - The base score of the booster.
    pub fn set_base_score(mut self, base_score: f64) -> Self {
        self.base_score = base_score;
        self
    }

    /// Set the number of bins on the booster.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_max_bin(mut self, max_bin: u16) -> Self {
        self.cfg.max_bin = max_bin;
        self
    }

    /// Set the number of threads on the booster.
    /// * `num_threads` - Set the number of threads to be used during training.
    pub fn set_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.cfg.num_threads = num_threads;
        self
    }

    /// Set the monotone_constraints on the booster.
    /// * `monotone_constraints` - The monotone constraints of the booster.
    pub fn set_monotone_constraints(mut self, monotone_constraints: Option<ConstraintMap>) -> Self {
        self.cfg.monotone_constraints = monotone_constraints;
        self
    }

    /// Set the force_children_to_bound_parent on the booster.
    /// * `force_children_to_bound_parent` - Set force children to bound parent.
    pub fn set_force_children_to_bound_parent(mut self, force_children_to_bound_parent: bool) -> Self {
        self.cfg.force_children_to_bound_parent = force_children_to_bound_parent;
        self
    }

    /// Set missing value of the booster
    /// * `missing` - Float value to consider as missing.
    pub fn set_missing(mut self, missing: f64) -> Self {
        self.cfg.missing = missing;
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.cfg.allow_missing_splits = allow_missing_splits;
        self
    }

    /// Set create missing value of the booster
    /// * `create_missing_branch` - Bool specifying if missing should get it's own branch.
    pub fn set_create_missing_branch(mut self, create_missing_branch: bool) -> Self {
        self.cfg.create_missing_branch = create_missing_branch;
        self
    }

    /// Set the features where whose missing nodes should
    /// always be terminated.
    /// * `terminate_missing_features` - Hashset of the feature indices for the features that should always terminate the missing node, if create_missing_branch is true.
    pub fn set_terminate_missing_features(mut self, terminate_missing_features: HashSet<usize>) -> Self {
        self.cfg.terminate_missing_features = terminate_missing_features;
        self
    }

    /// Set the missing_node_treatment on the booster.
    /// * `missing_node_treatment` - The missing node treatment of the booster.
    pub fn set_missing_node_treatment(mut self, missing_node_treatment: MissingNodeTreatment) -> Self {
        self.cfg.missing_node_treatment = missing_node_treatment;
        self
    }

    /// Set the log iterations on the booster.
    /// * `log_iterations` - The number of log iterations of the booster.
    pub fn set_log_iterations(mut self, log_iterations: usize) -> Self {
        self.cfg.log_iterations = log_iterations;
        self
    }

    /// Set the log iterations on the booster.
    /// * `log_iterations` - The number of log iterations of the booster.
    pub fn set_ref_log_iterations(mut self, log_iterations: usize) -> Self {
        self.cfg.log_iterations = log_iterations;
        self
    }

    /// Set the seed on the booster.
    /// * `seed` - Integer value used to see any randomness used in the algorithm.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self
    }

    /// Set the quantile on the booster.
    /// * `quantile` - used only in quantile regression.
    pub fn set_quantile(mut self, quantile: Option<f64>) -> Self {
        self.cfg.quantile = quantile;
        self
    }

    /// Set the reset on the booster.
    /// * `reset` - Reset the model or continue training.
    pub fn set_reset(mut self, reset: Option<bool>) -> Self {
        self.cfg.reset = reset;
        self
    }

    /// Set the categorical features on the booster.
    /// * `categorical_features` - categorical features.
    pub fn set_categorical_features(mut self, categorical_features: Option<HashSet<usize>>) -> Self {
        self.cfg.categorical_features = categorical_features;
        self
    }

    /// Set the timeout on the booster.
    /// * `timeout` - fit timeout limit in seconds.
    pub fn set_timeout(mut self, timeout: Option<f32>) -> Self {
        self.cfg.timeout = timeout;
        self
    }

    /// Set the iteration limit on the booster.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    pub fn set_iteration_limit(mut self, iteration_limit: Option<usize>) -> Self {
        self.cfg.iteration_limit = iteration_limit;
        self
    }

    /// Set the memory limit on the booster.
    /// * `memory_limit` - optional limit for memory allocation.
    pub fn set_memory_limit(mut self, memory_limit: Option<f32>) -> Self {
        self.cfg.memory_limit = memory_limit;
        self
    }

    /// Set the stopping rounds on the booster.
    /// * `stopping_rounds` - optional limit for auto stopping rounds.
    pub fn set_stopping_rounds(mut self, stopping_rounds: Option<usize>) -> Self {
        self.cfg.stopping_rounds = stopping_rounds;
        self
    }
}
