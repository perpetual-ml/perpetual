//! Univariate Booster
//! 
//! 

use std::collections::{HashMap, HashSet};
use crate::bin::Bin;
use crate::binning::bin_matrix;
use crate::constants::{
    FREE_MEM_ALLOC_FACTOR, GENERALIZATION_THRESHOLD_RELAXED, ITER_LIMIT, MIN_COL_AMOUNT, N_NODES_ALLOC_MAX,
    N_NODES_ALLOC_MIN, STOPPING_ROUNDS,
};
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::PerpetualError;
use crate::histogram::{update_cuts, NodeHistogram, NodeHistogramOwned};
use crate::objective_functions::{calc_init_callables, gradient_hessian_callables, loss_callables};
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, SplitInfo, SplitInfoSlice, Splitter};
use crate::tree::tree::{Tree, TreeStopper};
use crate::booster::config::*;
use core::{f32, f64};
use log::{info, warn};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::{mem};
use sysinfo::System;
use crate::objective_functions::{ObjectiveFunction, Objective, CustomObjective};
use std::sync::Arc;

type ImportanceFn = fn(&Tree, &mut HashMap<usize, (f32, usize)>);


/// Perpetual Booster object
#[derive(Clone, Serialize, Deserialize)]
pub struct UnivariateBooster {
    pub cfg: BoosterConfig,
    pub base_score: f64,
    pub eta: f32,
    pub trees: Vec<Tree>,
    pub cal_models: HashMap<String, [(UnivariateBooster, f64); 2]>,
    pub metadata: HashMap<String, String>,
}

impl Default for UnivariateBooster {
    fn default() -> Self {
        UnivariateBooster {
            cfg: BoosterConfig::default(),
            base_score: f64::NAN,
            eta: f32::NAN,
            trees: Vec::new(),
            cal_models: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl UnivariateBooster {
    /// Perpetual Booster object
    ///
    /// * `objective` - The name of objective function used to optimize. Valid options are:
    ///      "LogLoss" to use logistic loss as the objective function,
    ///      "SquaredLoss" to use Squared Error as the objective function,
    ///      "QuantileLoss" for quantile regression.
    /// * `budget` - budget to fit the model.
    /// * `base_score` - The initial prediction value of the model. If set to None, it will be calculated based on the objective function at fit time.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///     a smaller number, will result in faster training time, while potentially sacrificing
    ///     accuracy. If there are more bins, than unique values in a column, all unique values
    ///     will be used.
    /// * `num_threads` - Number of threads to use during training
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///     between the training features and the target variable.
    /// * `force_children_to_bound_parent` - force_children_to_bound_parent.
    /// * `missing` - Value to consider missing.
    /// * `allow_missing_splits` - Should the algorithm allow splits that completed seperate out missing
    ///     and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    ///     is true, setting this to true will result in the missing branch being further split.
    /// * `create_missing_branch` - Should missing be split out its own separate branch?
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
        objective: Objective,
        budget: f32,
        base_score: f64,
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

        let cfg = BoosterConfig {
            objective,
            budget,
            max_bin,
            num_threads,
            monotone_constraints,
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
            custom_objective
        };

        let booster = UnivariateBooster {
            cfg,
            base_score,
            eta: f32::NAN,
            trees: Vec::new(),
            cal_models: HashMap::new(),
            metadata: HashMap::new(),
        };

        booster.validate_parameters()?;
        Ok(booster)
    }


    pub fn validate_parameters(&self) -> Result<(), PerpetualError> {
        Ok(())
    }

    pub fn reset(&mut self) {
        self.trees = Vec::new();
    }

    


    /// Fit the gradient booster on a provided dataset.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `y` - Either a Polars or Pandas Series, or a 1 dimensional Numpy array.
    /// * `sample_weight` - Instance weights to use when training the model.
    pub fn fit(&mut self, data: &Matrix<f64>, y: &[f64], sample_weight: Option<&[f64]>) -> Result<(), PerpetualError> {
        let constraints_map = self
            .cfg.monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();

        self.set_eta(self.cfg.budget);

        if self.cfg.create_missing_branch {
            let splitter = MissingBranchSplitter::new(
                self.eta,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.terminate_missing_features.clone(),
                self.cfg.missing_node_treatment,
                self.cfg.force_children_to_bound_parent,
            );
            self.fit_trees(data, y, &splitter, sample_weight)?;
        } else {
            let splitter = MissingImputerSplitter::new(self.eta, self.cfg.allow_missing_splits, constraints_map);
            self.fit_trees(data, y, &splitter, sample_weight)?;
        };

        Ok(())
    }

    fn fit_trees<T: Splitter>(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        splitter: &T,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {

        // NOTE: is_cost_hess is unused BUT
        // it is going to be used at some point.
        let (calc_grad_hess, calc_loss, calc_init, constant_gradient, _metric) =
            if let Some(custom) = &self.cfg.custom_objective {
                (
                    Arc::clone(&custom.grad_hess),
                    Arc::clone(&custom.loss),
                    Arc::clone(&custom.init),
                    custom.hessian_constant,
                    custom.metric,
                )
            } else {
                let inst = self.cfg.objective.instantiate();
                (
                    gradient_hessian_callables(inst.clone()),
                    loss_callables(inst.clone()),
                    calc_init_callables(inst.clone()),
                    inst.hessian_is_constant(),
                    inst.default_metric(),
                )
            };


        let start = Instant::now();

        let n_threads_available = std::thread::available_parallelism().unwrap().get();
        let num_threads = match self.cfg.num_threads {
            Some(num_threads) => num_threads,
            None => n_threads_available,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // let calc_loss = loss_callables(&self.cfg.objective);

        // If reset, reset the trees. Otherwise continue training.
        let mut yhat;
        if self.cfg.reset.unwrap_or(true) || self.trees.len() == 0 {
            self.cfg.reset;
            if self.base_score.is_nan() {
                self.base_score = calc_init(y, sample_weight);
            }
            yhat = vec![self.base_score; y.len()];
        } else {
            yhat = self.predict(data, true);
        }

        //let calc_grad_hess = gradient_hessian_callables(&self.cfg.objective);
        let (mut grad, mut hess) = calc_grad_hess(y, &yhat, sample_weight);

        let mut loss = calc_loss(y, &yhat, sample_weight);

        let loss_base = calc_loss(y, &vec![self.base_score; y.len()], sample_weight);
        let loss_avg = loss_base.iter().sum::<f32>() / loss_base.len() as f32;

        let base = 10.0_f32;
        let n = base / self.cfg.budget;
        let reciprocals_of_powers = n / (n - 1.0);
        let truncated_series_sum = reciprocals_of_powers - (1.0 + 1.0 / n);
        let c = 1.0 / n - truncated_series_sum;
        let target_loss_decrement = c * base.powf(-self.cfg.budget) * loss_avg;

        
        let is_const_hess = match sample_weight {
            Some(_sample_weight) => false,
            None => constant_gradient,
        };

        // Generate binned data
        //
        // In scikit-learn, they sample 200_000 records for generating the bins.
        // we could consider that, especially if this proved to be a large bottleneck...
        let binned_data = bin_matrix(
            data,
            sample_weight,
            self.cfg.max_bin,
            self.cfg.missing,
            self.cfg.categorical_features.as_ref(),
        )?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut stopping = 0 as usize;
        let mut n_low_loss_rounds = 0;

        let mut rng = StdRng::seed_from_u64(self.cfg.seed);

        // Column sampling is only applied when (n_rows / n_columns) < ROW_COLUMN_RATIO_LIMIT.
        // ROW_COLUMN_RATIO_LIMIT is calculated using budget.
        // budget = 1.0 -> ROW_COLUMN_RATIO_LIMIT = 100
        // budget = 2.0 -> ROW_COLUMN_RATIO_LIMIT = 10
        let row_column_ratio_limit = 10.0_f32.powf(-self.cfg.budget) * 1000.0;
        let colsample_bytree = (data.rows as f32 / data.cols as f32) / row_column_ratio_limit;

        let col_amount = (((col_index.len() as f32) * colsample_bytree).floor() as usize)
            .clamp(usize::min(MIN_COL_AMOUNT, col_index.len()), col_index.len());

        let mem_bin = mem::size_of::<Bin>();
        let mem_hist: usize;
        if col_amount == col_index.len() {
            mem_hist = mem_bin * binned_data.nunique.iter().sum::<usize>();
        } else {
            mem_hist = mem_bin * self.cfg.max_bin as usize * col_amount;
        }
        let sys = System::new_all();
        let mem_available = match self.cfg.memory_limit {
            Some(mem_limit) => mem_limit * (1e9 as f32),
            None => match sys.cgroup_limits() {
                Some(limits) => limits.free_memory as f32,
                None => sys.available_memory() as f32,
            },
        };

        let mut n_nodes_alloc: usize;
        if self.cfg.memory_limit.is_none() {
            n_nodes_alloc = (FREE_MEM_ALLOC_FACTOR * (mem_available / (mem_hist as f32))) as usize;
            n_nodes_alloc = n_nodes_alloc.clamp(N_NODES_ALLOC_MIN, N_NODES_ALLOC_MAX);
        } else {
            n_nodes_alloc = (FREE_MEM_ALLOC_FACTOR * (mem_available / (mem_hist as f32))) as usize;
        }

        let mut hist_tree_owned: Vec<NodeHistogramOwned>;
        if col_amount == col_index.len() {
            hist_tree_owned = (0..n_nodes_alloc)
                .map(|_| NodeHistogramOwned::empty_from_cuts(&binned_data.cuts, &col_index, is_const_hess, false))
                .collect();
        } else {
            hist_tree_owned = (0..n_nodes_alloc)
                .map(|_| NodeHistogramOwned::empty(self.cfg.max_bin, col_amount, is_const_hess, false))
                .collect();
        }

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_amount).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        for i in 0..self.cfg.iteration_limit.unwrap_or(ITER_LIMIT) {
            let verbose = if self.cfg.log_iterations == 0 {
                false
            } else {
                i % self.cfg.log_iterations == 0
            };

            let tld = if n_low_loss_rounds > (self.cfg.stopping_rounds.unwrap_or(STOPPING_ROUNDS) + 1) {
                None
            } else {
                Some(target_loss_decrement)
            };

            let col_index_sample: Vec<usize> = if col_amount == col_index.len() {
                Vec::new()
            } else {
                let mut v: Vec<usize> = col_index
                    .iter()
                    .choose_multiple(&mut rng, col_amount)
                    .iter()
                    .map(|i| **i)
                    .collect();
                v.sort();
                v
            };

            let col_index_fit = if col_amount == col_index.len() {
                &col_index
            } else {
                &col_index_sample
            };

            if col_amount != col_index.len() {
                hist_tree.iter().for_each(|h| {
                    update_cuts(h, col_index_fit, &binned_data.cuts, true);
                })
            }

            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                data.index.to_owned(),
                &col_index_fit,
                &mut grad,
                hess.as_deref_mut(),
                splitter,
                &pool,
                tld,
                &loss,
                y,
                calc_loss.clone(),
                &yhat,
                sample_weight,
                is_const_hess,
                &mut hist_tree,
                self.cfg.categorical_features.as_ref(),
                &split_info_slice,
                n_nodes_alloc,
            );

            self.update_predictions_inplace(&mut yhat, &tree, data);

            if tree.nodes.len() < 5 {
                let generalization = tree
                    .nodes
                    .values()
                    .map(|n| n.generalization.unwrap_or(0.0))
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);
                if generalization < GENERALIZATION_THRESHOLD_RELAXED && tree.stopper != TreeStopper::StepSize {
                    stopping += 1;
                    // If root node cannot be split due to no positive split gain, stop boosting.
                    if tree.nodes.len() == 1 {
                        break;
                    }
                }
            }

            if tree.stopper != TreeStopper::StepSize {
                n_low_loss_rounds += 1;
            } else {
                n_low_loss_rounds = 0;
            }

            (grad, hess) = calc_grad_hess(y, &yhat, sample_weight);
            loss = calc_loss(y, &yhat, sample_weight);

            if verbose {
                info!(
                    "round {:0?}, tree.nodes: {:1?}, tree.depth: {:2?}, tree.stopper: {:3?}, loss: {:4?}",
                    i,
                    tree.nodes.len(),
                    tree.depth,
                    tree.stopper,
                    loss.iter().sum::<f32>() / loss.len() as f32,
                );
            }

            self.trees.push(tree);

            if stopping >= self.cfg.stopping_rounds.unwrap_or(STOPPING_ROUNDS) {
                info!("Auto stopping since stopping round limit reached.");
                break;
            }

            if let Some(t) = self.cfg.timeout {
                if start.elapsed().as_secs_f32() > t {
                    warn!("Reached timeout limit before auto stopping. Try to decrease the budget or increase the timeout for the best performance.");
                    break;
                }
            }

            if i == self.cfg.iteration_limit.unwrap_or(ITER_LIMIT) - 1 {
                warn!("Reached iteration limit before auto stopping. Try to decrease the budget for the best performance.");
            }
        }

        if self.cfg.log_iterations > 0 {
            info!(
                "Finished training a booster with {0} trees in {1} seconds.",
                self.trees.len(),
                start.elapsed().as_secs()
            );
        }

        Ok(())
    }

    fn update_predictions_inplace(&self, yhat: &mut [f64], tree: &Tree, data: &Matrix<f64>) {
        let preds = tree.predict(data, true, &self.cfg.missing);
        yhat.iter_mut().zip(preds).for_each(|(i, j)| *i += j);
    }

    /// Set model fitting eta which is step size to use at each iteration.
    /// Each leaf weight is multiplied by this number.
    /// The smaller the value, the more conservative the weights will be.
    /// * `budget` - A positive number for fitting budget.
    pub fn set_eta(&mut self, budget: f32) {
        let budget = f32::max(0.0, budget);
        let power = budget * -1.0;
        let base = 10_f32;
        self.eta = base.powf(power);
    }

    /// Get reference to the trees
    pub fn get_prediction_trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        let pd: f64 = if true {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.value_partial_dependence(feature, value, &self.cfg.missing))
                .sum()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.value_partial_dependence(feature, value, &self.cfg.missing))
                .sum()
        };
        pd + self.base_score
    }

    /// Calculate feature importance measure for the features
    /// in the model.
    /// - `method`: variable importance method to use.
    /// - `n_features`: The number of features to calculate the importance for.
    pub fn calculate_feature_importance(&self, method: ImportanceMethod, normalize: bool) -> HashMap<usize, f32> {
        let (average, importance_fn): (bool, ImportanceFn) = match method {
            ImportanceMethod::Weight => (false, Tree::calculate_importance_weight),
            ImportanceMethod::Gain => (true, Tree::calculate_importance_gain),
            ImportanceMethod::TotalGain => (false, Tree::calculate_importance_gain),
            ImportanceMethod::Cover => (true, Tree::calculate_importance_cover),
            ImportanceMethod::TotalCover => (false, Tree::calculate_importance_cover),
        };
        let mut stats = HashMap::new();
        for tree in self.trees.iter() {
            importance_fn(tree, &mut stats)
        }

        let importance = stats
            .iter()
            .map(|(k, (v, c))| if average { (*k, v / (*c as f32)) } else { (*k, *v) })
            .collect::<HashMap<usize, f32>>();

        if normalize {
            // To make deterministic, sort values and then sum.
            // Otherwise we were getting them in different orders, and
            // floating point error was creeping in.
            let mut values: Vec<f32> = importance.values().copied().collect();
            // We are OK to unwrap because we know we will never have missing.
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let total: f32 = values.iter().sum();
            importance.iter().map(|(k, v)| (*k, v / total)).collect()
        } else {
            importance
        }
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
