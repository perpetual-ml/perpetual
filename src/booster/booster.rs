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
use crate::objective_functions::{calc_init_callables, gradient_hessian_callables, loss_callables, Objective};
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, SplitInfo, SplitInfoSlice, Splitter};
use crate::tree::tree::{Tree, TreeStopper};
use core::{f32, f64};
use log::{info, warn};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use std::{fs, mem};
use sysinfo::System;

type ImportanceFn = fn(&Tree, &mut HashMap<usize, (f32, usize)>);

#[derive(Serialize, Deserialize)]
pub enum ContributionsMethod {
    /// This method will use the internal leaf weights, to calculate the contributions. This is the same as what is described by Saabas [here](https://blog.datadive.net/interpreting-random-forests/).
    Weight,
    /// If this option is specified, the average internal node values are calculated, this is equivalent to the `approx_contribs` parameter in XGBoost.
    Average,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the other non-missing branch. This method does not have the property where the contributions summed is equal to the final prediction of the model.
    BranchDifference,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the mid-point between the right and left node weighted by the cover of each node. This method does not have the property where the contributions summed is equal to the final prediction of the model.
    MidpointDifference,
    /// This method will calculate contributions by subtracting the weight of the node the record will travel down by the weight of the node with the largest cover (the mode node). This method does not have the property where the contributions summed is equal to the final prediction of the model.
    ModeDifference,
    /// This method is only valid when the objective type is set to "LogLoss". This method will calculate contributions as the change in a records probability of being 1 moving from a parent node to a child node. The sum of the returned contributions matrix, will be equal to the probability a record will be 1. For example, given a model, `model.predict_contributions(X, method="ProbabilityChange") == 1 / (1 + np.exp(-model.predict(X)))`
    ProbabilityChange,
    /// This method computes the Shapley values for each record, and feature.
    Shapley,
}

/// Method to calculate variable importance.
#[derive(Serialize, Deserialize, Clone)]
pub enum ImportanceMethod {
    /// The number of times a feature is used to split the data across all trees.
    Weight,
    /// The average split gain across all splits the feature is used in.
    Gain,
    /// The average coverage across all splits the feature is used in.
    Cover,
    /// The total gain across all splits the feature is used in.
    TotalGain,
    /// The total coverage across all splits the feature is used in.
    TotalCover,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum MissingNodeTreatment {
    /// Calculate missing node weight values without any constraints.
    None,
    /// Assign the weight of the missing node to that of the parent.
    AssignToParent,
    /// After training each tree, starting from the bottom of the tree, assign the missing node weight to the weighted average of the left and right child nodes. Next assign the parent to the weighted average of the children nodes. This is performed recursively up through the entire tree. This is performed as a post processing step on each tree after it is built, and prior to updating the predictions for which to train the next tree.
    AverageLeafWeight,
    /// Set the missing node to be equal to the weighted average weight of the left and the right nodes.
    AverageNodeWeight,
}

/// Perpetual Booster object
#[derive(Deserialize, Serialize, Clone)]
pub struct PerpetualBooster {
    /// The name of objective function used to optimize. Valid options are:
    /// "LogLoss" to use logistic loss as the objective function,
    /// "SquaredLoss" to use Squared Error as the objective function,
    /// "QuantileLoss" for quantile regression.
    pub objective: Objective,
    /// Budget to fit the model.
    #[serde(default = "default_budget")]
    pub budget: f32,
    /// The initial prediction value of the model. Calculated from y and sample_weight if nan.
    pub base_score: f64,
    /// Number of bins to calculate to partition the data. Setting this to
    /// a smaller number, will result in faster training time, while potentially sacrificing
    /// accuracy. If there are more bins, than unique values in a column, all unique values
    /// will be used.
    pub max_bin: u16,
    /// Number of threads to use during training.
    pub num_threads: Option<usize>,
    /// Constraints that are used to enforce a specific relationship
    /// between the training features and the target variable.
    pub monotone_constraints: Option<ConstraintMap>,
    /// Should the children nodes contain the parent node in their bounds, setting this to true, will result in no children being created that result in the higher and lower child values both being greater than, or less than the parent weight.
    #[serde(default = "default_force_children_to_bound_parent")]
    pub force_children_to_bound_parent: bool,
    /// Value to consider missing.
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    /// Should the algorithm allow splits that completed seperate out missing
    /// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    /// is true, setting this to true will result in the missin branch being further split.
    pub allow_missing_splits: bool,
    /// Should missing be split out it's own separate branch?
    pub create_missing_branch: bool,
    /// A set of features for which the missing node will always be terminated, even
    /// if `allow_missing_splits` is set to true. This value is only valid if
    /// `create_missing_branch` is also True.
    #[serde(default = "default_terminate_missing_features")]
    pub terminate_missing_features: HashSet<usize>,
    /// How the missing nodes weights should be treated at training time.
    #[serde(default = "default_missing_node_treatment")]
    pub missing_node_treatment: MissingNodeTreatment,
    /// Should the model be trained showing output.
    #[serde(default = "default_log_iterations")]
    pub log_iterations: usize,
    // Members internal to the booster object, and not parameters set by the user.
    // Trees is public, just to interact with it directly in the python wrapper.
    pub trees: Vec<Tree>,
    // Metadata for the booster
    metadata: HashMap<String, String>,
    /// Step size to use at each iteration. Each
    /// leaf weight is multiplied by this number. The smaller the value, the more
    /// conservative the weights will be.
    eta: f32,
    /// Integer value used to seed any randomness used in the algorithm.
    pub seed: u64,
    /// Used only in quantile regression.
    #[serde(default = "default_quantile")]
    pub quantile: Option<f64>,
    /// Reset the model or continue training.
    #[serde(default = "default_reset")]
    pub reset: Option<bool>,
    /// Features to be treated as categorical.
    #[serde(default = "default_categorical_features")]
    pub categorical_features: Option<HashSet<usize>>,
    /// Fit timeout limit in seconds.
    #[serde(default = "default_timeout")]
    pub timeout: Option<f32>,
    /// Optional limit for the number of boosting rounds.
    /// The algorithm will stop automatically before this limit if budget is low enough.
    #[serde(default = "default_iteration_limit")]
    pub iteration_limit: Option<usize>,
    /// Optional limit for memory allocation.
    /// This will limit the number of allocated nodes for a tree.
    /// The number of nodes in a final tree will be limited by this,
    /// if it is not limited by step size control and generalization control.
    #[serde(default = "default_memory_limit")]
    pub memory_limit: Option<f32>,
    /// Optional limit for auto stopping rounds.
    #[serde(default = "default_stopping_rounds")]
    pub stopping_rounds: Option<usize>,
    /// Calibration models for conformal prediction. Created with `calibrate` method.
    #[serde(default = "default_cal_models")]
    pub(crate) cal_models: HashMap<String, [(PerpetualBooster, f64); 2]>,
}

fn default_cal_models() -> HashMap<String, [(PerpetualBooster, f64); 2]> {
    HashMap::new()
}
fn default_budget() -> f32 {
    0.5
}
fn default_quantile() -> Option<f64> {
    None
}
fn default_reset() -> Option<bool> {
    None
}
fn default_categorical_features() -> Option<HashSet<usize>> {
    None
}
fn default_timeout() -> Option<f32> {
    None
}
fn default_iteration_limit() -> Option<usize> {
    None
}
fn default_memory_limit() -> Option<f32> {
    None
}
fn default_stopping_rounds() -> Option<usize> {
    None
}
fn default_terminate_missing_features() -> HashSet<usize> {
    HashSet::new()
}
fn default_missing_node_treatment() -> MissingNodeTreatment {
    MissingNodeTreatment::AssignToParent
}
fn default_log_iterations() -> usize {
    0
}
fn default_force_children_to_bound_parent() -> bool {
    false
}
fn parse_missing<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or(f64::NAN))
}

impl Default for PerpetualBooster {
    fn default() -> Self {
        Self::new(
            Objective::LogLoss,
            0.5,
            f64::NAN,
            256,
            None,
            None,
            false,
            f64::NAN,
            true,
            false,
            HashSet::new(),
            MissingNodeTreatment::AssignToParent,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }
}

impl PerpetualBooster {
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
    ) -> Result<Self, PerpetualError> {
        let booster = PerpetualBooster {
            objective,
            budget,
            base_score,
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
            trees: Vec::new(),
            metadata: HashMap::new(),
            eta: f32::NAN,
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
            cal_models: HashMap::new(),
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
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();

        self.set_eta(self.budget);

        if self.create_missing_branch {
            let splitter = MissingBranchSplitter::new(
                self.eta,
                self.allow_missing_splits,
                constraints_map,
                self.terminate_missing_features.clone(),
                self.missing_node_treatment,
                self.force_children_to_bound_parent,
            );
            self.fit_trees(data, y, &splitter, sample_weight)?;
        } else {
            let splitter = MissingImputerSplitter::new(self.eta, self.allow_missing_splits, constraints_map);
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
        let start = Instant::now();

        let n_threads_available = std::thread::available_parallelism().unwrap().get();
        let num_threads = match self.num_threads {
            Some(num_threads) => num_threads,
            None => n_threads_available,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let calc_loss = loss_callables(&self.objective);

        // If reset, reset the trees. Otherwise continue training.
        let mut yhat;
        if self.reset.unwrap_or(true) || self.trees.len() == 0 {
            self.reset();
            if self.base_score.is_nan() {
                self.base_score = calc_init_callables(&self.objective)(y, sample_weight, self.quantile);
            }
            yhat = vec![self.base_score; y.len()];
        } else {
            yhat = self.predict(data, true);
        }

        let calc_grad_hess = gradient_hessian_callables(&self.objective);
        let (mut grad, mut hess) = calc_grad_hess(y, &yhat, sample_weight, self.quantile);

        let mut loss = calc_loss(y, &yhat, sample_weight, self.quantile);

        let loss_base = calc_loss(y, &vec![self.base_score; y.len()], sample_weight, self.quantile);
        let loss_avg = loss_base.iter().sum::<f32>() / loss_base.len() as f32;

        let base = 10.0_f32;
        let n = base / self.budget;
        let reciprocals_of_powers = n / (n - 1.0);
        let truncated_series_sum = reciprocals_of_powers - (1.0 + 1.0 / n);
        let c = 1.0 / n - truncated_series_sum;
        let target_loss_decrement = c * base.powf(-self.budget) * loss_avg;

        let is_const_hess = match sample_weight {
            Some(_sample_weight) => false,
            None => match &self.objective {
                Objective::LogLoss => false,
                Objective::AdaptiveHuberLoss => false,
                Objective::HuberLoss => false,
                _ => true,
            },
        };

        // Generate binned data
        //
        // In scikit-learn, they sample 200_000 records for generating the bins.
        // we could consider that, especially if this proved to be a large bottleneck...
        let binned_data = bin_matrix(
            data,
            sample_weight,
            self.max_bin,
            self.missing,
            self.categorical_features.as_ref(),
        )?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut stopping = 0 as usize;
        let mut n_low_loss_rounds = 0;

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Column sampling is only applied when (n_rows / n_columns) < ROW_COLUMN_RATIO_LIMIT.
        // ROW_COLUMN_RATIO_LIMIT is calculated using budget.
        // budget = 1.0 -> ROW_COLUMN_RATIO_LIMIT = 100
        // budget = 2.0 -> ROW_COLUMN_RATIO_LIMIT = 10
        let row_column_ratio_limit = 10.0_f32.powf(-self.budget) * 1000.0;
        let colsample_bytree = (data.rows as f32 / data.cols as f32) / row_column_ratio_limit;

        let col_amount = (((col_index.len() as f32) * colsample_bytree).floor() as usize)
            .clamp(usize::min(MIN_COL_AMOUNT, col_index.len()), col_index.len());

        let mem_bin = mem::size_of::<Bin>();
        let mem_hist: usize;
        if col_amount == col_index.len() {
            mem_hist = mem_bin * binned_data.nunique.iter().sum::<usize>();
        } else {
            mem_hist = mem_bin * self.max_bin as usize * col_amount;
        }
        let sys = System::new_all();
        let mem_available = match self.memory_limit {
            Some(mem_limit) => mem_limit * (1e9 as f32),
            None => match sys.cgroup_limits() {
                Some(limits) => limits.free_memory as f32,
                None => sys.available_memory() as f32,
            },
        };

        let mut n_nodes_alloc: usize;
        if self.memory_limit.is_none() {
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
                .map(|_| NodeHistogramOwned::empty(self.max_bin, col_amount, is_const_hess, false))
                .collect();
        }

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_amount).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        for i in 0..self.iteration_limit.unwrap_or(ITER_LIMIT) {
            let verbose = if self.log_iterations == 0 {
                false
            } else {
                i % self.log_iterations == 0
            };

            let tld = if n_low_loss_rounds > (self.stopping_rounds.unwrap_or(STOPPING_ROUNDS) + 1) {
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
                calc_loss,
                &yhat,
                sample_weight,
                self.quantile,
                is_const_hess,
                &mut hist_tree,
                self.categorical_features.as_ref(),
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

            (grad, hess) = calc_grad_hess(y, &yhat, sample_weight, self.quantile);
            loss = calc_loss(y, &yhat, sample_weight, self.quantile);

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

            if stopping >= self.stopping_rounds.unwrap_or(STOPPING_ROUNDS) {
                info!("Auto stopping since stopping round limit reached.");
                break;
            }

            if let Some(t) = self.timeout {
                if start.elapsed().as_secs_f32() > t {
                    warn!("Reached timeout limit before auto stopping. Try to decrease the budget or increase the timeout for the best performance.");
                    break;
                }
            }

            if i == self.iteration_limit.unwrap_or(ITER_LIMIT) - 1 {
                warn!("Reached iteration limit before auto stopping. Try to decrease the budget for the best performance.");
            }
        }

        if self.log_iterations > 0 {
            info!(
                "Finished training a booster with {0} trees in {1} seconds.",
                self.trees.len(),
                start.elapsed().as_secs()
            );
        }

        Ok(())
    }

    fn update_predictions_inplace(&self, yhat: &mut [f64], tree: &Tree, data: &Matrix<f64>) {
        let preds = tree.predict(data, true, &self.missing);
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
                .map(|t| t.value_partial_dependence(feature, value, &self.missing))
                .sum()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.value_partial_dependence(feature, value, &self.missing))
                .sum()
        };
        pd + self.base_score
    }

    /// Calculate feature importance measure for the features
    /// in the model.
    /// - `method`: variable importance method to use.
    /// - `normalize`: whether to normalize the importance values with the sum.
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

    /// Save a booster as a json object to a file.
    ///
    /// * `path` - Path to save booster.
    pub fn save_booster(&self, path: &str) -> Result<(), PerpetualError> {
        let model = self.json_dump()?;
        match fs::write(path, model) {
            Err(e) => Err(PerpetualError::UnableToWrite(e.to_string())),
            Ok(_) => Ok(()),
        }
    }

    /// Dump a booster as a json object
    pub fn json_dump(&self) -> Result<String, PerpetualError> {
        match serde_json::to_string(self) {
            Ok(s) => Ok(s),
            Err(e) => Err(PerpetualError::UnableToWrite(e.to_string())),
        }
    }

    /// Load a booster from Json string
    ///
    /// * `json_str` - String object, which can be serialized to json.
    pub fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        let model = serde_json::from_str::<PerpetualBooster>(json_str);
        match model {
            Ok(m) => Ok(m),
            Err(e) => Err(PerpetualError::UnableToRead(e.to_string())),
        }
    }

    /// Load a booster from a path to a json booster object.
    ///
    /// * `path` - Path to load booster from.
    pub fn load_booster(path: &str) -> Result<Self, PerpetualError> {
        let json_str = match fs::read_to_string(path) {
            Ok(s) => Ok(s),
            Err(e) => Err(PerpetualError::UnableToRead(e.to_string())),
        }?;
        Self::from_json(&json_str)
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

#[cfg(test)]
mod tests {
    use crate::utils::between;

    use super::*;
    use approx::assert_relative_eq;
    use polars::io::SerReader;
    use polars::prelude::{CsvReadOptions, DataType};
    use std::error::Error;
    use std::fs;
    use std::sync::Arc;

    #[test]
    fn test_booster_fit_subsample() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        let mut booster = PerpetualBooster::default()
            .set_max_bin(300)
            .set_base_score(0.5)
            .set_budget(0.3);
        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);

        let mut booster = PerpetualBooster::default().set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit_no_fitted_base_score() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance-fare.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);

        let mut booster = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(300)
            .set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_tree_save() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, 891, 5);

        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = PerpetualBooster::default()
            .set_max_bin(300)
            .set_base_score(0.5)
            .set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, true);

        booster.save_booster("resources/model64.json").unwrap();
        let booster2 = PerpetualBooster::load_booster("resources/model64.json").unwrap();
        assert_eq!(booster2.predict(&data, true)[0..10], preds[0..10]);

        // Test with non-NAN missing.
        booster.missing = 0.0;
        booster.save_booster("resources/modelmissing.json").unwrap();
        let booster3 = PerpetualBooster::load_booster("resources/modelmissing.json").unwrap();
        assert_eq!(booster3.missing, 0.);
        assert_eq!(booster3.missing, booster.missing);
    }

    #[test]
    fn test_gbm_categorical() -> Result<(), Box<dyn Error>> {
        let n_columns = 13;

        let file = fs::read_to_string("resources/titanic_test_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file =
            fs::read_to_string("resources/titanic_test_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let mut booster = PerpetualBooster::default()
            .set_budget(0.1)
            .set_categorical_features(Some(cat_index));

        booster.fit(&data, &y, None).unwrap();

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let probabilities = booster.predict_proba(&data, true);

        let accuracy = probabilities
            .iter()
            .zip(y.iter())
            .map(|(p, y)| if p.round() == *y { 1 } else { 0 })
            .sum::<usize>() as f32
            / y.len() as f32;

        println!("accuracy: {}", accuracy);
        assert!(between(0.76, 0.78, accuracy));

        Ok(())
    }

    #[test]
    fn test_gbm_parallel() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_train = Arc::new(all_names.clone());
        let column_names_test = Arc::new(all_names.clone());

        let df_train = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_train))
            .try_into_reader_with_file_path(Some("resources/cal_housing_train.csv".into()))?
            .finish()
            .unwrap();

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...
        let id_vars_train: Vec<&str> = Vec::new();
        let mdf_train = df_train.unpivot(feature_names.clone(), &id_vars_train)?;
        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_train = Vec::from_iter(
            mdf_train
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_train = Vec::from_iter(
            df_train
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_train = Matrix::new(&data_train, y_train.len(), 8);
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model1 = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_num_threads(Some(1))
            .set_budget(0.1);
        let mut model2 = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_num_threads(Some(2))
            .set_budget(0.1);

        model1.fit(&matrix_test, &y_test, None)?;
        model2.fit(&matrix_test, &y_test, None)?;

        let trees1 = model1.get_prediction_trees();
        let trees2 = model2.get_prediction_trees();
        assert_eq!(trees1.len(), trees2.len());

        let n_leaves1: usize = trees1.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
        let n_leaves2: usize = trees2.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
        assert_eq!(n_leaves1, n_leaves2);

        println!("{}", trees1.last().unwrap());
        println!("{}", trees2.last().unwrap());

        let y_pred1 = model1.predict(&matrix_train, true);
        let y_pred2 = model2.predict(&matrix_train, true);

        let mse1 = y_pred1
            .iter()
            .zip(y_train.iter())
            .map(|(y1, y2)| (y1 - y2) * (y1 - y2))
            .sum::<f64>()
            / y_train.len() as f64;
        let mse2 = y_pred2
            .iter()
            .zip(y_train.iter())
            .map(|(y1, y2)| (y1 - y2) * (y1 - y2))
            .sum::<f64>()
            / y_train.len() as f64;
        assert_relative_eq!(mse1, mse2, max_relative = 0.99);

        Ok(())
    }

    #[test]
    fn test_gbm_sensory() -> Result<(), Box<dyn Error>> {
        let n_columns = 11;
        let iter_limit = 10;

        let file = fs::read_to_string("resources/sensory_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/sensory_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let cat_index = HashSet::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let mut booster = PerpetualBooster::default()
            .set_log_iterations(1)
            .set_objective(Objective::SquaredLoss)
            .set_categorical_features(Some(cat_index))
            .set_iteration_limit(Some(iter_limit))
            .set_memory_limit(Some(0.00003))
            .set_budget(1.0);

        booster.fit(&data, &y, None).unwrap();

        let split_features_test = vec![6, 6, 6, 1, 6, 1, 6, 9, 1, 6];
        let split_gains_test = vec![
            31.172100067138672,
            25.249399185180664,
            20.45199966430664,
            17.50349998474121,
            16.566099166870117,
            14.345199584960938,
            13.418600082397461,
            12.505200386047363,
            12.23270034790039,
            10.869000434875488,
        ];
        for (i, tree) in booster.get_prediction_trees().iter().enumerate() {
            let nodes = &tree.nodes;
            let root_node = nodes.get(&0).unwrap();
            println!("i: {}", i);
            println!("nodes.len: {}", nodes.len());
            println!("root_node.split_feature: {}", root_node.split_feature);
            println!("root_node.split_gain: {}", root_node.split_gain);
            assert_eq!(3, nodes.len());
            assert_eq!(root_node.split_feature, split_features_test[i]);
            assert_relative_eq!(root_node.split_gain, split_gains_test[i], max_relative = 0.99);
        }
        assert_eq!(iter_limit, booster.get_prediction_trees().len());

        let pred_nodes = booster.predict_nodes(&data, true);
        println!("pred_nodes.len: {}", pred_nodes.len());
        assert_eq!(booster.get_prediction_trees().len(), pred_nodes.len());
        assert_eq!(data.rows, pred_nodes[0].len());

        Ok(())
    }

    #[test]
    fn test_adaptive_huber_loss() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_test = Arc::new(all_names.clone());

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...

        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::AdaptiveHuberLoss)
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 31);

        Ok(())
    }

    #[test]
    fn test_huber_loss() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_test = Arc::new(all_names.clone());

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...

        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::HuberLoss)
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 45);

        Ok(())
    }
}
