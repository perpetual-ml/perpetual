use crate::binning::bin_matrix;
use crate::constraints::ConstraintMap;
use crate::data::Matrix;
use crate::errors::ForustError;
use crate::grower::GrowPolicy;
use crate::histogram::HistogramMatrix;
use crate::metric::Metric;
use crate::objective::{calc_init_callables, gradient_hessian_callables, loss_callables, Objective};
use crate::sampler::SampleMethod;
use crate::shapley::predict_contributions_row_shapley;
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, Splitter};
use crate::tree::{Tree, TreeStopper};
use crate::utils::{odds, validate_positive_float_field};
use hashbrown::HashMap as BrownHashMap;
use log::info;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;

pub type CalData<'a> = (Matrix<'a, f64>, &'a [f64], &'a [f32]); // (x_flat_data, rows, cols), y, alpha
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
#[derive(Serialize, Deserialize)]
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
#[derive(Deserialize, Serialize)]
pub struct PerpetualBooster {
    /// The name of objective function used to optimize.
    /// Valid options include "LogLoss" to use logistic loss as the objective function,
    /// or "SquaredLoss" to use Squared Error as the objective function.
    pub objective: Objective,
    /// Total number of trees to train in the ensemble.
    iterations: usize,
    /// Step size to use at each iteration. Each
    /// leaf weight is multiplied by this number. The smaller the value, the more
    /// conservative the weights will be.
    learning_rate: f32,
    /// Maximum depth of an individual tree. Valid values are 0 to infinity.
    max_depth: usize,
    /// Maximum number of leaves allowed on a tree. Valid values
    /// are 0 to infinity. This is the total number of final nodes.
    max_leaves: usize,
    /// L1 regularization term applied to the weights of the tree. Valid values
    /// are 0 to infinity. 0 Means no regularization applied.
    #[serde(default = "default_l1")]
    l1: f32,
    /// L2 regularization term applied to the weights of the tree. Valid values
    /// are 0 to infinity.
    l2: f32,
    /// The minimum amount of loss required to further split a node.
    /// Valid values are 0 to infinity.
    gamma: f32,
    /// Maximum delta step allowed at each leaf. This is the maximum magnitude a leaf can take. Setting to 0 results in no constrain.
    #[serde(default = "default_max_delta_step")]
    max_delta_step: f32,
    /// Minimum sum of the hessian values of the loss function
    /// required to be in a node.
    min_leaf_weight: f32,
    /// The initial prediction value of the model.
    pub base_score: f64,
    /// Number of bins to calculate to partition the data. Setting this to
    /// a smaller number, will result in faster training time, while potentially sacrificing
    /// accuracy. If there are more bins, than unique values in a column, all unique values
    /// will be used.
    pub nbins: u16,
    pub parallel: bool,
    /// Should the algorithm allow splits that completed seperate out missing
    /// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    /// is true, setting this to true will result in the missin branch being further split.
    pub allow_missing_splits: bool,
    /// Constraints that are used to enforce a specific relationship
    /// between the training features and the target variable.
    pub monotone_constraints: Option<ConstraintMap>,
    /// Percent of records to randomly sample at each iteration when training a tree.
    subsample: f32,
    /// Used only in goss. The retain ratio of large gradient data.
    #[serde(default = "default_top_rate")]
    top_rate: f64,
    /// Used only in goss. the retain ratio of small gradient data.
    #[serde(default = "default_other_rate")]
    other_rate: f64,
    /// Specify the fraction of columns that should be sampled at each iteration, valid values are in the range (0.0,1.0].
    #[serde(default = "default_colsample_bytree")]
    colsample_bytree: f64,
    /// Integer value used to seed any randomness used in the algorithm.
    seed: u64,
    /// Value to consider missing.
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    /// Should missing be split out it's own separate branch?
    pub create_missing_branch: bool,
    /// Specify the method that records should be sampled when training?
    #[serde(default = "default_sample_method")]
    sample_method: SampleMethod,
    /// Growth policy to use when training a tree, this is how the next node is selected.
    #[serde(default = "default_grow_policy")]
    grow_policy: GrowPolicy,
    /// Define the evaluation metric to record at each iterations.
    #[serde(default = "default_evaluation_metric")]
    evaluation_metric: Option<Metric>,
    /// Number of rounds where the evaluation metric value must improve in
    /// to keep training.
    #[serde(default = "default_early_stopping_rounds")]
    early_stopping_rounds: usize,
    /// If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
    #[serde(default = "default_initialize_base_score")]
    initialize_base_score: bool,
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
    /// Should the children nodes contain the parent node in their bounds, setting this to true, will result in no children being created that result in the higher and lower child values both being greater than, or less than the parent weight.
    #[serde(default = "default_force_children_to_bound_parent")]
    pub force_children_to_bound_parent: bool,
    // Members internal to the booster object, and not parameters set by the user.
    // Trees is public, just to interact with it directly in the python wrapper.
    pub trees: Vec<Tree>,
    // Metadata for the booster
    metadata: HashMap<String, String>,
    // Growth control
    growth_control: f32,
}

fn default_l1() -> f32 {
    0.0
}
fn default_max_delta_step() -> f32 {
    0.0
}

fn default_initialize_base_score() -> bool {
    false
}

fn default_grow_policy() -> GrowPolicy {
    GrowPolicy::LossGuide
}

fn default_top_rate() -> f64 {
    0.1
}
fn default_other_rate() -> f64 {
    0.2
}
fn default_sample_method() -> SampleMethod {
    SampleMethod::None
}
fn default_evaluation_metric() -> Option<Metric> {
    None
}
fn default_early_stopping_rounds() -> usize {
    3
}
fn default_terminate_missing_features() -> HashSet<usize> {
    HashSet::new()
}
fn default_colsample_bytree() -> f64 {
    1.0
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
            100000,
            0.1,
            99,
            usize::MAX,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            256,
            true,
            true,
            None,
            1.0,
            0.1,
            0.2,
            1.0,
            0,
            f64::NAN,
            false,
            SampleMethod::None,
            GrowPolicy::LossGuide,
            None,
            3,
            true,
            HashSet::new(),
            MissingNodeTreatment::AssignToParent,
            0,
            false,
            0.1,
        )
        .unwrap()
    }
}

impl PerpetualBooster {
    /// Perpetual Booster object
    ///
    /// * `objective` - The name of objective function used to optimize.
    ///   Valid options include "LogLoss" to use logistic loss as the objective function,
    ///   or "SquaredLoss" to use Squared Error as the objective function.
    /// * `iterations` - Total number of trees to train in the ensemble.
    /// * `learning_rate` - Step size to use at each iteration. Each
    ///   leaf weight is multiplied by this number. The smaller the value, the more
    ///   conservative the weights will be.
    /// * `max_depth` - Maximum depth of an individual tree. Valid values
    ///   are 0 to infinity.
    /// * `max_leaves` - Maximum number of leaves allowed on a tree. Valid values
    ///   are 0 to infinity. This is the total number of final nodes.
    /// * `l2` - L2 regularization term applied to the weights of the tree. Valid values
    ///   are 0 to infinity.
    /// * `gamma` - The minimum amount of loss required to further split a node.
    ///   Valid values are 0 to infinity.
    /// * `min_leaf_weight` - Minimum sum of the hessian values of the loss function
    ///   required to be in a node.
    /// * `base_score` - The initial prediction value of the model. If set to None the parameter `initialize_base_score` will automatically be set to `true`, in which case the base score will be chosen based on the objective function at fit time.
    /// * `nbins` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    /// * `parallel` - Should the algorithm be run in parallel?
    /// * `allow_missing_splits` - Should the algorithm allow splits that completed seperate out missing
    /// and non-missing values, in the case where `create_missing_branch` is false. When `create_missing_branch`
    /// is true, setting this to true will result in the missin branch being further split.
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///   between the training features and the target variable.
    /// * `subsample` - Percent of records to randomly sample at each iteration when training a tree.
    /// * `top_rate` - Used only in goss. The retain ratio of large gradient data.
    /// * `other_rate` - Used only in goss. the retain ratio of small gradient data.
    /// * `colsample_bytree` - Specify the fraction of columns that should be sampled at each iteration, valid values are in the range (0.0,1.0].
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `missing` - Value to consider missing.
    /// * `create_missing_branch` - Should missing be split out it's own separate branch?
    /// * `sample_method` - Specify the method that records should be sampled when training?
    /// * `evaluation_metric` - Define the evaluation metric to record at each iterations.
    /// * `early_stopping_rounds` - Number of rounds that must
    /// * `initialize_base_score` - If this is specified, the base_score will be calculated using the sample_weight and y data in accordance with the requested objective_type.
    /// * `missing_node_treatment` - specify how missing nodes should be handled during training.
    /// * `log_iterations` - Setting to a value (N) other than zero will result in information being logged about ever N iterations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        objective: Objective,
        iterations: usize,
        learning_rate: f32,
        max_depth: usize,
        max_leaves: usize,
        l1: f32,
        l2: f32,
        gamma: f32,
        max_delta_step: f32,
        min_leaf_weight: f32,
        base_score: f64,
        nbins: u16,
        parallel: bool,
        allow_missing_splits: bool,
        monotone_constraints: Option<ConstraintMap>,
        subsample: f32,
        top_rate: f64,
        other_rate: f64,
        colsample_bytree: f64,
        seed: u64,
        missing: f64,
        create_missing_branch: bool,
        sample_method: SampleMethod,
        grow_policy: GrowPolicy,
        evaluation_metric: Option<Metric>,
        early_stopping_rounds: usize,
        initialize_base_score: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: MissingNodeTreatment,
        log_iterations: usize,
        force_children_to_bound_parent: bool,
        growth_control: f32,
    ) -> Result<Self, ForustError> {
        let booster = PerpetualBooster {
            objective,
            iterations,
            learning_rate,
            max_depth,
            max_leaves,
            l1,
            l2,
            gamma,
            max_delta_step,
            min_leaf_weight,
            base_score,
            nbins,
            parallel,
            allow_missing_splits,
            monotone_constraints,
            subsample,
            top_rate,
            other_rate,
            colsample_bytree,
            seed,
            missing,
            create_missing_branch,
            sample_method,
            grow_policy,
            evaluation_metric,
            early_stopping_rounds,
            initialize_base_score,
            terminate_missing_features,
            missing_node_treatment,
            log_iterations,
            force_children_to_bound_parent,
            trees: Vec::new(),
            metadata: HashMap::new(),
            growth_control,
        };
        booster.validate_parameters()?;
        Ok(booster)
    }

    pub fn validate_parameters(&self) -> Result<(), ForustError> {
        validate_positive_float_field!(self.learning_rate);
        validate_positive_float_field!(self.l1);
        validate_positive_float_field!(self.l2);
        validate_positive_float_field!(self.gamma);
        validate_positive_float_field!(self.max_delta_step);
        validate_positive_float_field!(self.min_leaf_weight);
        validate_positive_float_field!(self.subsample);
        validate_positive_float_field!(self.top_rate);
        validate_positive_float_field!(self.other_rate);
        validate_positive_float_field!(self.colsample_bytree);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.trees = Vec::new();
    }

    /// Fit the gradient booster on a provided dataset.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `y` - Either a pandas Series, or a 1 dimensional numpy array.
    /// * `sample_weight` - Instance weights to use when training the model.
    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f32>,
        budget: f32,
        reset: Option<bool>,
        categorical_features: Option<&[u64]>,
    ) -> Result<(), ForustError> {
        let constraints_map = self
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();

        self.set_budget(budget);

        if self.create_missing_branch {
            let splitter = MissingBranchSplitter {
                l1: self.l1,
                l2: self.l2,
                max_delta_step: self.max_delta_step,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
                terminate_missing_features: self.terminate_missing_features.clone(),
                missing_node_treatment: self.missing_node_treatment,
                force_children_to_bound_parent: self.force_children_to_bound_parent,
            };
            self.fit_trees(y, sample_weight, data, &splitter, quantile, reset, categorical_features)?;
        } else {
            let splitter = MissingImputerSplitter {
                l1: self.l1,
                l2: self.l2,
                max_delta_step: self.max_delta_step,
                gamma: self.gamma,
                min_leaf_weight: self.min_leaf_weight,
                learning_rate: self.learning_rate,
                allow_missing_splits: self.allow_missing_splits,
                constraints_map,
            };
            self.fit_trees(y, sample_weight, data, &splitter, quantile, reset, categorical_features)?;
        };

        Ok(())
    }

    fn fit_trees<T: Splitter>(
        &mut self,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        data: &Matrix<f64>,
        splitter: &T,
        alpha: Option<f32>,
        reset: Option<bool>,
        categorical_features: Option<&[u64]>,
    ) -> Result<(), ForustError> {
        let calc_loss = loss_callables(&self.objective);

        // If reset, reset the trees. Otherwise continue training.
        let mut yhat;
        if reset.unwrap_or(true) || self.trees.len() == 0 {
            self.reset();
            self.base_score = calc_init_callables(&self.objective)(y, sample_weight, alpha);
            yhat = vec![self.base_score; y.len()];
        } else {
            // self.lumber(data, y, sample_weight, alpha)?;
            yhat = self.predict(data, self.parallel, None);
        }

        let calc_grad_hess = gradient_hessian_callables(&self.objective);
        let (mut grad, mut hess) = calc_grad_hess(y, &yhat, sample_weight, alpha);

        let mut loss = calc_loss(y, &yhat, sample_weight, alpha);

        let loss_base = calc_loss(y, &vec![self.base_score; y.len()], sample_weight, alpha);
        // let loss_base = loss.clone();
        let loss_avg = loss_base.iter().sum::<f32>() / loss_base.len() as f32;
        let lr = splitter.get_learning_rate();
        let target_loss_decrement = self.growth_control * lr * loss_avg;

        let is_const_hess = match sample_weight {
            Some(_sample_weight) => false,
            None => match &self.objective {
                Objective::LogLoss => false,
                _ => true,
            },
        };

        // Generate binned data
        //
        // In scikit-learn, they sample 200_000 records for generating the bins.
        // we could consider that, especially if this proved to be a large bottleneck...
        let binned_data = bin_matrix(data, sample_weight, self.nbins, self.missing, categorical_features)?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut es = 0 as usize;
        let mut n_low_loss_rounds = 0;

        let hist_node = HistogramMatrix::empty(&bdata, &binned_data.cuts, &col_index, is_const_hess, self.parallel);
        let hist_capacity = 10000;
        let mut hist_tree: BrownHashMap<usize, HistogramMatrix> = BrownHashMap::with_capacity(hist_capacity);
        for i in 0..hist_capacity {
            hist_tree.insert(i, hist_node.clone());
        }

        for i in 0..self.iterations {
            let verbose = if self.log_iterations == 0 {
                false
            } else {
                i % self.log_iterations == 0
            };
            let tld = if n_low_loss_rounds > (self.early_stopping_rounds + 1) {
                None
            } else {
                Some(target_loss_decrement)
            };

            let mut tree = Tree::new();
            tree.fit(
                &bdata,
                data.index.to_owned(),
                &col_index,
                &binned_data.cuts,
                &grad,
                hess.as_deref(),
                splitter,
                self.max_leaves,
                self.max_depth,
                self.parallel,
                tld,
                &loss,
                y,
                calc_loss,
                &yhat,
                sample_weight,
                alpha,
                is_const_hess,
                &mut hist_tree,
                categorical_features,
            );

            self.update_predictions_inplace(&mut yhat, &tree, data);

            if tree.nodes.len() < 5 && tree.nodes.values().last().unwrap().generalization < Some(1.0) {
                es += 1;
                println!(
                    "round {0}, tree.nodes: {1}, tree.depth: {2}, early stopping: {3}",
                    i,
                    tree.nodes.len(),
                    tree.depth,
                    es
                );
            } else {
                // es = 0;
                println!(
                    "round {0}, tree.nodes: {1}, tree.depth: {2}",
                    i,
                    tree.nodes.len(),
                    tree.depth,
                );
            }

            if tree.stopper != TreeStopper::LossDecr {
                n_low_loss_rounds += 1;
            } else {
                n_low_loss_rounds = 0;
            }

            self.trees.push(tree);

            if es >= self.early_stopping_rounds {
                break;
            }

            (grad, hess) = calc_grad_hess(y, &yhat, sample_weight, alpha);
            loss = calc_loss(y, &yhat, sample_weight, alpha);

            if verbose {
                info!("Completed iteration {} of {}", i, self.iterations);
            }
        }

        if self.log_iterations > 0 {
            info!("Finished training booster with {} iterations.", self.trees.len());
        }
        Ok(())
    }

    fn update_predictions_inplace(&self, yhat: &mut [f64], tree: &Tree, data: &Matrix<f64>) {
        let preds = tree.predict(data, self.parallel, &self.missing);
        yhat.iter_mut().zip(preds).for_each(|(i, j)| *i += j);
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `score` -  Optional unceartinity score.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool, score: Option<f64>) -> Vec<f64> {
        let init_pred = match score {
            None => self.base_score,
            Some(score) => self.base_score + score,
        };
        let mut init_preds = vec![init_pred; data.rows];
        self.get_prediction_trees().iter().for_each(|tree| {
            for (p_, val) in init_preds.iter_mut().zip(tree.predict(data, parallel, &self.missing)) {
                *p_ += val;
            }
        });
        init_preds
    }

    /// Predict the contributions matrix for the provided dataset.
    pub fn predict_contributions(&self, data: &Matrix<f64>, method: ContributionsMethod, parallel: bool) -> Vec<f64> {
        match method {
            ContributionsMethod::Average => self.predict_contributions_average(data, parallel),
            ContributionsMethod::ProbabilityChange => {
                match self.objective {
                    Objective::LogLoss => {}
                    _ => panic!("ProbabilityChange contributions method is only valid when LogLoss objective is used."),
                }
                self.predict_contributions_probability_change(data, parallel)
            }
            _ => self.predict_contributions_tree_alone(data, parallel, method),
        }
    }

    // All of the contribution calculation methods, except for average are calculated
    // using just the model, so we don't need to have separate methods, we can instead
    // just have this one method, that dispatches to each one respectively.
    fn predict_contributions_tree_alone(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
        method: ContributionsMethod,
    ) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        let row_pred_fn = match method {
            ContributionsMethod::Weight => Tree::predict_contributions_row_weight,
            ContributionsMethod::BranchDifference => Tree::predict_contributions_row_branch_difference,
            ContributionsMethod::MidpointDifference => Tree::predict_contributions_row_midpoint_difference,
            ContributionsMethod::ModeDifference => Tree::predict_contributions_row_mode_difference,
            ContributionsMethod::Shapley => predict_contributions_row_shapley,
            ContributionsMethod::Average | ContributionsMethod::ProbabilityChange => unreachable!(),
        };
        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.missing);
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().for_each(|t| {
                        row_pred_fn(t, &r_, c, &self.missing);
                    });
                });
        }

        contribs
    }

    /// Get the a reference to the trees for predicting, ensureing that the right number of
    /// trees are used.
    pub fn get_prediction_trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Generate predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    fn predict_contributions_average(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let weights: Vec<HashMap<usize, f64>> = if parallel {
            self.get_prediction_trees()
                .par_iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        } else {
            self.get_prediction_trees()
                .iter()
                .map(|t| t.distribute_leaf_weights())
                .collect()
        };
        let mut contribs = vec![0.0; (data.cols + 1) * data.rows];

        // Add the bias term to every bias value...
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += self.base_score);

        // Clean this up..
        // materializing a row, and then passing that to all of the
        // trees seems to be the fastest approach (5X faster), we should test
        // something like this for normal predictions.
        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.missing);
                        });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees()
                        .iter()
                        .zip(weights.iter())
                        .for_each(|(t, w)| {
                            t.predict_contributions_row_average(&r_, c, w, &self.missing);
                        });
                });
        }

        contribs
    }

    fn predict_contributions_probability_change(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        let bias_idx = data.cols + 1;
        contribs
            .iter_mut()
            .skip(bias_idx - 1)
            .step_by(bias_idx)
            .for_each(|v| *v += odds(self.base_score));

        if parallel {
            data.index
                .par_iter()
                .zip(contribs.par_chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.missing, acc)
                    });
                });
        } else {
            data.index
                .iter()
                .zip(contribs.chunks_mut(data.cols + 1))
                .for_each(|(row, c)| {
                    let r_ = data.get_row(*row);
                    self.get_prediction_trees().iter().fold(self.base_score, |acc, t| {
                        t.predict_contributions_row_probability_change(&r_, c, &self.missing, acc)
                    });
                });
        }
        contribs
    }

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        let pd: f64 = if self.parallel {
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

    /// Save a booster as a json object to a file.
    ///
    /// * `path` - Path to save booster.
    pub fn save_booster(&self, path: &str) -> Result<(), ForustError> {
        let model = self.json_dump()?;
        match fs::write(path, model) {
            Err(e) => Err(ForustError::UnableToWrite(e.to_string())),
            Ok(_) => Ok(()),
        }
    }

    /// Dump a booster as a json object
    pub fn json_dump(&self) -> Result<String, ForustError> {
        match serde_json::to_string(self) {
            Ok(s) => Ok(s),
            Err(e) => Err(ForustError::UnableToWrite(e.to_string())),
        }
    }

    /// Load a booster from Json string
    ///
    /// * `json_str` - String object, which can be serialized to json.
    pub fn from_json(json_str: &str) -> Result<Self, ForustError> {
        let model = serde_json::from_str::<PerpetualBooster>(json_str);
        match model {
            Ok(m) => Ok(m),
            Err(e) => Err(ForustError::UnableToRead(e.to_string())),
        }
    }

    /// Load a booster from a path to a json booster object.
    ///
    /// * `path` - Path to load booster from.
    pub fn load_booster(path: &str) -> Result<Self, ForustError> {
        let json_str = match fs::read_to_string(path) {
            Ok(s) => Ok(s),
            Err(e) => Err(ForustError::UnableToRead(e.to_string())),
        }?;
        Self::from_json(&json_str)
    }

    // Set methods for paramters
    /// Set the objective_type on the booster.
    /// * `objective_type` - The objective type of the booster.
    pub fn set_objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Set model fitting budget.
    /// * `budget` - A fractional number for fitting.
    pub fn set_budget(&mut self, budget: f32) {
        let budget = f32::max(0.0, f32::min(1.0, budget));
        let power = budget * -3.0;
        let base = 10_f32;
        self.learning_rate = base.powf(power);
    }

    /// Set the iterations on the booster.
    /// * `iterations` - The number of iterations of the booster.
    pub fn set_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the log iterations on the booster.
    /// * `log_iterations` - The number of log iterations of the booster.
    pub fn set_log_iterations(mut self, log_iterations: usize) -> Self {
        self.log_iterations = log_iterations;
        self
    }

    /// Set the learning_rate on the booster.
    /// * `learning_rate` - The learning rate of the booster.
    pub fn set_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the max_depth on the booster.
    /// * `max_depth` - The maximum tree depth of the booster.
    pub fn set_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the max_leaves on the booster.
    /// * `max_leaves` - The maximum number of leaves of the booster.
    pub fn set_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Set the number of nbins on the booster.
    /// * `max_leaves` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_nbins(mut self, nbins: u16) -> Self {
        self.nbins = nbins;
        self
    }

    /// Set the l1 on the booster.
    /// * `l1` - The l1 regulation term of the booster.
    pub fn set_l1(mut self, l1: f32) -> Self {
        self.l1 = l1;
        self
    }

    /// Set the l2 on the booster.
    /// * `l2` - The l2 regulation term of the booster.
    pub fn set_l2(mut self, l2: f32) -> Self {
        self.l2 = l2;
        self
    }

    /// Set the gamma on the booster.
    /// * `gamma` - The gamma value of the booster.
    pub fn set_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the max_delta_step on the booster.
    /// * `max_delta_step` - The max_delta_step value of the booster.
    pub fn set_max_delta_step(mut self, max_delta_step: f32) -> Self {
        self.max_delta_step = max_delta_step;
        self
    }

    /// Set the min_leaf_weight on the booster.
    /// * `min_leaf_weight` - The minimum sum of the hession values allowed in the
    ///     node of a tree of the booster.
    pub fn set_min_leaf_weight(mut self, min_leaf_weight: f32) -> Self {
        self.min_leaf_weight = min_leaf_weight;
        self
    }

    /// Set the growth_control on the booster.
    pub fn set_growth_control(mut self, growth_control: f32) -> Self {
        self.growth_control = growth_control;
        self
    }

    /// Set the base_score on the booster.
    /// * `base_score` - The base score of the booster.
    pub fn set_base_score(mut self, base_score: f64) -> Self {
        self.base_score = base_score;
        self
    }

    /// Set the base_score on the booster.
    /// * `base_score` - The base score of the booster.
    pub fn set_initialize_base_score(mut self, initialize_base_score: bool) -> Self {
        self.initialize_base_score = initialize_base_score;
        self
    }

    /// Set the parallel on the booster.
    /// * `parallel` - Set if the booster should be trained in parallels.
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the force_children_to_bound_parent on the booster.
    /// * `force_children_to_bound_parent` - Set force children to bound parent.
    pub fn set_force_children_to_bound_parent(mut self, force_children_to_bound_parent: bool) -> Self {
        self.force_children_to_bound_parent = force_children_to_bound_parent;
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.allow_missing_splits = allow_missing_splits;
        self
    }

    /// Set the monotone_constraints on the booster.
    /// * `monotone_constraints` - The monotone constraints of the booster.
    pub fn set_monotone_constraints(mut self, monotone_constraints: Option<ConstraintMap>) -> Self {
        self.monotone_constraints = monotone_constraints;
        self
    }

    /// Set the missing_node_treatment on the booster.
    /// * `missing_node_treatment` - The missing node treatment of the booster.
    pub fn set_missing_node_treatment(mut self, missing_node_treatment: MissingNodeTreatment) -> Self {
        self.missing_node_treatment = missing_node_treatment;
        self
    }

    /// Set the subsample on the booster.
    /// * `subsample` - Percent of the data to randomly sample when training each tree.
    pub fn set_subsample(mut self, subsample: f32) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set the colsample_bytree on the booster.
    /// * `colsample_bytree` - Percent of the columns to randomly sample when training each tree.
    pub fn set_colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.colsample_bytree = colsample_bytree;
        self
    }

    /// Set the seed on the booster.
    /// * `seed` - Integer value used to see any randomness used in the algorithm.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set missing value of the booster
    /// * `missing` - Float value to consider as missing.
    pub fn set_missing(mut self, missing: f64) -> Self {
        self.missing = missing;
        self
    }

    /// Set create missing value of the booster
    /// * `create_missing_branch` - Bool specifying if missing should get it's own
    /// branch.
    pub fn set_create_missing_branch(mut self, create_missing_branch: bool) -> Self {
        self.create_missing_branch = create_missing_branch;
        self
    }

    /// Set sample method on the booster.
    /// * `sample_method` - Sample method.
    pub fn set_sample_method(mut self, sample_method: SampleMethod) -> Self {
        self.sample_method = sample_method;
        self
    }

    /// Set sample method on the booster.
    /// * `evaluation_metric` - Sample method.
    pub fn set_evaluation_metric(mut self, evaluation_metric: Option<Metric>) -> Self {
        self.evaluation_metric = evaluation_metric;
        self
    }

    /// Set early stopping rounds.
    /// * `early_stopping_rounds` - Early stoppings rounds.
    pub fn set_early_stopping_rounds(mut self, early_stopping_rounds: usize) -> Self {
        self.early_stopping_rounds = early_stopping_rounds;
        self
    }

    /// Set prediction iterations.
    /// * `early_stopping_rounds` - Early stoppings rounds.
    ///pub fn set_prediction_iteration(mut self, prediction_iteration: Option<usize>) -> Self {
    ///     self.prediction_iteration = prediction_iteration.map(|i| i + 1);
    ///      self
    ///  }

    /// Set the features where whose missing nodes should
    /// always be terminated.
    /// * `terminate_missing_features` - Hashset of the feature indices for the
    /// features that should always terminate the missing node, if create_missing_branch
    /// is true.
    pub fn set_terminate_missing_features(mut self, terminate_missing_features: HashSet<usize>) -> Self {
        self.terminate_missing_features = terminate_missing_features;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs;

    #[test]
    fn test_booster_fit_subsample() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = PerpetualBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_subsample(0.5)
            .set_base_score(0.5)
            .set_initialize_base_score(false);
        booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
        let preds = booster.predict(&data, false, None);
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
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = PerpetualBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
        let preds = booster.predict(&data, false, None);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit_continual() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = PerpetualBooster::default()
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
        booster.fit(&data, &y, None, None, 0.9, Some(false), None).unwrap();
        let preds = booster.predict(&data, false, None);
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
        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_initialize_base_score(true);

        booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
        let preds = booster.predict(&data, false, None);
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
            .set_iterations(10)
            .set_nbins(300)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
        let preds = booster.predict(&data, true, None);

        booster.save_booster("resources/model64.json").unwrap();
        let booster2 = PerpetualBooster::load_booster("resources/model64.json").unwrap();
        assert_eq!(booster2.predict(&data, true, None)[0..10], preds[0..10]);

        // Test with non-NAN missing.
        booster.missing = 0.0;
        booster.save_booster("resources/modelmissing.json").unwrap();
        let booster3 = PerpetualBooster::load_booster("resources/modelmissing.json").unwrap();
        assert_eq!(booster3.missing, 0.);
        assert_eq!(booster3.missing, booster.missing);
    }

    #[test]
    fn test_gbm_categorical() -> Result<(), Box<dyn Error>> {
        let n_bins = 256;
        let n_rows = 39073;
        let n_columns = 14;

        let file = fs::read_to_string("resources/adult_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);

        let file = fs::read_to_string("resources/adult_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let cat_index = vec![1, 3, 5, 6, 7, 8, 13];

        let mut booster = PerpetualBooster::default()
            .set_iterations(10)
            .set_nbins(n_bins)
            .set_max_depth(3)
            .set_base_score(0.5)
            .set_initialize_base_score(false);

        booster.fit(&data, &y, None, None, 0.3, None, Some(&cat_index)).unwrap();
        let preds = booster.predict(&data, true, None);

        println!("{:?}", &preds[..10]);
        Ok(())
    }
}
