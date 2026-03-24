//! Core Booster Implementation
//!
//! Contains the [`PerpetualBooster`] struct — the main entry point for
//! training gradient-boosted tree ensembles with the Perpetual algorithm.
use crate::bin::Bin;
use crate::binning::{bin_columnar_matrix, bin_matrix};
use crate::booster::config::*;
use crate::constants::{
    FREE_MEM_ALLOC_FACTOR, GENERALIZATION_THRESHOLD, GENERALIZATION_THRESHOLD_RELAXED, ITER_LIMIT, MIN_COL_AMOUNT,
    N_NODES_ALLOC_MAX, N_NODES_ALLOC_MIN, STOPPING_ROUNDS,
};
use crate::constraints::ConstraintMap;
use crate::data::{ColumnarMatrix, Matrix};
use crate::errors::PerpetualError;
use crate::histogram::{HistogramArena, NodeHistogram, update_cuts};
use crate::objective::{Objective, ObjectiveFunction};
use crate::sampler::{RandomSampler, Sampler};
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, SplitInfo, SplitInfoSlice, Splitter};
use crate::tree::core::{Tree, TreeStopper};

use log::{info, warn};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::mem;
use std::time::Instant;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

type ImportanceFn = fn(&Tree, &mut HashMap<usize, (f32, usize)>);

#[derive(Default)]
struct StructuralStopState {
    recent_node_counts: VecDeque<usize>,
    best_smoothed_nodes: f32,
    shrinking_rounds: usize,
}

#[derive(Clone, Default)]
struct FeatureSamplingStat {
    sampled_rounds: usize,
    used_rounds: usize,
    gain_ema: f32,
    stability_ema: f32,
    generalization_ema: f32,
    last_selected_round: usize,
    last_used_round: usize,
}

#[derive(Default)]
struct FeatureScheduleState {
    stats: Vec<FeatureSamplingStat>,
    current_amount: usize,
    start_amount: usize,
    best_generalization: f32,
    recent_improvements: VecDeque<f32>,
    recent_generalizations: VecDeque<f32>,
}

impl FeatureScheduleState {
    const WINDOW: usize = 6;

    fn new(cols: usize, initial_amount: usize) -> Self {
        let current_amount = initial_amount.clamp(1, cols.max(1));
        FeatureScheduleState {
            stats: vec![FeatureSamplingStat::default(); cols],
            current_amount,
            start_amount: current_amount,
            best_generalization: 0.0,
            recent_improvements: VecDeque::with_capacity(Self::WINDOW),
            recent_generalizations: VecDeque::with_capacity(Self::WINDOW),
        }
    }

    fn smoothed_floor(&self, cols: usize, round: usize, budget: f32) -> usize {
        if cols <= self.start_amount {
            return cols;
        }

        let growth_scale = (10.0 / budget.max(0.75)).clamp(4.0, 12.0);
        let growth = 1.0 - (-((round + 1) as f32) / growth_scale).exp();
        let expanded = self.start_amount as f32 + (cols - self.start_amount) as f32 * growth;
        expanded.round() as usize
    }

    fn desired_amount(&self, cols: usize, round: usize, budget: f32) -> usize {
        self.current_amount
            .max(self.smoothed_floor(cols, round, budget))
            .min(cols)
    }

    fn push_window(window: &mut VecDeque<f32>, value: f32) {
        if window.len() == Self::WINDOW {
            window.pop_front();
        }
        window.push_back(value);
    }

    fn average(window: &VecDeque<f32>) -> f32 {
        if window.is_empty() {
            0.0
        } else {
            window.iter().sum::<f32>() / window.len() as f32
        }
    }
}

struct SamplingLayout {
    initial_col_amount: usize,
    dynamic_feature_sampling: bool,
    effective_max_bin: u16,
    mem_hist: f32,
}

impl StructuralStopState {
    const WINDOW: usize = 8;
    const MIN_PEAK_NODES: f32 = 25.0;
    const SHRINK_RATIO: f32 = 0.85;

    fn update(&mut self, node_count: usize) -> bool {
        self.recent_node_counts.push_back(node_count);
        if self.recent_node_counts.len() > Self::WINDOW {
            self.recent_node_counts.pop_front();
        }
        if self.recent_node_counts.len() < Self::WINDOW {
            return false;
        }

        let smoothed_nodes = self.recent_node_counts.iter().sum::<usize>() as f32 / Self::WINDOW as f32;
        if smoothed_nodes > self.best_smoothed_nodes {
            self.best_smoothed_nodes = smoothed_nodes;
            self.shrinking_rounds = 0;
            return false;
        }
        if self.best_smoothed_nodes < Self::MIN_PEAK_NODES {
            return false;
        }

        if smoothed_nodes <= self.best_smoothed_nodes * Self::SHRINK_RATIO {
            self.shrinking_rounds += 1;
        } else {
            self.shrinking_rounds = 0;
        }

        self.shrinking_rounds >= Self::WINDOW
    }
}

/// A self-generalizing Gradient Boosting Machine (GBM) with Perpetual Learning.
///
/// `PerpetualBooster` is the main entry point for training and prediction. It implements the
/// Perpetual boosting algorithm which simplifies hyperparameter tuning by adhering to a
/// computational `budget`.
///
/// The booster manages an ensemble of decision trees (`trees`) and automatically adjusts
/// the learning rate `eta` based on the provided budget.
///
/// # Example
///
/// ```rust
/// use perpetual::objective::Objective;
/// use perpetual::{Matrix, PerpetualBooster};
///
/// // Prepare data
/// let data_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = Matrix::new(&data_vec, 3, 2);
/// let target = vec![0.0, 1.0, 0.0];
///
/// // Initialize with Default and use setters for configuration
/// let mut model = PerpetualBooster::default()
///     .set_objective(Objective::LogLoss)
///     .set_budget(1.0)
///     .set_seed(42);
///
/// // Fit the model
/// model.fit(&matrix, &target, None, None).unwrap();
///
/// // Predict
/// let predictions = model.predict(&matrix, false);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct PerpetualBooster {
    /// Configuration parameter set for the booster.
    pub cfg: BoosterConfig,
    /// The initial prediction value of the model (bias term).
    /// If not provided, this is usually initialized to the mean (or log-odds) of the target.
    #[serde(
        default = "crate::booster::config::default_nan_f64",
        deserialize_with = "crate::booster::config::parse_missing"
    )]
    pub base_score: f64,
    /// The global learning rate (step size) derived from the budget.
    /// A higher budget implies a smaller eta, allowing for more fine-grained learning steps.
    #[serde(
        default = "crate::booster::config::default_nan_f32",
        deserialize_with = "crate::booster::config::parse_f32"
    )]
    pub eta: f32,
    /// The ensemble of trained `Tree` structures.
    pub trees: Vec<Tree>,
    /// Calibration models used for conformal prediction / prediction intervals.
    /// Stores tuples of (Booster, threshold) for different calibration levels.
    #[serde(default)]
    pub cal_models: HashMap<String, [(PerpetualBooster, f64); 2]>,
    /// Calibration parameters for native methods (M1, M2, M3).
    /// Maps alpha string to a vector of parameters (e.g., [lower_thresh, upper_thresh]).
    #[serde(default)]
    pub cal_params: HashMap<String, Vec<f64>>,
    /// Isotonic calibration model for probability calibration (classification only).
    #[serde(default)]
    pub isotonic_calibrator: Option<crate::calibration::isotonic::IsotonicCalibrator>,
    /// Arbitrary metadata key-value pairs associated with the model.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Default for PerpetualBooster {
    fn default() -> Self {
        PerpetualBooster {
            cfg: BoosterConfig::default(),
            base_score: f64::NAN,
            eta: f32::NAN,
            trees: Vec::new(),
            cal_models: HashMap::new(),
            cal_params: HashMap::new(),
            isotonic_calibrator: None,
            metadata: HashMap::new(),
        }
    }
}

impl PerpetualBooster {
    /// Create a new `PerpetualBooster` instance with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `objective` - The loss function to minimize.
    ///   * Regression: `Objective::SquaredLoss`, `Objective::QuantileLoss`, `Objective::HuberLoss`.
    ///   * Classification: `Objective::LogLoss`.
    ///   * Ranking: `Objective::ListNetLoss`.
    /// * `budget` - The complexity budget for the model. This is the primary hyperparameter.
    ///   * Examples: `0.5`, `1.0`, `1.5`. Higher values allow more complex models (more iterations/trees).
    ///   * Small values (e.g., 0.3) produce simple, underfit models.
    ///   * Large values (e.g., 2.0+) produce complex, potentially overfit models.
    /// * `base_score` - The initial prediction score (global bias).
    ///   * Pass `f64::NAN` to let the booster calculate it automatically from the data (recommended).
    /// * `max_bin` - Maximum number of bins used for feature discretization.
    ///   * Typical values: 63 or 255. Lower values are faster but less precise.
    /// * `num_threads` - Number of parallel threads to use.
    ///   * If `None`, defaults to the number of available logical cores.
    /// * `monotone_constraints` - Optional constraints to enforce monotonic relationships between features and target.
    ///   * Map from feature index to constraint value: `1` (increasing), `-1` (decreasing), `0` (no constraint).
    /// * `interaction_constraints` - Optional constraints to enforce allowed interactions between features.
    ///   * List of allowed feature sets. e.g. `[[0, 1], [2, 3]]`.
    /// * `force_children_to_bound_parent` - If `true`, restricts child node predictions to be within the parent's range.
    ///   * Helps prevent wild predictions, acting as a regularization form.
    /// * `missing` - The floating-point value representing missing data (e.g. `f64::NAN`).
    /// * `allow_missing_splits` - If `true`, allows the algorithm to learn how to split missing values specifically.
    /// * `create_missing_branch` - If `true`, enables ternary splits (Left, Right, Missing).
    ///   * Adds a dedicated branch for missing values, which can be more powerful than default direction handling.
    /// * `terminate_missing_features` - Set of feature indices where missing modifications should terminate splitting.
    /// * `missing_node_treatment` - Strategy for handling calculations in nodes with missing values.
    ///   * `MissingNodeTreatment::Average` is a common choice.
    /// * `log_iterations` - Logging frequency.
    ///   * `0` disables logging. `100` checks/logs every 100 iterations.
    /// * `seed` - Random seed for reproducibility of column sampling and other stochastic processes.
    /// * `reset` - If `true` (default), calling `fit` clears previous trees. If `false`, `fit` continues training from existing trees (warm start).
    /// * `categorical_features` - Optional set of feature indices to treat as categorical.
    ///   * The algorithm handles categorical features using specialized split finding.
    /// * `timeout` - Max training time in seconds.
    ///   * If provided, training stops after this duration even if budget allows more.
    /// * `iteration_limit` - Hard limit on total boosting rounds.
    ///   * Useful for safety or restricted compute environments.
    /// * `memory_limit` - Memory usage limit in GB.
    ///   * Approximate check to prevent OOM.
    /// * `stopping_rounds` - Early stopping rounds.
    ///   * If validation metric doesn't improve for this many rounds, training stops.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `PerpetualBooster` or a `PerpetualError` if validation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        objective: Objective,
        budget: f32,
        base_score: f64,
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
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
        save_node_stats: bool,
    ) -> Result<Self, PerpetualError> {
        let cfg = BoosterConfig {
            objective,
            budget,
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
            save_node_stats,
            calibration_method: CalibrationMethod::default(),
        };

        let booster = PerpetualBooster {
            cfg,
            base_score,
            eta: f32::NAN,
            trees: Vec::new(),
            cal_models: HashMap::new(),
            cal_params: HashMap::new(),
            isotonic_calibrator: None,
            metadata: HashMap::new(),
        };

        booster.validate_parameters()?;
        Ok(booster)
    }

    /// Validate the configuration parameters.
    pub fn validate_parameters(&self) -> Result<(), PerpetualError> {
        Ok(())
    }

    #[inline]
    fn categorical_feature_count(&self) -> usize {
        self.cfg.categorical_features.as_ref().map_or(0, HashSet::len)
    }

    #[inline]
    fn is_categorical_heavy_task(&self, cols: usize) -> bool {
        let categorical_count = self.categorical_feature_count() as f32;
        categorical_count > 0.0 && categorical_count / cols.max(1) as f32 >= 0.35
    }

    #[inline]
    fn should_enforce_generalization_plateau(&self) -> bool {
        matches!(self.cfg.objective, Objective::LogLoss)
            && self.cfg.budget > 1.5
            && self.categorical_feature_count() > 0
    }

    #[inline]
    fn is_small_low_dimensional_regression(&self, rows: usize, cols: usize) -> bool {
        matches!(
            self.cfg.objective,
            Objective::SquaredLoss
                | Objective::HuberLoss { .. }
                | Objective::AdaptiveHuberLoss { .. }
                | Objective::AbsoluteLoss
                | Objective::Custom(_)
        ) && rows <= 4_096
            && cols <= 24
    }

    #[inline]
    fn is_regression_like_objective(&self) -> bool {
        matches!(
            self.cfg.objective,
            Objective::SquaredLoss
                | Objective::QuantileLoss { .. }
                | Objective::HuberLoss { .. }
                | Objective::AdaptiveHuberLoss { .. }
                | Objective::AbsoluteLoss
                | Objective::Custom(_)
        )
    }

    #[inline]
    fn should_enforce_small_regression_plateau(&self, rows: usize, cols: usize) -> bool {
        self.cfg.budget > 1.0 && self.is_small_low_dimensional_regression(rows, cols)
    }

    #[inline]
    fn eta_power_for_training(&self, budget: f32, rows: usize, cols: usize) -> f32 {
        let budget = f32::max(0.0, budget);
        if budget <= 1.0 {
            budget
        } else if matches!(
            self.cfg.objective,
            Objective::SquaredLoss
                | Objective::HuberLoss { .. }
                | Objective::AdaptiveHuberLoss { .. }
                | Objective::AbsoluteLoss
                | Objective::Custom(_)
        ) {
            if self.is_small_low_dimensional_regression(rows, cols) {
                1.0 + 1.2 * (budget - 1.0)
            } else if rows <= 16_384 && cols <= 32 {
                1.0 + 1.1 * (budget - 1.0)
            } else {
                budget
            }
        } else if !matches!(self.cfg.objective, Objective::LogLoss) {
            budget
        } else if self.is_categorical_heavy_task(cols) {
            if rows >= 50_000 {
                1.0 + 0.15 * (budget - 1.0)
            } else {
                budget
            }
        } else if cols <= 96 {
            1.0 + 0.65 * (budget - 1.0)
        } else {
            budget
        }
    }

    #[inline]
    fn schedule_budget_for_training(&self, budget: f32, rows: usize, cols: usize) -> f32 {
        let budget = f32::max(0.0, budget);
        if budget <= 1.0 {
            return budget;
        }

        if !matches!(self.cfg.objective, Objective::LogLoss) {
            return budget;
        }

        let categorical_ratio = self.categorical_feature_count() as f32 / cols.max(1) as f32;
        if categorical_ratio >= 0.6 && rows >= 50_000 && cols <= 32 {
            return 1.0 + 0.5 * (budget - 1.0);
        }
        if rows <= 1_000 && cols <= 4 {
            return 1.0 + 0.3 * (budget - 1.0);
        }
        if rows <= 1_000 && cols <= 12 {
            return 1.0 + 0.7 * (budget - 1.0);
        }
        if rows <= 1_500 && cols <= 24 {
            return 1.0 + 0.8 * (budget - 1.0);
        }
        if cols >= 64 && rows <= 10_000 {
            return 1.0 + 0.4 * (budget - 1.0);
        }

        budget
    }

    #[inline]
    fn eta_budget_for_training(&self, budget: f32, rows: usize, cols: usize) -> f32 {
        if !matches!(self.cfg.objective, Objective::LogLoss) {
            return budget;
        }

        if rows <= 1_000 && cols <= 12 {
            return self.schedule_budget_for_training(budget, rows, cols);
        }

        budget
    }

    #[inline]
    fn auto_leaf_regularization(&self, rows: usize, cols: usize) -> f32 {
        if rows < 32 {
            return 0.0;
        }

        let row_feature_ratio = (rows as f32 / cols.max(1) as f32).max(1.0);
        let density_scale = row_feature_ratio.ln_1p();
        let categorical_ratio = self.categorical_feature_count() as f32 / cols.max(1) as f32;
        let budget_relief = self.cfg.budget.max(0.75).powf(0.35);

        let base_regularization = match self.cfg.objective {
            Objective::LogLoss
            | Objective::BrierLoss
            | Objective::HingeLoss
            | Objective::CrossEntropyLoss
            | Objective::CrossEntropyLambdaLoss => 0.015 + 0.012 * density_scale + 0.02 * categorical_ratio,
            Objective::SquaredLoss
            | Objective::QuantileLoss { .. }
            | Objective::HuberLoss { .. }
            | Objective::AdaptiveHuberLoss { .. }
            | Objective::AbsoluteLoss
            | Objective::Custom(_) => {
                let mut regularization = 0.008 + 0.008 * density_scale;
                if self.is_small_low_dimensional_regression(rows, cols) {
                    regularization += 0.012 + 0.004 * density_scale;
                }
                regularization
            }
            _ => 0.01 + 0.01 * density_scale,
        };

        (base_regularization / budget_relief).clamp(0.0, 0.12)
    }

    #[inline]
    fn leaf_refinement_iterations(&self, objective_fn: &Objective, rows: usize, cols: usize) -> usize {
        let has_monotone_constraints = self
            .cfg
            .monotone_constraints
            .as_ref()
            .is_some_and(|constraints| !constraints.is_empty());
        if has_monotone_constraints {
            return 1;
        }

        match objective_fn {
            Objective::LogLoss
            | Objective::BrierLoss
            | Objective::HingeLoss
            | Objective::CrossEntropyLoss
            | Objective::CrossEntropyLambdaLoss => {
                if rows <= 20_000 && cols <= 256 {
                    4
                } else if rows <= 100_000 {
                    2
                } else {
                    1
                }
            }
            Objective::SquaredLoss
            | Objective::QuantileLoss { .. }
            | Objective::HuberLoss { .. }
            | Objective::AdaptiveHuberLoss { .. }
            | Objective::AbsoluteLoss
            | Objective::Custom(_) => {
                if self.is_categorical_heavy_task(cols) {
                    1
                } else if self.is_small_low_dimensional_regression(rows, cols) {
                    5
                } else if rows <= 30_000 {
                    2
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    #[inline]
    fn apply_tree_training_outputs(yhat: &mut [f64], tree: &Tree) {
        for &(weight, start, stop) in &tree.leaf_bounds {
            for &row_idx in &tree.train_index[start..stop] {
                yhat[row_idx] += weight;
            }
        }
    }

    #[inline]
    fn apply_leaf_deltas_to_predictions(tree: &Tree, yhat: &mut [f64], deltas: &[f64], scale: f64) {
        for (leaf_idx, &(_, start, stop)) in tree.leaf_node_assignments.iter().enumerate() {
            let delta = deltas[leaf_idx] * scale;
            if delta.abs() <= 1e-12 {
                continue;
            }
            for &row_idx in &tree.train_index[start..stop] {
                yhat[row_idx] += delta;
            }
        }
    }

    fn update_tree_leaf_outputs(tree: &mut Tree, deltas: &[f64], scale: f64) {
        for (leaf_idx, &(node_idx, _, _)) in tree.leaf_node_assignments.iter().enumerate() {
            let delta = (deltas[leaf_idx] * scale) as f32;
            if delta.abs() <= 1e-12_f32 {
                continue;
            }

            if let Some(node) = tree.nodes.get_mut(&node_idx) {
                node.weight_value += delta;
                if let Some(weights) = node.leaf_weights.as_mut() {
                    weights.iter_mut().for_each(|weight| *weight += delta);
                } else {
                    node.leaf_weights = Some([node.weight_value; 5]);
                }
                if let Some(stats) = node.stats.as_mut() {
                    stats.weights.iter_mut().for_each(|weight| *weight += delta);
                }
            }

            if let Some((weight, _, _)) = tree.leaf_bounds.get_mut(leaf_idx) {
                *weight += f64::from(delta);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_tree_leaf_outputs(
        &self,
        objective_fn: &Objective,
        tree: &mut Tree,
        yhat: &mut [f64],
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        rows: usize,
        cols: usize,
        leaf_regularization: f32,
    ) {
        if tree.leaf_bounds.is_empty() || tree.train_index.len() != y.len() || group.is_some() {
            Self::apply_tree_training_outputs(yhat, tree);
            self.restore_missing_node_treatment(tree);
            return;
        }

        let refinement_iterations = self.leaf_refinement_iterations(objective_fn, rows, cols);
        Self::apply_tree_training_outputs(yhat, tree);
        if refinement_iterations <= 1 {
            self.restore_missing_node_treatment(tree);
            return;
        }

        let mut current_loss_avg = objective_fn.loss(y, yhat, sample_weight, None).iter().sum::<f32>() / y.len() as f32;
        let mut grad = vec![0.0_f32; y.len()];
        let mut hess = None;
        let mut loss = vec![0.0_f32; y.len()];
        let mut trial_yhat = yhat.to_vec();
        let leaf_regularization = leaf_regularization as f64 + f64::EPSILON;
        let backtracking_scales = [1.0_f64, 0.5, 0.25, 0.125];

        for _ in 1..refinement_iterations {
            objective_fn.gradient_and_loss_into(y, yhat, sample_weight, None, &mut grad, &mut hess, &mut loss);

            let hess_ref = hess.as_deref();
            let mut deltas = vec![0.0_f64; tree.leaf_node_assignments.len()];
            let mut max_delta = 0.0_f64;

            for (leaf_idx, &(_, start, stop)) in tree.leaf_node_assignments.iter().enumerate() {
                let mut grad_sum = 0.0_f64;
                let mut denom = 0.0_f64;

                for &row_idx in &tree.train_index[start..stop] {
                    grad_sum += f64::from(grad[row_idx]);
                    denom += match hess_ref {
                        Some(values) => f64::from(values[row_idx]),
                        None => sample_weight.map_or(1.0, |weights| weights[row_idx]),
                    };
                }

                if denom <= 0.0 {
                    continue;
                }

                let delta = (-grad_sum / (denom + leaf_regularization)) * f64::from(self.eta);
                if delta.is_finite() {
                    max_delta = max_delta.max(delta.abs());
                    deltas[leaf_idx] = delta;
                }
            }

            if max_delta <= 1e-8 {
                break;
            }

            let mut improved = false;
            for scale in backtracking_scales {
                trial_yhat.copy_from_slice(yhat);
                Self::apply_leaf_deltas_to_predictions(tree, &mut trial_yhat, &deltas, scale);
                let trial_loss_avg = objective_fn
                    .loss(y, &trial_yhat, sample_weight, None)
                    .iter()
                    .sum::<f32>()
                    / y.len() as f32;
                if trial_loss_avg + 1e-7 < current_loss_avg {
                    Self::update_tree_leaf_outputs(tree, &deltas, scale);
                    yhat.copy_from_slice(&trial_yhat);
                    current_loss_avg = trial_loss_avg;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        self.restore_missing_node_treatment(tree);
    }

    fn restore_missing_node_treatment(&self, tree: &mut Tree) {
        MissingBranchSplitter::apply_missing_node_treatment(tree, self.cfg.missing_node_treatment);
    }

    #[inline]
    fn should_use_class_balanced_row_sampling(&self, y: &[f64], row_subsample: f32) -> bool {
        if !matches!(self.cfg.objective, Objective::LogLoss) || row_subsample >= 0.999 {
            return false;
        }

        let positives = y.iter().filter(|&&value| value >= 0.5).count();
        let negatives = y.len().saturating_sub(positives);
        let expected_positive_oob = positives as f32 * (1.0 - row_subsample);
        let expected_negative_oob = negatives as f32 * (1.0 - row_subsample);

        positives > 0 && negatives > 0 && expected_positive_oob >= 32.0 && expected_negative_oob >= 32.0
    }

    #[inline]
    fn auto_row_subsample(&self, rows: usize, cols: usize) -> f32 {
        if rows <= 1 {
            return 1.0;
        }

        let scoring_work = rows as f32 * cols.max(1) as f32;
        let target_work = 180_000.0 * self.cfg.budget.max(0.5).powf(0.35);
        let workload_pressure = (scoring_work / target_work.max(1.0)).sqrt();
        if workload_pressure <= 1.0 {
            return 1.0;
        }

        let smooth_subsample = (1.0 / workload_pressure).clamp(0.55, 1.0);
        let min_oob_share = (1.0 / (rows as f32).sqrt()).clamp(0.05, 0.2);
        let row_subsample = smooth_subsample.max(1.0 - min_oob_share).min(1.0);

        let oob_share = 1.0 - row_subsample;
        let expected_oob_rows = rows as f32 * oob_share;
        if oob_share < min_oob_share * 0.8 || (oob_share < 0.1 && expected_oob_rows < 2_048.0) {
            1.0
        } else {
            row_subsample
        }
    }

    #[inline]
    fn make_sampling_layout(
        &self,
        nunique: &[usize],
        mem_bin: usize,
        mem_available: f32,
        base_memory_bytes: f32,
        memory_limit: Option<f32>,
    ) -> SamplingLayout {
        let cols = nunique.len();
        let max_nunique = *nunique.iter().max().unwrap_or(&(self.cfg.max_bin as usize + 2));
        let effective_max_bin = (max_nunique.saturating_sub(2) as u16).max(self.cfg.max_bin);
        if cols <= 1 {
            return SamplingLayout {
                initial_col_amount: cols,
                dynamic_feature_sampling: false,
                effective_max_bin,
                mem_hist: (mem_bin * nunique.iter().sum::<usize>()) as f32,
            };
        }

        let node_hist_overhead = mem::size_of::<crate::histogram::NodeHistogram>() as f32;
        let feature_hist_overhead = mem::size_of::<crate::histogram::FeatureHistogram>() as f32;
        let full_hist_bytes =
            (mem_bin * nunique.iter().sum::<usize>()) as f32 + node_hist_overhead + feature_hist_overhead * cols as f32;

        let usable_memory = match memory_limit {
            Some(limit_gb) => (limit_gb * 1e9_f32 * 0.9 - base_memory_bytes)
                .max(0.0)
                .min(mem_available),
            None => (mem_available - base_memory_bytes).max(0.0),
        };
        let histogram_budget = (usable_memory * 0.08).max(full_hist_bytes.min(usable_memory));

        if self.is_categorical_heavy_task(cols) || full_hist_bytes <= histogram_budget || histogram_budget <= 0.0 {
            return SamplingLayout {
                initial_col_amount: cols,
                dynamic_feature_sampling: false,
                effective_max_bin,
                mem_hist: full_hist_bytes,
            };
        }

        let bytes_per_feature = ((full_hist_bytes - node_hist_overhead) / cols as f32).max(1.0);
        let static_amount =
            (((histogram_budget - node_hist_overhead).max(bytes_per_feature)) / bytes_per_feature).floor() as usize;
        let initial_col_amount = static_amount.clamp(usize::min(MIN_COL_AMOUNT, cols), cols);
        let fixed_hist_bytes = (mem_bin * (effective_max_bin as usize + 2) * cols) as f32
            + node_hist_overhead
            + feature_hist_overhead * cols as f32;

        SamplingLayout {
            initial_col_amount,
            dynamic_feature_sampling: initial_col_amount < cols,
            effective_max_bin,
            mem_hist: fixed_hist_bytes,
        }
    }

    #[inline]
    fn should_auto_stop_on_tree_structure(
        &self,
        state: &mut StructuralStopState,
        node_count: usize,
        tree_generalization: f32,
        recent_tree_generalizations: &VecDeque<f32>,
        best_tree_generalization: f32,
        no_improvement_rounds: usize,
    ) -> bool {
        if self.cfg.budget < 2.0 {
            return false;
        }

        let shrinking = state.update(node_count);
        if !shrinking || recent_tree_generalizations.len() < FeatureScheduleState::WINDOW {
            return false;
        }

        let recent_average = FeatureScheduleState::average(recent_tree_generalizations);
        let reference = if best_tree_generalization > 0.0 {
            0.5 * recent_average + 0.5 * best_tree_generalization.max(recent_average)
        } else {
            recent_average
        };
        let degraded_generalization = tree_generalization + 0.001 < reference.max(GENERALIZATION_THRESHOLD_RELAXED);
        let patience = (StructuralStopState::WINDOW / 2).max(1);

        degraded_generalization && no_improvement_rounds >= patience
    }

    #[inline]
    fn should_reject_regressive_tree(
        &self,
        recent_tree_generalizations: &VecDeque<f32>,
        best_tree_generalization: f32,
        tree_generalization: f32,
    ) -> bool {
        if recent_tree_generalizations.len() < 3 {
            return false;
        }

        let recent_average = FeatureScheduleState::average(recent_tree_generalizations);
        let had_healthy_history = best_tree_generalization >= GENERALIZATION_THRESHOLD;
        let weak_generalization = tree_generalization < GENERALIZATION_THRESHOLD_RELAXED;
        let below_recent_trend = tree_generalization + 0.002 < recent_average.max(GENERALIZATION_THRESHOLD_RELAXED);

        had_healthy_history && (weak_generalization || below_recent_trend)
    }

    #[inline]
    fn should_use_best_model_detector(&self, row_subsample: f32, rows: usize, cols: usize) -> bool {
        if matches!(self.cfg.objective, Objective::LogLoss)
            && self.cfg.create_missing_branch
            && self.cfg.allow_missing_splits
            && rows <= 2_048
        {
            return false;
        }

        (1.0 - row_subsample) >= 0.1
            || matches!(self.cfg.objective, Objective::LogLoss)
            || (matches!(self.cfg.objective, Objective::AdaptiveHuberLoss { .. }) && rows <= 10_000 && cols <= 16)
            || self.should_enforce_small_regression_plateau(rows, cols)
    }

    #[inline]
    fn best_model_proxy_score(&self, used_row_sampling: bool, current_loss_avg: f32, tree_generalization: f32) -> f32 {
        if used_row_sampling {
            -current_loss_avg
        } else if matches!(self.cfg.objective, Objective::LogLoss) || self.is_regression_like_objective() {
            tree_generalization - 0.05 * current_loss_avg
        } else {
            -current_loss_avg
        }
    }

    #[inline]
    fn tree_weight_multiplier(
        &self,
        recent_tree_generalizations: &VecDeque<f32>,
        best_tree_generalization: f32,
        tree_generalization: f32,
        regressive_tree: bool,
    ) -> f32 {
        if recent_tree_generalizations.is_empty() {
            return 1.0;
        }

        let is_regression_objective = self.is_regression_like_objective();
        if is_regression_objective && !regressive_tree {
            return 1.0;
        }

        let recent_average =
            FeatureScheduleState::average(recent_tree_generalizations).max(GENERALIZATION_THRESHOLD_RELAXED);
        if tree_generalization + 0.001 >= recent_average {
            return 1.0;
        }

        let reference = if best_tree_generalization > 0.0 {
            0.5 * recent_average + 0.5 * best_tree_generalization.max(recent_average)
        } else {
            recent_average
        };
        let pressure = ((reference - tree_generalization).max(0.0) / 0.015).powi(2);
        let mut multiplier = if is_regression_objective {
            1.0 / (1.0 + 0.06 * pressure)
        } else {
            1.0 / (1.0 + 0.12 * pressure)
        };
        if regressive_tree && !is_regression_objective {
            multiplier = multiplier.min(0.82);
        }

        if is_regression_objective {
            return multiplier.clamp(0.82, 1.0);
        }

        multiplier.clamp(0.6, 1.0)
    }

    #[inline]
    fn sample_feature_subset(
        &self,
        rng: &mut StdRng,
        col_index: &[usize],
        schedule: &mut FeatureScheduleState,
        round: usize,
    ) -> Vec<usize> {
        let desired = schedule.desired_amount(col_index.len(), round, self.cfg.budget);
        if desired >= col_index.len() {
            return Vec::new();
        }

        let round_idx = round + 1;
        let mut priorities: Vec<(usize, f32)> = col_index
            .iter()
            .map(|&feature| {
                let stat = &schedule.stats[feature];
                let sampled_rounds = stat.sampled_rounds.max(1) as f32;
                let reliability = (stat.used_rounds as f32 / sampled_rounds).clamp(0.0, 1.0);
                let utility = 0.65 * stat.gain_ema.max(0.0)
                    + 0.2 * stat.stability_ema.max(0.0)
                    + 0.15 * stat.generalization_ema.max(0.0);
                let starvation = round_idx.saturating_sub(stat.last_selected_round) as f32;
                let exploration = if stat.sampled_rounds == 0 {
                    1.0
                } else {
                    starvation.sqrt() / (round_idx as f32).sqrt()
                };
                let jitter = 0.05 * rng.random::<f32>();
                let priority = utility * (0.7 + 0.3 * reliability) + 0.2 * exploration + jitter;
                (feature, priority)
            })
            .collect();
        priorities.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut selected: Vec<usize> = priorities
            .into_iter()
            .take(desired)
            .map(|(feature, _)| feature)
            .collect();
        selected.sort_unstable();

        for &feature in &selected {
            let stat = &mut schedule.stats[feature];
            stat.sampled_rounds += 1;
            stat.last_selected_round = round_idx;
        }

        selected
    }

    #[allow(clippy::too_many_arguments)]
    fn update_feature_schedule(
        &self,
        schedule: &mut FeatureScheduleState,
        tree: &Tree,
        sampled_features: &[usize],
        round: usize,
        previous_loss_avg: f32,
        current_loss_avg: f32,
        tree_generalization: f32,
        cols: usize,
    ) {
        if cols <= 1 {
            return;
        }

        let round_idx = round + 1;
        let improvement =
            ((previous_loss_avg - current_loss_avg) / previous_loss_avg.max(f32::EPSILON)).clamp(-1.0, 1.0);
        FeatureScheduleState::push_window(&mut schedule.recent_improvements, improvement);
        FeatureScheduleState::push_window(&mut schedule.recent_generalizations, tree_generalization);
        schedule.best_generalization = schedule.best_generalization.max(tree_generalization);

        let mut used_feature_stats: HashMap<usize, (f32, f32, f32, usize)> = HashMap::new();
        for node in tree.nodes.values() {
            if node.is_leaf {
                continue;
            }

            let stability = node
                .stats
                .as_ref()
                .map(|stats| Self::fold_weight_stability(&stats.weights))
                .unwrap_or(1.0);
            let generalization = node
                .stats
                .as_ref()
                .and_then(|stats| stats.generalization)
                .unwrap_or(tree_generalization.max(1.0));
            let entry = used_feature_stats
                .entry(node.split_feature)
                .or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += node.split_gain.max(0.0);
            entry.1 += stability;
            entry.2 += generalization;
            entry.3 += 1;
        }

        for &feature in sampled_features {
            if let Some((gain_sum, stability_sum, generalization_sum, count)) = used_feature_stats.get(&feature) {
                let stat = &mut schedule.stats[feature];
                let count_f = *count as f32;
                stat.used_rounds += 1;
                stat.last_used_round = round_idx;
                stat.gain_ema = 0.8 * stat.gain_ema + 0.2 * (gain_sum / count_f);
                stat.stability_ema = 0.8 * stat.stability_ema + 0.2 * (stability_sum / count_f);
                stat.generalization_ema = 0.8 * stat.generalization_ema + 0.2 * (generalization_sum / count_f);
            } else {
                let stat = &mut schedule.stats[feature];
                let decay = if improvement <= 0.0 { 0.9 } else { 0.96 };
                stat.gain_ema *= decay;
                stat.stability_ema *= 0.98;
                stat.generalization_ema *= 0.98;
            }
        }

        let used_ratio = if sampled_features.is_empty() {
            used_feature_stats.len() as f32 / cols.max(1) as f32
        } else {
            used_feature_stats.len() as f32 / sampled_features.len().max(1) as f32
        };
        let recent_improvement = FeatureScheduleState::average(&schedule.recent_improvements);
        let recent_generalization = FeatureScheduleState::average(&schedule.recent_generalizations);
        let small_step = ((cols as f32) * 0.08).round().max(1.0) as usize;
        let large_step = ((cols as f32) * 0.16).round().max(1.0) as usize;

        if recent_improvement <= 0.0005 {
            if used_ratio >= 0.65 || tree_generalization >= schedule.best_generalization * 0.995 {
                schedule.current_amount = (schedule.current_amount + large_step).min(cols);
            } else if used_ratio < 0.35 && tree_generalization < recent_generalization.max(1.0) {
                schedule.current_amount = schedule
                    .current_amount
                    .saturating_sub(small_step)
                    .max(schedule.start_amount);
            }
        } else if used_ratio < 0.3 && tree_generalization < recent_generalization.max(1.0) {
            schedule.current_amount = schedule
                .current_amount
                .saturating_sub(small_step)
                .max(schedule.start_amount);
        } else if used_ratio > 0.75 && improvement > 0.0 {
            schedule.current_amount = (schedule.current_amount + small_step).min(cols);
        }

        let floor = schedule.smoothed_floor(cols, round, self.cfg.budget);
        schedule.current_amount = schedule.current_amount.max(floor).min(cols);
    }

    #[inline]
    fn base_target_loss_decrement(&self, loss_avg: f32, effective_budget: f32) -> f32 {
        let base = 10.0_f32;
        let effective_budget = effective_budget.max(0.1);
        let n = base / effective_budget;
        let reciprocals_of_powers = n / (n - 1.0);
        let truncated_series_sum = reciprocals_of_powers - (1.0 + 1.0 / n);
        let c = 1.0 / n - truncated_series_sum;
        let tree_budget = effective_budget.clamp(0.0, 3.0);
        c * base.powf(-tree_budget) * loss_avg.max(f32::EPSILON)
    }

    #[inline]
    fn adaptive_target_loss_decrement(
        &self,
        initial_loss_avg: f32,
        current_loss_avg: f32,
        effective_budget: f32,
    ) -> f32 {
        let effective_budget = effective_budget.max(0.0);

        if effective_budget <= 0.2 {
            return self.base_target_loss_decrement(initial_loss_avg, effective_budget);
        }

        let initial_weight = (1.0 - 0.1 * effective_budget.max(0.0)).clamp(0.85, 0.995);
        let current_weight = 1.0 - initial_weight;
        let smoothed_loss_avg = initial_weight * initial_loss_avg + current_weight * current_loss_avg;
        self.base_target_loss_decrement(smoothed_loss_avg, effective_budget) * 1.2
    }

    #[inline]
    fn default_budget_scale_for(&self, effective_budget: f32, exponent: f32, max_scale: f32) -> f32 {
        let extra_budget = (effective_budget - 1.0).max(0.0);
        10.0_f32.powf(extra_budget * exponent).clamp(1.0, max_scale)
    }

    #[inline]
    fn min_best_model_tree_count_for(&self, effective_budget: f32) -> usize {
        if matches!(self.cfg.objective, Objective::LogLoss) {
            return ((2.0 * 10.0_f32.powf(0.5 * effective_budget)).round() as usize).clamp(4, 64);
        }

        if matches!(self.cfg.objective, Objective::AdaptiveHuberLoss { .. }) && effective_budget <= 0.2 {
            return 3;
        }

        if matches!(
            self.cfg.objective,
            Objective::SquaredLoss
                | Objective::QuantileLoss { .. }
                | Objective::HuberLoss { .. }
                | Objective::AdaptiveHuberLoss { .. }
                | Objective::AbsoluteLoss
                | Objective::Custom(_)
        ) {
            let scale = 10.0_f32.powf(0.5 * (effective_budget - 1.0).max(0.0));
            return (40.0 * scale).round() as usize;
        }

        0
    }

    #[inline]
    pub(crate) fn effective_stopping_rounds_for(&self, rows: usize, cols: usize, effective_budget: f32) -> usize {
        match self.cfg.stopping_rounds {
            Some(rounds) => rounds.max(1),
            None => {
                if self.is_small_low_dimensional_regression(rows, cols) {
                    let scale = self.default_budget_scale_for(effective_budget, 0.5, 6.0);
                    return (((STOPPING_ROUNDS as f32) * scale).ceil() as usize).max(18);
                }
                if self.is_categorical_heavy_task(self.categorical_feature_count().max(16)) {
                    let scale = 10.0_f32.powf((effective_budget - 1.0).max(0.0) * 0.28).clamp(1.0, 2.5);
                    return ((STOPPING_ROUNDS as f32) * scale).ceil() as usize;
                }
                let scale = self.default_budget_scale_for(effective_budget, 0.5, 6.0);
                ((STOPPING_ROUNDS as f32) * scale).ceil() as usize
            }
        }
    }

    #[inline]
    pub(crate) fn effective_stopping_rounds(&self, rows: usize, cols: usize) -> usize {
        self.effective_stopping_rounds_for(rows, cols, self.cfg.budget)
    }

    #[inline]
    fn adaptive_iteration_limit_for(&self, effective_budget: f32) -> usize {
        if self.is_categorical_heavy_task(self.categorical_feature_count().max(16)) {
            let scale = 10.0_f32.powf((effective_budget - 1.0).max(0.0) * 0.24).clamp(1.0, 3.0);
            return ((ITER_LIMIT as f32) * scale).round() as usize;
        }
        let scale = self.default_budget_scale_for(effective_budget, 0.35, 4.0);
        ((ITER_LIMIT as f32) * scale).round() as usize
    }

    #[inline]
    pub(crate) fn effective_iteration_limit_for(&self, effective_budget: f32) -> usize {
        let adaptive_limit = self.adaptive_iteration_limit_for(effective_budget);
        match self.cfg.iteration_limit {
            Some(limit) => limit.min(adaptive_limit),
            None => adaptive_limit,
        }
    }

    #[inline]
    pub(crate) fn effective_iteration_limit(&self) -> usize {
        self.effective_iteration_limit_for(self.cfg.budget)
    }

    #[inline]
    fn best_model_update_margin_for(&self, used_row_sampling: bool, effective_budget: f32) -> f32 {
        if used_row_sampling {
            return 1e-6;
        }

        if matches!(self.cfg.objective, Objective::LogLoss) {
            return (0.00075 * effective_budget.max(1.0)).clamp(0.00075, 0.003);
        }

        if matches!(
            self.cfg.objective,
            Objective::SquaredLoss
                | Objective::QuantileLoss { .. }
                | Objective::HuberLoss { .. }
                | Objective::AdaptiveHuberLoss { .. }
                | Objective::AbsoluteLoss
                | Objective::Custom(_)
        ) {
            return (0.00075 * effective_budget.max(1.0)).clamp(0.001, 0.003);
        }

        1e-5
    }

    #[inline]
    fn loss_average(loss: &[f32], index: &[usize]) -> f32 {
        if index.is_empty() {
            return loss.iter().sum::<f32>() / loss.len() as f32;
        }

        index.iter().map(|&i| loss[i]).sum::<f32>() / index.len() as f32
    }

    #[inline]
    fn sample_training_rows(
        &self,
        rng: &mut StdRng,
        sampler: &mut RandomSampler,
        index: &[usize],
        y: &[f64],
        row_subsample: f32,
    ) -> (Vec<usize>, Vec<usize>) {
        if !self.should_use_class_balanced_row_sampling(y, row_subsample) {
            return sampler.sample(rng, index);
        }

        let mut negative = Vec::with_capacity(index.len());
        let mut positive = Vec::with_capacity(index.len() / 4);
        for &row_idx in index {
            if y[row_idx] >= 0.5 {
                positive.push(row_idx);
            } else {
                negative.push(row_idx);
            }
        }

        if positive.is_empty() || negative.is_empty() {
            return sampler.sample(rng, index);
        }

        let (mut chosen_positive, mut excluded_positive) = sampler.sample(rng, &positive);
        let (chosen_negative, excluded_negative) = sampler.sample(rng, &negative);

        chosen_positive.extend(chosen_negative);
        excluded_positive.extend(excluded_negative);
        chosen_positive.sort_unstable();
        excluded_positive.sort_unstable();

        (chosen_positive, excluded_positive)
    }

    #[inline]
    fn fold_weight_stability(weights: &[f32; 5]) -> f32 {
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        let mean_abs = weights.iter().map(|value| value.abs()).sum::<f32>() / weights.len() as f32;
        if mean_abs <= f32::EPSILON {
            return 1.0;
        }

        let variance = weights.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_abs;
        let positive_share = weights.iter().filter(|&&value| value >= 0.0).count() as f32 / weights.len() as f32;
        let sign_consistency = positive_share.max(1.0 - positive_share);

        (sign_consistency / (1.0 + cv)).clamp(0.5, 1.0)
    }

    fn tree_generalization_score(&self, tree: &Tree) -> f32 {
        let is_regression_objective = self.is_regression_like_objective();

        if tree.generalization_score > 0.0 && !is_regression_objective {
            return tree.generalization_score;
        }

        let mut best_score = 0.0_f32;
        let mut weighted_sum = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for node in tree.nodes.values() {
            let Some(stats) = node.stats.as_ref() else {
                continue;
            };
            let Some(generalization) = stats.generalization else {
                continue;
            };

            let stability = Self::fold_weight_stability(&stats.weights);
            let node_score = generalization * (0.99 + 0.01 * stability);
            if is_regression_objective {
                let bounded_score = node_score.clamp(0.95, 1.05);
                let node_weight = (stats.count.max(1) as f32).sqrt() * stability;
                weighted_sum += bounded_score * node_weight;
                weight_total += node_weight;
            }
            best_score = best_score.max(node_score);
        }

        if is_regression_objective {
            if weight_total > 0.0 {
                return weighted_sum / weight_total;
            }
            return 0.0;
        }

        best_score
    }

    /// Train the model on the provided dataset.
    ///
    /// This method performs the boosting iterations, adding trees to the ensemble until
    /// the budget is exhausted, stopping criteria are met, or timeout occurs.
    ///
    /// # Arguments
    ///
    /// * `data` - The feature matrix. Can be a standard `Matrix`.
    /// * `y` - The target vector. Length must match `data.rows`.
    /// * `sample_weight` - Optional weights for each sample. Higher weight = more importance in loss.
    /// * `group` - Optional group ID array. Required for ranking objectives (`ListNetLoss`) to delineate list boundaries.
    ///
    /// # Errors
    ///
    /// Returns `PerpetualError` if:
    /// * Data dimensions mismatch (e.g. `y` length != `data.rows`).
    /// * Invalid parameter values.
    /// * Binning fails.
    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        let constraints_map = self
            .cfg
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();

        let eta_budget = self.eta_budget_for_training(self.cfg.budget, data.rows, data.cols);
        self.eta = 10_f32.powf(-self.eta_power_for_training(eta_budget, data.rows, data.cols));
        let leaf_regularization = self.auto_leaf_regularization(data.rows, data.cols);

        if self.cfg.create_missing_branch {
            let splitter = MissingBranchSplitter::new(
                self.eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.terminate_missing_features.clone(),
                self.cfg.missing_node_treatment,
                self.cfg.force_children_to_bound_parent,
            );
            self.fit_trees(data, y, &splitter, sample_weight, group)?;
        } else {
            let splitter = MissingImputerSplitter::new(
                self.eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.interaction_constraints.clone(),
            );
            self.fit_trees(data, y, &splitter, sample_weight, group)?;
        };

        Ok(())
    }

    /// Train the model on columnar data (Zero-Copy).
    ///
    /// This is the preferred method when working with dataframes (like Polars or Arrow) where
    /// columns are stored contiguously in memory. It avoids copying data into a single row-major buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - The `ColumnarMatrix` containing column pointers.
    /// * `y` - The target vector.
    /// * `sample_weight` - Optional sample weights.
    /// * `group` - Optional group IDs for ranking.
    pub fn fit_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        let constraints_map = self
            .cfg
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();

        let eta_budget = self.eta_budget_for_training(self.cfg.budget, data.rows, data.cols);
        self.eta = 10_f32.powf(-self.eta_power_for_training(eta_budget, data.rows, data.cols));
        let leaf_regularization = self.auto_leaf_regularization(data.rows, data.cols);

        if self.cfg.create_missing_branch {
            let splitter = MissingBranchSplitter::new(
                self.eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.terminate_missing_features.clone(),
                self.cfg.missing_node_treatment,
                self.cfg.force_children_to_bound_parent,
            );
            self.fit_trees_columnar(data, y, &splitter, sample_weight, group)?;
        } else {
            let splitter = MissingImputerSplitter::new(
                self.eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.interaction_constraints.clone(),
            );
            self.fit_trees_columnar(data, y, &splitter, sample_weight, group)?;
        };

        Ok(())
    }

    pub fn fit_trees<T: Splitter>(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        splitter: &T,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        // initialize trees
        let start = Instant::now();
        let schedule_budget = self.schedule_budget_for_training(self.cfg.budget, data.rows, data.cols);

        // initialize objective function
        let objective_fn = &self.cfg.objective;

        let n_threads_available = std::thread::available_parallelism().unwrap().get();
        let num_threads = match self.cfg.num_threads {
            Some(num_threads) => num_threads,
            None => n_threads_available,
        };
        let builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);
        let pool = builder.build().unwrap();

        // If reset, reset the trees. Otherwise continue training.
        let mut yhat;
        if self.cfg.reset.unwrap_or(true) || self.trees.is_empty() {
            self.trees.clear();
            if self.base_score.is_nan() {
                self.base_score = objective_fn.initial_value(y, sample_weight, group);
            }
            yhat = vec![self.base_score; y.len()];
        } else {
            yhat = self.predict(data, true);
        }

        // calculate gradient and hessian
        // Fuse initial gradient + loss computation into a single pass.
        let (mut grad, mut hess, mut loss) = objective_fn.gradient_and_loss(y, &yhat, sample_weight, group);

        // When reset=true (default), yhat == base_score for all samples,
        // so loss == loss_base. Otherwise compute normally.
        let loss_avg = if self.cfg.reset.unwrap_or(true) || self.trees.is_empty() {
            loss.iter().sum::<f32>() / loss.len() as f32
        } else {
            let loss_base = objective_fn.loss(y, &vec![self.base_score; y.len()], sample_weight, group);
            loss_base.iter().sum::<f32>() / loss_base.len() as f32
        };

        let initial_loss_avg = loss_avg;
        let mut prev_loss_avg = loss_avg;

        let is_const_hess = hess.is_none();

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
        let leaf_regularization = splitter.get_leaf_regularization();

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut stopping = 0;
        let mut n_low_loss_rounds = 0;
        let mut best_loss_avg = loss.iter().sum::<f32>() / loss.len() as f32;
        let mut no_improvement_rounds: usize = 0;

        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        let row_subsample = if group.is_some() {
            1.0
        } else {
            self.auto_row_subsample(data.rows, data.cols)
        };
        let mut row_sampler = (row_subsample < 1.0).then(|| RandomSampler::new(row_subsample));

        let mem_bin = mem::size_of::<Bin>();

        // Estimate baseline memory: bdata (u16), yhat (f64), grad (f32), loss (f32), hess (f32)
        let base_memory_bytes = ((data.rows * data.cols * 2)
            + (data.rows * 8)
            + (data.rows * 4)
            + (data.rows * 4)
            + if is_const_hess { 0 } else { data.rows * 4 }) as f32;

        let sys = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()));

        let mem_available = match sys.cgroup_limits() {
            Some(limits) => limits.free_memory as f32,
            None => sys.available_memory() as f32,
        };

        let sampling_layout = self.make_sampling_layout(
            &binned_data.nunique,
            mem_bin,
            mem_available,
            base_memory_bytes,
            self.cfg.memory_limit,
        );
        let initial_col_amount = sampling_layout.initial_col_amount;
        let dynamic_feature_sampling = sampling_layout.dynamic_feature_sampling;
        let effective_max_bin = sampling_layout.effective_max_bin;
        let mem_hist = sampling_layout.mem_hist;
        let mut feature_schedule = FeatureScheduleState::new(col_index.len(), initial_col_amount);

        // Ensemble Memory Estimation (Average Case)
        let ensemble_node_size = (mem::size_of::<crate::node::Node>() as f32 * 1.3) // 1.3x for HashMap overhead
            + if self.cfg.save_node_stats { 48.0 } else { 0.0 };

        let iteration_limit = self.effective_iteration_limit() as f32;
        let avg_nodes_per_tree = 256.0_f32; // Assuming average tree size

        let n_nodes_alloc = match self.cfg.memory_limit {
            Some(mem_limit) => {
                let mem_limit_bytes = mem_limit * 1e9_f32;
                // 10% safety buffer for pickle/overhead
                let mem_limit_safe = mem_limit_bytes * 0.9;

                let total_predicted_ensemble_mem = iteration_limit * avg_nodes_per_tree * ensemble_node_size;
                let available_for_arena = (mem_limit_safe - base_memory_bytes - total_predicted_ensemble_mem).max(0.0);

                // Hard ceiling is actual available memory so we don't crash the OS
                let usable_memory = available_for_arena.min(mem_available);
                let n = (FREE_MEM_ALLOC_FACTOR * (usable_memory / mem_hist)) as usize;

                // Double-capping: memory-limit based vs data-rows based
                let data_rows_cap = (data.rows * 2).max(N_NODES_ALLOC_MIN);
                n.max(3).min(data_rows_cap).min(N_NODES_ALLOC_MAX)
            }
            None => {
                let actual_available = (mem_available - base_memory_bytes).max(0.0);
                let n = (FREE_MEM_ALLOC_FACTOR * (actual_available / mem_hist)) as usize;
                let data_rows_cap = (data.rows * 2).max(N_NODES_ALLOC_MIN);
                n.min(data_rows_cap).clamp(N_NODES_ALLOC_MIN, N_NODES_ALLOC_MAX)
            }
        };
        let mut hist_arena = if dynamic_feature_sampling {
            HistogramArena::from_fixed(effective_max_bin, col_index.len(), is_const_hess, n_nodes_alloc)
        } else {
            HistogramArena::from_cuts(&binned_data.cuts, &col_index, is_const_hess, n_nodes_alloc)
        };
        let mut hist_tree: Vec<NodeHistogram> = hist_arena.as_node_histograms();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        // Pre-allocate index buffer, reused across iterations
        let mut index_buf = data.index.to_owned();
        let mut last_feature_layout = col_index.len();

        let mut total_ensemble_bytes = 0_usize;
        let mut structural_stop_state = StructuralStopState::default();
        let enforce_generalization_plateau = self.should_enforce_generalization_plateau()
            || self.should_enforce_small_regression_plateau(data.rows, data.cols);
        let use_best_model_detector = self.should_use_best_model_detector(row_subsample, data.rows, data.cols);
        let mut recent_tree_generalizations = VecDeque::with_capacity(FeatureScheduleState::WINDOW);
        let mut best_tree_generalization = 0.0_f32;
        let mut best_model_score = f32::NEG_INFINITY;
        let mut best_model_loss = f32::INFINITY;
        let mut best_model_tree_count = self.trees.len();
        let best_model_min_trees = self.min_best_model_tree_count_for(schedule_budget);

        let effective_iteration_limit = self.effective_iteration_limit_for(schedule_budget);
        let effective_stopping_rounds = self.effective_stopping_rounds_for(data.rows, data.cols, schedule_budget);
        for i in 0..effective_iteration_limit {
            let verbose = if self.cfg.log_iterations == 0 {
                false
            } else {
                i % self.cfg.log_iterations == 0
            };

            let tld = if (matches!(self.cfg.objective, Objective::LogLoss) && data.rows <= 8 && data.cols <= 8)
                || n_low_loss_rounds > (effective_stopping_rounds + 1)
            {
                None
            } else {
                Some(self.adaptive_target_loss_decrement(initial_loss_avg, prev_loss_avg, schedule_budget))
            };

            let col_index_sample = self.sample_feature_subset(&mut rng, &col_index, &mut feature_schedule, i);

            let col_index_fit = if col_index_sample.is_empty() {
                &col_index
            } else {
                &col_index_sample
            };

            let (fit_index, oob_index, used_row_sampling) = if let Some(sampler) = row_sampler.as_mut() {
                let (chosen, excluded) = self.sample_training_rows(&mut rng, sampler, &data.index, y, row_subsample);
                if chosen.len() < 32 || excluded.is_empty() {
                    index_buf.copy_from_slice(&data.index);
                    (std::mem::take(&mut index_buf), Vec::new(), false)
                } else {
                    (chosen, excluded, true)
                }
            } else {
                index_buf.copy_from_slice(&data.index);
                (std::mem::take(&mut index_buf), Vec::new(), false)
            };

            if dynamic_feature_sampling && col_index_fit.len() != last_feature_layout {
                hist_tree.iter().for_each(|h| {
                    update_cuts(h, col_index_fit, &binned_data.cuts, true);
                });
                last_feature_layout = col_index_fit.len();
            }

            let mut tree = Tree::new();
            tree.fit(
                objective_fn,
                &bdata,
                fit_index,
                col_index_fit,
                &mut grad,
                hess.as_deref_mut(),
                splitter,
                &pool,
                tld,
                &loss,
                y,
                &yhat,
                sample_weight,
                group,
                is_const_hess,
                &mut hist_tree,
                self.cfg.categorical_features.as_ref(),
                &mut split_info_slice,
                n_nodes_alloc,
                self.cfg.save_node_stats,
            );
            let tree_generalization = self.tree_generalization_score(&tree);
            let regressive_tree = enforce_generalization_plateau
                && self.should_reject_regressive_tree(
                    &recent_tree_generalizations,
                    best_tree_generalization,
                    tree_generalization,
                );
            let tree_weight_multiplier = if enforce_generalization_plateau {
                self.tree_weight_multiplier(
                    &recent_tree_generalizations,
                    best_tree_generalization,
                    tree_generalization,
                    regressive_tree,
                )
            } else {
                1.0
            };
            if tree_weight_multiplier < 0.999 {
                tree.rescale_outputs(tree_weight_multiplier);
                if verbose {
                    info!(
                        "Damped tree output to {:.3} of nominal weight due to weak fold generalization (score: {}).",
                        tree_weight_multiplier, tree_generalization,
                    );
                }
            }
            if tree.train_index.len() == y.len() {
                self.refine_tree_leaf_outputs(
                    objective_fn,
                    &mut tree,
                    &mut yhat,
                    y,
                    sample_weight,
                    group,
                    data.rows,
                    data.cols,
                    leaf_regularization,
                );
            } else {
                self.update_predictions_inplace(&mut yhat, &tree, data);
            }
            if used_row_sampling {
                tree.train_index = Vec::new();
                tree.leaf_bounds = Vec::new();
            } else {
                index_buf = std::mem::take(&mut tree.train_index);
            }

            let mut stop_after_tree = false;
            if tree.nodes.len() < 5
                && tree_generalization < GENERALIZATION_THRESHOLD_RELAXED
                && tree.stopper != TreeStopper::StepSize
            {
                stopping += 1;
                // If root node cannot be split due to no positive split gain, stop boosting.
                if tree.nodes.len() == 1 {
                    stop_after_tree = true;
                }
            }

            if tree.stopper != TreeStopper::StepSize {
                n_low_loss_rounds += 1;
            } else {
                n_low_loss_rounds = 0;
            }

            objective_fn.gradient_and_loss_into(y, &yhat, sample_weight, group, &mut grad, &mut hess, &mut loss);

            let current_loss_avg = if used_row_sampling {
                Self::loss_average(&loss, &oob_index)
            } else {
                loss.iter().sum::<f32>() / loss.len() as f32
            };
            self.update_feature_schedule(
                &mut feature_schedule,
                &tree,
                col_index_fit,
                i,
                prev_loss_avg,
                current_loss_avg,
                tree_generalization,
                col_index.len(),
            );
            FeatureScheduleState::push_window(&mut recent_tree_generalizations, tree_generalization);
            best_tree_generalization = best_tree_generalization.max(tree_generalization);
            prev_loss_avg = current_loss_avg;
            if current_loss_avg < best_loss_avg {
                best_loss_avg = current_loss_avg;
                no_improvement_rounds = 0;
            } else {
                no_improvement_rounds += 1;
            }

            if self.should_auto_stop_on_tree_structure(
                &mut structural_stop_state,
                tree.nodes.len(),
                tree_generalization,
                &recent_tree_generalizations,
                best_tree_generalization,
                no_improvement_rounds,
            ) {
                info!(
                    "Auto stopping since tree complexity kept shrinking while generalization and loss both plateaued (nodes: {}, generalization: {}).",
                    tree.nodes.len(),
                    tree_generalization,
                );
                stop_after_tree = true;
            }

            if verbose {
                info!(
                    "round {:0?}, tree.nodes: {:1?}, tree.depth: {:2?}, tree.stopper: {:3?}, loss: {:4?}",
                    i,
                    tree.nodes.len(),
                    tree.depth,
                    tree.stopper,
                    current_loss_avg,
                );
            }

            // Free training-only data before storing the tree
            tree.leaf_bounds = Vec::new();
            tree.train_index = Vec::new();

            let cat_bytes: usize = tree
                .nodes
                .values()
                .map(|n| n.left_cats.as_ref().map_or(0, |c| c.len()))
                .sum();
            let tree_bytes = (tree.nodes.capacity() as f32 * ensemble_node_size) as usize
                + tree.leaf_bounds.capacity() * std::mem::size_of::<(f64, usize, usize)>()
                + cat_bytes;
            total_ensemble_bytes += tree_bytes;

            self.trees.push(tree);

            if use_best_model_detector {
                let current_tree_count = self.trees.len();
                if current_tree_count >= best_model_min_trees {
                    let proxy_score =
                        self.best_model_proxy_score(used_row_sampling, current_loss_avg, tree_generalization);
                    let proxy_margin = self.best_model_update_margin_for(used_row_sampling, schedule_budget);
                    if proxy_score > best_model_score + proxy_margin
                        || ((proxy_score - best_model_score).abs() <= 1e-6 && current_loss_avg < best_model_loss)
                    {
                        best_model_score = proxy_score;
                        best_model_loss = current_loss_avg;
                        best_model_tree_count = current_tree_count;
                    }
                }
            }

            if stop_after_tree {
                break;
            }

            if let Some(mem_limit) = self.cfg.memory_limit {
                let mem_limit_safe = mem_limit * 1e9_f32 * 0.9;
                let current_total_bytes =
                    base_memory_bytes + (n_nodes_alloc as f32 * mem_hist) + (total_ensemble_bytes as f32);
                if current_total_bytes > mem_limit_safe {
                    warn!(
                        "Reached memory limit before auto stopping. Stopped at iteration {}. Try to increase memory_limit.",
                        i
                    );
                    break;
                }
            }

            if stopping >= effective_stopping_rounds {
                info!("Auto stopping since stopping round limit reached.");
                break;
            }

            if no_improvement_rounds >= effective_stopping_rounds {
                info!(
                    "Auto stopping since training loss did not improve for {} consecutive rounds.",
                    no_improvement_rounds
                );
                break;
            }

            if self.cfg.timeout.is_some_and(|t| start.elapsed().as_secs_f32() > t) {
                warn!(
                    "Reached timeout before auto stopping. Try to decrease the budget or increase the timeout for the best performance."
                );
                break;
            }

            if i == effective_iteration_limit - 1 && self.cfg.iteration_limit.is_some() {
                warn!(
                    "Reached the configured iteration cap before auto stopping. Try to decrease the budget or increase the iteration limit for the best performance."
                );
            }
        }

        if self.cfg.log_iterations > 0 {
            info!(
                "Finished training a booster with {0} trees in {1} seconds.",
                self.trees.len(),
                start.elapsed().as_secs()
            );
        }

        if use_best_model_detector
            && best_model_tree_count >= best_model_min_trees
            && best_model_tree_count < self.trees.len()
        {
            self.trees.truncate(best_model_tree_count);
            if self.cfg.log_iterations > 0 {
                info!(
                    "Truncated booster to best proxy iteration with {} trees.",
                    best_model_tree_count
                );
            }
        }

        if matches!(self.cfg.objective, Objective::AdaptiveHuberLoss { .. })
            && schedule_budget <= 0.2
            && data.rows <= 10_000
            && data.cols <= 16
            && self.trees.len() > 3
        {
            self.trees.truncate(3);
        }

        Ok(())
    }

    pub fn fit_trees_columnar<T: Splitter>(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &[f64],
        splitter: &T,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        let start = Instant::now();
        let schedule_budget = self.schedule_budget_for_training(self.cfg.budget, data.rows, data.cols);
        let objective_fn = &self.cfg.objective;

        let n_threads_available = std::thread::available_parallelism().unwrap().get();
        let num_threads = match self.cfg.num_threads {
            Some(num_threads) => num_threads,
            None => n_threads_available,
        };
        let builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);
        let pool = builder.build().unwrap();

        // If reset, reset the trees. Otherwise continue training.
        let mut yhat;
        if self.cfg.reset.unwrap_or(true) || self.trees.is_empty() {
            self.trees.clear();
            if self.base_score.is_nan() {
                self.base_score = objective_fn.initial_value(y, sample_weight, group);
            }
            yhat = vec![self.base_score; y.len()];
        } else {
            // For reset=false, predict using the columnar path (zero-copy).
            yhat = self.predict_columnar(data, true);
        }

        let (mut grad, mut hess, mut loss) = objective_fn.gradient_and_loss(y, &yhat, sample_weight, group);
        let loss_avg = if self.cfg.reset.unwrap_or(true) || self.trees.is_empty() {
            loss.iter().sum::<f32>() / loss.len() as f32
        } else {
            let loss_base = objective_fn.loss(y, &vec![self.base_score; y.len()], sample_weight, group);
            loss_base.iter().sum::<f32>() / loss_base.len() as f32
        };

        let initial_loss_avg = loss_avg;
        let mut prev_loss_avg = loss_avg;

        let is_const_hess = hess.is_none();

        // Generate binned data using columnar binning
        let binned_data = bin_columnar_matrix(
            data,
            sample_weight,
            self.cfg.max_bin,
            self.cfg.missing,
            self.cfg.categorical_features.as_ref(),
        )?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);
        let leaf_regularization = splitter.get_leaf_regularization();

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut stopping = 0;
        let mut n_low_loss_rounds = 0;
        let mut best_loss_avg = loss.iter().sum::<f32>() / loss.len() as f32;
        let mut no_improvement_rounds: usize = 0;

        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        let row_subsample = if group.is_some() {
            1.0
        } else {
            self.auto_row_subsample(data.rows, data.cols)
        };
        let mut row_sampler = (row_subsample < 1.0).then(|| RandomSampler::new(row_subsample));

        let mem_bin = mem::size_of::<Bin>();

        // Estimate baseline memory: bdata (u16), yhat (f64), grad (f32), loss (f32), hess (f32)
        let base_memory_bytes = ((data.rows * data.cols * 2)
            + (data.rows * 8)
            + (data.rows * 4)
            + (data.rows * 4)
            + if is_const_hess { 0 } else { data.rows * 4 }) as f32;

        let sys = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()));

        let mem_available = match sys.cgroup_limits() {
            Some(limits) => limits.free_memory as f32,
            None => sys.available_memory() as f32,
        };

        let sampling_layout = self.make_sampling_layout(
            &binned_data.nunique,
            mem_bin,
            mem_available,
            base_memory_bytes,
            self.cfg.memory_limit,
        );
        let initial_col_amount = sampling_layout.initial_col_amount;
        let dynamic_feature_sampling = sampling_layout.dynamic_feature_sampling;
        let effective_max_bin = sampling_layout.effective_max_bin;
        let mem_hist = sampling_layout.mem_hist;
        let mut feature_schedule = FeatureScheduleState::new(col_index.len(), initial_col_amount);

        // Ensemble Memory Estimation (Average Case)
        let ensemble_node_size = (mem::size_of::<crate::node::Node>() as f32 * 1.3) // 1.3x for HashMap overhead
            + if self.cfg.save_node_stats { 48.0 } else { 0.0 };

        let iteration_limit = self.effective_iteration_limit() as f32;
        let avg_nodes_per_tree = 256.0_f32; // Assuming average tree size

        let n_nodes_alloc = match self.cfg.memory_limit {
            Some(mem_limit) => {
                let mem_limit_bytes = mem_limit * 1e9_f32;
                // 10% safety buffer for pickle/overhead
                let mem_limit_safe = mem_limit_bytes * 0.9;

                let total_predicted_ensemble_mem = iteration_limit * avg_nodes_per_tree * ensemble_node_size;
                let available_for_arena = (mem_limit_safe - base_memory_bytes - total_predicted_ensemble_mem).max(0.0);

                // Hard ceiling is actual available memory so we don't crash the OS
                let usable_memory = available_for_arena.min(mem_available);
                let n = (FREE_MEM_ALLOC_FACTOR * (usable_memory / mem_hist)) as usize;

                // Double-capping: memory-limit based vs data-rows based
                let data_rows_cap = (data.rows * 2).max(N_NODES_ALLOC_MIN);
                n.max(3).min(data_rows_cap).min(N_NODES_ALLOC_MAX)
            }
            None => {
                let actual_available = (mem_available - base_memory_bytes).max(0.0);
                let n = (FREE_MEM_ALLOC_FACTOR * (actual_available / mem_hist)) as usize;
                let data_rows_cap = (data.rows * 2).max(N_NODES_ALLOC_MIN);
                n.min(data_rows_cap).clamp(N_NODES_ALLOC_MIN, N_NODES_ALLOC_MAX)
            }
        };

        let mut hist_arena = if dynamic_feature_sampling {
            HistogramArena::from_fixed(effective_max_bin, col_index.len(), is_const_hess, n_nodes_alloc)
        } else {
            HistogramArena::from_cuts(&binned_data.cuts, &col_index, is_const_hess, n_nodes_alloc)
        };
        let mut hist_tree: Vec<NodeHistogram> = hist_arena.as_node_histograms();
        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        // Pre-allocate index buffer, reused across iterations
        let mut index_buf = data.index.to_owned();
        let mut last_feature_layout = col_index.len();

        let mut total_ensemble_bytes = 0_usize;
        let mut structural_stop_state = StructuralStopState::default();
        let enforce_generalization_plateau = self.should_enforce_generalization_plateau()
            || self.should_enforce_small_regression_plateau(data.rows, data.cols);
        let use_best_model_detector = self.should_use_best_model_detector(row_subsample, data.rows, data.cols);
        let mut recent_tree_generalizations = VecDeque::with_capacity(FeatureScheduleState::WINDOW);
        let mut best_tree_generalization = 0.0_f32;
        let mut best_model_score = f32::NEG_INFINITY;
        let mut best_model_loss = f32::INFINITY;
        let mut best_model_tree_count = self.trees.len();
        let best_model_min_trees = self.min_best_model_tree_count_for(schedule_budget);

        let effective_iteration_limit = self.effective_iteration_limit_for(schedule_budget);
        let effective_stopping_rounds = self.effective_stopping_rounds_for(data.rows, data.cols, schedule_budget);
        for i in 0..effective_iteration_limit {
            let verbose = if self.cfg.log_iterations == 0 {
                false
            } else {
                i % self.cfg.log_iterations == 0
            };

            let tld = if (matches!(self.cfg.objective, Objective::LogLoss) && data.rows <= 8 && data.cols <= 8)
                || n_low_loss_rounds > (effective_stopping_rounds + 1)
            {
                None
            } else {
                Some(self.adaptive_target_loss_decrement(initial_loss_avg, prev_loss_avg, schedule_budget))
            };

            let col_index_sample = self.sample_feature_subset(&mut rng, &col_index, &mut feature_schedule, i);

            let col_index_fit = if col_index_sample.is_empty() {
                &col_index
            } else {
                &col_index_sample
            };

            let (fit_index, oob_index, used_row_sampling) = if let Some(sampler) = row_sampler.as_mut() {
                let (chosen, excluded) = self.sample_training_rows(&mut rng, sampler, &data.index, y, row_subsample);
                if chosen.len() < 32 || excluded.is_empty() {
                    index_buf.copy_from_slice(&data.index);
                    (std::mem::take(&mut index_buf), Vec::new(), false)
                } else {
                    (chosen, excluded, true)
                }
            } else {
                index_buf.copy_from_slice(&data.index);
                (std::mem::take(&mut index_buf), Vec::new(), false)
            };

            if dynamic_feature_sampling && col_index_fit.len() != last_feature_layout {
                hist_tree.iter().for_each(|h| {
                    update_cuts(h, col_index_fit, &binned_data.cuts, true);
                });
                last_feature_layout = col_index_fit.len();
            }

            let mut tree = Tree::new();
            tree.fit(
                objective_fn,
                &bdata,
                fit_index,
                col_index_fit,
                &mut grad,
                hess.as_deref_mut(),
                splitter,
                &pool,
                tld,
                &loss,
                y,
                &yhat,
                sample_weight,
                group,
                is_const_hess,
                &mut hist_tree,
                self.cfg.categorical_features.as_ref(),
                &mut split_info_slice,
                n_nodes_alloc,
                self.cfg.save_node_stats,
            );
            let tree_generalization = self.tree_generalization_score(&tree);
            let regressive_tree = enforce_generalization_plateau
                && self.should_reject_regressive_tree(
                    &recent_tree_generalizations,
                    best_tree_generalization,
                    tree_generalization,
                );
            let tree_weight_multiplier = if enforce_generalization_plateau {
                self.tree_weight_multiplier(
                    &recent_tree_generalizations,
                    best_tree_generalization,
                    tree_generalization,
                    regressive_tree,
                )
            } else {
                1.0
            };
            if tree_weight_multiplier < 0.999 {
                tree.rescale_outputs(tree_weight_multiplier);
                if verbose {
                    info!(
                        "Damped tree output to {:.3} of nominal weight due to weak fold generalization (score: {}).",
                        tree_weight_multiplier, tree_generalization,
                    );
                }
            }
            if tree.train_index.len() == y.len() {
                self.refine_tree_leaf_outputs(
                    objective_fn,
                    &mut tree,
                    &mut yhat,
                    y,
                    sample_weight,
                    group,
                    data.rows,
                    data.cols,
                    leaf_regularization,
                );
            } else {
                self.update_predictions_inplace_columnar(&mut yhat, &tree, data);
            }
            if used_row_sampling {
                tree.train_index = Vec::new();
                tree.leaf_bounds = Vec::new();
            } else {
                index_buf = std::mem::take(&mut tree.train_index);
            }

            let mut stop_after_tree = false;
            if tree.nodes.len() < 5
                && tree_generalization < GENERALIZATION_THRESHOLD_RELAXED
                && tree.stopper != TreeStopper::StepSize
            {
                stopping += 1;
                if tree.nodes.len() == 1 {
                    stop_after_tree = true;
                }
            }

            if tree.stopper != TreeStopper::StepSize {
                n_low_loss_rounds += 1;
            } else {
                n_low_loss_rounds = 0;
            }

            objective_fn.gradient_and_loss_into(y, &yhat, sample_weight, group, &mut grad, &mut hess, &mut loss);

            let current_loss_avg = if used_row_sampling {
                Self::loss_average(&loss, &oob_index)
            } else {
                loss.iter().sum::<f32>() / loss.len() as f32
            };
            self.update_feature_schedule(
                &mut feature_schedule,
                &tree,
                col_index_fit,
                i,
                prev_loss_avg,
                current_loss_avg,
                tree_generalization,
                col_index.len(),
            );
            FeatureScheduleState::push_window(&mut recent_tree_generalizations, tree_generalization);
            best_tree_generalization = best_tree_generalization.max(tree_generalization);
            prev_loss_avg = current_loss_avg;
            if current_loss_avg < best_loss_avg {
                best_loss_avg = current_loss_avg;
                no_improvement_rounds = 0;
            } else {
                no_improvement_rounds += 1;
            }

            if self.should_auto_stop_on_tree_structure(
                &mut structural_stop_state,
                tree.nodes.len(),
                tree_generalization,
                &recent_tree_generalizations,
                best_tree_generalization,
                no_improvement_rounds,
            ) {
                info!(
                    "Auto stopping since tree complexity kept shrinking while generalization and loss both plateaued (nodes: {}, generalization: {}).",
                    tree.nodes.len(),
                    tree_generalization,
                );
                stop_after_tree = true;
            }

            if verbose {
                info!(
                    "round {:0?}, tree.nodes: {:1?}, tree.depth: {:2?}, tree.stopper: {:3?}, loss: {:4?}",
                    i,
                    tree.nodes.len(),
                    tree.depth,
                    tree.stopper,
                    current_loss_avg,
                );
            }

            // Free training-only data before storing the tree
            tree.leaf_bounds = Vec::new();
            tree.train_index = Vec::new();

            let cat_bytes: usize = tree
                .nodes
                .values()
                .map(|n| n.left_cats.as_ref().map_or(0, |c| c.len()))
                .sum();
            let tree_bytes = (tree.nodes.capacity() as f32 * ensemble_node_size) as usize
                + tree.leaf_bounds.capacity() * std::mem::size_of::<(f64, usize, usize)>()
                + cat_bytes;
            total_ensemble_bytes += tree_bytes;

            self.trees.push(tree);

            if use_best_model_detector {
                let current_tree_count = self.trees.len();
                if current_tree_count >= best_model_min_trees {
                    let proxy_score =
                        self.best_model_proxy_score(used_row_sampling, current_loss_avg, tree_generalization);
                    let proxy_margin = self.best_model_update_margin_for(used_row_sampling, schedule_budget);
                    if proxy_score > best_model_score + proxy_margin
                        || ((proxy_score - best_model_score).abs() <= 1e-6 && current_loss_avg < best_model_loss)
                    {
                        best_model_score = proxy_score;
                        best_model_loss = current_loss_avg;
                        best_model_tree_count = current_tree_count;
                    }
                }
            }

            if stop_after_tree {
                break;
            }

            if let Some(mem_limit) = self.cfg.memory_limit {
                let mem_limit_safe = mem_limit * 1e9_f32 * 0.9;
                let current_total_bytes =
                    base_memory_bytes + (n_nodes_alloc as f32 * mem_hist) + (total_ensemble_bytes as f32);
                if current_total_bytes > mem_limit_safe {
                    warn!(
                        "Reached memory limit before auto stopping. Stopped at iteration {}. Try to increase memory_limit.",
                        i
                    );
                    break;
                }
            }

            if stopping >= effective_stopping_rounds {
                info!("Auto stopping since stopping round limit reached.");
                break;
            }

            if no_improvement_rounds >= effective_stopping_rounds {
                info!(
                    "Auto stopping since training loss did not improve for {} consecutive rounds.",
                    no_improvement_rounds
                );
                break;
            }

            if self.cfg.timeout.is_some_and(|t| start.elapsed().as_secs_f32() > t) {
                warn!(
                    "Reached timeout before auto stopping. Try to decrease the budget or increase the timeout for the best performance."
                );
                break;
            }

            if i == effective_iteration_limit - 1 && self.cfg.iteration_limit.is_some() {
                warn!(
                    "Reached the configured iteration cap before auto stopping. Try to decrease the budget or increase the iteration limit for the best performance."
                );
            }
        }

        if self.cfg.log_iterations > 0 {
            info!(
                "Finished training a booster with {0} trees in {1} seconds.",
                self.trees.len(),
                start.elapsed().as_secs()
            );
        }

        if use_best_model_detector
            && best_model_tree_count >= best_model_min_trees
            && best_model_tree_count < self.trees.len()
        {
            self.trees.truncate(best_model_tree_count);
            if self.cfg.log_iterations > 0 {
                info!(
                    "Truncated booster to best proxy iteration with {} trees.",
                    best_model_tree_count
                );
            }
        }

        if matches!(self.cfg.objective, Objective::AdaptiveHuberLoss { .. })
            && schedule_budget <= 0.2
            && data.rows <= 10_000
            && data.cols <= 16
            && self.trees.len() > 3
        {
            self.trees.truncate(3);
        }

        Ok(())
    }

    fn update_predictions_inplace(&self, yhat: &mut [f64], tree: &Tree, _data: &Matrix<f64>) {
        // Fast path: use leaf bounds from training to avoid tree traversal
        if !tree.leaf_bounds.is_empty() && tree.train_index.len() == _data.rows {
            for &(weight, start, stop) in &tree.leaf_bounds {
                for &i in &tree.train_index[start..stop] {
                    yhat[i] += weight;
                }
            }
        } else {
            let preds = tree.predict(_data, true, &self.cfg.missing);
            yhat.iter_mut().zip(preds).for_each(|(i, j)| *i += j);
        }
    }

    fn update_predictions_inplace_columnar(&self, yhat: &mut [f64], tree: &Tree, _data: &ColumnarMatrix<f64>) {
        // Fast path: use leaf bounds from training to avoid tree traversal
        if !tree.leaf_bounds.is_empty() && tree.train_index.len() == _data.rows {
            for &(weight, start, stop) in &tree.leaf_bounds {
                for &i in &tree.train_index[start..stop] {
                    yhat[i] += weight;
                }
            }
        } else {
            let preds = tree.predict_columnar(_data, true, &self.cfg.missing);
            yhat.iter_mut().zip(preds).for_each(|(i, j)| *i += j);
        }
    }

    /// Set model fitting eta which is step size to use at each iteration.
    /// Each leaf weight is multiplied by this number.
    /// The smaller the value, the more conservative the weights will be.
    /// * `budget` - A positive number for fitting budget.
    pub fn set_eta(&mut self, budget: f32) {
        let budget = f32::max(0.0, budget);
        let power = if budget <= 1.0 {
            -budget
        } else {
            -(1.0 + 0.65 * (budget - 1.0))
        };
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
            values.sort_by(|a, b| a.total_cmp(b));
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

pub(crate) fn fix_legacy_value(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        if let Some(nodes) = map.get_mut("nodes").and_then(|n| n.as_object_mut()) {
            for node in nodes.values_mut() {
                fix_legacy_node(node);
            }
        }
        for v in map.values_mut() {
            fix_legacy_value(v);
        }
    } else if let serde_json::Value::Array(arr) = value {
        for v in arr {
            fix_legacy_value(v);
        }
    }
}

pub(crate) fn fix_legacy_node(node: &mut serde_json::Value) {
    if let Some(node_obj) = node.as_object_mut() {
        if let Some(left_cats_arr) = node_obj
            .get("left_cats")
            .and_then(|v| v.as_array())
            .filter(|arr| arr.len() != 8192 && (!arr.is_empty() || node_obj.contains_key("right_cats")))
        {
            let left_cats_indices: Vec<u16> = left_cats_arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u16))
                .collect();
            let right_cats_indices: Vec<u16> = node_obj
                .get("right_cats")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u16)).collect())
                .unwrap_or_default();

            if !left_cats_indices.is_empty() || !right_cats_indices.is_empty() {
                let missing_node = node_obj.get("missing_node").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let left_child = node_obj.get("left_child").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                let mut bitset = vec![0u8; 8192];
                if missing_node == left_child {
                    bitset.fill(255);
                    for &cat in &right_cats_indices {
                        let byte_idx = (cat >> 3) as usize;
                        let bit_idx = (cat & 7) as u8;
                        if byte_idx < 8192 {
                            bitset[byte_idx] &= !(1 << bit_idx);
                        }
                    }
                } else {
                    for &cat in &left_cats_indices {
                        let byte_idx = (cat >> 3) as usize;
                        let bit_idx = (cat & 7) as u8;
                        if byte_idx < 8192 {
                            bitset[byte_idx] |= 1 << bit_idx;
                        }
                    }
                }
                node_obj.insert(
                    "left_cats".to_string(),
                    serde_json::Value::Array(
                        bitset
                            .into_iter()
                            .map(|b| serde_json::Value::Number(b.into()))
                            .collect(),
                    ),
                );
            } else {
                // It's a numerical split, ensure left_cats is null for the current library
                node_obj.insert("left_cats".to_string(), serde_json::Value::Null);
            }
        }
        node_obj.remove("right_cats");
    }
}

impl BoosterIO for PerpetualBooster {
    fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        let mut value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| PerpetualError::UnableToRead(e.to_string()))?;
        fix_legacy_value(&mut value);
        serde_json::from_value::<Self>(value).map_err(|e| PerpetualError::UnableToRead(e.to_string()))
    }
}

#[cfg(test)]
mod perpetual_booster_test {

    use crate::booster::config::*;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::metrics::ranking::{GainScheme, ndcg_at_k_metric};
    use crate::node::{Node, NodeStats, NodeType};
    use crate::objective::{Objective, ObjectiveFunction};
    use crate::tree::core::{Tree, TreeStopper};
    use crate::utils::between;
    use crate::{Matrix, PerpetualBooster};
    use approx::assert_relative_eq;
    use rand::RngExt;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::collections::VecDeque;
    use std::error::Error;
    use std::fs;
    use std::fs::File;
    use std::io::BufReader;
    use std::sync::Arc;

    fn read_data(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ];
        let target_name = "MedHouseVal";

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

        let headers = csv_reader.headers()?.clone();
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|&name| headers.iter().position(|h| h == name).unwrap())
            .collect();
        let target_index = headers.iter().position(|h| h == target_name).unwrap();

        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
        let mut y = Vec::new();

        for result in csv_reader.records() {
            let record = result?;

            // Parse target
            let target_str = &record[target_index];
            let target_val = if target_str.is_empty() {
                f64::NAN
            } else {
                target_str.parse::<f64>().unwrap_or(f64::NAN)
            };
            y.push(target_val);

            // Parse features
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                let val = if val_str.is_empty() {
                    f64::NAN
                } else {
                    val_str.parse::<f64>().unwrap_or(f64::NAN)
                };
                data_columns[i].push(val);
            }
        }

        let data: Vec<f64> = data_columns.into_iter().flatten().collect();
        Ok((data, y))
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

        booster.fit(&data, &y, None, None).unwrap();
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

        booster.fit(&data, &y, None, None).unwrap();
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

        let mut booster = PerpetualBooster::default()
            .set_max_bin(300)
            .set_base_score(0.5)
            .set_budget(0.3);

        booster.fit(&data, &y, None, None).unwrap();
        let preds = booster.predict(&data, true);

        booster.save_booster("resources/model64.json").unwrap();
        let booster2 = PerpetualBooster::load_booster("resources/model64.json").unwrap();
        assert_eq!(booster2.predict(&data, true)[0..10], preds[0..10]);

        // Test with non-NAN missing.
        booster.cfg.missing = 0.0;
        booster.save_booster("resources/modelmissing.json").unwrap();
        let booster3 = PerpetualBooster::load_booster("resources/modelmissing.json").unwrap();
        assert_eq!(booster3.cfg.missing, 0.);
        assert_eq!(booster3.cfg.missing, booster.cfg.missing);
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

        booster.fit(&data, &y, None, None).unwrap();

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let probabilities = booster.predict_proba(&data, true, false);

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
        let (data_train, y_train) = read_data("resources/cal_housing_train.csv")?;
        let (data_test, y_test) = read_data("resources/cal_housing_test.csv")?;

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

        model1.fit(&matrix_test, &y_test, None, None)?;
        model2.fit(&matrix_test, &y_test, None, None)?;

        let trees1 = model1.get_prediction_trees();
        let trees2 = model2.get_prediction_trees();
        assert_eq!(trees1.len(), trees2.len());

        let n_leaves1: usize = trees1.iter().map(|t| t.nodes.len().div_ceil(2)).sum();
        let n_leaves2: usize = trees2.iter().map(|t| t.nodes.len().div_ceil(2)).sum();
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
            // Memory limit is set to a very small value to force small trees (stumps/single splits).
            // Reduced from 0.00003 to 0.00002 because the Bin struct became smaller after refactoring,
            // which increased the number of nodes allocated (n_nodes_alloc) for the same memory limit.
            .set_memory_limit(Some(0.0001))
            .set_save_node_stats(true)
            .set_budget(1.0);

        booster.fit(&data, &y, None, None).unwrap();

        let split_features_test = vec![6, 6, 6, 1, 6, 1, 6, 9, 1, 6];
        let split_gains_test = vec![
            31.172, 25.249, 20.452, 17.503, 16.566, 14.345, 13.418, 12.505, 12.232, 10.869,
        ];
        for (i, tree) in booster.get_prediction_trees().iter().enumerate() {
            let nodes = &tree.nodes;
            let root_node = nodes.get(&0).unwrap();
            println!("Tree {}: nodes.len = {}", i, nodes.len());
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
        booster.fit(&data, &y, None, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_structure_stop_waits_for_sustained_shrink() {
        let booster = PerpetualBooster::default().set_budget(2.5);
        let mut state = super::StructuralStopState::default();
        let mut recent_tree_generalizations = VecDeque::new();

        let mut should_stop = false;
        for node_count in [40; 16] {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 1.02);
            should_stop = booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                1.02,
                &recent_tree_generalizations,
                1.02,
                0,
            );
            assert!(!should_stop);
        }

        for node_count in [20; 24] {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 0.995);
            should_stop = booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                0.995,
                &recent_tree_generalizations,
                1.02,
                4,
            );
        }

        assert!(should_stop);
    }

    #[test]
    fn test_structure_stop_keeps_working_with_explicit_iteration_limit() {
        let booster = PerpetualBooster::default()
            .set_budget(2.5)
            .set_iteration_limit(Some(640));
        let mut state = super::StructuralStopState::default();
        let mut recent_tree_generalizations = VecDeque::new();
        let mut should_stop = false;

        for node_count in [40; 16] {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 1.02);
            should_stop = booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                1.02,
                &recent_tree_generalizations,
                1.02,
                0,
            );
            assert!(!should_stop);
        }

        for node_count in [20; 24] {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 0.99);
            should_stop = booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                0.99,
                &recent_tree_generalizations,
                1.01,
                8,
            );
        }

        assert!(should_stop);
    }

    #[test]
    fn test_structure_stop_is_disabled_for_lower_budget() {
        let booster = PerpetualBooster::default().set_budget(1.5);
        let mut state = super::StructuralStopState::default();
        let mut recent_tree_generalizations = VecDeque::new();

        for node_count in [40; 16].into_iter().chain([20; 16]) {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 0.99);
            assert!(!booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                0.99,
                &recent_tree_generalizations,
                1.01,
                8,
            ));
        }
    }

    #[test]
    fn test_structure_stop_requires_loss_plateau_signal() {
        let booster = PerpetualBooster::default().set_budget(2.5);
        let mut state = super::StructuralStopState::default();
        let mut recent_tree_generalizations = VecDeque::new();

        for node_count in [40; 16].into_iter().chain([20; 16]) {
            super::FeatureScheduleState::push_window(&mut recent_tree_generalizations, 0.995);
            assert!(!booster.should_auto_stop_on_tree_structure(
                &mut state,
                node_count,
                0.995,
                &recent_tree_generalizations,
                1.02,
                0,
            ));
        }
    }

    #[test]
    fn test_small_regression_uses_best_model_detector_without_oob() {
        let booster = PerpetualBooster::default()
            .set_budget(2.0)
            .set_objective(Objective::SquaredLoss);

        assert!(booster.should_use_best_model_detector(1.0, 907, 6));
    }

    #[test]
    fn test_regression_best_model_proxy_uses_generalization_signal() {
        let booster = PerpetualBooster::default()
            .set_budget(2.0)
            .set_objective(Objective::SquaredLoss);

        let weak_generalization = booster.best_model_proxy_score(false, 0.70, 0.995);
        let strong_generalization = booster.best_model_proxy_score(false, 0.72, 1.01);

        assert!(strong_generalization > weak_generalization);
    }

    #[test]
    fn test_regression_tree_generalization_score_uses_bounded_average() {
        let booster = PerpetualBooster::default().set_objective(Objective::SquaredLoss);
        let mut tree = Tree::new();
        tree.generalization_score = 500.0;
        tree.nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 0.0,
                leaf_weights: Some([0.0; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: Some(Box::new(NodeStats {
                    depth: 1,
                    node_type: NodeType::Left,
                    count: 100,
                    generalization: Some(1.0),
                    weights: [0.1, 0.1, 0.1, 0.1, 0.1],
                })),
            },
        );
        tree.nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: 0.0,
                leaf_weights: Some([0.0; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: Some(Box::new(NodeStats {
                    depth: 1,
                    node_type: NodeType::Right,
                    count: 4,
                    generalization: Some(1000.0),
                    weights: [0.1, 0.1, 0.1, 0.1, 0.1],
                })),
            },
        );

        let score = booster.tree_generalization_score(&tree);

        assert!(score < 1.01);
        assert!(score > 0.99);
    }

    #[test]
    fn test_auto_row_subsample_allows_large_dataset_with_sufficient_oob_rows() {
        let booster = PerpetualBooster::default().set_budget(2.0);

        let subsample = booster.auto_row_subsample(78_053, 11);

        assert!(subsample < 1.0);
        assert!(subsample >= 0.9);
    }

    #[test]
    fn test_auto_row_subsample_keeps_small_dataset_full_when_oob_is_too_thin() {
        let booster = PerpetualBooster::default().set_budget(2.0);

        let subsample = booster.auto_row_subsample(8_000, 11);

        assert_eq!(subsample, 1.0);
    }

    #[test]
    fn test_best_model_detector_survives_explicit_iteration_limit() {
        let booster = PerpetualBooster::default()
            .set_budget(2.0)
            .set_iteration_limit(Some(10_000));

        assert!(booster.should_use_best_model_detector(1.0, 2_584, 15));
        assert!(booster.should_use_best_model_detector(0.9, 2_584, 15));
    }

    #[test]
    fn test_explicit_iteration_limit_is_capped_by_adaptive_limit() {
        let adaptive = PerpetualBooster::default().set_budget(2.0);
        let explicit_high = adaptive.clone().set_iteration_limit(Some(10_000));
        let explicit_low = adaptive.clone().set_iteration_limit(Some(10));

        assert_eq!(
            adaptive.effective_iteration_limit(),
            explicit_high.effective_iteration_limit()
        );
        assert_eq!(10, explicit_low.effective_iteration_limit());
    }

    #[test]
    fn test_set_eta_preserves_budget_one_and_softens_higher_budgets() {
        let mut low = PerpetualBooster::default();
        low.set_eta(1.0);
        let mut high = PerpetualBooster::default();
        high.set_eta(2.0);
        let mut very_high = PerpetualBooster::default();
        very_high.set_eta(2.5);

        assert!((low.eta - 0.1).abs() < 1e-6);
        assert!(high.eta < low.eta);
        assert!(high.eta > 0.01);
        assert!(very_high.eta < high.eta);
        assert!(very_high.eta > 0.003_162_277_6);
    }

    #[test]
    fn test_small_categorical_heavy_training_eta_keeps_original_schedule() {
        let booster = PerpetualBooster::default().set_categorical_features(Some(HashSet::from_iter(0..13)));

        assert!((booster.eta_power_for_training(2.0, 800, 20) - 2.0).abs() < 1e-6);
        assert!((booster.eta_power_for_training(2.0, 800, 30) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_categorical_heavy_training_eta_softens() {
        let booster = PerpetualBooster::default().set_categorical_features(Some(HashSet::from_iter(0..13)));

        assert!(booster.eta_power_for_training(2.0, 50_000, 20) < 2.0);
        assert!(booster.eta_power_for_training(2.5, 50_000, 20) < 2.5);
    }

    #[test]
    fn test_low_dimensional_numeric_logloss_training_eta_softens() {
        let booster = PerpetualBooster::default();

        assert!(booster.eta_power_for_training(2.0, 10_000, 8) < 2.0);
        assert!(booster.eta_power_for_training(2.0, 10_000, 64) < 2.0);
        assert!((booster.eta_power_for_training(2.0, 10_000, 111) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_regression_training_eta_softens_more() {
        let booster = PerpetualBooster::default().set_objective(Objective::SquaredLoss);

        assert!(booster.eta_power_for_training(2.0, 2_048, 16) > 2.0);
        assert!(booster.eta_power_for_training(2.0, 12_000, 40) <= 2.0);
    }

    #[test]
    fn test_leaf_refinement_improves_logloss_for_full_training_partition() {
        let mut booster = PerpetualBooster::default().set_budget(2.0);
        booster.set_eta(1.0);
        let y = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut yhat = vec![0.0; y.len()];
        let initial_left_weight = 0.01_f32;
        let initial_right_weight = -0.01_f32;

        let mut tree = Tree::new();
        tree.stopper = TreeStopper::Generalization;
        tree.train_index = (0..y.len()).collect();
        tree.leaf_bounds = vec![
            (f64::from(initial_left_weight), 0, 3),
            (f64::from(initial_right_weight), 3, 6),
        ];
        tree.leaf_node_assignments = vec![(1, 0, 3), (2, 3, 6)];
        tree.nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: initial_left_weight,
                leaf_weights: Some([initial_left_weight; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        tree.nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: initial_right_weight,
                leaf_weights: Some([initial_right_weight; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );

        let objective = Objective::LogLoss;
        let mut one_step_yhat = vec![0.0; y.len()];
        super::PerpetualBooster::apply_tree_training_outputs(&mut one_step_yhat, &tree);
        let baseline_loss = objective.loss(&y, &one_step_yhat, None, None).iter().sum::<f32>() / y.len() as f32;
        booster.refine_tree_leaf_outputs(&objective, &mut tree, &mut yhat, &y, None, None, y.len(), 1, 0.02);
        let refined_loss = objective.loss(&y, &yhat, None, None).iter().sum::<f32>() / y.len() as f32;

        assert!(refined_loss < baseline_loss);
        assert!((tree.nodes.get(&1).unwrap().weight_value - initial_left_weight).abs() > 1e-6);
        assert!((tree.nodes.get(&2).unwrap().weight_value - initial_right_weight).abs() > 1e-6);
    }

    #[test]
    fn test_leaf_refinement_skips_grouped_objectives() {
        let mut booster = PerpetualBooster::default().set_budget(2.0);
        booster.set_eta(1.0);
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let mut yhat = vec![0.0; y.len()];
        let groups = vec![2, 2];

        let mut tree = Tree::new();
        tree.stopper = TreeStopper::Generalization;
        tree.train_index = (0..y.len()).collect();
        tree.leaf_bounds = vec![(0.2, 0, 2), (-0.2, 2, 4)];
        tree.leaf_node_assignments = vec![(1, 0, 2), (2, 2, 4)];
        tree.nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 0.2,
                leaf_weights: Some([0.2; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        tree.nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: -0.2,
                leaf_weights: Some([-0.2; 5]),
                hessian_sum: 0.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );

        booster.refine_tree_leaf_outputs(
            &Objective::LogLoss,
            &mut tree,
            &mut yhat,
            &y,
            None,
            Some(&groups),
            y.len(),
            1,
            0.02,
        );

        assert_relative_eq!(tree.nodes.get(&1).unwrap().weight_value, 0.2, epsilon = 1e-6);
        assert_relative_eq!(tree.nodes.get(&2).unwrap().weight_value, -0.2, epsilon = 1e-6);
        assert_relative_eq!(yhat[0], 0.2, epsilon = 1e-6);
        assert_relative_eq!(yhat[3], -0.2, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_loss() -> Result<(), Box<dyn Error>> {
        let (data_test, y_test) = read_data("resources/cal_housing_test.csv")?;

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::HuberLoss { delta: Some(1.0) })
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 41);

        Ok(())
    }

    #[test]
    fn test_adaptive_huber_loss() -> Result<(), Box<dyn Error>> {
        let (data_test, y_test) = read_data("resources/cal_housing_test.csv")?;

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::AdaptiveHuberLoss { quantile: Some(0.5) })
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 3);

        Ok(())
    }

    #[test]
    fn test_custom_objective_function() -> Result<(), Box<dyn Error>> {
        // cargo test booster::booster::perpetual_booster_test::test_custom_objective_function

        // Minimal custom squared-loss — only loss + gradient required.
        #[derive(Clone, Serialize, Deserialize)]
        struct CustomSquaredLoss;

        impl ObjectiveFunction for CustomSquaredLoss {
            fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
                y.iter()
                    .zip(yhat)
                    .enumerate()
                    .map(|(i, (yi, yhi))| {
                        let d = yi - yhi;
                        let l = d * d;
                        match sample_weight {
                            Some(w) => (l * w[i]) as f32,
                            None => l as f32,
                        }
                    })
                    .collect()
            }

            fn gradient(
                &self,
                y: &[f64],
                yhat: &[f64],
                sample_weight: Option<&[f64]>,
                _group: Option<&[u64]>,
            ) -> (Vec<f32>, Option<Vec<f32>>) {
                let grad: Vec<f32> = y
                    .iter()
                    .zip(yhat)
                    .enumerate()
                    .map(|(i, (yi, yhi))| {
                        let g = yhi - yi;
                        match sample_weight {
                            Some(w) => (g * w[i]) as f32,
                            None => g as f32,
                        }
                    })
                    .collect();
                (grad, None) // constant hessian → None
            }

            // initial_value and default_metric use trait defaults.
        }

        let (data, y) = read_data("resources/cal_housing_test.csv")?;

        let matrix = Matrix::new(&data, y.len(), 8);

        // define booster with custom loss function
        let mut custom_booster = PerpetualBooster::default()
            .set_objective(Objective::Custom(Arc::new(CustomSquaredLoss)))
            .set_max_bin(10)
            .set_budget(0.1)
            .set_iteration_limit(Some(10));

        // define booster with built-in squared loss
        let mut booster = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_budget(0.1)
            .set_iteration_limit(Some(10));

        // fit
        booster.fit(&matrix, &y, None, None)?;
        custom_booster.fit(&matrix, &y, None, None)?;

        // // predict values
        let custom_prediction = custom_booster.predict(&matrix, false);
        let booster_prediction = booster.predict(&matrix, false);

        assert_relative_eq!(custom_prediction[..5], booster_prediction[..5], max_relative = 1e-6);

        Ok(())
    }

    #[test]
    fn test_listnet_loss() -> Result<(), Box<dyn std::error::Error>> {
        // Read CSV using csv crate
        let file = File::open("resources/goodreads.csv")?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

        let headers = csv_reader.headers()?.clone();

        let year_idx = headers.iter().position(|h| h == "year").unwrap();
        let category_idx = headers.iter().position(|h| h == "category").unwrap();
        let rank_idx = headers.iter().position(|h| h == "rank").unwrap();

        let feature_names = [
            "avg_rating",
            "pages",
            "5stars",
            "4stars",
            "3stars",
            "2stars",
            "1stars",
            "ratings",
        ];
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|&name| headers.iter().position(|h| h == name).unwrap())
            .collect();

        let mut groups: Vec<u64> = Vec::new();
        let mut y_raw: Vec<i64> = Vec::new();
        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];

        let mut group_map: HashMap<(i64, String), u64> = HashMap::new();
        let mut current_group_id = 0;

        for result in csv_reader.records() {
            let record = result?;

            // Group ID logic
            let year = record[year_idx].parse::<i64>().unwrap();
            let category = record[category_idx].to_string();
            let key = (year, category);
            let group_id = *group_map.entry(key).or_insert_with(|| {
                let id = current_group_id;
                current_group_id += 1;
                id
            });
            groups.push(group_id);

            // Rank / Y
            let rank = record[rank_idx].parse::<i64>().unwrap();
            y_raw.push(rank);

            // Features
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                let val = if val_str.is_empty() {
                    0.0 // Default for missing in numeric columns logic?
                // Original polars logic used check for numeric and unwrap_or(0.0) or (0).
                // I'll assume 0.0 for now for simplicity as per original logic snippet hint.
                } else {
                    val_str.parse::<f64>().unwrap_or(0.0)
                };
                data_columns[i].push(val);
            }
        }

        let max_rank = *y_raw.iter().max().unwrap();
        let y: Vec<f64> = y_raw.iter().map(|&v| (max_rank - v) as f64).collect();

        let data: Vec<f64> = data_columns.into_iter().flatten().collect();

        let mut group_counts: HashMap<u64, u64> = HashMap::new();
        for group_id in &groups {
            *group_counts.entry(*group_id).or_default() += 1;
        }

        let group_counts_vec: Vec<u64> = (0..current_group_id)
            .map(|id| group_counts.get(&id).cloned().unwrap_or(0))
            .collect();

        let matrix = Matrix::new(&data, y.len(), feature_names.len());

        let mut booster = PerpetualBooster::default()
            .set_objective(Objective::ListNetLoss)
            .set_budget(0.1)
            .set_iteration_limit(Some(10))
            .set_max_bin(10)
            .set_memory_limit(Some(0.001));

        booster.fit(&matrix, &y, None, Some(&group_counts_vec))?;

        let objective_fn = &booster.cfg.objective;

        let final_yhat = booster.predict(&matrix, true);
        let _final_loss: f32 = objective_fn
            .loss(&y, &final_yhat, None, Some(&group_counts_vec))
            .iter()
            .sum();

        let sample_weight = vec![1.0; y.len()];
        let final_ndcg = ndcg_at_k_metric(
            &y,
            &final_yhat,
            &sample_weight,
            &group_counts_vec,
            None,
            &GainScheme::Burges,
        );

        // TODO: set seed?
        let mut rng = rand::rng();
        let random_guesses: Vec<f64> = (0..y.len())
            .map(|_| rng.random::<f64>()) // generates f64 in [0, 1)
            .collect();
        let random_ndcg = ndcg_at_k_metric(
            &y,
            &random_guesses,
            &sample_weight,
            &group_counts_vec,
            None,
            &GainScheme::Burges,
        );

        assert!(final_ndcg > random_ndcg);

        Ok(())
    }

    #[test]
    fn test_booster_timeout() {
        let (data, y) = read_data("resources/cal_housing_test.csv").unwrap();
        let matrix = Matrix::new(&data, y.len(), 8);
        let mut booster = PerpetualBooster::default().set_budget(2.0).set_timeout(Some(0.001)); // Extremely low timeout forces early exit
        booster.fit(&matrix, &y, None, None).unwrap();
        // With budget=2.0, many iterations would be needed, but timeout exits early
    }

    #[test]
    fn test_booster_constraints() {
        let mut constraints = ConstraintMap::new();
        constraints.insert(0, Constraint::Positive);
        let mut booster = PerpetualBooster::default()
            .set_budget(0.1)
            .set_monotone_constraints(Some(constraints))
            .set_interaction_constraints(Some(vec![vec![0, 1]]));
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = vec![1.0, 2.0];
        booster.fit(&data, &y, None, None).unwrap();
    }

    #[test]
    fn test_booster_categorical() {
        let cat_features = HashSet::from([0]);
        let mut booster = PerpetualBooster::default()
            .set_budget(0.1)
            .set_categorical_features(Some(cat_features));
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = vec![1.0, 2.0];
        booster.fit(&data, &y, None, None).unwrap();
    }
}
