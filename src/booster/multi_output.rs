//! Multi-Output Booster
//!
//! A wrapper around multiple [`PerpetualBooster`] instances for multi-target
//! regression or classification tasks.
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::booster::config::MissingNodeTreatment;
use crate::booster::config::*;
use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};

/// Multi-Output Gradient Boosting Machine.
///
/// Wraps `n_boosters` independent [`PerpetualBooster`] instances — one per target column —
/// and exposes a unified `fit` / `predict` API.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiOutputBooster {
    /// Number of independent boosters (one per output).
    pub n_boosters: usize,
    /// Shared configuration applied to every sub-booster.
    pub cfg: BoosterConfig,
    /// The individual [`PerpetualBooster`] instances.
    pub boosters: Vec<PerpetualBooster>,
    /// Arbitrary metadata key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Default for MultiOutputBooster {
    fn default() -> Self {
        let cfg = BoosterConfig::default();
        let n_boosters = 1;
        let boosters = vec![{
            PerpetualBooster {
                cfg: cfg.clone(),
                ..Default::default()
            }
        }];

        Self {
            n_boosters,
            cfg,
            boosters,
            metadata: HashMap::new(),
        }
    }
}

impl MultiOutputBooster {
    /// Create a new `MultiOutputBooster`.
    ///
    /// * `n_boosters` - Number of independent boosters (one per target column).
    /// * `objective` - The name of objective function used to optimize. Valid options are:
    ///   "LogLoss" to use logistic loss as the objective function,
    ///   "SquaredLoss" to use Squared Error as the objective function,
    ///   "QuantileLoss" for quantile regression.
    ///   "AdaptiveHuberLoss" for adaptive huber loss regression.
    ///   "HuberLoss" for huber loss regression.
    ///   "ListNetLoss" for listnet loss ranking.
    /// * `budget` - budget to fit the model.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    /// * `num_threads` - Number of threads to be used during training
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///   between the training features and the target variable.
    /// * `force_children_to_bound_parent` - force_children_to_bound_parent.
    /// * `missing` - Value to consider missing.
    /// * `allow_missing_splits` - Whether the algorithm allows splits that completely separate
    ///   missing and non-missing values. When `create_missing_branch` is true, setting this to
    ///   true will result in the missing branch being further split.
    /// * `create_missing_branch` - Should missing be split out into its own separate branch?
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
    /// * `save_node_stats` - whether to save node statistics during training.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_boosters: usize,
        objective: Objective,
        budget: f32,
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
        save_node_stats: bool,
        calibration_method: CalibrationMethod,
    ) -> Result<Self, PerpetualError> {
        // Build the common configuration object.
        let cfg = BoosterConfig {
            objective: objective.clone(),
            budget,
            max_bin,
            num_threads,
            monotone_constraints: monotone_constraints.clone(),
            interaction_constraints: interaction_constraints.clone(),
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
            save_node_stats,
            calibration_method,
        };

        // Base booster template that child boosters will clone.
        let template_booster = {
            PerpetualBooster {
                cfg: cfg.clone(),
                ..Default::default()
            }
        };
        template_booster.validate_parameters()?;

        // Assemble the wrapper with `n_boosters` copies.
        let boosters = vec![template_booster; n_boosters.max(1)];

        Ok(MultiOutputBooster {
            n_boosters: n_boosters.max(1),
            cfg,
            boosters,
            metadata: HashMap::new(),
        })
    }

    /// Reset all boosters.
    pub fn reset(&mut self) {
        for b in &mut self.boosters {
            b.reset();
        }
    }

    /// Fit the multi-output booster.
    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            let _ = self.boosters[i].fit(data, y.get_col(i), sample_weight, group);
        }
        Ok(())
    }

    /// Fit the multi-output booster on columnar data.
    pub fn fit_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            let _ = self.boosters[i].fit_columnar(data, y.get_col(i), sample_weight, group);
        }
        Ok(())
    }

    /// Prune the trees in the boosters.
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            let _ = self.boosters[i].prune(data, y.get_col(i), sample_weight, group);
        }
        Ok(())
    }

    /// Calibrate the boosters using a selected non-conformal method.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - A tuple of (features, targets, alphas) representing the dedicated calibration set.
    pub fn calibrate(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&Matrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        self.cfg.calibration_method = method;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate(method, (x_cal, y_cal_col, alpha))?;
        }
        Ok(())
    }

    /// Calibrate the boosters using Conformal Prediction (CQR).
    pub fn calibrate_conformal(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        data_cal: (&Matrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        self.cfg.calibration_method = CalibrationMethod::Conformal;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_conformal(
                data,
                y.get_col(i),
                sample_weight,
                group,
                (x_cal, y_cal_col, alpha),
            )?;
        }
        Ok(())
    }

    /// Calibrate the boosters on columnar data using a selected non-conformal method.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - Dedicated calibration set (features, targets, alphas).
    pub fn calibrate_columnar(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&ColumnarMatrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        self.cfg.calibration_method = method;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_columnar(method, (x_cal, y_cal_col, alpha))?;
        }
        Ok(())
    }

    /// Calibrate the boosters on columnar data using Conformal Prediction (CQR).
    pub fn calibrate_conformal_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        data_cal: (&ColumnarMatrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        self.cfg.calibration_method = CalibrationMethod::Conformal;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_conformal_columnar(
                data,
                y.get_col(i),
                sample_weight,
                group,
                (x_cal, y_cal_col, alpha),
            )?;
        }
        Ok(())
    }

    /// Get the boosters
    pub fn get_boosters(&self) -> &[PerpetualBooster] {
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
        let tree_objective = objective.clone();

        self.boosters = self
            .boosters
            .into_iter()
            .map(|b| b.set_objective(tree_objective.clone()))
            .collect();

        self.cfg.objective = objective;

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

    /// Set the interaction_constraints on the booster.
    /// * `interaction_constraints` - The interaction constraints of the booster.
    pub fn set_interaction_constraints(mut self, interaction_constraints: Option<Vec<Vec<usize>>>) -> Self {
        self.cfg.interaction_constraints = interaction_constraints.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_interaction_constraints(interaction_constraints.clone()))
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
    ///   branch.
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
    /// * `terminate_missing_features` - Hashset of the feature indices for the features that should always terminate the missing node, if create_missing_branch is true.
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
            .map(|b| b.clone().set_missing_node_treatment(missing_node_treatment))
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

    /// Set whether to save node stats on the booster.
    /// * `save_node_stats` - Whether to save node statistics during training.
    pub fn set_save_node_stats(mut self, save_node_stats: bool) -> Self {
        self.cfg.save_node_stats = save_node_stats;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_save_node_stats(save_node_stats))
            .collect();
        self
    }

    /// Set the calibration_method on the booster.
    /// * `calibration_method` - The calibration method of the booster.
    pub fn set_calibration_method(mut self, calibration_method: CalibrationMethod) -> Self {
        self.cfg.calibration_method = calibration_method;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_calibration_method(calibration_method))
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

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        self.boosters
            .iter()
            .map(|b| b.value_partial_dependence(feature, value))
            .sum::<f64>()
            / self.n_boosters as f64
    }

    /// Calculate feature importance measure for the features
    /// in the model.
    /// - `method`: variable importance method to use.
    /// - `normalize`: whether to normalize the importance values with the sum.
    pub fn calculate_feature_importance(&self, method: ImportanceMethod, normalize: bool) -> HashMap<usize, f32> {
        let cumulative_importance = self.boosters.iter().fold(HashMap::new(), |mut acc, booster| {
            let importance = booster.calculate_feature_importance(method.clone(), normalize);
            for (feature, value) in importance {
                *acc.entry(feature).or_insert(0.0) += value;
            }
            acc
        });
        cumulative_importance
            .into_iter()
            .map(|(k, v)| (k, v / self.n_boosters as f32))
            .collect()
    }
}

impl BoosterIO for MultiOutputBooster {
    fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        let mut value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| PerpetualError::UnableToRead(e.to_string()))?;
        crate::booster::core::fix_legacy_value(&mut value);
        serde_json::from_value::<Self>(value).map_err(|e| PerpetualError::UnableToRead(e.to_string()))
    }
}

#[cfg(test)]
mod multi_output_booster_test {

    use crate::Matrix;
    use crate::objective::Objective;
    use crate::{MultiOutputBooster, utils::between};
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;

    fn read_data(path: &str, feature_names: &[&str]) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let target_name = "Cover_Type";

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
    fn test_multi_output_booster() -> Result<(), Box<dyn Error>> {
        let n_classes = 7;
        let n_columns = 54;
        let n_rows = 500;
        let max_bin = 5;

        let mut features: Vec<&str> = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area_0",
            "Wilderness_Area_1",
            "Wilderness_Area_2",
            "Wilderness_Area_3",
        ]
        .to_vec();

        let soil_types = (0..40).map(|i| format!("{}_{}", "Soil_Type", i)).collect::<Vec<_>>();
        let s_types = soil_types.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        features.extend(s_types);

        // Read data using csv crate
        // NOTE: The original test performed a `.head(Some(n_rows))` operation via polars.
        // We will read all and then slice, or we can just use all if n_rows is small enough.
        // n_rows is 500. `resources/cover_types_test.csv` might be larger.
        // But since this is a test, let's just use 500 rows to match original behavior exactly for performance.

        // Actually, slicing column-major data is tedious.
        // Let's modify `read_data` to take a limit optionally?
        // Or just read everything and slice `y` and `data`.
        // `data` is [col1_val1...col1_valN, col2_val1...].
        // To slice 500 rows, we need to reconstruct new vector.

        let (data_full, y_full) = read_data("resources/cover_types_test.csv", &features)?;

        let rows_full = y_full.len();
        let limit = n_rows.min(rows_full);

        let mut data = Vec::new();
        // Extract n_columns columns
        for c in 0..n_columns {
            let col_start = c * rows_full;
            data.extend_from_slice(&data_full[col_start..col_start + limit]);
        }
        let y_test = y_full[0..limit].to_vec();

        // Create Matrix from ndarray.
        let data_matrix = Matrix::new(&data, y_test.len(), n_columns);

        let mut y_vec: Vec<Vec<f64>> = Vec::new();
        for i in 0..n_classes {
            y_vec.push(
                y_test
                    .iter()
                    .map(|y| if (*y as usize) == (i + 1) { 1.0 } else { 0.0 })
                    .collect(),
            );
        }
        let y_data = y_vec.into_iter().flatten().collect::<Vec<f64>>();
        let y = Matrix::new(&y_data, y_test.len(), n_classes);

        let mut booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_max_bin(max_bin)
            .set_n_boosters(n_classes)
            .set_budget(0.1)
            .set_iteration_limit(Some(5))
            .set_memory_limit(Some(0.001));

        println!("The number of boosters: {:?}", booster.get_boosters().len());
        assert!(booster.get_boosters().len() == n_classes);

        booster.fit(&data_matrix, &y, None, None).unwrap();

        let probas = booster.predict_proba(&data_matrix, true);

        assert!(between(0.999, 1.001, probas[0..n_classes].iter().sum::<f64>() as f32));

        Ok(())
    }
}
