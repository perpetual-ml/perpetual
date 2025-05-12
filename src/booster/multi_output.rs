use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::objective_functions::Objective;
use crate::{Matrix, PerpetualBooster};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;

use super::booster::MissingNodeTreatment;

/// Perpetual Booster object
#[derive(Deserialize, Serialize, Clone)]
pub struct MultiOutputBooster {
    /// The number of boosters to fit.
    pub n_boosters: usize,
    /// The name of objective function used to optimize.
    /// Valid options include "LogLoss" to use logistic loss as the objective function,
    /// or "SquaredLoss" to use Squared Error as the objective function.
    pub objective: Objective,
    /// Budget to fit the model.
    #[serde(default = "default_budget")]
    pub budget: f32,
    /// Number of bins to calculate to partition the data. Setting this to
    /// a smaller number, will result in faster training time, while potentially sacrificing
    /// accuracy. If there are more bins, than unique values in a column, all unique values
    /// will be used.
    pub max_bin: u16,
    /// Number of threads to be used during training.
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
    // Members internal to the multi output booster object, and not parameters set by the user.
    // Boosters is public, just to interact with it directly in the python wrapper.
    pub boosters: Vec<PerpetualBooster>,
    // Metadata for the multi output booster
    metadata: HashMap<String, String>,
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

impl Default for MultiOutputBooster {
    fn default() -> Self {
        Self::new(
            1,
            Objective::LogLoss,
            0.5,
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

impl MultiOutputBooster {
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
    ) -> Result<Self, PerpetualError> {
        let booster_objective = objective.clone();
        let booster_monotone_constraints = monotone_constraints.clone();
        let booster_terminate_missing_features = terminate_missing_features.clone();
        let booster_categorical_features = categorical_features.clone();

        let booster_timeout = match timeout {
            Some(t) => Some(t / n_boosters as f32),
            None => None,
        };

        let mut multi_output_booster = MultiOutputBooster {
            n_boosters,
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
            boosters: Vec::new(),
            metadata: HashMap::new(),
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        };

        let booster = PerpetualBooster::default()
            .set_objective(booster_objective)
            .set_budget(budget)
            .set_max_bin(max_bin)
            .set_num_threads(num_threads)
            .set_monotone_constraints(booster_monotone_constraints)
            .set_force_children_to_bound_parent(force_children_to_bound_parent)
            .set_missing(missing)
            .set_allow_missing_splits(allow_missing_splits)
            .set_create_missing_branch(create_missing_branch)
            .set_terminate_missing_features(booster_terminate_missing_features)
            .set_missing_node_treatment(missing_node_treatment)
            .set_log_iterations(log_iterations)
            .set_seed(seed)
            .set_quantile(quantile)
            .set_reset(reset)
            .set_categorical_features(booster_categorical_features)
            .set_timeout(booster_timeout)
            .set_iteration_limit(iteration_limit)
            .set_memory_limit(memory_limit)
            .set_stopping_rounds(stopping_rounds);

        booster.validate_parameters()?;

        for _ in 0..n_boosters {
            multi_output_booster.boosters.push(booster.clone());
        }

        Ok(multi_output_booster)
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

    /// Generate predictions on data using the multi-output booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        self.boosters
            .iter()
            .map(|b| b.predict(data, parallel))
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>()
    }

    /// Generate probabilities on data using the multi-output booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_proba(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        let log_odds = self.predict(data, parallel);
        let data_log_odds = Matrix::new(&log_odds, data.rows, data.cols);
        let mut preds = Vec::with_capacity(log_odds.len());
        for row in 0..data.rows {
            let y_p_exp = data_log_odds.get_row(row).iter().map(|e| e.exp()).collect::<Vec<f64>>();
            let y_p_exp_sum = y_p_exp.iter().sum::<f64>();
            let probabilities = y_p_exp.iter().map(|e| e / y_p_exp_sum).collect::<Vec<f64>>();
            preds.extend(probabilities);
        }
        preds
    }

    /// Generate node predictions on data using the gradient booster.
    ///
    /// * `data` -  Either a Polars or Pandas DataFrame, or a 2 dimensional Numpy array.
    /// * `parallel` -  Predict in parallel.
    pub fn predict_nodes(&self, data: &Matrix<f64>, parallel: bool) -> Vec<Vec<Vec<HashSet<usize>>>> {
        self.boosters.iter().map(|b| b.predict_nodes(data, parallel)).collect()
    }

    /// Get the boosters
    pub fn get_boosters(&self) -> &[PerpetualBooster] {
        &self.boosters
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

    /// Load a multi-output booster from Json string
    ///
    /// * `json_str` - String object, which can be serialized to json.
    pub fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        let model = serde_json::from_str::<MultiOutputBooster>(json_str);
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
        self.objective = objective.clone();
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
        self.budget = budget;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_budget(budget)).collect();
        self
    }

    /// Set the number of bins on the booster.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_max_bin(mut self, max_bin: u16) -> Self {
        self.max_bin = max_bin;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_max_bin(max_bin)).collect();
        self
    }

    /// Set the number of threads on the booster.
    /// * `num_threads` - Set the number of threads to be used during training.
    pub fn set_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.num_threads = num_threads;
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
        self.monotone_constraints = monotone_constraints.clone();
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
        self.force_children_to_bound_parent = force_children_to_bound_parent;
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
        self.missing = missing;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_missing(missing)).collect();
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.create_missing_branch = allow_missing_splits;
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
        self.create_missing_branch = create_missing_branch;
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
        self.terminate_missing_features = terminate_missing_features.clone();
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
        self.missing_node_treatment = missing_node_treatment;
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
        self.log_iterations = log_iterations;
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
        self.seed = seed;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_seed(seed)).collect();
        self
    }

    /// Set the quantile on the booster.
    /// * `quantile` - used only in quantile regression.
    pub fn set_quantile(mut self, quantile: Option<f64>) -> Self {
        self.quantile = quantile;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_quantile(quantile)).collect();
        self
    }

    /// Set the reset on the booster.
    /// * `reset` - Reset the model or continue training.
    pub fn set_reset(mut self, reset: Option<bool>) -> Self {
        self.reset = reset;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_reset(reset)).collect();
        self
    }

    /// Set the categorical features on the booster.
    /// * `categorical_features` - categorical features.
    pub fn set_categorical_features(mut self, categorical_features: Option<HashSet<usize>>) -> Self {
        self.categorical_features = categorical_features.clone();
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
        self.timeout = timeout;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_timeout(timeout)).collect();
        self
    }

    /// Set the iteration limit on the booster.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    pub fn set_iteration_limit(mut self, iteration_limit: Option<usize>) -> Self {
        self.iteration_limit = iteration_limit;
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
        self.memory_limit = memory_limit;
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
        self.stopping_rounds = stopping_rounds;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::between;
    use polars::{
        io::SerReader,
        prelude::{CsvReadOptions, DataType},
    };
    use std::error::Error;

    #[test]
    fn test_multi_output_booster() -> Result<(), Box<dyn Error>> {
        let n_classes = 7;
        let n_columns = 54;
        let n_rows = 1000;
        let max_bin = 10;

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

        let mut features_and_target = features.clone();
        features_and_target.push("Cover_Type");

        let features_and_target_arc = features_and_target
            .iter()
            .map(|s| String::from(s.to_owned()))
            .collect::<Vec<String>>()
            .into();

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(features_and_target_arc))
            .try_into_reader_with_file_path(Some("resources/cover_types_test.csv".into()))?
            .finish()
            .unwrap()
            .head(Some(n_rows));

        // Get data in column major format...
        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(&features, &id_vars_test)?;

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
                .column("Cover_Type")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let data = Matrix::new(&data_test, y_test.len(), n_columns);

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
            .set_timeout(Some(60.0));

        println!("The number of boosters: {:?}", booster.get_boosters().len());
        assert!(booster.get_boosters().len() == n_classes);

        booster.fit(&data, &y, None).unwrap();

        let probas = booster.predict_proba(&data, true);

        assert!(between(0.999, 1.001, probas[0..n_classes].iter().sum::<f64>() as f32));

        Ok(())
    }
}
