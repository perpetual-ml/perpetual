//! Booster Configuration
//!
//! Defines the configuration structures and enums used by the Perpetual Booster,
//! including contribution methods, importance methods, and missing-value strategies.
use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Methods for calculating feature contributions.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum ContributionsMethod {
    /// Saabas-style contributions using leaf weights.
    Weight,
    /// Internal node averages (equivalent to XGBoost's `approx_contribs`).
    Average,
    /// Difference between the weight of the traveled branch and the non-traveled branch.
    BranchDifference,
    /// Weighted difference between branches relative to their midpoint.
    MidpointDifference,
    /// Difference from the node with the largest coverage (the mode).
    ModeDifference,
    /// Probability change for LogLoss tasks.
    ProbabilityChange,
    /// Exact tree-SHAP values.
    Shapley,
}

/// Method to calculate variable importance.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
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

/// Strategies for handling missing values during training.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum MissingNodeTreatment {
    /// No constraints on missing node weights.
    None,
    /// Inherit the parent's weight.
    AssignToParent,
    /// Weighted average of leaf children, recursively updated up the tree.
    AverageLeafWeight,
    /// Simple weighted average of immediate left and right children.
    AverageNodeWeight,
}

/// Methods for prediction interval calibration.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
pub enum CalibrationMethod {
    /// Method 1: Min-Max interval from fold weights.
    MinMax,
    /// Method 2: Global Relative Position interpolation.
    GRP,
    /// Method 3: Weight Variance (Standard Deviation of fold weights).
    #[default]
    WeightVariance,
    /// Conformal Prediction (CQR).
    Conformal,
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
pub(crate) fn parse_missing<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or(f64::NAN))
}

pub(crate) fn parse_f32<'de, D>(d: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(d).map(|x: Option<_>| x.unwrap_or(f32::NAN))
}
fn default_interaction_constraints() -> Option<Vec<Vec<usize>>> {
    None
}

// Common Booster configuration
// across
/// Configuration for the `PerpetualBooster`.
#[derive(Clone, Serialize, Deserialize)]
pub struct BoosterConfig {
    /// Learning objective.
    pub objective: Objective,
    /// Fitting budget.
    #[serde(default = "default_budget")]
    pub budget: f32,
    /// Maximum number of bins for discretization.
    pub max_bin: u16,
    /// Number of threads for parallel tasks.
    pub num_threads: Option<usize>,
    /// Monotonicity constraints.
    pub monotone_constraints: Option<ConstraintMap>,
    /// Interaction constraints.
    #[serde(default = "default_interaction_constraints")]
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    /// Whether to restrict child node weights within parent range.
    #[serde(default = "default_force_children_to_bound_parent")]
    pub force_children_to_bound_parent: bool,
    /// Representation of missing values.
    #[serde(deserialize_with = "parse_missing")]
    pub missing: f64,
    /// Whether to allow splits isolating missing values.
    pub allow_missing_splits: bool,
    /// Whether to use ternary trees with explicit missing branches.
    pub create_missing_branch: bool,
    /// Features for which missing branches are terminated early.
    #[serde(default = "default_terminate_missing_features")]
    pub terminate_missing_features: HashSet<usize>,
    /// Strategy for calculating missing node weights.
    #[serde(default = "default_missing_node_treatment")]
    pub missing_node_treatment: MissingNodeTreatment,
    /// Logging frequency (every N iterations).
    #[serde(default = "default_log_iterations")]
    pub log_iterations: usize,
    /// Seed for random number generation.
    pub seed: u64,
    /// Quantile for quantile regression.
    #[serde(default = "default_quantile")]
    pub quantile: Option<f64>,
    /// Whether to reset or continue training on fit.
    #[serde(default = "default_reset")]
    pub reset: Option<bool>,
    /// Features to treat as categorical.
    #[serde(default = "default_categorical_features")]
    pub categorical_features: Option<HashSet<usize>>,
    /// Hard limit for training time (seconds).
    #[serde(default = "default_timeout")]
    pub timeout: Option<f32>,
    /// Hard limit for number of iterations.
    #[serde(default = "default_iteration_limit")]
    pub iteration_limit: Option<usize>,
    /// Memory limit for training (GB).
    #[serde(default = "default_memory_limit")]
    pub memory_limit: Option<f32>,
    /// Number of rounds for early stopping.
    #[serde(default = "default_stopping_rounds")]
    pub stopping_rounds: Option<usize>,
    /// Save node statistics for debugging.
    #[serde(default)]
    pub save_node_stats: bool,
    /// Calibration method used for prediction intervals.
    #[serde(default)]
    pub calibration_method: CalibrationMethod,
}

// Default booster base configuration
impl Default for BoosterConfig {
    fn default() -> Self {
        BoosterConfig {
            objective: Objective::LogLoss,
            budget: 0.5,
            max_bin: 256,
            num_threads: None,
            monotone_constraints: None,
            interaction_constraints: None,
            force_children_to_bound_parent: false,
            missing: f64::NAN,
            allow_missing_splits: true,
            create_missing_branch: false,
            terminate_missing_features: HashSet::new(),
            missing_node_treatment: MissingNodeTreatment::AssignToParent,
            log_iterations: 0,
            seed: 0,
            quantile: None,
            reset: None,
            categorical_features: None,
            timeout: None,
            iteration_limit: None,
            memory_limit: None,
            stopping_rounds: None,
            save_node_stats: false,
            calibration_method: CalibrationMethod::WeightVariance,
        }
    }
}

/// IO
pub trait BoosterIO: Serialize + DeserializeOwned + Sized {
    /// Save a booster as a json object to a file.
    ///
    /// * `path` - Path to save booster.
    fn save_booster<P: AsRef<Path>>(&self, path: P) -> Result<(), PerpetualError> {
        fs::write(path, self.json_dump()?).map_err(|e| PerpetualError::UnableToWrite(e.to_string()))
    }

    /// Dump a booster as a json object
    fn json_dump(&self) -> Result<String, PerpetualError> {
        serde_json::to_string(self).map_err(|e| PerpetualError::UnableToWrite(e.to_string()))
    }

    /// Load a booster from Json string
    ///
    /// * `json_str` - String object, which can be serialized to json.
    fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        serde_json::from_str::<Self>(json_str).map_err(|e| PerpetualError::UnableToRead(e.to_string()))
    }

    /// Load a booster from a path to a json booster object.
    ///
    /// * `path` - Path to load booster from.
    fn load_booster<P: AsRef<Path>>(path: P) -> Result<Self, PerpetualError> {
        let json_str = fs::read_to_string(path).map_err(|e| PerpetualError::UnableToRead(e.to_string()))?;
        Self::from_json(&json_str)
    }
}

impl BoosterIO for BoosterConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_booster_config_default() {
        let config = BoosterConfig::default();
        assert_eq!(config.budget, 0.5);
        assert_eq!(config.max_bin, 256);
        assert!(config.missing.is_nan());
        assert_eq!(config.calibration_method, CalibrationMethod::WeightVariance);
    }

    #[test]
    fn test_booster_io_json() {
        let config = BoosterConfig::default();
        let json = config.json_dump().unwrap();
        let config2 = BoosterConfig::from_json(&json).unwrap();
        assert_eq!(config.budget, config2.budget);
        assert_eq!(config.max_bin, config2.max_bin);
    }

    #[test]
    fn test_booster_io_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("booster.json");
        let config = BoosterConfig::default();
        config.save_booster(&file_path).unwrap();
        let config2 = BoosterConfig::load_booster(&file_path).unwrap();
        assert_eq!(config.budget, config2.budget);
        assert_eq!(config.max_bin, config2.max_bin);
    }

    #[test]
    fn test_parse_missing() {
        let json = r#"{"objective": "LogLoss", "max_bin": 256, "seed": 0, "allow_missing_splits": true, "create_missing_branch": false, "save_node_stats": false, "missing": null}"#;
        let config: BoosterConfig = serde_json::from_str(json).unwrap();
        assert!(config.missing.is_nan());

        let json2 = r#"{"objective": "LogLoss", "max_bin": 256, "seed": 0, "allow_missing_splits": true, "create_missing_branch": false, "save_node_stats": false, "missing": 123.45}"#;
        let config2: BoosterConfig = serde_json::from_str(json2).unwrap();
        assert_eq!(config2.missing, 123.45);
    }

    #[test]
    fn test_parse_f32() {
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct TestStruct {
            #[serde(deserialize_with = "parse_f32")]
            val: f32,
        }
        let json = r#"{"val": null}"#;
        let s: TestStruct = serde_json::from_str(json).unwrap();
        assert!(s.val.is_nan());

        let json2 = r#"{"val": 0.5}"#;
        let s2: TestStruct = serde_json::from_str(json2).unwrap();
        assert_eq!(s2.val, 0.5);
    }
}
