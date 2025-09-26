//! Booster configurations
//!
//! Booster specific configurations.
//!
//!
//!
use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::objective_functions::objective::Objective;
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

// Common configuration
// across implementations

/// Contributions
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

/// Missing values
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

// Common functions
// fn default_cal_models() -> HashMap<String, [(PerpetualBooster, f64); 2]> {
//     HashMap::new()
// }
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

// Common Booster configuration
// across
/// Booster configurations
#[derive(Clone, Serialize, Deserialize)]
pub struct BoosterConfig {
    /// The name of objective function used to optimize. Valid options are:
    /// "LogLoss" to use logistic loss as the objective function,
    /// "SquaredLoss" to use Squared Error as the objective function,
    /// "QuantileLoss" for quantile regression.
    pub objective: Objective,
    /// Budget to fit the model.
    #[serde(default = "default_budget")]
    pub budget: f32,
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

// Default booster base configuration
impl Default for BoosterConfig {
    fn default() -> Self {
        BoosterConfig {
            objective: Objective::LogLoss,
            budget: 0.5,
            max_bin: 256,
            num_threads: None,
            monotone_constraints: None,
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

impl<T> BoosterIO for T where T: Serialize + DeserializeOwned + Sized {}
