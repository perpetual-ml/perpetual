use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Monotonicity constraint for a feature.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum Constraint {
    /// Constrain the relationship to be monotonically increasing.
    Positive,
    /// Constrain the relationship to be monotonically decreasing.
    Negative,
    /// No monotonicity constraint.
    Unconstrained,
}

/// A map covering the constraints for each feature, key is the feature index.
pub type ConstraintMap = HashMap<usize, Constraint>;
