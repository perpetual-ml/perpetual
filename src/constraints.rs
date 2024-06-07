use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum Constraint {
    Positive,
    Negative,
    Unconstrained,
}

pub type ConstraintMap = HashMap<usize, Constraint>;
