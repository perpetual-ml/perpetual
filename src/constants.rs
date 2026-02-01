/// The number of rounds to look back for auto stopping.
pub const STOPPING_ROUNDS: usize = 3;
/// Factor to use when allocating memory for free nodes.
pub const FREE_MEM_ALLOC_FACTOR: f32 = 0.9;
/// Minimum number of nodes to allocate.
pub const N_NODES_ALLOC_MIN: usize = 100;
/// Maximum number of nodes to allocate.
pub const N_NODES_ALLOC_MAX: usize = 10000;
/// Default iteration limit.
pub const ITER_LIMIT: usize = 1000;
/// Threshold for generalization capability.
pub const GENERALIZATION_THRESHOLD: f32 = 1.0;
/// Relaxed threshold for generalization capability.
pub const GENERALIZATION_THRESHOLD_RELAXED: f32 = 0.99;
/// Minimum amount of columns to sample.
pub const MIN_COL_AMOUNT: usize = 40;
/// Epsilon value for Hessian to prevent division by zero.
pub const HESSIAN_EPS: f32 = 1e-8;
