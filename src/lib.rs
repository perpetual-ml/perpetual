#![feature(get_many_mut)]
#![feature(array_ptr_get)]

mod node;
mod partial_dependence;
mod shapley;

// Modules
pub mod bin;
pub mod binning;
pub mod booster;
pub mod constants;
pub mod constraints;
pub mod data;
pub mod errors;
pub mod grower;
pub mod histogram;
pub mod metric;
pub mod multi_output;
pub mod objective;
pub mod sampler;
pub mod splitter;
pub mod tree;
pub mod utils;

// Individual classes, and functions
pub use booster::PerpetualBooster;
pub use data::Matrix;
pub use multi_output::MultiOutputBooster;
