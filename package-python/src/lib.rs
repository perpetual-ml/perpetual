//! Python bindings for the Perpetual gradient boosting library.
//!
//! This crate exposes [`PerpetualBooster`], [`MultiOutputBooster`], [`UpliftBooster`],
//! and [`IVBooster`] to Python via [PyO3](https://pyo3.rs).
mod booster;
mod custom_objective;
mod iv;
mod multi_output;
mod uplift;
mod utils;

use crate::booster::PerpetualBooster;
use crate::iv::IVBooster;
use crate::multi_output::MultiOutputBooster;
use crate::uplift::UpliftBooster;
use crate::utils::percentiles;
use crate::utils::print_matrix;
use pyo3::prelude::*;

#[pymodule]
fn perpetual(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(print_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(percentiles, m)?)?;

    m.add_class::<PerpetualBooster>()?;
    m.add_class::<MultiOutputBooster>()?;
    m.add_class::<UpliftBooster>()?;
    m.add_class::<IVBooster>()?;

    Ok(())
}
