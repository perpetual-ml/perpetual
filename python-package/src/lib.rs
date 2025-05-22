mod booster;
mod multi_output;
mod utils;

use crate::booster::PerpetualBooster;
use crate::multi_output::MultiOutputBooster;
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

    Ok(())
}
