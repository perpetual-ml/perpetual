use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use perpetual_rs::constraints::{Constraint, ConstraintMap};
use perpetual_rs::data::Matrix;
use perpetual_rs::utils::percentiles as crate_percentiles;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

pub fn int_map_to_constraint_map(int_map: HashMap<usize, i8>) -> PyResult<ConstraintMap> {
    let mut constraints: ConstraintMap = HashMap::new();
    for (f, c) in int_map.iter() {
        let c_ = match c {
            -1 => Ok(Constraint::Negative),
            1 => Ok(Constraint::Positive),
            0 => Ok(Constraint::Unconstrained),
            _ => Err(PyValueError::new_err(format!(
                "Valid monotone constraints are -1, 1 or 0, but '{}' was provided for feature number {}.",
                c, f
            ))),
        }?;
        constraints.insert(*f, c_);
    }
    Ok(constraints)
}

pub fn to_value_error<T, E: std::fmt::Display>(value: Result<T, E>) -> Result<T, PyErr> {
    match value {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}
#[pyfunction]
pub fn print_matrix(x: PyReadonlyArray1<f32>, rows: usize, cols: usize) -> PyResult<()> {
    let m = Matrix::new(x.as_slice()?, rows, cols);
    println!("{}", m);
    Ok(())
}

#[pyfunction]
pub fn percentiles<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<f64>,
    sample_weight: PyReadonlyArray1<f64>,
    percentiles: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v_ = v.as_slice()?;
    let sample_weight_ = sample_weight.as_slice()?;
    let percentiles_ = percentiles.as_slice()?;
    let p = crate_percentiles(v_, sample_weight_, percentiles_);
    Ok(p.into_pyarray_bound(py))
}
