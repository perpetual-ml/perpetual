use numpy::PyReadonlyArray1;
use perpetual_rs::causal::fairness::{FairnessObjective as InnerFairnessObjective, FairnessType};
use pyo3::prelude::*;

#[pyclass(name = "FairnessObjective")]
#[derive(Clone)]
pub struct PyFairnessObjective {
    pub inner: InnerFairnessObjective,
}

#[pymethods]
impl PyFairnessObjective {
    #[new]
    #[pyo3(signature = (sensitive_attr, lambda, fairness_type))]
    pub fn new(sensitive_attr: PyReadonlyArray1<i32>, lambda: f32, fairness_type: &str) -> PyResult<Self> {
        let s = sensitive_attr.as_slice()?.to_vec();

        let ftype = match fairness_type {
            "demographic_parity" => FairnessType::DemographicParity,
            "equalized_odds" => FairnessType::EqualizedOdds,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown fairness type: {}. Expected 'demographic_parity' or 'equalized_odds'.",
                    fairness_type
                )));
            }
        };

        Ok(PyFairnessObjective {
            inner: InnerFairnessObjective::with_type(s, lambda, ftype),
        })
    }
}
