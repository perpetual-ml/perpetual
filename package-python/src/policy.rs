use numpy::PyReadonlyArray1;
use perpetual_rs::causal::policy::{PolicyMode, PolicyObjective as InnerPolicyObjective};
use pyo3::prelude::*;

#[pyclass(name = "PolicyObjective")]
#[derive(Clone)]
pub struct PyPolicyObjective {
    pub inner: InnerPolicyObjective,
}

#[pymethods]
impl PyPolicyObjective {
    #[new]
    #[pyo3(signature=(treatment, propensity, mode, mu_hat))]
    pub fn new(
        treatment: PyReadonlyArray1<u8>,
        propensity: PyReadonlyArray1<f64>,
        mode: &str,
        mu_hat: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let treatment_vec = treatment.as_slice()?.to_vec();
        let propensity_vec = propensity.as_slice()?.to_vec();

        let policy_mode = match mode {
            "ipw" => PolicyMode::IPW,
            "aipw" => {
                let mu = mu_hat
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("mu_hat required for AIPW mode"))?
                    .as_slice()?
                    .to_vec();
                PolicyMode::AIPW { mu_hat: mu }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown mode: {}",
                    mode
                )))
            }
        };

        Ok(PyPolicyObjective {
            inner: InnerPolicyObjective {
                treatment: treatment_vec,
                propensity: propensity_vec,
                mode: policy_mode,
            },
        })
    }
}
