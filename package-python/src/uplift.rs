use crate::utils::{int_map_to_constraint_map, to_value_error};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use perpetual_rs::causal::uplift::UpliftBooster as CrateUpliftBooster;
use perpetual_rs::data::Matrix;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct UpliftBooster {
    booster: CrateUpliftBooster,
}

#[pymethods]
impl UpliftBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        outcome_budget,
        propensity_budget,
        effect_budget,
        max_bin,
        num_threads,
        monotone_constraints,
        interaction_constraints,
        force_children_to_bound_parent,
        missing,
        allow_missing_splits,
        create_missing_branch,
        terminate_missing_features,
        missing_node_treatment,
        log_iterations,
        quantile,
        reset,
        categorical_features,
        timeout,
        iteration_limit,
        memory_limit,
        stopping_rounds,
    ))]
    pub fn new(
        outcome_budget: f32,
        propensity_budget: f32,
        effect_budget: f32,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: HashMap<usize, i8>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: &str,
        log_iterations: usize,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> PyResult<Self> {
        let missing_node_treatment_ = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let monotone_constraints_ = int_map_to_constraint_map(monotone_constraints)?;

        let booster = to_value_error(CrateUpliftBooster::new(
            outcome_budget,
            propensity_budget,
            effect_budget,
            max_bin,
            num_threads,
            Some(monotone_constraints_),
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            missing_node_treatment_,
            log_iterations,
            42, // seed - TODO: pass this?
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        ))?;
        Ok(UpliftBooster { booster })
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        w: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_data_slice = flat_data.as_slice()?;
        let data = Matrix::new(flat_data_slice, rows, cols);
        let w_slice = w.as_slice()?;
        let y_slice = y.as_slice()?;

        to_value_error(self.booster.fit(&data, w_slice, y_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_data_slice = flat_data.as_slice()?;
        let data = Matrix::new(flat_data_slice, rows, cols);
        let preds = self.booster.predict(&data);
        Ok(preds.into_pyarray(py))
    }

    pub fn json_dump(&self) -> PyResult<String> {
        to_value_error(serde_json::to_string(&self.booster).map_err(|e| e.to_string()))
    }

    #[classmethod]
    pub fn from_json(_: &Bound<'_, PyType>, json_str: &str) -> PyResult<Self> {
        let booster: CrateUpliftBooster = to_value_error(serde_json::from_str(json_str).map_err(|e| e.to_string()))?;
        Ok(UpliftBooster { booster })
    }
}
