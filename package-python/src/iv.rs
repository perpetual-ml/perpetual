use crate::utils::{int_map_to_constraint_map, to_value_error};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use perpetual_rs::causal::iv::IVBooster as CrateIVBooster;
use perpetual_rs::data::Matrix;
use perpetual_rs::objective_functions::Objective;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyType;
use pyo3::IntoPyObjectExt;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct IVBooster {
    booster: CrateIVBooster,
}

#[pymethods]
impl IVBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        treatment_objective,
        outcome_objective,
        stage1_budget,
        stage2_budget,
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
        treatment_objective: &str,
        outcome_objective: &str,
        stage1_budget: f32,
        stage2_budget: f32,
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
        let treatment_obj = to_value_error(serde_plain::from_str(treatment_objective))?;
        let outcome_obj = to_value_error(serde_plain::from_str(outcome_objective))?;
        let missing_node_treatment_ = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let monotone_constraints_ = int_map_to_constraint_map(monotone_constraints)?;

        let booster = to_value_error(CrateIVBooster::new(
            treatment_obj,
            outcome_obj,
            stage1_budget,
            stage2_budget,
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
        Ok(IVBooster { booster })
    }

    pub fn fit(
        &mut self,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        flat_z: PyReadonlyArray1<f64>,
        z_rows: usize,
        z_cols: usize,
        y: PyReadonlyArray1<f64>,
        w: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);

        let flat_z_slice = flat_z.as_slice()?;
        let z = Matrix::new(flat_z_slice, z_rows, z_cols);

        // Ensure dimensionality logic if needed?
        // Matrix::new doesn't check length vs rows*cols heavily unless we access it.
        // But let's trust the python side passed correct dims.

        let y_slice = y.as_slice()?;
        let w_slice = w.as_slice()?;

        to_value_error(self.booster.fit(&x, &z, y_slice, w_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        w_counterfactual: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let w_slice = w_counterfactual.as_slice()?;

        let preds = self.booster.predict(&x, w_slice);
        Ok(preds.into_pyarray(py))
    }

    pub fn json_dump(&self) -> PyResult<String> {
        to_value_error(serde_json::to_string(&self.booster).map_err(|e| e.to_string()))
    }

    #[classmethod]
    pub fn from_json(_: &Bound<'_, PyType>, json_str: &str) -> PyResult<Self> {
        let booster: CrateIVBooster = to_value_error(serde_json::from_str(json_str).map_err(|e| e.to_string()))?;
        Ok(IVBooster { booster })
    }

    pub fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        let treatment_objective = to_value_error(serde_plain::to_string::<Objective>(
            &self.booster.treatment_model.cfg.objective,
        ))?;
        let outcome_objective = to_value_error(serde_plain::to_string::<Objective>(
            &self.booster.outcome_model.cfg.objective,
        ))?;

        let key_vals: Vec<(&str, Py<PyAny>)> = vec![
            ("treatment_objective", treatment_objective.into_py_any(py).unwrap()),
            ("outcome_objective", outcome_objective.into_py_any(py).unwrap()),
            ("stage1_budget", self.booster.stage1_budget.into_py_any(py).unwrap()),
            ("stage2_budget", self.booster.stage2_budget.into_py_any(py).unwrap()),
        ];
        let dict = key_vals.into_py_dict(py)?;
        Ok(dict.into_py_any(py)?)
    }
}
