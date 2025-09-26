use crate::custom_objective::CustomObjective;
use crate::utils::int_map_to_constraint_map;
use crate::utils::to_value_error;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use perpetual_rs::booster::config::BoosterIO;
use perpetual_rs::booster::config::MissingNodeTreatment;
use perpetual_rs::conformal::cqr::CalData;
use perpetual_rs::constraints::Constraint;
use perpetual_rs::data::Matrix;
use perpetual_rs::objective_functions::Objective;
use perpetual_rs::PerpetualBooster as CratePerpetualBooster;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyType;
use pyo3::IntoPyObjectExt;
use pyo3::{Bound, Python};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[pyclass(subclass)]
pub struct PerpetualBooster {
    booster: CratePerpetualBooster,
}

#[pymethods]
impl PerpetualBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        objective,
        budget,
        max_bin,
        num_threads,
        monotone_constraints,
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
        loss,
        grad,
        init,
    ))]
    pub fn new<'py>(
        objective: Option<&str>,
        budget: f32,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: HashMap<usize, i8>,
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
        loss: Option<Bound<'py, PyAny>>,
        grad: Option<Bound<'py, PyAny>>,
        init: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let objective_ = match objective {
            Some(obj) => to_value_error(serde_plain::from_str(obj))?,
            None => Objective::Custom(Arc::new(CustomObjective {
                loss: loss.unwrap().to_owned().into(),
                grad: grad.unwrap().to_owned().into(),
                init: init.unwrap().to_owned().into(),
            })),
        };
        let missing_node_treatment_ = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let monotone_constraints_ = int_map_to_constraint_map(monotone_constraints)?;

        let booster = CratePerpetualBooster::default()
            .set_objective(objective_)
            .set_budget(budget)
            .set_max_bin(max_bin)
            .set_num_threads(num_threads)
            .set_monotone_constraints(Some(monotone_constraints_))
            .set_force_children_to_bound_parent(force_children_to_bound_parent)
            .set_missing(missing)
            .set_allow_missing_splits(allow_missing_splits)
            .set_create_missing_branch(create_missing_branch)
            .set_terminate_missing_features(terminate_missing_features)
            .set_missing_node_treatment(missing_node_treatment_)
            .set_log_iterations(log_iterations)
            .set_quantile(quantile)
            .set_reset(reset)
            .set_categorical_features(categorical_features)
            .set_timeout(timeout)
            .set_iteration_limit(iteration_limit)
            .set_memory_limit(memory_limit)
            .set_stopping_rounds(stopping_rounds);

        to_value_error(booster.validate_parameters())?;

        Ok(PerpetualBooster { booster })
    }

    #[setter]
    fn set_objective(&mut self, value: &str) -> PyResult<()> {
        let objective_ = to_value_error(serde_plain::from_str(value))?;
        self.booster.cfg.objective = objective_;
        Ok(())
    }
    #[setter]
    fn set_budget(&mut self, value: f32) -> PyResult<()> {
        self.booster.cfg.budget = value;
        Ok(())
    }
    #[setter]
    fn set_max_bin(&mut self, value: u16) -> PyResult<()> {
        self.booster.cfg.max_bin = value;
        Ok(())
    }
    #[setter]
    fn set_num_threads(&mut self, value: Option<usize>) -> PyResult<()> {
        self.booster.cfg.num_threads = value;
        Ok(())
    }
    #[setter]
    fn set_monotone_constraints(&mut self, value: HashMap<usize, i8>) -> PyResult<()> {
        let map = int_map_to_constraint_map(value)?;
        self.booster.cfg.monotone_constraints = Some(map);
        Ok(())
    }
    #[setter]
    fn set_force_children_to_bound_parent(&mut self, value: bool) -> PyResult<()> {
        self.booster.cfg.force_children_to_bound_parent = value;
        Ok(())
    }
    #[setter]
    fn set_missing(&mut self, value: f64) -> PyResult<()> {
        self.booster.cfg.missing = value;
        Ok(())
    }
    #[setter]
    fn set_allow_missing_splits(&mut self, value: bool) -> PyResult<()> {
        self.booster.cfg.allow_missing_splits = value;
        Ok(())
    }
    #[setter]
    fn set_create_missing_branch(&mut self, value: bool) -> PyResult<()> {
        self.booster.cfg.create_missing_branch = value;
        Ok(())
    }
    #[setter]
    fn set_terminate_missing_features(&mut self, value: HashSet<usize>) -> PyResult<()> {
        self.booster.cfg.terminate_missing_features = value;
        Ok(())
    }
    #[setter]
    fn set_missing_node_treatment(&mut self, value: &str) -> PyResult<()> {
        let missing_node_treatment_ = to_value_error(serde_plain::from_str(value))?;
        self.booster.cfg.missing_node_treatment = missing_node_treatment_;
        Ok(())
    }
    #[setter]
    fn set_log_iterations(&mut self, value: usize) -> PyResult<()> {
        self.booster.cfg.log_iterations = value;
        Ok(())
    }
    #[setter]
    fn set_quantile(&mut self, value: Option<f64>) -> PyResult<()> {
        self.booster.cfg.quantile = value;
        Ok(())
    }
    #[setter]
    fn set_reset(&mut self, value: Option<bool>) -> PyResult<()> {
        self.booster.cfg.reset = value;
        Ok(())
    }
    #[setter]
    fn set_categorical_features(&mut self, value: Option<HashSet<usize>>) -> PyResult<()> {
        self.booster.cfg.categorical_features = value;
        Ok(())
    }
    #[setter]
    fn set_timeout(&mut self, value: Option<f32>) -> PyResult<()> {
        self.booster.cfg.timeout = value;
        Ok(())
    }
    #[setter]
    fn set_iteration_limit(&mut self, value: Option<usize>) -> PyResult<()> {
        self.booster.cfg.iteration_limit = value;
        Ok(())
    }
    #[setter]
    fn set_memory_limit(&mut self, value: Option<f32>) -> PyResult<()> {
        self.booster.cfg.memory_limit = value;
        Ok(())
    }
    #[setter]
    fn set_stopping_rounds(&mut self, value: Option<usize>) -> PyResult<()> {
        self.booster.cfg.stopping_rounds = value;
        Ok(())
    }

    #[getter]
    fn base_score(&self) -> PyResult<f64> {
        Ok(self.booster.base_score)
    }
    #[getter]
    fn number_of_trees(&self) -> PyResult<usize> {
        Ok(self.booster.get_prediction_trees().len())
    }

    pub fn fit(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
        group: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight_ = match sample_weight.as_ref() {
            Some(sw) => {
                let sw_slice = sw.as_slice()?;
                Some(sw_slice)
            }
            None => None,
        };
        let group_ = match group.as_ref() {
            Some(gr) => {
                let gr_slice = gr.as_slice()?;
                Some(gr_slice)
            }
            None => None,
        };

        match self.booster.fit(&data, y, sample_weight_, group_) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;

        Ok(())
    }

    pub fn prune(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
        group: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight_ = match sample_weight.as_ref() {
            Some(sw) => {
                let sw_slice = sw.as_slice()?;
                Some(sw_slice)
            }
            None => None,
        };
        let group_ = match group.as_ref() {
            Some(gr) => {
                let gr_slice = gr.as_slice()?;
                Some(gr_slice)
            }
            None => None,
        };

        match self.booster.prune(&data, y, sample_weight_, group_) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;

        Ok(())
    }

    pub fn calibrate(
        &mut self,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
        flat_data_cal: PyReadonlyArray1<f64>,
        rows_cal: usize,
        cols_cal: usize,
        y_cal: PyReadonlyArray1<f64>,
        alpha: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
        group: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let y = y.as_slice()?;
        let sample_weight_ = match sample_weight.as_ref() {
            Some(sw) => {
                let sw_slice = sw.as_slice()?;
                Some(sw_slice)
            }
            None => None,
        };
        let group_ = match group.as_ref() {
            Some(gr) => {
                let gr_slice = gr.as_slice()?;
                Some(gr_slice)
            }
            None => None,
        };

        let flat_data_cal = flat_data_cal.as_slice()?;
        let data_cal = Matrix::new(flat_data_cal, rows_cal, cols_cal);
        let y_cal = y_cal.as_slice()?;

        let cal_data: CalData = (data_cal, y_cal, alpha.as_slice()?);

        match self.booster.calibrate(&data, y, sample_weight_, group_, cal_data) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;

        Ok(())
    }

    pub fn predict_intervals<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);

        let predictions: HashMap<String, Vec<Vec<f64>>> = self.booster.predict_intervals(&data, parallel);

        let py_dict = PyDict::new(py);
        for (key, value) in predictions.iter() {
            let py_array = PyArray2::from_vec2(py, value)?;
            py_dict.set_item(key, py_array)?;
        }

        Ok(py_dict.into_py_dict(py)?)
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        Ok(self.booster.predict(&data, parallel).into_pyarray(py))
    }

    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        Ok(self.booster.predict_proba(&data, parallel).into_pyarray(py))
    }

    pub fn predict_nodes<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        parallel: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);

        let value: Vec<Vec<HashSet<usize>>> = self.booster.predict_nodes(&data, parallel);

        value.into_py_any(py)
    }

    pub fn predict_contributions<'py>(
        &self,
        py: Python<'py>,
        flat_data: PyReadonlyArray1<f64>,
        rows: usize,
        cols: usize,
        method: &str,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_data = flat_data.as_slice()?;
        let data = Matrix::new(flat_data, rows, cols);
        let parallel = parallel.unwrap_or(true);
        let method_ = to_value_error(serde_plain::from_str(method))?;
        Ok(self
            .booster
            .predict_contributions(&data, method_, parallel)
            .into_pyarray(py))
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> PyResult<HashMap<usize, f32>> {
        let method_ = to_value_error(serde_plain::from_str(method))?;
        Ok(self.booster.calculate_feature_importance(method_, normalize))
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> PyResult<f64> {
        Ok(self.booster.value_partial_dependence(feature, value))
    }

    pub fn text_dump(&self) -> PyResult<Vec<String>> {
        let mut trees = Vec::new();
        for t in &self.booster.trees {
            trees.push(format!("{}", t));
        }
        Ok(trees)
    }

    pub fn save_booster(&self, path: &str) -> PyResult<()> {
        match self.booster.save_booster(path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    pub fn json_dump(&self) -> PyResult<String> {
        match self.booster.json_dump() {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    pub fn insert_metadata(&mut self, key: String, value: String) -> PyResult<()> {
        self.booster.insert_metadata(key, value);
        Ok(())
    }

    pub fn get_metadata(&self, key: String) -> PyResult<String> {
        match self.booster.get_metadata(&key) {
            Some(m) => Ok(m),
            None => Err(PyKeyError::new_err(format!(
                "No value associated with provided key {}",
                key
            ))),
        }
    }

    #[classmethod]
    pub fn load_booster(_: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let booster = match CratePerpetualBooster::load_booster(path.as_str()) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(PerpetualBooster { booster })
    }

    #[classmethod]
    pub fn from_json(_: &Bound<'_, PyType>, json_str: &str) -> PyResult<Self> {
        let booster = match CratePerpetualBooster::from_json(json_str) {
            Ok(m) => Ok(m),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }?;
        Ok(PerpetualBooster { booster })
    }

    pub fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        let objective_ = to_value_error(serde_plain::to_string::<Objective>(&self.booster.cfg.objective))?;
        let missing_node_treatment_ = to_value_error(serde_plain::to_string::<MissingNodeTreatment>(
            &self.booster.cfg.missing_node_treatment,
        ))?;
        let monotone_constraints_: HashMap<usize, i8> = self
            .booster
            .cfg
            .monotone_constraints
            .as_ref()
            .unwrap_or(&HashMap::new())
            .iter()
            .map(|(f, c)| {
                let c_ = match c {
                    Constraint::Negative => -1,
                    Constraint::Positive => 1,
                    Constraint::Unconstrained => 0,
                };
                (*f, c_)
            })
            .collect();

        let key_vals: Vec<(&str, Py<PyAny>)> = vec![
            ("objective", objective_.into_py_any(py).unwrap()),
            ("budget", self.booster.cfg.budget.into_py_any(py).unwrap()),
            ("num_threads", self.booster.cfg.num_threads.into_py_any(py).unwrap()),
            (
                "allow_missing_splits",
                self.booster.cfg.allow_missing_splits.into_py_any(py).unwrap(),
            ),
            ("monotone_constraints", monotone_constraints_.into_py_any(py).unwrap()),
            ("missing", self.booster.cfg.missing.into_py_any(py).unwrap()),
            (
                "create_missing_branch",
                self.booster.cfg.create_missing_branch.into_py_any(py).unwrap(),
            ),
            (
                "terminate_missing_features",
                self.booster
                    .cfg
                    .terminate_missing_features
                    .clone()
                    .into_py_any(py)
                    .unwrap(),
            ),
            (
                "missing_node_treatment",
                missing_node_treatment_.into_py_any(py).unwrap(),
            ),
            (
                "log_iterations",
                self.booster.cfg.log_iterations.into_py_any(py).unwrap(),
            ),
            (
                "force_children_to_bound_parent",
                self.booster.cfg.force_children_to_bound_parent.into_py_any(py).unwrap(),
            ),
            ("quantile", self.booster.cfg.quantile.into_py_any(py).unwrap()),
            ("reset", self.booster.cfg.reset.into_py_any(py).unwrap()),
            (
                "categorical_features",
                self.booster.cfg.categorical_features.clone().into_py_any(py).unwrap(),
            ),
            ("timeout", self.booster.cfg.timeout.into_py_any(py).unwrap()),
            (
                "iteration_limit",
                self.booster.cfg.iteration_limit.into_py_any(py).unwrap(),
            ),
            ("memory_limit", self.booster.cfg.memory_limit.into_py_any(py).unwrap()),
            (
                "stopping_rounds",
                self.booster.cfg.stopping_rounds.into_py_any(py).unwrap(),
            ),
        ];
        let dict = key_vals.into_py_dict(py);

        Ok(dict?.into_py_any(py)?)
    }
}
