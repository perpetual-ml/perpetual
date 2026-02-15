//! Python bindings for meta-learners.
use crate::utils::{int_map_to_constraint_map, to_value_error};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use perpetual_rs::booster::config::ImportanceMethod;
use perpetual_rs::causal::metalearners::{
    DRLearner as CrateDRLearner, SLearner as CrateSLearner, TLearner as CrateTLearner, XLearner as CrateXLearner,
};
use perpetual_rs::data::Matrix;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct SLearner {
    pub inner: CrateSLearner,
}

#[pymethods]
impl SLearner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
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
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> PyResult<Self> {
        let mnt = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let mc = int_map_to_constraint_map(monotone_constraints)?;
        let inner = to_value_error(CrateSLearner::new(
            budget,
            max_bin,
            num_threads,
            Some(mc),
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            mnt,
            log_iterations,
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        ))?;
        Ok(Self { inner })
    }

    pub fn fit(
        &mut self,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        w: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let w_slice = w.as_slice()?;
        let y_slice = y.as_slice()?;
        to_value_error(self.inner.fit(&x, w_slice, y_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let preds = self.inner.predict(&x);
        Ok(preds.into_pyarray(py))
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> PyResult<HashMap<usize, f32>> {
        let method_: ImportanceMethod = to_value_error(serde_plain::from_str(method))?;
        Ok(self.inner.model.calculate_feature_importance(method_, normalize))
    }
}

#[pyclass]
pub struct TLearner {
    pub inner: CrateTLearner,
}

#[pymethods]
impl TLearner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
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
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> PyResult<Self> {
        let mnt = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let mc = int_map_to_constraint_map(monotone_constraints)?;
        let inner = to_value_error(CrateTLearner::new(
            budget,
            max_bin,
            num_threads,
            Some(mc),
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            mnt,
            log_iterations,
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        ))?;
        Ok(Self { inner })
    }

    pub fn fit(
        &mut self,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        w: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let w_slice = w.as_slice()?;
        let y_slice = y.as_slice()?;
        to_value_error(self.inner.fit(&x, w_slice, y_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let preds = self.inner.predict(&x);
        Ok(preds.into_pyarray(py))
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> PyResult<HashMap<usize, f32>> {
        let method_: ImportanceMethod = to_value_error(serde_plain::from_str(method))?;
        let imp0 = self.inner.mu0.calculate_feature_importance(method_.clone(), false);
        let imp1 = self.inner.mu1.calculate_feature_importance(method_, false);

        let mut combined = HashMap::new();
        for (f, v) in imp0 {
            combined.insert(f, v);
        }
        for (f, v) in imp1 {
            *combined.entry(f).or_insert(0.0) += v;
        }
        if normalize {
            let total: f32 = combined.values().copied().sum::<f32>();
            if total > 0.0 {
                for v in combined.values_mut() {
                    *v /= total;
                }
            }
        }
        Ok(combined)
    }
}

#[pyclass]
pub struct XLearner {
    pub inner: CrateXLearner,
}

#[pymethods]
impl XLearner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        propensity_budget: Option<f32>,
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
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> PyResult<Self> {
        let mnt = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let mc = int_map_to_constraint_map(monotone_constraints)?;
        let inner = to_value_error(CrateXLearner::new(
            budget,
            propensity_budget,
            max_bin,
            num_threads,
            Some(mc),
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            mnt,
            log_iterations,
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        ))?;
        Ok(Self { inner })
    }

    pub fn fit(
        &mut self,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        w: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let w_slice = w.as_slice()?;
        let y_slice = y.as_slice()?;
        to_value_error(self.inner.fit(&x, w_slice, y_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let preds = self.inner.predict(&x);
        Ok(preds.into_pyarray(py))
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> PyResult<HashMap<usize, f32>> {
        let method_: ImportanceMethod = to_value_error(serde_plain::from_str(method))?;
        // For X-Learner, the "effect" models are tau0 and tau1
        let imp0 = self.inner.tau0.calculate_feature_importance(method_.clone(), false);
        let imp1 = self.inner.tau1.calculate_feature_importance(method_, false);

        let mut combined = HashMap::new();
        for (f, v) in imp0 {
            combined.insert(f, v);
        }
        for (f, v) in imp1 {
            *combined.entry(f).or_insert(0.0) += v;
        }
        if normalize {
            let total: f32 = combined.values().copied().sum::<f32>();
            if total > 0.0 {
                for v in combined.values_mut() {
                    *v /= total;
                }
            }
        }
        Ok(combined)
    }
}

#[pyclass]
pub struct DRLearner {
    pub inner: CrateDRLearner,
}

#[pymethods]
impl DRLearner {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: f32,
        propensity_budget: Option<f32>,
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
        seed: u64,
        quantile: Option<f64>,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
    ) -> PyResult<Self> {
        let mnt = to_value_error(serde_plain::from_str(missing_node_treatment))?;
        let mc = int_map_to_constraint_map(monotone_constraints)?;
        let inner = to_value_error(CrateDRLearner::new(
            budget,
            propensity_budget,
            max_bin,
            num_threads,
            Some(mc),
            interaction_constraints,
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features,
            mnt,
            log_iterations,
            seed,
            quantile,
            reset,
            categorical_features,
            timeout,
            iteration_limit,
            memory_limit,
            stopping_rounds,
        ))?;
        Ok(Self { inner })
    }

    pub fn fit(
        &mut self,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
        w: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let w_slice = w.as_slice()?;
        let y_slice = y.as_slice()?;
        to_value_error(self.inner.fit(&x, w_slice, y_slice))?;
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        flat_x: PyReadonlyArray1<f64>,
        x_rows: usize,
        x_cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let flat_x_slice = flat_x.as_slice()?;
        let x = Matrix::new(flat_x_slice, x_rows, x_cols);
        let preds = self.inner.predict(&x);
        Ok(preds.into_pyarray(py))
    }

    pub fn calculate_feature_importance(&self, method: &str, normalize: bool) -> PyResult<HashMap<usize, f32>> {
        let method_: ImportanceMethod = to_value_error(serde_plain::from_str(method))?;
        Ok(self.inner.effect.calculate_feature_importance(method_, normalize))
    }
}
