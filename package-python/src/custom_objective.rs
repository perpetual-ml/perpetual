//! Custom Objective
//!
//! Wraps user-supplied Python callables (loss, gradient, initial_value) into
//! a Rust [`ObjectiveFunction`] implementation for use with the booster.
use numpy::PyArray;
use perpetual_rs::objective::ObjectiveFunction;
use pyo3::prelude::*;

/// A user-defined objective backed by Python callables.
pub struct CustomObjective {
    /// Python callable: `loss(y, yhat, sample_weight, group) -> List[float]`.
    pub loss: Py<PyAny>,
    /// Python callable: `grad(y, yhat, sample_weight, group) -> (List[float], Optional[List[float]])`.
    pub grad: Py<PyAny>,
    /// Python callable: `init(y, sample_weight, group) -> float`.
    pub init: Py<PyAny>,
}

impl ObjectiveFunction for CustomObjective {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        Python::attach(|py| -> PyResult<Vec<f32>> {
            let py_y = PyArray::from_slice(py, y);
            let py_yhat = PyArray::from_slice(py, yhat);
            let py_sample_weight = sample_weight.map(|sw| PyArray::from_slice(py, sw));
            let py_group = group.map(|gr| PyArray::from_slice(py, gr));
            let args = (py_y, py_yhat, py_sample_weight, py_group);
            let result = self.loss.call1(py, args)?;
            result.extract(py)
        })
        .expect("Python loss function failed")
    }

    #[inline]
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        Python::attach(|py| -> PyResult<(Vec<f32>, Option<Vec<f32>>)> {
            let py_y = PyArray::from_slice(py, y);
            let py_yhat = PyArray::from_slice(py, yhat);
            let py_sample_weight = sample_weight.map(|sw| PyArray::from_slice(py, sw));
            let py_group = group.map(|gr| PyArray::from_slice(py, gr));
            let args = (py_y, py_yhat, py_sample_weight, py_group);
            let result = self.grad.call1(py, args)?;
            result.extract(py)
        })
        .expect("Python gradient function failed")
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64 {
        Python::attach(|py| -> PyResult<f64> {
            let py_y = PyArray::from_slice(py, y);
            let py_sample_weight = sample_weight.map(|sw| PyArray::from_slice(py, sw));
            let py_group = group.map(|gr| PyArray::from_slice(py, gr));
            let args = (py_y, py_sample_weight, py_group);
            let result = self.init.call1(py, args)?;
            result.extract(py)
        })
        .expect("Python initial_value function failed")
    }

    fn requires_batch_evaluation(&self) -> bool {
        true
    }

    // Uses the trait default (`RootMeanSquaredError`).
}
