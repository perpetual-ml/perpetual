use numpy::PyArray;
use perpetual_rs::metrics::evaluation::Metric;
use perpetual_rs::objective_functions::ObjectiveFunction;
use pyo3::prelude::*;

pub struct CustomObjective {
    pub loss: Py<PyAny>,
    pub grad: Py<PyAny>,
    pub init: Py<PyAny>,
}

impl ObjectiveFunction for CustomObjective {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        Python::attach(|py| -> PyResult<Vec<f32>> {
            let py_y = PyArray::from_slice(py, y);
            let py_yhat = PyArray::from_slice(py, yhat);
            let py_sample_weight = match sample_weight {
                Some(sw) => Some(PyArray::from_slice(py, sw)),
                None => None,
            };
            let py_group = match group {
                Some(gr) => Some(PyArray::from_slice(py, gr)),
                None => None,
            };
            let args = (py_y, py_yhat, py_sample_weight, py_group);
            let result = self.loss.call1(py, args)?;
            let extracted: Vec<f32> = result.extract(py)?;
            Ok(extracted)
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
            let py_sample_weight = match sample_weight {
                Some(sw) => Some(PyArray::from_slice(py, sw)),
                None => None,
            };
            let py_group = match group {
                Some(gr) => Some(PyArray::from_slice(py, gr)),
                None => None,
            };
            let args = (py_y, py_yhat, py_sample_weight, py_group);
            let result = self.grad.call1(py, args)?;
            let extracted: (Vec<f32>, Option<Vec<f32>>) = result.extract(py)?;
            Ok(extracted)
        })
        .expect("Python gradient function failed")
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> f64 {
        Python::attach(|py| -> PyResult<f64> {
            let py_y = PyArray::from_slice(py, y);
            let py_sample_weight = match sample_weight {
                Some(sw) => Some(PyArray::from_slice(py, sw)),
                None => None,
            };
            let py_group = match group {
                Some(gr) => Some(PyArray::from_slice(py, gr)),
                None => None,
            };
            let args = (py_y, py_sample_weight, py_group);
            let result = self.init.call1(py, args)?;
            let extracted: f64 = result.extract(py)?;
            Ok(extracted)
        })
        .expect("Python initial_value function failed")
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
    }
}
