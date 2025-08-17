//! ListNet Loss function
//!
//!
use super::ObjectiveFunction;
use crate::metrics::Metric;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct ListNetLoss {}

impl ObjectiveFunction for ListNetLoss {
    #[inline]
    fn loss(&self, _y: &[f64], _yhat: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        todo!()
    }

    #[inline]
    fn gradient(
        &self,
        _y: &[f64],
        _yhat: &[f64],
        _sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        todo!()
    }

    #[inline]
    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        todo!()
    }

    fn default_metric(&self) -> Metric {
        Metric::NDCG { k: None }
    }
}
