use crate::metrics::*;

pub struct NDCGMetric {
    #[allow(dead_code)]
    k: Option<u64>,
}

impl NDCGMetric {
    pub fn new(k: Option<u64>) -> Self {
        Self { k }
    }
}

impl EvaluationMetric for NDCGMetric {
    fn calculate_metric(_y: &[f64], _yhat: &[f64], _sample_weight: &[f64], _alpha: Option<f32>) -> f64 {
        todo!()
    }
    fn maximize() -> bool {
        true
    }
}
