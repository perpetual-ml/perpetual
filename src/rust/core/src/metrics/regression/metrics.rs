use crate::metrics::evaluation::EvaluationMetric;

pub struct QuantileLossMetric {}
impl EvaluationMetric for QuantileLossMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], alpha: Option<f32>) -> f64 {
        quantile_loss(y, yhat, sample_weight, alpha)
    }
    fn maximize() -> bool {
        false
    }
}

pub struct RootMeanSquaredLogErrorMetric {}
impl EvaluationMetric for RootMeanSquaredLogErrorMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], _alpha: Option<f32>) -> f64 {
        root_mean_squared_log_error(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub struct RootMeanSquaredErrorMetric {}
impl EvaluationMetric for RootMeanSquaredErrorMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], _alpha: Option<f32>) -> f64 {
        root_mean_squared_error(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub fn quantile_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64], alpha: Option<f32>) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            let _alpha = alpha.unwrap() as f64;
            let s = *y_ - *yhat_;
            let l = if s >= 0.0 { _alpha * s } else { (1.0 - _alpha) * s };
            l * *w_
        })
        .sum::<f64>();
    res / w_sum
}

pub fn root_mean_squared_log_error(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            (y_.ln_1p() - yhat_.ln_1p()).powi(2) * *w_
        })
        .sum::<f64>();
    (res / w_sum).sqrt()
}

pub fn root_mean_squared_error(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            (y_ - yhat_).powi(2) * *w_
        })
        .sum::<f64>();
    (res / w_sum).sqrt()
}
