use crate::data::FloatData;
use crate::metrics::evaluation::EvaluationMetric;

pub struct LogLossMetric {}
impl EvaluationMetric for LogLossMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], _alpha: Option<f32>) -> f64 {
        log_loss(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        false
    }
}

pub struct AUCMetric {}
impl EvaluationMetric for AUCMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], _alpha: Option<f32>) -> f64 {
        roc_auc_score(y, yhat, sample_weight)
    }
    fn maximize() -> bool {
        true
    }
}

pub fn log_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
            -(*y_ * yhat_.ln() + (f64::ONE - *y_) * ((f64::ONE - yhat_).ln())) * *w_
        })
        .sum::<f64>();
    res / w_sum
}

fn trapezoid_area(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    (x0 - x1).abs() * (y0 + y1) * 0.5
}

pub fn roc_auc_score(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut indices = (0..y.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| yhat[b].total_cmp(&yhat[a]));
    let mut auc: f64 = 0.0;

    let mut label = y[indices[0]];
    let mut w = sample_weight[indices[0]];
    let mut fp = (1.0 - label) * w;
    let mut tp: f64 = label * w;
    let mut tp_prev: f64 = 0.0;
    let mut fp_prev: f64 = 0.0;

    for i in 1..indices.len() {
        if yhat[indices[i]] != yhat[indices[i - 1]] {
            auc += trapezoid_area(fp_prev, fp, tp_prev, tp);
            tp_prev = tp;
            fp_prev = fp;
        }
        label = y[indices[i]];
        w = sample_weight[indices[i]];
        fp += (1.0 - label) * w;
        tp += label * w;
    }

    auc += trapezoid_area(fp_prev, fp, tp_prev, tp);
    if fp <= 0.0 || tp <= 0.0 {
        auc = 0.0;
        fp = 0.0;
        tp = 0.0;
    }

    auc / (tp * fp)
}
