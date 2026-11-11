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

pub struct BrierLossMetric {}
impl EvaluationMetric for BrierLossMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], _group: &[u64], _alpha: Option<f32>) -> f64 {
        brier_loss(y, yhat, sample_weight)
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

pub fn brier_loss(y: &[f64], yhat: &[f64], sample_weight: &[f64]) -> f64 {
    let mut w_sum = 0.;
    let res = y
        .iter()
        .zip(yhat)
        .zip(sample_weight)
        .map(|((y_, yhat_), w_)| {
            w_sum += *w_;
            let p = f64::ONE / (f64::ONE + (-*yhat_).exp());
            (*y_ - p).powi(2) * *w_
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_loss_metric() {
        let y = vec![1.0, 0.0];
        let yhat = vec![1.0, -1.0]; // p approx 0.73, 0.27
        let w = vec![1.0, 1.0];
        let loss = log_loss(&y, &yhat, &w);
        // p = 1/(1+e^-1) = 0.7310585
        // loss = -(1*ln(p) + 0*ln(1-p)) = 0.313261
        // For two samples with same diff: 0.313261
        assert!((loss - 0.3132617).abs() < 1e-6);
    }

    #[test]
    fn test_brier_loss_metric() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0]; // p = 0.5
        let w = vec![1.0, 1.0];
        let loss = brier_loss(&y, &yhat, &w);
        assert_eq!(loss, 0.25);
    }

    #[test]
    fn test_auc_metric() {
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let yhat = vec![0.8, 0.2, 0.7, 0.1];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let auc = roc_auc_score(&y, &yhat, &w);
        assert_eq!(auc, 1.0);

        let yhat2 = vec![0.1, 0.9, 0.2, 0.8];
        let auc2 = roc_auc_score(&y, &yhat2, &w);
        assert_eq!(auc2, 0.0);
    }

    #[test]
    fn test_auc_metric_tied() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.5, 0.5];
        let w = vec![1.0, 1.0];
        let auc = roc_auc_score(&y, &yhat, &w);
        assert_eq!(auc, 0.5);
    }

    #[test]
    fn test_auc_all_same() {
        let y = vec![1.0, 1.0];
        let yhat = vec![0.5, 0.5];
        let w = vec![1.0, 1.0];
        assert!(roc_auc_score(&y, &yhat, &w).is_nan());

        let y2 = vec![0.0, 0.0];
        assert!(roc_auc_score(&y2, &yhat, &w).is_nan());
    }

    #[test]
    fn test_metric_traits() {
        let y = vec![1.0, 0.0];
        let yhat = vec![0.0, 0.0];
        let w = vec![1.0, 1.0];
        let g: Vec<u64> = vec![0, 0];

        assert!(!LogLossMetric::maximize());
        assert!(LogLossMetric::calculate_metric(&y, &yhat, &w, &g, None::<f32>) > 0.0);

        assert!(!BrierLossMetric::maximize());
        assert_eq!(BrierLossMetric::calculate_metric(&y, &yhat, &w, &g, None::<f32>), 0.25);

        assert!(AUCMetric::maximize());
        assert_eq!(AUCMetric::calculate_metric(&y, &yhat, &w, &g, None::<f32>), 0.5);
    }
}
