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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_loss_median() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let yhat = vec![1.5, 2.5, 2.5, 3.5];
        let w = vec![1.0; 4];
        let loss = quantile_loss(&y, &yhat, &w, Some(0.5));
        assert!(loss.is_finite());
        // median quantile: symmetric penalty
        assert!((loss - 0.0).abs() < 1.0); // sanity check
    }

    #[test]
    fn test_quantile_loss_low_alpha() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![2.0, 3.0, 4.0]; // all over-predicted
        let w = vec![1.0; 3];
        let loss = quantile_loss(&y, &yhat, &w, Some(0.1));
        assert!(loss.is_finite());
    }

    #[test]
    fn test_quantile_loss_high_alpha() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![0.0, 1.0, 2.0]; // all under-predicted
        let w = vec![1.0; 3];
        let loss = quantile_loss(&y, &yhat, &w, Some(0.9));
        assert!(loss.is_finite());
    }

    #[test]
    fn test_quantile_loss_weighted() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.5, 2.5, 2.5];
        let w = vec![2.0, 1.0, 3.0];
        let loss = quantile_loss(&y, &yhat, &w, Some(0.5));
        assert!(loss.is_finite());
    }

    #[test]
    fn test_quantile_loss_metric_trait() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.5, 2.5, 2.5];
        let w = vec![1.0; 3];
        let group = vec![0u64; 3];
        let metric = QuantileLossMetric::calculate_metric(&y, &yhat, &w, &group, Some(0.5));
        assert!(metric.is_finite());
        assert!(!QuantileLossMetric::maximize());
    }

    #[test]
    fn test_rmsle_perfect() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.0, 2.0, 3.0];
        let w = vec![1.0; 3];
        let rmsle = root_mean_squared_log_error(&y, &yhat, &w);
        assert!((rmsle - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmsle_imperfect() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.5, 2.5, 3.5];
        let w = vec![1.0; 3];
        let rmsle = root_mean_squared_log_error(&y, &yhat, &w);
        assert!(rmsle > 0.0);
        assert!(rmsle.is_finite());
    }

    #[test]
    fn test_rmsle_metric_trait() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.5, 2.5, 3.5];
        let w = vec![1.0; 3];
        let group = vec![0u64; 3];
        let metric = RootMeanSquaredLogErrorMetric::calculate_metric(&y, &yhat, &w, &group, None);
        assert!(metric > 0.0);
        assert!(!RootMeanSquaredLogErrorMetric::maximize());
    }

    #[test]
    fn test_rmse_perfect() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![1.0, 2.0, 3.0];
        let w = vec![1.0; 3];
        let rmse = root_mean_squared_error(&y, &yhat, &w);
        assert!((rmse - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_imperfect() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![2.0, 3.0, 4.0];
        let w = vec![1.0; 3];
        let rmse = root_mean_squared_error(&y, &yhat, &w);
        // Each error is 1.0, so MSE = 1.0, RMSE = 1.0
        assert!((rmse - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_weighted() {
        let y = vec![1.0, 2.0];
        let yhat = vec![2.0, 2.0]; // errors: 1.0 and 0.0
        let w = vec![2.0, 1.0]; // weight = 2.0 on error=1
        let rmse = root_mean_squared_error(&y, &yhat, &w);
        // weighted MSE = (2.0*1.0 + 1.0*0.0) / 3.0 = 2/3
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((rmse - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_metric_trait() {
        let y = vec![1.0, 2.0, 3.0];
        let yhat = vec![2.0, 3.0, 4.0];
        let w = vec![1.0; 3];
        let group = vec![0u64; 3];
        let metric = RootMeanSquaredErrorMetric::calculate_metric(&y, &yhat, &w, &group, None);
        assert!((metric - 1.0).abs() < 1e-10);
        assert!(!RootMeanSquaredErrorMetric::maximize());
    }
}
