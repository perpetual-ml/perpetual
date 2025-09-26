use crate::errors::PerpetualError;
use crate::metrics::classification;
pub use crate::metrics::ranking::GainScheme;
use crate::metrics::{ranking, regression};
use crate::utils::items_to_strings;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub type MetricFn = fn(&[f64], &[f64], &[f64], &[u64], Option<f32>) -> f64;

/// Compare to metric values, determining if b is better.
/// If one of them is NaN favor the non NaN value.
/// If both are NaN, consider the first value to be better.
pub fn is_comparison_better(value: f64, comparison: f64, maximize: bool) -> bool {
    match (value.is_nan(), comparison.is_nan()) {
        // Both nan, comparison is not better,
        // Or comparison is nan, also not better
        (true, true) | (false, true) => false,
        // comparison is not Nan, it's better
        (true, false) => true,
        // Perform numerical comparison.
        (false, false) => {
            // If we are maximizing is the comparison
            // greater, than the current value
            if maximize {
                value < comparison
            // If we are minimizing is the comparison
            // less than the current value.
            } else {
                value > comparison
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum Metric {
    AUC,
    LogLoss,
    RootMeanSquaredLogError,
    RootMeanSquaredError,
    QuantileLoss,
    NDCG { k: Option<u64>, gain: GainScheme },
}

fn get_parse_error(s: &str) -> PerpetualError {
    PerpetualError::ParseString(
        s.to_string(),
        "Metric".to_string(),
        items_to_strings(vec![
            "AUC",
            "LogLoss",
            "RootMeanSquaredLogError",
            "RootMeanSquaredError",
            "NDCG",
            "NDCG@k",
        ]),
    )
}

impl FromStr for Metric {
    type Err = PerpetualError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "AUC" => Ok(Metric::AUC),
            "LogLoss" => Ok(Metric::LogLoss),
            "RootMeanSquaredLogError" => Ok(Metric::RootMeanSquaredLogError),
            "RootMeanSquaredError" => Ok(Metric::RootMeanSquaredError),

            // TODO: also parse gain scheme?
            "NDCG" => Ok(Metric::NDCG {
                k: None,
                gain: GainScheme::Burges,
            }),

            _ if s.starts_with("NDCG@") => {
                let k_str = &s["NDCG@".len()..];
                let k = k_str.parse().map_err(|_| get_parse_error(s))?;
                Ok(Metric::NDCG {
                    k: Some(k),
                    gain: GainScheme::Burges,
                })
            }

            _ => Err(get_parse_error(s)),
        }
    }
}

pub fn metric_callables(metric_type: &Metric) -> (MetricFn, bool) {
    match metric_type {
        Metric::AUC => (
            classification::AUCMetric::calculate_metric,
            classification::AUCMetric::maximize(),
        ),
        Metric::LogLoss => (
            classification::LogLossMetric::calculate_metric,
            classification::LogLossMetric::maximize(),
        ),
        Metric::RootMeanSquaredLogError => (
            regression::RootMeanSquaredLogErrorMetric::calculate_metric,
            regression::RootMeanSquaredLogErrorMetric::maximize(),
        ),
        Metric::RootMeanSquaredError => (
            regression::RootMeanSquaredErrorMetric::calculate_metric,
            regression::RootMeanSquaredErrorMetric::maximize(),
        ),
        Metric::QuantileLoss => (
            regression::QuantileLossMetric::calculate_metric,
            regression::QuantileLossMetric::maximize(),
        ),

        // TODO: decide if and/or how to do this
        Metric::NDCG { k: _k, gain: _gain } => (ranking::NDCGMetric::calculate_metric, ranking::NDCGMetric::maximize()),
    }
}

pub trait EvaluationMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], group: &[u64], alpha: Option<f32>) -> f64;
    fn maximize() -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::classification::*;
    use crate::metrics::ranking::*;
    use crate::metrics::regression::*;
    use crate::utils::precision_round;

    #[test]
    fn test_root_mean_squared_log_error() {
        let y = vec![1., 3., 4., 5., 2., 4., 6.];
        let yhat = vec![3., 2., 3., 4., 4., 4., 4.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = root_mean_squared_log_error(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 4), 0.3549);
    }
    #[test]
    fn test_root_mean_squared_error() {
        let y = vec![1., 3., 4., 5., 2., 4., 6.];
        let yhat = vec![3., 2., 3., 4., 4., 4., 4.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = root_mean_squared_error(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 6), 1.452966);
    }

    #[test]
    fn test_log_loss() {
        let y = vec![1., 0., 1., 0., 0., 0., 0.];
        let yhat = vec![0.5, 0.01, -0., 1.05, 0., -4., 0.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = log_loss(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 5), 0.59235);
    }

    #[test]
    fn test_auc_real_data() {
        let y = vec![1., 0., 1., 0., 0., 0., 0.];
        let yhat = vec![0.5, 0.01, -0., 1.05, 0., -4., 0.];
        let sample_weight = vec![1., 1., 1., 1., 1., 2., 2.];
        let res = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(precision_round(res, 5), 0.67857);
    }

    #[test]
    fn test_auc_generc() {
        let sample_weight: Vec<f64> = vec![1.; 2];

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![0., 1.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.);

        let y: Vec<f64> = vec![0., 1.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 1.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![1., 0.];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 1.0);

        let y: Vec<f64> = vec![1., 0.];
        let yhat: Vec<f64> = vec![0.5, 0.5];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert_eq!(auc_score, 0.5);

        let y: Vec<f64> = vec![0., 0.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());

        let y: Vec<f64> = vec![1., 1.];
        let yhat: Vec<f64> = vec![0.25, 0.75];
        let auc_score = roc_auc_score(&y, &yhat, &sample_weight);
        assert!(auc_score.is_nan());
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let y = vec![3.0, 2.0, 1.0, 0.0];
        let yhat = vec![1.0, 0.8, 0.6, 0.4];
        let weights = vec![1.0; 4];
        let group = vec![4];

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Burges);
        assert!((ndcg - 1.0).abs() < 1e-10);

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Jarvelin);
        assert!((ndcg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_reversed_ranking() {
        let y = vec![3.0, 2.0, 1.0, 0.0];
        let yhat = vec![0.4, 0.6, 0.8, 1.0];
        let weights = vec![1.0; 4];
        let group = vec![4];

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Burges);
        assert!(ndcg < 1.0 && ndcg >= 0.0);

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Jarvelin);
        assert!(ndcg < 1.0 && ndcg >= 0.0);
    }

    #[test]
    fn test_ndcg_at_k() {
        let y = vec![3.0, 2.0, 1.0, 0.0, 1.0];
        let yhat = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let weights = vec![1.0; 5];
        let group = vec![5];

        let ndcg_full = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Burges);
        let ndcg_at_3 = ndcg_at_k_metric(&y, &yhat, &weights, &group, Some(3), &GainScheme::Burges);

        assert!((ndcg_full - ndcg_at_3).abs() > 1e-10);

        let ndcg_full = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Jarvelin);
        let ndcg_at_3 = ndcg_at_k_metric(&y, &yhat, &weights, &group, Some(3), &GainScheme::Jarvelin);

        assert!((ndcg_full - ndcg_at_3).abs() > 1e-10);
    }

    #[test]
    fn test_multiple_groups() {
        let y = vec![2.0, 1.0, 3.0, 1.0, 0.0];
        let yhat = vec![0.8, 0.6, 1.0, 0.4, 0.2];
        let weights = vec![1.0; 5];
        let group = vec![2, 3];

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Burges);
        assert!(ndcg >= 0.0 && ndcg <= 1.0);

        let ndcg = ndcg_at_k_metric(&y, &yhat, &weights, &group, None, &GainScheme::Jarvelin);
        assert!(ndcg >= 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_ndcg_parsing_no_k() {
        let metric = Metric::from_str("NDCG").unwrap();
        match metric {
            Metric::NDCG { k, gain } => {
                assert_eq!(k, None);
                assert_eq!(gain, GainScheme::Burges);
            }
            _ => panic!("Expected Metric::NDCG"),
        }
    }

    #[test]
    fn test_ndcg_parsing_with_k() {
        let metric = Metric::from_str("NDCG@10").unwrap();
        match metric {
            Metric::NDCG { k, gain } => {
                assert_eq!(k, Some(10));
                assert_eq!(gain, GainScheme::Burges);
            }
            _ => panic!("Expected Metric::NDCG"),
        }
    }

    #[test]
    fn test_ndcg_parsing_invalid_k() {
        let result = Metric::from_str("NDCG@foo");
        assert!(result.is_err(), "Expected error for invalid k");
    }

    #[test]
    fn test_ndcg_parsing_invalid_format() {
        let result = Metric::from_str("NDCG@@10");
        assert!(result.is_err(), "Expected error for invalid format");
    }
}
