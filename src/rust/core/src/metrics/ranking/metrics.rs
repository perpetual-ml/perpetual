use crate::metrics::evaluation::EvaluationMetric;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, PartialEq)]
pub enum GainScheme {
    Jarvelin,
    Burges,
}

pub struct NDCGMetric {
    k: Option<u64>,
    gain: GainScheme,
}

impl NDCGMetric {
    pub fn new(k: Option<u64>, gain: GainScheme) -> Self {
        Self { k, gain }
    }
}

impl EvaluationMetric for NDCGMetric {
    fn calculate_metric(y: &[f64], yhat: &[f64], sample_weight: &[f64], group: &[u64], _alpha: Option<f32>) -> f64 {
        // Use self.k and self.gain instead of hardcoded values
        // Note: This requires changing the trait to accept &self, or you can implement a method on NDCGMetric itself.
        // For demonstration, let's add a method to NDCGMetric:
        // You need an instance of NDCGMetric to call the method with &self
        let metric = NDCGMetric {
            k: None,
            gain: GainScheme::Burges,
        };
        metric.calculate_metric_with_params(y, yhat, sample_weight, group, None, &GainScheme::Burges)
    }
    fn maximize() -> bool {
        true
    }
}

// Add a method to use the struct fields
impl NDCGMetric {
    pub fn calculate_metric_with_params(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: &[f64],
        group: &[u64],
        _alpha: Option<f32>,
        _default_gain: &GainScheme,
    ) -> f64 {
        ndcg_at_k_metric(y, yhat, sample_weight, group, self.k, &self.gain)
    }
}

#[inline]
fn compute_discount(rank: usize) -> f64 {
    1.0 / ((rank + 2) as f64).log2()
}

#[inline]
fn compute_gain(relevance: f64, scheme: &GainScheme) -> f64 {
    match scheme {
        GainScheme::Jarvelin => relevance,
        GainScheme::Burges => 2_f64.powf(relevance) - 1.0,
    }
}

fn compute_group_dcg(relevance_scores: &[f64], k: Option<u64>, weights: &[f64], scheme: &GainScheme) -> f64 {
    let limit = k
        .map(|k| k as usize)
        .unwrap_or(relevance_scores.len())
        .min(relevance_scores.len());

    relevance_scores
        .iter()
        .zip(weights)
        .take(limit)
        .enumerate()
        .map(|(rank, (&relevance, &weight))| {
            let gain = compute_gain(relevance, scheme) * weight;
            let discount = compute_discount(rank);
            gain * discount
        })
        .sum()
}

fn compute_group_ndcg(
    y_group: &[f64],
    yhat_group: &[f64],
    weights_group: &[f64],
    k: Option<u64>,
    scheme: &GainScheme,
) -> f64 {
    if y_group.is_empty() {
        return 0.0;
    }

    // Create (relevance, prediction, weight) tuples with original indices
    let mut items: Vec<(f64, f64, f64, usize)> = y_group
        .iter()
        .zip(yhat_group)
        .zip(weights_group)
        .enumerate()
        .map(|(idx, ((&y, &yhat), &weight))| (y, yhat, weight, idx))
        .collect();

    // Sort by predictions (descending) to get predicted ranking
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Extract relevance scores and weights in predicted order
    let predicted_relevance: Vec<f64> = items.iter().map(|(y, _, _, _)| *y).collect();
    let predicted_weights: Vec<f64> = items.iter().map(|(_, _, weight, _)| *weight).collect();

    // Compute DCG for predicted ranking
    let dcg = compute_group_dcg(&predicted_relevance, k, &predicted_weights, scheme);

    // Compute IDCG (Ideal DCG) by sorting by true relevance
    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let ideal_relevance: Vec<f64> = items.iter().map(|(y, _, _, _)| *y).collect();
    let ideal_weights: Vec<f64> = items.iter().map(|(_, _, weight, _)| *weight).collect();

    let idcg = compute_group_dcg(&ideal_relevance, k, &ideal_weights, scheme);

    // Return NDCG
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

pub fn ndcg_at_k_metric(
    y: &[f64],
    yhat: &[f64],
    sample_weight: &[f64],
    group: &[u64],
    k: Option<u64>,
    scheme: &GainScheme,
) -> f64 {
    if y.is_empty() {
        return 0.0;
    }

    let mut start = 0;
    let mut total_ndcg = 0.0;
    let mut total_weight = 0.0;

    for &group_size in group {
        let end = start + group_size as usize;

        if end > y.len() {
            break;
        }

        let y_group = &y[start..end];
        let yhat_group = &yhat[start..end];
        let weights_group = &sample_weight[start..end];

        let group_ndcg = compute_group_ndcg(y_group, yhat_group, weights_group, k, scheme);

        // Weight each group's NDCG by the sum of its sample weights
        // TODO: is the the desired behaviour?
        let group_weight: f64 = weights_group.iter().sum();
        total_ndcg += group_ndcg * group_weight;
        total_weight += group_weight;

        start = end;
    }

    if total_weight > 0.0 {
        total_ndcg / total_weight
    } else {
        // TODO: I dont know what would be logical/optimal here
        0.0
    }
}
