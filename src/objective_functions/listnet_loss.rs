//! ListNet Loss function
//!
//!
use crate::metrics::ranking::GainScheme;
use crate::{metrics::evaluation::Metric, objective_functions::objective::ObjectiveFunction};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct ListNetLoss {}

const LOSS_FOR_SINGLE_GROUP: f32 = f32::INFINITY;
const EPSILON: f32 = 1e-15;

#[inline]
fn compute_softmax_inplace(input: &[f64], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;

    // First pass: compute exp and sum
    for (i, &val) in input.iter().enumerate() {
        let exp_val = ((val - max_val) as f32).exp();
        output[i] = exp_val;
        sum += exp_val;
    }

    // Second pass: normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in output.iter_mut() {
            *val *= inv_sum;
        }
    }
}

#[inline]
fn compute_listnet_loss(softmax_y: &[f32], softmax_yhat: &[f32], weights: Option<&[f64]>) -> f32 {
    match weights {
        Some(w) => softmax_y
            .iter()
            .zip(softmax_yhat)
            .zip(w)
            .map(|((p_y, p_yhat), weight)| {
                if *p_y > 0.0 {
                    -p_y * p_yhat.max(EPSILON).ln() * (*weight as f32)
                } else {
                    0.0
                }
            })
            .sum(),
        None => softmax_y
            .iter()
            .zip(softmax_yhat)
            .map(|(p_y, p_yhat)| {
                if *p_y > 0.0 {
                    -p_y * p_yhat.max(EPSILON).ln()
                } else {
                    0.0
                }
            })
            .sum(),
    }
}

#[inline]
fn compute_group_gradients(softmax_y: &[f32], softmax_yhat: &[f32], weights: Option<&[f64]>, output: &mut [f32]) {
    match weights {
        Some(w) => {
            for (i, ((p_yhat, p_y), weight)) in softmax_yhat.iter().zip(softmax_y).zip(w).enumerate() {
                output[i] = (p_yhat - p_y) * (*weight as f32);
            }
        }
        None => {
            for (i, (p_yhat, p_y)) in softmax_yhat.iter().zip(softmax_y).enumerate() {
                output[i] = p_yhat - p_y;
            }
        }
    }
}

#[inline]
fn compute_group_hessian(softmax_yhat: &[f32], weights: Option<&[f64]>, output: &mut [f32]) {
    // For ListNet, the hessian is H_ii = p_i * (1 - p_i) * weight_i
    match weights {
        Some(w) => {
            for (i, (p_yhat, weight)) in softmax_yhat.iter().zip(w).enumerate() {
                output[i] = p_yhat * (1.0 - p_yhat) * (*weight as f32);
            }
        }
        None => {
            for (i, p_yhat) in softmax_yhat.iter().enumerate() {
                output[i] = p_yhat * (1.0 - p_yhat);
            }
        }
    }
}

impl ObjectiveFunction for ListNetLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, group: Option<&[u64]>) -> Vec<f32> {
        if y.len() < 2 {
            return vec![LOSS_FOR_SINGLE_GROUP; y.len()];
        }
        let mut losses = vec![0.0f32; y.len()];

        if let Some(group_sizes) = group {
            let mut start = 0;
            for &group_size in group_sizes {
                let end = start + group_size as usize;
                let group_len = group_size as usize;

                let y_group = &y[start..end];
                let yhat_group = &yhat[start..end];
                let weight_group = sample_weight.map(|w| &w[start..end]);

                let mut softmax_y = vec![0.0f32; group_len];
                let mut softmax_yhat = vec![0.0f32; group_len];

                compute_softmax_inplace(y_group, &mut softmax_y);
                compute_softmax_inplace(yhat_group, &mut softmax_yhat);

                let group_loss = compute_listnet_loss(&softmax_y, &softmax_yhat, weight_group);

                let per_sample_loss = group_loss / (group_size as f32);
                losses[start..end].fill(per_sample_loss);
                start = end;
            }
        } else {
            let mut softmax_y = vec![0.0f32; y.len()];
            let mut softmax_yhat = vec![0.0f32; y.len()];

            compute_softmax_inplace(y, &mut softmax_y);
            compute_softmax_inplace(yhat, &mut softmax_yhat);

            let total_loss = compute_listnet_loss(&softmax_y, &softmax_yhat, sample_weight);

            let per_sample_loss = total_loss / (y.len() as f32);
            losses.fill(per_sample_loss);
        }

        losses
    }

    #[inline]
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        if y.len() < 2 {
            return (vec![0.0f32; y.len()], None);
        }

        let mut gradients = vec![0.0f32; y.len()];
        let mut hessians = vec![0.0f32; y.len()];

        if let Some(group_sizes) = group {
            let mut start = 0;
            for &group_size in group_sizes {
                let end = start + group_size as usize;
                let group_len = group_size as usize;

                let y_group = &y[start..end];
                let yhat_group = &yhat[start..end];
                let weight_group = sample_weight.map(|w| &w[start..end]);

                let mut softmax_y = vec![0.0f32; group_len];
                let mut softmax_yhat = vec![0.0f32; group_len];

                compute_softmax_inplace(y_group, &mut softmax_y);
                compute_softmax_inplace(yhat_group, &mut softmax_yhat);

                compute_group_gradients(&softmax_y, &softmax_yhat, weight_group, &mut gradients[start..end]);

                compute_group_hessian(&softmax_yhat, weight_group, &mut hessians[start..end]);

                start = end;
            }
        } else {
            let mut softmax_y = vec![0.0f32; y.len()];
            let mut softmax_yhat = vec![0.0f32; y.len()];

            compute_softmax_inplace(y, &mut softmax_y);
            compute_softmax_inplace(yhat, &mut softmax_yhat);

            compute_group_gradients(&softmax_y, &softmax_yhat, sample_weight, &mut gradients);

            compute_group_hessian(&softmax_yhat, sample_weight, &mut hessians);
        }

        (gradients, Some(hessians))
    }

    #[inline]
    fn initial_value(&self, _y: &[f64], _sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        0.0
    }

    fn default_metric(&self) -> Metric {
        Metric::NDCG {
            k: None,
            gain: GainScheme::Burges,
        }
    }
}
