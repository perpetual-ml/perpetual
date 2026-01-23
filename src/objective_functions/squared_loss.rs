//! Squared Loss function
//!
//!
use crate::{metrics::evaluation::Metric, objective_functions::objective::ObjectiveFunction, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct SquaredLoss {}

impl ObjectiveFunction for SquaredLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let s = *y_ - *yhat_;
                    (s * s * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let s = *y_ - *yhat_;
                    (s * s) as f32
                })
                .collect(),
        }
    }

    #[inline]
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let len = y.len();
        let mut g = Vec::with_capacity(len);
        let mut h = Vec::with_capacity(len);

        match sample_weight {
            Some(w) => {
                for i in 0..len {
                    let diff = (yhat[i] - y[i]) as f32;
                    let w_val = w[i] as f32;
                    g.push(diff * w_val);
                    h.push(w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let diff = (yhat[i] - y[i]) as f32;
                    g.push(diff);
                }
                (g, None)
            }
        }
    }

    #[inline]
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        match sample_weight {
            Some(sample_weight) => {
                let mut ytot: f64 = 0.;
                let mut ntot: f64 = 0.;
                for i in 0..y.len() {
                    ytot += sample_weight[i] * y[i];
                    ntot += sample_weight[i];
                }
                ytot / ntot
            }
            None => fast_sum(y) / y.len() as f64,
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredLogError
    }
}
