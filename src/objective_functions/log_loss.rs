//! Negative Logloss function
//!
//!
// Update the import path to match the actual location of ObjectiveFunction
use crate::objective_functions::ObjectiveFunction;
use crate::{data::FloatData, metrics::evaluation::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                    (-(*y_ * yhat_.ln() + (f64::ONE - *y_) * ((f64::ONE - yhat_).ln())) * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                    (-(*y_ * yhat_.ln() + (f64::ONE - *y_) * ((f64::ONE - yhat_).ln()))) as f32
                })
                .collect(),
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
                f64::ln(ytot / (ntot - ytot))
            }
            None => {
                let ytot = fast_sum(y);
                let ntot = y.len() as f64;
                f64::ln(ytot / (ntot - ytot))
            }
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
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;
                    let w_val = w[i] as f32;

                    // Sigmoid in f32
                    let p = 1.0 / (1.0 + (-yhat_val).exp());

                    g.push((p - y_val) * w_val);
                    h.push(p * (1.0 - p) * w_val);
                }
                (g, Some(h))
            }
            None => {
                for i in 0..len {
                    let y_val = y[i] as f32;
                    let yhat_val = yhat[i] as f32;

                    // Sigmoid in f32
                    let p = 1.0 / (1.0 + (-yhat_val).exp());

                    g.push(p - y_val);
                    h.push(p * (1.0 - p));
                }
                (g, Some(h))
            }
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }
}
