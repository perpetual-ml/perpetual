//! Negative Logloss function
//! 
//! 
use super::ObjectiveFunction;
use crate::{data::FloatData, metrics::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub struct LogLoss {}
impl ObjectiveFunction for LogLoss {

    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> Vec<f32> {

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
    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>) -> f64 {

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
    fn gradient(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> (Vec<f32>, Option<Vec<f32>>, bool) {

        match sample_weight {
            Some(sample_weight) => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .zip(sample_weight)
                    .map(|((y_, yhat_), w_)| {
                        let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                        (((yhat_ - *y_) * *w_) as f32, (yhat_ * (f64::ONE - yhat_) * *w_) as f32)
                    })
                    .unzip();
                (g, Some(h), false)
            }
            None => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .map(|(y_, yhat_)| {
                        let yhat_ = f64::ONE / (f64::ONE + (-*yhat_).exp());
                        ((yhat_ - *y_) as f32, (yhat_ * (f64::ONE - yhat_)) as f32)
                    })
                    .unzip();
                (g, Some(h), false)
            }
        }

    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }

    fn hessian_is_constant(&self) -> bool {
        false
    }

    fn constant_hessian(&self, weights_flag: bool) -> bool {
        if weights_flag {
            false
        } else {
            false
        }
    }

}
