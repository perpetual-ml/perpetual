use super::ObjectiveFunction;
use crate::{data::FloatData, metrics::Metric, utils::fast_sum};

#[derive(Default)]
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _quantile: Option<f64>) -> Vec<f32> {
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

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, _quantile: Option<f64>) -> f64 {
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
    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _quantile: Option<f64>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
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
                (g, Some(h))
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
                (g, Some(h))
            }
        }
    }

    fn default_metric() -> Metric {
        Metric::LogLoss
    }
}
