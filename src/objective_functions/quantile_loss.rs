use super::ObjectiveFunction;
use crate::metrics::Metric;

#[derive(Default)]
pub struct QuantileLoss {}

impl ObjectiveFunction for QuantileLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let _quantile = quantile.unwrap();
                    let s = *y_ - *yhat_;
                    let l = if s >= 0.0 { _quantile * s } else { (_quantile - 1.0) * s };
                    (l * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let _quantile = quantile.unwrap();
                    let s = *y_ - *yhat_;
                    let l = if s >= 0.0 { _quantile * s } else { (_quantile - 1.0) * s };
                    l as f32
                })
                .collect(),
        }
    }

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>) -> f64 {
        match sample_weight {
            Some(sample_weight) => {
                let mut indices = (0..y.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
                let w_tot: f64 = sample_weight.iter().sum();
                let w_target = w_tot * quantile.unwrap() as f64;
                let mut w_cum = 0.0_f64;
                let mut init_value = f64::NAN;
                for i in indices {
                    w_cum += sample_weight[i];
                    if w_cum >= w_target {
                        init_value = y[i];
                        break;
                    }
                }
                init_value
            }
            None => {
                let mut indices = (0..y.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
                let w_tot: f64 = y.len() as f64;
                let w_target = w_tot * quantile.unwrap() as f64;
                let mut w_cum = 0.0_f64;
                let mut init_value = f64::NAN;
                for i in indices {
                    w_cum += 1.0;
                    if w_cum >= w_target {
                        init_value = y[i];
                        break;
                    }
                }
                init_value
            }
        }
    }

    #[inline]
    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f64>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        match sample_weight {
            Some(sample_weight) => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .zip(sample_weight)
                    .map(|((y_, yhat_), w_)| {
                        let _quantile = quantile.unwrap();
                        let delta = yhat_ - *y_;
                        let g = if delta >= 0.0 {
                            (1.0 - _quantile) * w_
                        } else {
                            -1.0 * _quantile * w_
                        };
                        (g as f32, *w_ as f32)
                    })
                    .unzip();
                (g, Some(h))
            }
            None => {
                let g = y
                    .iter()
                    .zip(yhat)
                    .map(|(y_, yhat_)| {
                        let _quantile = quantile.unwrap();
                        let delta = yhat_ - *y_;
                        let g = if delta >= 0.0 {
                            1.0 - _quantile
                        } else {
                            -1.0 * _quantile
                        };
                        g as f32
                    })
                    .collect();
                (g, None)
            }
        }
    }

    fn default_metric() -> Metric {
        Metric::QuantileLoss
    }
}
