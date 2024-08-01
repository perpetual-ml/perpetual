use crate::{data::FloatData, metric::Metric, utils::fast_sum};
use serde::{Deserialize, Serialize};

type ObjFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f32>) -> (Vec<f32>, Option<Vec<f32>>);
type LossFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f32>) -> Vec<f32>;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Objective {
    LogLoss,
    SquaredLoss,
    QuantileLoss,
}

pub fn loss_callables(objective: &Objective) -> LossFn {
    match objective {
        Objective::LogLoss => LogLoss::calc_loss,
        Objective::SquaredLoss => SquaredLoss::calc_loss,
        Objective::QuantileLoss => QuantileLoss::calc_loss,
    }
}

pub fn gradient_hessian_callables(objective: &Objective) -> ObjFn {
    match objective {
        Objective::LogLoss => LogLoss::calc_grad_hess,
        Objective::SquaredLoss => SquaredLoss::calc_grad_hess,
        Objective::QuantileLoss => QuantileLoss::calc_grad_hess,
    }
}

pub fn calc_init_callables(objective: &Objective) -> fn(&[f64], Option<&[f64]>, Option<f32>) -> f64 {
    match objective {
        Objective::LogLoss => LogLoss::calc_init,
        Objective::SquaredLoss => SquaredLoss::calc_init,
        Objective::QuantileLoss => QuantileLoss::calc_init,
    }
}

pub trait ObjectiveFunction {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, alpha: Option<f32>) -> Vec<f32>;
    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        alpha: Option<f32>,
    ) -> (Vec<f32>, Option<Vec<f32>>);
    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, alpha: Option<f32>) -> f64;
    fn default_metric() -> Metric;
}

#[derive(Default)]
pub struct LogLoss {}

impl ObjectiveFunction for LogLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _alpha: Option<f32>) -> Vec<f32> {
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

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, _alpha: Option<f32>) -> f64 {
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
        _alpha: Option<f32>,
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

#[derive(Default)]
pub struct SquaredLoss {}

impl ObjectiveFunction for SquaredLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _alpha: Option<f32>) -> Vec<f32> {
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

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, _alpha: Option<f32>) -> f64 {
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

    #[inline]
    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _alpha: Option<f32>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        match sample_weight {
            Some(sample_weight) => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .zip(sample_weight)
                    .map(|((y_, yhat_), w_)| (((yhat_ - *y_) * *w_) as f32, *w_ as f32))
                    .unzip();
                (g, Some(h))
            }
            None => (
                y.iter().zip(yhat).map(|(y_, yhat_)| (yhat_ - *y_) as f32).collect(),
                None,
            ),
        }
    }

    fn default_metric() -> Metric {
        Metric::RootMeanSquaredLogError
    }
}

#[derive(Default)]
pub struct QuantileLoss {}

impl ObjectiveFunction for QuantileLoss {
    #[inline]
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, alpha: Option<f32>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let _alpha = alpha.unwrap();
                    let s = (*y_ - *yhat_) as f32;
                    let l = if s >= 0.0 { _alpha * s } else { (_alpha - 1.0) * s };
                    l * *w_ as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let _alpha = alpha.unwrap();
                    let s = (*y_ - *yhat_) as f32;
                    let l = if s >= 0.0 { _alpha * s } else { (_alpha - 1.0) * s };
                    l
                })
                .collect(),
        }
    }

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, alpha: Option<f32>) -> f64 {
        match sample_weight {
            Some(sample_weight) => {
                let mut indices = (0..y.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
                let w_tot: f64 = sample_weight.iter().sum();
                let w_target = w_tot * alpha.unwrap() as f64;
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
                let w_target = w_tot * alpha.unwrap() as f64;
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
        alpha: Option<f32>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        match sample_weight {
            Some(sample_weight) => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .zip(sample_weight)
                    .map(|((y_, yhat_), w_)| {
                        let _alpha = alpha.unwrap();
                        let _w = *w_ as f32;
                        let delta = yhat_ - *y_;
                        let g = if delta >= 0.0 {
                            (1.0 - _alpha) * _w
                        } else {
                            -1.0 * _alpha * _w
                        };
                        (g, _w)
                    })
                    .unzip();
                (g, Some(h))
            }
            None => {
                let g = y
                    .iter()
                    .zip(yhat)
                    .map(|(y_, yhat_)| {
                        let _alpha = alpha.unwrap();
                        let delta = yhat_ - *y_;
                        let g = if delta >= 0.0 { 1.0 - _alpha } else { -1.0 * _alpha };
                        g
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_logloss_loss() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let l1 = LogLoss::calc_loss(&y, &yhat1, None, None);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let l2 = LogLoss::calc_loss(&y, &yhat2, None, None);
        assert!(l1.iter().sum::<f32>() < l2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_grad() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let yhat1 = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let (g1, _) = LogLoss::calc_grad_hess(&y, &yhat1, None, None);
        let yhat2 = vec![0.0, 0.0, -1.0, 1.0, 0.0, 1.0];
        let (g2, _) = LogLoss::calc_grad_hess(&y, &yhat2, None, None);
        assert!(g1.iter().sum::<f32>() < g2.iter().sum::<f32>());
    }

    #[test]
    fn test_logloss_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let l1 = LogLoss::calc_init(&y, None, None);
        assert!(l1 == 0.);

        let y = vec![1.0; 6];
        let l2 = LogLoss::calc_init(&y, None, None);
        assert!(l2 == f64::INFINITY);

        let y = vec![0.0; 6];
        let l3 = LogLoss::calc_init(&y, None, None);
        assert!(l3 == f64::NEG_INFINITY);

        let y = vec![0., 0., 0., 0., 1., 1.];
        let l4 = LogLoss::calc_init(&y, None, None);
        assert!(l4 == f64::ln(2. / 4.));
    }

    #[test]
    fn test_mse_init() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let l1 = SquaredLoss::calc_init(&y, None, None);
        assert!(l1 == 0.5);

        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let l2 = SquaredLoss::calc_init(&y, None, None);
        assert!(l2 == 1.);

        let y = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let l3 = SquaredLoss::calc_init(&y, None, None);
        assert!(l3 == -1.);

        let y = vec![-1.0, -1.0, -1.0, 1., 1., 1.];
        let l4 = SquaredLoss::calc_init(&y, None, None);
        assert!(l4 == 0.);
    }

    #[test]
    fn test_quantile_init() {
        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let w = vec![0.0, 0.5, 1.0, 0.3, 0.5];
        let l1 = QuantileLoss::calc_init(&y, Some(&w), Some(0.1));
        println!("{}", l1);
        assert!(l1 == 2.0);

        let y = vec![1.0, 2.0, 9.0, 3.2, 4.0];
        let w = vec![0.0, 0.5, 1.0, 0.3, 0.5];
        let l2 = QuantileLoss::calc_init(&y, Some(&w), Some(0.9));
        println!("{}", l2);
        assert!(l2 == 9.0);
    }
}
