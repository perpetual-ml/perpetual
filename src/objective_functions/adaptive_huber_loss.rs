use super::ObjectiveFunction;
use crate::metrics::Metric;

/// Adaptive Huber Loss
#[derive(Default)]
pub struct AdaptiveHuberLoss {}
impl ObjectiveFunction for AdaptiveHuberLoss {
    fn calc_loss(y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, quantile: Option<f64>) -> Vec<f32> {
        // default alpha: 0.5
        // if not passed explicitly
        let alpha = quantile.unwrap_or(0.5);
        let n = y.len();

        let mut abs_res = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect::<Vec<_>>();
        abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((n as f64) * alpha).floor() as usize;
        let delta = abs_res[idx.min(n - 1)];

        match sample_weight {
            Some(weights) => y
                .iter()
                .zip(yhat.iter())
                .enumerate()
                .map(|(i, (&yi, &yh))| {
                    let r = yi - yh;
                    let ar = r.abs();
                    let base = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    (base * weights[i]) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat.iter())
                .map(|(&yi, &yh)| {
                    let r = yi - yh;
                    let ar = r.abs();
                    let loss = if ar <= delta {
                        0.5 * r * r
                    } else {
                        delta * (ar - 0.5 * delta)
                    };
                    loss as f32
                })
                .collect(),
        }
    }

    fn calc_grad_hess(
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f64>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        // default alpha: 0.5
        // if not passed explicitly
        let alpha = quantile.unwrap_or(0.5);
        let n = y.len();

        let mut abs_res = y
            .iter()
            .zip(yhat.iter())
            .map(|(&yi, &yh)| (yi - yh).abs())
            .collect::<Vec<_>>();
        abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((n as f64) * alpha).floor() as usize;
        let delta = abs_res[idx.min(n - 1)];

        match sample_weight {
            Some(weights) => {
                let (grad, hess): (Vec<f32>, Vec<f32>) = y
                    .iter()
                    .zip(yhat.iter())
                    .enumerate()
                    .map(|(i, (&yi, &yh))| {
                        let r = yi - yh;
                        let ar = r.abs();
                        let sign = (yh - yi).signum();
                        let g = if ar <= delta {
                            (yh - yi) * weights[i]
                        } else {
                            delta * sign * weights[i]
                        };
                        let h = if ar <= delta { weights[i] } else { 0.0 };
                        (g as f32, h as f32)
                    })
                    .unzip();
                (grad, Some(hess))
            }
            None => {
                let (grad, hess): (Vec<f32>, Vec<f32>) = y
                    .iter()
                    .zip(yhat.iter())
                    .map(|(&yi, &yh)| {
                        let r = yi - yh;
                        let ar = r.abs();
                        let sign = (yh - yi).signum();
                        let g = if ar <= delta { yh - yi } else { delta * sign };
                        let h = if ar <= delta { 1.0 } else { 0.0 };
                        (g as f32, h as f32)
                    })
                    .unzip();
                (grad, Some(hess))
            }
        }
    }

    fn calc_init(y: &[f64], sample_weight: Option<&[f64]>, _quantile: Option<f64>) -> f64 {
        let mut idxs = (0..y.len()).collect::<Vec<_>>();
        idxs.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap());

        let total_w = sample_weight.map(|w| w.iter().sum::<f64>()).unwrap_or(y.len() as f64);
        let target = total_w * 0.5;

        // find weighted median via scan()
        let median = idxs
            .iter()
            .scan(0.0, |cum, &i| {
                *cum += sample_weight.map_or(1.0, |w| w[i]);
                Some((i, *cum))
            })
            .find(|&(_i, cum)| cum >= target)
            .map(|(i, _)| y[i])
            .unwrap_or(y[idxs[y.len() / 2]]);

        median
    }

    fn default_metric() -> Metric {
        Metric::RootMeanSquaredError
    }
}
