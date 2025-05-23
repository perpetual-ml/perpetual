use crate::{errors::PerpetualError, utils::percentiles, Matrix, UnivariateBooster};
use crate::objective_functions::{ObjectiveFunction, Objective, CustomObjective};
use std::collections::HashMap;

pub type CalData<'a> = (Matrix<'a, f64>, &'a [f64], &'a [f64]); // (x_flat_data, rows, cols), y, alpha

impl UnivariateBooster {
    /// Calibrate models to get prediction intervals
    /// * `alpha` - Alpha list to train calibration models for
    pub fn calibrate(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        data_cal: CalData,
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha) = data_cal;

        for alpha_ in alpha {
            let lower_quantile = Some(alpha_ / 2.0);
            let mut model_lower = UnivariateBooster::default()
                .set_objective(Objective::QuantileLoss { quantile: lower_quantile } );
            model_lower.fit(&data, &y, sample_weight)?;

            let upper_quantile = Some(1.0 - alpha_ / 2.0);
            let mut model_upper = UnivariateBooster::default()
                .set_objective(Objective::QuantileLoss { quantile: upper_quantile } );

            model_upper.fit(&data, &y, sample_weight)?;

            let y_cal_pred_lower = model_lower.predict(&x_cal, true);
            let y_cal_pred_upper = model_upper.predict(&x_cal, true);
            let mut scores: Vec<f64> = Vec::with_capacity(y_cal.len());
            for i in 0..y_cal.len() {
                scores.push(f64::max(y_cal_pred_lower[i] - y_cal[i], y_cal[i] - y_cal_pred_upper[i]));
            }
            let perc = (1.0 - (*alpha_ as f64)) * (1.0 + 1.0 * (1.0 / (scores.len() as f64)));
            let score = percentiles(&scores, &vec![1.0; scores.len()], &vec![perc])[0];
            self.cal_models
                .insert(alpha_.to_string(), [(model_lower, -score), (model_upper, score)]);
        }
        Ok(())
    }

    pub fn predict_intervals(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        for (alpha, value) in &self.cal_models {
            let (model_lower, score_lower) = &value[0];
            let (model_upper, score_upper) = &value[1];
            let lower_preds = model_lower
                .predict(data, parallel)
                .iter()
                .map(|p| p + score_lower)
                .collect();
            let upper_preds = model_upper
                .predict(data, parallel)
                .iter()
                .map(|p| p + score_upper)
                .collect();
            intervals.insert(alpha.to_string(), vec![lower_preds, upper_preds]);
        }
        intervals
    }
}