use crate::booster::config::CalibrationMethod;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};
use std::collections::HashMap;

impl PerpetualBooster {
    /// Calculate calibration scores for the given data based on the configured calibration method.
    ///
    /// These scores are used as input to the Isotonic Calibrator or other calibration models.
    /// The specific derivation of the score depends on the `calibration_method` set in the configuration.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features as a `Matrix`.
    /// * `parallel` - Whether to parallelize the prediction across samples.
    pub fn get_calibration_scores(&self, data: &Matrix<f64>, parallel: bool) -> Vec<f64> {
        match self.cfg.calibration_method {
            CalibrationMethod::Conformal => self.predict_proba(data, parallel, false),
            CalibrationMethod::WeightVariance => {
                let fold_weights = self.predict_fold_weights(data, parallel);
                fold_weights
                    .iter()
                    .map(|row| self.compute_score_weight_variance(row))
                    .collect()
            }
            CalibrationMethod::MinMax => {
                let fold_weights = self.predict_fold_weights(data, parallel);
                fold_weights.iter().map(|row| self.compute_score_min_max(row)).collect()
            }
            CalibrationMethod::GRP => {
                let fold_weights = self.predict_fold_weights(data, parallel);
                let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
                fold_weights
                    .iter()
                    .map(|row| self.compute_score_grp(row, &stat_q))
                    .collect()
            }
        }
    }

    /// Calculate calibration scores for the given data in columnar format.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features as a `ColumnarMatrix`.
    /// * `parallel` - Whether to parallelize the prediction.
    pub fn get_calibration_scores_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<f64> {
        match self.cfg.calibration_method {
            CalibrationMethod::Conformal => self.predict_proba_columnar(data, parallel, false),
            CalibrationMethod::WeightVariance => {
                let fold_weights = self.predict_fold_weights_columnar(data, parallel);
                fold_weights
                    .iter()
                    .map(|row| self.compute_score_weight_variance(row))
                    .collect()
            }
            CalibrationMethod::MinMax => {
                let fold_weights = self.predict_fold_weights_columnar(data, parallel);
                fold_weights.iter().map(|row| self.compute_score_min_max(row)).collect()
            }
            CalibrationMethod::GRP => {
                let fold_weights = self.predict_fold_weights_columnar(data, parallel);
                let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
                fold_weights
                    .iter()
                    .map(|row| self.compute_score_grp(row, &stat_q))
                    .collect()
            }
        }
    }

    /// Internal method to predict fold weights for each sample.
    ///
    /// For each sample, it retrieves the weights assigned by each tree in the ensemble
    /// across 5 virtual folds (derived from the internally stored node statistics).
    ///
    /// Returns a vector of arrays `[f64; 5]` for each sample, where each element is the sum
    /// of the corresponding fold weight across all trees, plus the base score.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features.
    /// * `parallel` - Whether to parallelize calculation across samples.
    pub fn predict_fold_weights(&self, data: &Matrix<f64>, parallel: bool) -> Vec<[f64; 5]> {
        let n_samples = data.rows;
        let mut results = vec![[self.base_score; 5]; n_samples];

        for tree in &self.trees {
            let tree_weights = tree.predict_weights(data, parallel, &self.cfg.missing);
            for (i, row_weights) in tree_weights.iter().enumerate() {
                let mut sorted_weights = *row_weights;
                sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for k in 0..5 {
                    results[i][k] += sorted_weights[k] as f64;
                }
            }
        }
        // Sanitize any non-finite values (defensive: replace NaN/inf with base_score)
        for row in results.iter_mut() {
            for v in row.iter_mut() {
                if !v.is_finite() {
                    *v = self.base_score;
                }
            }
        }
        results
    }

    /// Internal method to predict fold weights for each sample using columnar data.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features in columnar format.
    /// * `parallel` - Whether to parallelize calculation.
    pub fn predict_fold_weights_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> Vec<[f64; 5]> {
        let n_samples = data.index.len();
        let mut results = vec![[self.base_score; 5]; n_samples];

        for tree in &self.trees {
            let tree_weights = tree.predict_weights_columnar(data, parallel, &self.cfg.missing);
            for (i, row_weights) in tree_weights.iter().enumerate() {
                let mut sorted_weights = *row_weights;
                sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for k in 0..5 {
                    results[i][k] += sorted_weights[k] as f64;
                }
            }
        }
        // Sanitize any non-finite values (defensive: replace NaN/inf with base_score)
        for row in results.iter_mut() {
            for v in row.iter_mut() {
                if !v.is_finite() {
                    *v = self.base_score;
                }
            }
        }
        results
    }

    /// Calibrate the booster using a selected non-conformal method.
    ///
    /// This method performs calibration for the booster,
    /// calculating scaling factors or residual distributions based on the provided calibration data.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - A tuple of (features, targets, alphas) representing the dedicated calibration set.
    pub fn calibrate(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats && !matches!(method, CalibrationMethod::Conformal) {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        if matches!(self.cfg.objective, Objective::LogLoss) {
            return self.calibrate_classification(method, data_cal);
        }
        self.cfg.calibration_method = method;
        match method {
            CalibrationMethod::MinMax => self.calibrate_min_max(data_cal),
            CalibrationMethod::GRP => self.calibrate_grp(data_cal),
            CalibrationMethod::WeightVariance => self.calibrate_weight_variance(data_cal),
            CalibrationMethod::Conformal => Ok(()),
        }
    }

    /// Calibrate the booster on columnar data using a selected non-conformal method.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - A tuple of (features, targets, alphas) representing the dedicated calibration set.
    pub fn calibrate_columnar(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats && !matches!(method, CalibrationMethod::Conformal) {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        if matches!(self.cfg.objective, Objective::LogLoss) {
            return self.calibrate_classification_columnar(method, data_cal);
        }
        self.cfg.calibration_method = method;
        match method {
            CalibrationMethod::MinMax => self.calibrate_min_max_columnar(data_cal),
            CalibrationMethod::GRP => self.calibrate_grp_columnar(data_cal),
            CalibrationMethod::WeightVariance => self.calibrate_weight_variance_columnar(data_cal),
            CalibrationMethod::Conformal => Ok(()),
        }
    }

    /// Predict intervals for the given data using the fitted calibration models.
    ///
    /// Returns a `HashMap` where keys are the `alpha` values (as strings) and values are
    /// vectors of intervals `[lower, upper]` for each sample.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features.
    /// * `parallel` - Whether to parallelize prediction.
    pub fn predict_intervals(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        if !self.cal_models.is_empty() {
            return self.predict_intervals_conformal(data, parallel);
        }
        match self.cfg.calibration_method {
            CalibrationMethod::MinMax => self.predict_intervals_min_max(data, parallel),
            CalibrationMethod::GRP => self.predict_intervals_grp(data, parallel),
            CalibrationMethod::WeightVariance => self.predict_intervals_weight_variance(data, parallel),
            CalibrationMethod::Conformal => self.predict_intervals_conformal(data, parallel),
        }
    }

    /// Predict intervals for the given columnar data using the fitted calibration models.
    ///
    /// # Arguments
    ///
    /// * `data` - The input features in columnar format.
    /// * `parallel` - Whether to parallelize prediction.
    pub fn predict_intervals_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        if !self.cal_models.is_empty() {
            return self.predict_intervals_conformal_columnar(data, parallel);
        }
        match self.cfg.calibration_method {
            CalibrationMethod::MinMax => self.predict_intervals_min_max_columnar(data, parallel),
            CalibrationMethod::GRP => self.predict_intervals_grp_columnar(data, parallel),
            CalibrationMethod::WeightVariance => self.predict_intervals_weight_variance_columnar(data, parallel),
            CalibrationMethod::Conformal => self.predict_intervals_conformal_columnar(data, parallel),
        }
    }

    /// Performs Generalized Residual Prediction (GRP) interpolation.
    ///
    /// Given a probability `p`, it interpolates the corresponding value from the
    /// `vals` array (representing quantiles of fold predictions) using `stat_q` as the reference quantiles.
    ///
    /// # Arguments
    ///
    /// * `p` - The target probability/quantile.
    /// * `vals` - The 5 values representing the 0, 0.25, 0.5, 0.75, and 1.0 quantiles of fold predictions.
    /// * `stat_q` - The reference quantiles `[0.0, 0.25, 0.5, 0.75, 1.0]`.
    pub(crate) fn grp_interp(&self, p: f64, vals: &[f64; 5], stat_q: &[f64; 5]) -> f64 {
        if p <= 0.0 {
            let slope = (vals[1] - vals[0]) / (stat_q[1] - stat_q[0]);
            vals[0] + slope * p
        } else if p >= 1.0 {
            let slope = (vals[4] - vals[3]) / (stat_q[4] - stat_q[3]);
            vals[4] + slope * (p - 1.0)
        } else {
            let mut val = 0.0;
            for k in 0..4 {
                if p >= stat_q[k] && p <= stat_q[k + 1] {
                    let delta = stat_q[k + 1] - stat_q[k];
                    let frac = if delta > 1e-12 { (p - stat_q[k]) / delta } else { 0.5 };
                    val = vals[k] + frac * (vals[k + 1] - vals[k]);
                    break;
                }
            }
            val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;
    use crate::booster::config::CalibrationMethod;
    use std::error::Error;

    fn read_data(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut data = Vec::new();
        let mut y = Vec::new();
        for result in rdr.records() {
            let record = result?;
            for i in 0..8 {
                data.push(record[i].parse()?);
            }
            y.push(record[8].parse()?);
        }
        Ok((data, y))
    }

    #[test]
    fn test_base_calibration_methods() -> Result<(), Box<dyn Error>> {
        let (data_train, y_train) = read_data("resources/cal_housing_train.csv")?;
        let rows_full = y_train.len();
        let limit = 200.min(rows_full);
        let mut data_train_sub = Vec::new();
        for c in 0..8 {
            let col_start = c * rows_full;
            data_train_sub.extend_from_slice(&data_train[col_start..col_start + limit]);
        }
        let y_train_sub = y_train[0..limit].to_vec();
        let matrix_train = Matrix::new(&data_train_sub, y_train_sub.len(), 8);

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(5)
            .set_budget(0.1)
            .set_iteration_limit(Some(2))
            .set_save_node_stats(true);

        model.fit(&matrix_train, &y_train_sub, None, None)?;

        let alpha = vec![0.1];
        let data_cal = (&matrix_train, y_train_sub.as_slice(), alpha.as_slice());

        // Test MinMax
        model.calibrate(CalibrationMethod::MinMax, data_cal)?;
        let intervals = model.predict_intervals(&matrix_train, false);
        assert!(intervals.contains_key("0.1"));

        // Test GRP
        model.calibrate(CalibrationMethod::GRP, data_cal)?;
        let _ = model.predict_intervals(&matrix_train, false);

        // Test WeightVariance
        model.calibrate(CalibrationMethod::WeightVariance, data_cal)?;
        let _ = model.predict_intervals(&matrix_train, false);

        Ok(())
    }

    #[test]
    fn test_base_calibration_columnar() -> Result<(), Box<dyn Error>> {
        let (data_train, y_train) = read_data("resources/cal_housing_train.csv")?;
        let rows_full = y_train.len();
        let limit = 200.min(rows_full);
        let mut data_train_sub = Vec::new();
        for c in 0..8 {
            let col_start = c * rows_full;
            data_train_sub.extend_from_slice(&data_train[col_start..col_start + limit]);
        }
        let y_train_sub = y_train[0..limit].to_vec();
        let matrix_train = Matrix::new(&data_train_sub, y_train_sub.len(), 8);

        let columns_train: Vec<&[f64]> = (0..8).map(|j| matrix_train.get_col(j)).collect();
        let col_matrix_train = crate::ColumnarMatrix::new(columns_train, None, y_train_sub.len());

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(5)
            .set_budget(0.1)
            .set_iteration_limit(Some(2))
            .set_save_node_stats(true);

        model.fit_columnar(&col_matrix_train, &y_train_sub, None, None)?;

        let alpha = vec![0.1];
        let data_cal = (&col_matrix_train, y_train_sub.as_slice(), alpha.as_slice());

        // Test Columnar Calibrate
        model.calibrate_columnar(CalibrationMethod::MinMax, data_cal)?;
        let intervals = model.predict_intervals_columnar(&col_matrix_train, false);
        assert!(intervals.contains_key("0.1"));

        // Test Columnar Scores
        let scores = model.get_calibration_scores_columnar(&col_matrix_train, false);
        assert_eq!(scores.len(), y_train_sub.len());

        Ok(())
    }
}
