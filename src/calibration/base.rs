use crate::booster::config::CalibrationMethod;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};
use std::collections::HashMap;

impl PerpetualBooster {
    /// Calculate calibration scores for the given data based on the configured calibration method.
    /// These scores are used as input to the Isotonic Calibrator.
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
    /// Returns a vector of [f64; 5] for each sample, where each element is the sum
    /// of the corresponding fold weight across all trees, plus the base score.
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
