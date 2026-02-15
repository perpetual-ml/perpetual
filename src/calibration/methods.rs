use crate::booster::config::CalibrationMethod;
use crate::errors::PerpetualError;
use crate::objective::Objective;
use crate::utils::percentiles;
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

    fn compute_score_weight_variance(&self, log_odds: &[f64; 5]) -> f64 {
        let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        let std_p = (fold_probs.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
        let sigma = std_p.max(1e-6);
        mean_p / sigma
    }

    fn compute_score_min_max(&self, log_odds: &[f64; 5]) -> f64 {
        let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let diff = (max_p - min_p).max(1e-6);
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        mean_p / diff
    }

    fn compute_score_grp(&self, log_odds: &[f64; 5], stat_q: &[f64; 5]) -> f64 {
        let mut fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let fold_probs_arr: [f64; 5] = fold_probs.clone().try_into().unwrap();
        let p_low = self.grp_interp(0.0, &fold_probs_arr, stat_q);
        let p_high = self.grp_interp(1.0, &fold_probs_arr, stat_q);
        let spread = (p_high - p_low).max(1e-6);
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        mean_p / spread
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
            CalibrationMethod::MinMax => {
                // ... existing implementation ...
                let (x_cal, y_cal, alpha_vec) = data_cal;
                let fold_weights = self.predict_fold_weights_columnar(x_cal, true);
                let mut p_rels = Vec::with_capacity(y_cal.len());
                for (i, row) in fold_weights.iter().enumerate() {
                    let s_min = row.iter().copied().fold(f64::INFINITY, f64::min);
                    let s_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let denom = (s_max - s_min).max(1e-12);
                    let mut p_rel = (y_cal[i] - s_min) / denom;
                    if !p_rel.is_finite() {
                        p_rel = 0.0;
                    }
                    p_rels.push(p_rel);
                }
                for &alpha in alpha_vec {
                    let n = p_rels.len() as f64;
                    let low_q = if n > 0.0 {
                        ((alpha / 2.0) - 1.0 / n).max(0.0)
                    } else {
                        (alpha / 2.0).max(0.0)
                    };
                    let high_q = if n > 0.0 {
                        ((1.0 - alpha / 2.0) + 1.0 / n).min(1.0)
                    } else {
                        ((1.0 - alpha / 2.0) + 1.0 / (p_rels.len() as f64)).min(1.0)
                    };
                    let weights_ones = vec![1.0; p_rels.len()];
                    let percs = percentiles(&p_rels, &weights_ones, &[low_q, high_q]);
                    self.cal_params.insert(alpha.to_string(), percs);
                }
                Ok(())
            }
            CalibrationMethod::GRP => {
                // ... existing implementation ...
                let (x_cal, y_cal, alpha_vec) = data_cal;
                let fold_weights = self.predict_fold_weights_columnar(x_cal, true);
                let mut positions = Vec::with_capacity(y_cal.len());
                let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (i, row) in fold_weights.iter().enumerate() {
                    let mut vals = *row;
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let y = y_cal[i];
                    let mut pos = if y <= vals[0] {
                        let delta = vals[1] - vals[0];
                        (y - vals[0]) / (delta.max(1e-12) / (stat_q[1] - stat_q[0]))
                    } else if y >= vals[4] {
                        let delta = vals[4] - vals[3];
                        1.0 + (y - vals[4]) / (delta.max(1e-12) / (stat_q[4] - stat_q[3]))
                    } else {
                        let mut p = 0.0;
                        for k in 0..4 {
                            if y >= vals[k] && y <= vals[k + 1] {
                                let frac = (y - vals[k]) / (vals[k + 1] - vals[k]).max(1e-12);
                                p = stat_q[k] + frac * (stat_q[k + 1] - stat_q[k]);
                                break;
                            }
                        }
                        p
                    };
                    if !pos.is_finite() {
                        pos = 0.0;
                    }
                    positions.push(pos);
                }
                for &alpha in alpha_vec {
                    let n = positions.len() as f64;
                    let low_q = if n > 0.0 {
                        ((alpha / 2.0) - 1.0 / n).max(0.0)
                    } else {
                        (alpha / 2.0).max(0.0)
                    };
                    let high_q = if n > 0.0 {
                        ((1.0 - alpha / 2.0) + 1.0 / n).min(1.0)
                    } else {
                        ((1.0 - alpha / 2.0) + 1.0 / (positions.len() as f64)).min(1.0)
                    };
                    let weights_ones = vec![1.0; positions.len()];
                    let percs = percentiles(&positions, &weights_ones, &[low_q, high_q]);
                    self.cal_params.insert(alpha.to_string(), percs);
                }
                Ok(())
            }
            CalibrationMethod::WeightVariance => self.calibrate_weight_variance_columnar(data_cal),
            CalibrationMethod::Conformal => Ok(()),
        }
    }

    pub fn calibrate_min_max(&mut self, data_cal: (&Matrix<f64>, &[f64], &[f64])) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights(x_cal, true);

        // S_min, S_max for each sample
        let mut p_rels = Vec::with_capacity(y_cal.len());
        for (i, row) in fold_weights.iter().enumerate() {
            let s_min = row.iter().copied().fold(f64::INFINITY, f64::min);
            let s_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom = (s_max - s_min).max(1e-12);
            let mut p_rel = (y_cal[i] - s_min) / denom;
            if !p_rel.is_finite() {
                p_rel = 0.0;
            }
            p_rels.push(p_rel);
        }

        for &alpha in alpha_vec {
            let n = p_rels.len() as f64;
            let low_q = if n > 0.0 {
                ((alpha / 2.0) - 1.0 / n).max(0.0)
            } else {
                (alpha / 2.0).max(0.0)
            };
            let high_q = if n > 0.0 {
                ((1.0 - alpha / 2.0) + 1.0 / n).min(1.0)
            } else {
                ((1.0 - alpha / 2.0) + 1.0 / (p_rels.len() as f64)).min(1.0)
            };
            let weights_ones = vec![1.0; p_rels.len()];
            let percs = percentiles(&p_rels, &weights_ones, &[low_q, high_q]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    pub fn calibrate_grp(&mut self, data_cal: (&Matrix<f64>, &[f64], &[f64])) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights(x_cal, true);

        let mut positions = Vec::with_capacity(y_cal.len());
        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];

        for (i, row) in fold_weights.iter().enumerate() {
            let mut vals = *row;
            // Defensive: replace any non-finite values with base_score
            for v in vals.iter_mut() {
                if !v.is_finite() {
                    *v = self.base_score;
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let y = y_cal[i];

            let mut pos = if y <= vals[0] {
                let delta = vals[1] - vals[0];
                (y - vals[0]) / (delta.max(1e-12) / (stat_q[1] - stat_q[0]))
            } else if y >= vals[4] {
                let delta = vals[4] - vals[3];
                1.0 + (y - vals[4]) / (delta.max(1e-12) / (stat_q[4] - stat_q[3]))
            } else {
                let mut p = 0.0;
                for k in 0..4 {
                    if y >= vals[k] && y <= vals[k + 1] {
                        let frac = (y - vals[k]) / (vals[k + 1] - vals[k]).max(1e-12);
                        p = stat_q[k] + frac * (stat_q[k + 1] - stat_q[k]);
                        break;
                    }
                }
                p
            };
            if !pos.is_finite() {
                pos = 0.0;
            }
            positions.push(pos);
        }

        for &alpha in alpha_vec {
            // Filter out any non-finite positions defensively
            let mut filt_pos = Vec::with_capacity(positions.len());
            let mut filt_w = Vec::with_capacity(positions.len());
            for p in &positions {
                if p.is_finite() {
                    filt_pos.push(*p);
                    filt_w.push(1.0);
                }
            }
            if filt_pos.is_empty() {
                // fallback: use zeros
                filt_pos.push(0.0);
                filt_w.push(1.0);
            }
            let n = filt_pos.len() as f64;
            let low_q_val = if n > 0.0 {
                ((alpha / 2.0) - 1.0 / n).max(0.0)
            } else {
                (alpha / 2.0).max(0.0)
            };
            let high_q = if n > 0.0 {
                ((1.0 - alpha / 2.0) + 1.0 / n).min(1.0)
            } else {
                ((1.0 - alpha / 2.0) + 1.0 / (filt_pos.len() as f64)).min(1.0)
            };
            let percs = percentiles(&filt_pos, &filt_w, &[low_q_val, high_q]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    pub fn calibrate_weight_variance(
        &mut self,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let n_samples = x_cal.rows;
        let preds = self.predict(x_cal, true);
        let mut uncertainties = vec![0.0; n_samples];
        for tree in &self.trees {
            let tree_weights = tree.predict_weights(x_cal, true, &self.cfg.missing);
            for (i, row) in tree_weights.iter().enumerate() {
                let mean = row.iter().sum::<f32>() / 5.0;
                let var = row.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / 5.0;
                uncertainties[i] += var.sqrt() as f64;
            }
        }

        let mut scaling_factors = Vec::with_capacity(y_cal.len());
        for i in 0..y_cal.len() {
            let abs_res = (y_cal[i] - preds[i]).abs();
            let unc = uncertainties[i].max(1e-12);
            scaling_factors.push(abs_res / unc);
        }

        for &alpha in alpha_vec {
            // Use the same finite-sample correction as MinMax/GRP (1.0/n)
            let n = scaling_factors.len() as f64;
            let q = ((1.0 - alpha) + 1.0 / n).min(1.0);
            // Filter out any non-finite scaling factors defensively
            let mut filt_sf = Vec::with_capacity(scaling_factors.len());
            let mut filt_w = Vec::with_capacity(scaling_factors.len());
            for s in &scaling_factors {
                if s.is_finite() {
                    filt_sf.push(*s);
                    filt_w.push(1.0);
                }
            }
            if filt_sf.is_empty() {
                filt_sf.push(0.0);
                filt_w.push(1.0);
            }
            let perc = percentiles(&filt_sf, &filt_w, &[q]);
            self.cal_params.insert(alpha.to_string(), perc);
        }
        Ok(())
    }

    pub fn calibrate_weight_variance_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let n_samples = y_cal.len();
        let preds = self.predict_columnar(x_cal, true);
        let mut uncertainties = vec![0.0; n_samples];
        for tree in &self.trees {
            let tree_weights = tree.predict_weights_columnar(x_cal, true, &self.cfg.missing);
            for (i, row) in tree_weights.iter().enumerate() {
                let mean = row.iter().sum::<f32>() / 5.0;
                let var = row.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / 5.0;
                uncertainties[i] += var.sqrt() as f64;
            }
        }

        let mut scaling_factors = Vec::with_capacity(y_cal.len());
        for i in 0..y_cal.len() {
            let abs_res = (y_cal[i] - preds[i]).abs();
            let unc = uncertainties[i].max(1e-12);
            scaling_factors.push(abs_res / unc);
        }

        for &alpha in alpha_vec {
            // Same correction for the columnar variant.
            let n = scaling_factors.len() as f64;
            let q = ((1.0 - alpha) + 1.0 / n).min(1.0);
            let weights_ones = vec![1.0; scaling_factors.len()];
            let perc = percentiles(&scaling_factors, &weights_ones, &[q]);
            self.cal_params.insert(alpha.to_string(), perc);
        }
        Ok(())
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

    fn predict_intervals_min_max(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let fold_weights = self.predict_fold_weights(data, parallel);
        for (alpha_str, params) in &self.cal_params {
            let p_rel_lower = params[0];
            let p_rel_upper = params[1];
            let mut sample_intervals = Vec::with_capacity(data.rows);
            for row in &fold_weights {
                let s_min = row.iter().copied().fold(f64::INFINITY, f64::min);
                let s_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let diff = s_max - s_min;
                let lower = s_min + p_rel_lower * diff;
                let upper = s_min + p_rel_upper * diff;
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
    }

    fn predict_intervals_min_max_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let fold_weights = self.predict_fold_weights_columnar(data, parallel);
        let n_samples = data.index.len();
        for (alpha_str, params) in &self.cal_params {
            let p_rel_lower = params[0];
            let p_rel_upper = params[1];
            let mut sample_intervals = Vec::with_capacity(n_samples);
            for row in &fold_weights {
                let s_min = row.iter().copied().fold(f64::INFINITY, f64::min);
                let s_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let diff = s_max - s_min;
                let lower = s_min + p_rel_lower * diff;
                let upper = s_min + p_rel_upper * diff;
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
    }

    fn predict_intervals_grp(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let fold_weights = self.predict_fold_weights(data, parallel);
        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
        for (alpha_str, params) in &self.cal_params {
            let p_low = params[0];
            let p_high = params[1];
            let mut sample_intervals = Vec::with_capacity(data.rows);
            for row in &fold_weights {
                let mut vals = *row;
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lower = self.grp_interp(p_low, &vals, &stat_q);
                let upper = self.grp_interp(p_high, &vals, &stat_q);
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
    }

    fn predict_intervals_grp_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let fold_weights = self.predict_fold_weights_columnar(data, parallel);
        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
        let n_samples = data.index.len();
        for (alpha_str, params) in &self.cal_params {
            let p_low = params[0];
            let p_high = params[1];
            let mut sample_intervals = Vec::with_capacity(n_samples);
            for row in &fold_weights {
                let mut vals = *row;
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lower = self.grp_interp(p_low, &vals, &stat_q);
                let upper = self.grp_interp(p_high, &vals, &stat_q);
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
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

    fn predict_intervals_weight_variance(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let preds = self.predict(data, parallel);
        let mut uncertainties = vec![0.0; data.rows];
        for tree in &self.trees {
            let tree_weights = tree.predict_weights(data, parallel, &self.cfg.missing);
            for (i, row) in tree_weights.iter().enumerate() {
                let mean = row.iter().sum::<f32>() / 5.0;
                let var = row.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / 5.0;
                uncertainties[i] += var.sqrt() as f64;
            }
        }
        for (alpha_str, params) in &self.cal_params {
            let q_factor = params[0];
            let mut sample_intervals = Vec::with_capacity(data.rows);
            for i in 0..data.rows {
                let lower = preds[i] - q_factor * uncertainties[i];
                let upper = preds[i] + q_factor * uncertainties[i];
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
    }

    fn predict_intervals_weight_variance_columnar(
        &self,
        data: &ColumnarMatrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        let mut intervals = HashMap::new();
        let preds = self.predict_columnar(data, parallel);
        let n_samples = data.index.len();
        let mut uncertainties = vec![0.0; n_samples];
        for tree in &self.trees {
            let tree_weights = tree.predict_weights_columnar(data, parallel, &self.cfg.missing);
            for (i, row) in tree_weights.iter().enumerate() {
                let mean = row.iter().sum::<f32>() / 5.0;
                let var = row.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / 5.0;
                uncertainties[i] += var.sqrt() as f64;
            }
        }
        for (alpha_str, params) in &self.cal_params {
            let q_factor = params[0];
            let mut sample_intervals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let lower = preds[i] - q_factor * uncertainties[i];
                let upper = preds[i] + q_factor * uncertainties[i];
                sample_intervals.push(vec![lower, upper]);
            }
            intervals.insert(alpha_str.clone(), sample_intervals);
        }
        intervals
    }
}
