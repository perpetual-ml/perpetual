use crate::booster::config::CalibrationMethod;
use crate::errors::PerpetualError;
use crate::utils::percentiles;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};
use std::collections::HashMap;
use std::convert::TryInto;

impl PerpetualBooster {
    /// Calibrate the booster for classification.
    ///
    /// # Arguments
    /// * `method` - The calibration method.
    /// * `data_cal` - (features, targets, alphas). Targets should be 0 or 1.
    pub fn calibrate_classification(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        // Always fit Isotonic Calibrator for probability calibration
        let (x_cal, y_cal, _) = data_cal;

        // Temporarily set the method to the requested one so get_calibration_scores uses it
        // (Though it is passed as argument, get_calibration_scores reads from self.cfg)
        self.cfg.calibration_method = method;

        // Calculate scores using the method-specific logic
        let scores = self.get_calibration_scores(x_cal, false);
        self.isotonic_calibrator = Some(crate::calibration::isotonic::IsotonicCalibrator::new(&scores, y_cal));

        // self.cfg.calibration_method is already set above, but let's confirm logic flow
        match method {
            CalibrationMethod::Conformal => self.calibrate_classification_conformal(data_cal),
            CalibrationMethod::WeightVariance => self.calibrate_classification_weight_variance(data_cal),
            CalibrationMethod::MinMax => self.calibrate_classification_min_max(data_cal),
            CalibrationMethod::GRP => self.calibrate_classification_grp(data_cal),
        }
    }

    pub fn calibrate_classification_columnar(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        // Always fit Isotonic Calibrator for probability calibration
        let (x_cal, y_cal, _) = data_cal;

        self.cfg.calibration_method = method;

        let scores = self.get_calibration_scores_columnar(x_cal, false);
        self.isotonic_calibrator = Some(crate::calibration::isotonic::IsotonicCalibrator::new(&scores, y_cal));

        match method {
            CalibrationMethod::Conformal => self.calibrate_classification_conformal_columnar(data_cal),
            CalibrationMethod::WeightVariance => self.calibrate_classification_weight_variance_columnar(data_cal),
            CalibrationMethod::MinMax => self.calibrate_classification_min_max_columnar(data_cal),
            CalibrationMethod::GRP => self.calibrate_classification_grp_columnar(data_cal),
        }
    }

    fn calibrate_classification_conformal(
        &mut self,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let probs = self.predict_proba(x_cal, false, false); // Raw probabilities are used for finding score thresholds

        // Non-conformity scores: 1 - p(y)
        let mut scores = Vec::with_capacity(y_cal.len());
        for (i, &y) in y_cal.iter().enumerate() {
            let p = probs[i];
            let score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(score);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            // (1-alpha) quantile with finite sample correction
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    fn calibrate_classification_conformal_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let probs = self.predict_proba_columnar(x_cal, false, false);

        let mut scores = Vec::with_capacity(y_cal.len());
        for (i, &y) in y_cal.iter().enumerate() {
            let p = probs[i];
            let score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(score);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    /// Adaptive conformal classification scaled by standard deviation of fold probs.
    fn calibrate_classification_weight_variance(
        &mut self,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();

            let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
            let std_p = (fold_probs.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
            let sigma = std_p.max(1e-6);

            let y = y_cal[i];
            let p = mean_p;
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / sigma);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    fn calibrate_classification_weight_variance_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights_columnar(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();

            let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
            let std_p = (fold_probs.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
            let sigma = std_p.max(1e-6);

            let y = y_cal[i];
            let p = mean_p;
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / sigma);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    /// Adaptive conformal classification scaled by range of fold probs.
    fn calibrate_classification_min_max(
        &mut self,
        data_cal: (&Matrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();

            let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let diff = (max_p - min_p).max(1e-6);

            let y = y_cal[i];
            let p = fold_probs.iter().sum::<f64>() / 5.0;
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / diff);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    fn calibrate_classification_min_max_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights_columnar(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();

            let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let diff = (max_p - min_p).max(1e-6);

            let y = y_cal[i];
            let p = fold_probs.iter().sum::<f64>() / 5.0;
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / diff);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    /// Adaptive conformal classification scaled by GRP spread (percentiles of fold probs).
    fn calibrate_classification_grp(&mut self, data_cal: (&Matrix<f64>, &[f64], &[f64])) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());
        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            // Convert fold log-odds to probabilities
            let mut fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
            fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate spread using GRP interpolation
            // We use spread between effectively min and max (or high/low quantiles)
            // Here we use the full range estimate from GRP interpolation logic
            let fold_probs_arr: [f64; 5] = fold_probs.clone().try_into().unwrap();
            let p_low = self.grp_interp(0.0, &fold_probs_arr, &stat_q);
            let p_high = self.grp_interp(1.0, &fold_probs_arr, &stat_q);
            let spread = (p_high - p_low).max(1e-6);

            let y = y_cal[i];
            let p = fold_probs.iter().sum::<f64>() / 5.0; // Mean probability
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / spread);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }

    fn calibrate_classification_grp_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha_vec) = data_cal;
        let fold_weights = self.predict_fold_weights_columnar(x_cal, true);
        let mut scores = Vec::with_capacity(y_cal.len());
        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];

        for (i, row) in fold_weights.iter().enumerate() {
            let log_odds: Vec<f64> = row.to_vec();
            let mut fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
            fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let fold_probs_arr: [f64; 5] = fold_probs.clone().try_into().unwrap();
            let p_low = self.grp_interp(0.0, &fold_probs_arr, &stat_q);
            let p_high = self.grp_interp(1.0, &fold_probs_arr, &stat_q);
            let spread = (p_high - p_low).max(1e-6);

            let y = y_cal[i];
            let p = fold_probs.iter().sum::<f64>() / 5.0;
            let raw_score = if y > 0.5 { 1.0 - p } else { p };
            scores.push(raw_score / spread);
        }

        for &alpha in alpha_vec {
            let n = scores.len() as f64;
            let q_prob = ((1.0 - alpha) * (n + 1.0) / n).min(1.0);
            let weights = vec![1.0; scores.len()];
            let percs = percentiles(&scores, &weights, &[q_prob]);
            self.cal_params.insert(alpha.to_string(), percs);
        }
        Ok(())
    }
    /// Returns a map from alpha to a vector of sets (represented as bitmasks or vectors of labels).
    /// For binary, we'll return Vec<Vec<f64>> where each inner vec contains 0, 1, or both.
    pub fn predict_sets(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut results = HashMap::new();
        let fold_weights = self.predict_fold_weights(data, parallel);

        for (alpha_str, params) in &self.cal_params {
            let q = params[0];
            let mut sample_sets = Vec::with_capacity(data.rows);

            for row in &fold_weights {
                let log_odds: Vec<f64> = row.to_vec();
                let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                let p = fold_probs.iter().sum::<f64>() / 5.0;

                let mut set = Vec::new();

                match self.cfg.calibration_method {
                    CalibrationMethod::Conformal => {
                        // score(1) = 1 - p, score(0) = p
                        if 1.0 - p <= q {
                            set.push(1.0);
                        }
                        if p <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::GRP => {
                        // Calculate spread
                        let mut sorted_fold_probs = fold_probs.clone();
                        sorted_fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
                        let sorted_probs_arr: [f64; 5] = sorted_fold_probs.clone().try_into().unwrap();
                        let p_low = self.grp_interp(0.0, &sorted_probs_arr, &stat_q);
                        let p_high = self.grp_interp(1.0, &sorted_probs_arr, &stat_q);
                        let spread = (p_high - p_low).max(1e-6);

                        // Score = (1 - p(y)) / spread
                        if (1.0 - p) / spread <= q {
                            set.push(1.0);
                        }
                        if p / spread <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::WeightVariance => {
                        let mean_p = p;
                        let std_p = (fold_probs.iter().map(|&v| (v - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
                        let sigma = std_p.max(1e-6);
                        if (1.0 - p) / sigma <= q {
                            set.push(1.0);
                        }
                        if p / sigma <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::MinMax => {
                        let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let diff = (max_p - min_p).max(1e-6);
                        if (1.0 - p) / diff <= q {
                            set.push(1.0);
                        }
                        if p / diff <= q {
                            set.push(0.0);
                        }
                    }
                }

                // Ensure at least one label if we want non-empty sets, or allow empty?
                // Standard conformal allows empty sets.
                sample_sets.push(set);
            }
            results.insert(alpha_str.clone(), sample_sets);
        }
        results
    }

    pub fn predict_sets_columnar(&self, data: &ColumnarMatrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
        let mut results = HashMap::new();
        let fold_weights = self.predict_fold_weights_columnar(data, parallel);
        let n_samples = data.rows;

        for (alpha_str, params) in &self.cal_params {
            let q = params[0];
            let mut sample_sets = Vec::with_capacity(n_samples);

            for row in &fold_weights {
                let log_odds: Vec<f64> = row.to_vec();
                let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                let p = fold_probs.iter().sum::<f64>() / 5.0;

                let mut set = Vec::new();

                match self.cfg.calibration_method {
                    CalibrationMethod::Conformal => {
                        if 1.0 - p <= q {
                            set.push(1.0);
                        }
                        if p <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::GRP => {
                        // Calculate spread
                        let mut sorted_fold_probs = fold_probs.clone();
                        sorted_fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let stat_q = [0.0, 0.25, 0.5, 0.75, 1.0];
                        let sorted_probs_arr: [f64; 5] = sorted_fold_probs.clone().try_into().unwrap();
                        let p_low = self.grp_interp(0.0, &sorted_probs_arr, &stat_q);
                        let p_high = self.grp_interp(1.0, &sorted_probs_arr, &stat_q);
                        let spread = (p_high - p_low).max(1e-6);

                        if (1.0 - p) / spread <= q {
                            set.push(1.0);
                        }
                        if p / spread <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::WeightVariance => {
                        let mean_p = p;
                        let std_p = (fold_probs.iter().map(|&v| (v - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
                        let sigma = std_p.max(1e-6);
                        if (1.0 - p) / sigma <= q {
                            set.push(1.0);
                        }
                        if p / sigma <= q {
                            set.push(0.0);
                        }
                    }
                    CalibrationMethod::MinMax => {
                        let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
                        let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let diff = (max_p - min_p).max(1e-6);
                        if (1.0 - p) / diff <= q {
                            set.push(1.0);
                        }
                        if p / diff <= q {
                            set.push(0.0);
                        }
                    }
                }
                sample_sets.push(set);
            }
            results.insert(alpha_str.clone(), sample_sets);
        }
        results
    }
}
