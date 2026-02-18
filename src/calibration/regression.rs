use crate::errors::PerpetualError;
use crate::utils::percentiles;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};
use std::collections::HashMap;

impl PerpetualBooster {
    pub(crate) fn compute_score_weight_variance(&self, log_odds: &[f64; 5]) -> f64 {
        let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        let std_p = (fold_probs.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / 5.0).sqrt();
        let sigma = std_p.max(1e-6);
        mean_p / sigma
    }

    pub(crate) fn compute_score_min_max(&self, log_odds: &[f64; 5]) -> f64 {
        let fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let min_p = fold_probs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_p = fold_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let diff = (max_p - min_p).max(1e-6);
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        mean_p / diff
    }

    pub(crate) fn compute_score_grp(&self, log_odds: &[f64; 5], stat_q: &[f64; 5]) -> f64 {
        let mut fold_probs: Vec<f64> = log_odds.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        fold_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let fold_probs_arr: [f64; 5] = fold_probs.clone().try_into().unwrap();
        let p_low = self.grp_interp(0.0, &fold_probs_arr, stat_q);
        let p_high = self.grp_interp(1.0, &fold_probs_arr, stat_q);
        let spread = (p_high - p_low).max(1e-6);
        let mean_p = fold_probs.iter().sum::<f64>() / 5.0;
        mean_p / spread
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

    pub fn calibrate_min_max_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
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

    pub fn calibrate_grp_columnar(
        &mut self,
        data_cal: (&ColumnarMatrix<f64>, &[f64], &[f64]),
    ) -> Result<(), PerpetualError> {
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

    pub(crate) fn predict_intervals_min_max(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
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

    pub(crate) fn predict_intervals_min_max_columnar(
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

    pub(crate) fn predict_intervals_grp(&self, data: &Matrix<f64>, parallel: bool) -> HashMap<String, Vec<Vec<f64>>> {
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

    pub(crate) fn predict_intervals_grp_columnar(
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

    pub(crate) fn predict_intervals_weight_variance(
        &self,
        data: &Matrix<f64>,
        parallel: bool,
    ) -> HashMap<String, Vec<Vec<f64>>> {
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

    pub(crate) fn predict_intervals_weight_variance_columnar(
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
