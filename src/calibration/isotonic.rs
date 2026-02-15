use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IsotonicCalibrator {
    /// Thresholds (input probabilities)
    pub thresholds: Vec<f64>,
    /// Calibrated values (output probabilities)
    pub values: Vec<f64>,
    /// Min value in calibration set
    pub min_val: f64,
    /// Max value in calibration set
    pub max_val: f64,
}

impl IsotonicCalibrator {
    pub fn new(y_pred: &[f64], y_true: &[f64]) -> Self {
        if y_pred.is_empty() {
            return Self::default();
        }

        // Pair up (prediction, truth) and sort by prediction
        let mut data: Vec<(f64, f64)> = y_pred.iter().zip(y_true.iter()).map(|(&p, &t)| (p, t)).collect();
        // Sort by prediction primarily, then by truth
        data.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // PAVA algorithm
        // We use the "pool adjacent violators" algorithm to find the monotonic function
        // that minimizes squared error.

        // Stack stores indices of blocks: (sum_y, count, value, sum_x)
        let mut blocks_w_x: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(y_pred.len());

        // Re-run with sum_x tracking
        for (pred, target) in data {
            let mut current_sum_y = target;
            let mut current_sum_x = pred;
            let mut current_count = 1.0;

            // Merge down
            while let Some((prev_sum_y, prev_count, prev_val, prev_sum_x)) = blocks_w_x.last_mut() {
                let curr_val = current_sum_y / current_count;
                if *prev_val > curr_val {
                    // Merge
                    current_sum_y += *prev_sum_y;
                    current_count += *prev_count;
                    current_sum_x += *prev_sum_x;
                    blocks_w_x.pop();
                } else {
                    break;
                }
            }
            blocks_w_x.push((
                current_sum_y,
                current_count,
                current_sum_y / current_count,
                current_sum_x,
            ));
        }

        let mut thresholds = Vec::with_capacity(blocks_w_x.len());
        let mut values = Vec::with_capacity(blocks_w_x.len());

        for (sum_y, count, _, sum_x) in blocks_w_x {
            thresholds.push(sum_x / count);
            values.push(sum_y / count);
        }

        // Handle edge cases: extrapolating to 0 and 1 boundaries?
        // We will just clamp during prediction to the min/max threshold.

        let min_val = *values.first().unwrap_or(&0.0);
        let max_val = *values.last().unwrap_or(&1.0);

        IsotonicCalibrator {
            thresholds,
            values,
            min_val,
            max_val,
        }
    }

    pub fn transform(&self, y_pred: &[f64]) -> Vec<f64> {
        if self.thresholds.is_empty() {
            return y_pred.to_vec();
        }

        let mut calibrated = Vec::with_capacity(y_pred.len());
        for &p in y_pred {
            // Linear interpolation
            if p <= self.thresholds[0] {
                calibrated.push(self.values[0]);
            } else if p >= *self.thresholds.last().unwrap() {
                calibrated.push(*self.values.last().unwrap());
            } else {
                // Binary search for the interval
                let idx = match self.thresholds.binary_search_by(|t| t.partial_cmp(&p).unwrap()) {
                    Ok(i) => i,
                    Err(i) => i - 1,
                };

                let x0 = self.thresholds[idx];
                let x1 = self.thresholds[idx + 1];
                let y0 = self.values[idx];
                let y1 = self.values[idx + 1];

                let slope = (y1 - y0) / (x1 - x0);
                calibrated.push(y0 + slope * (p - x0));
            }
        }
        calibrated
    }
}
