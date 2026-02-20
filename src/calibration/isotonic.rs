use serde::{Deserialize, Serialize};

/// Isotonic Calibrator
///
/// Implements Isotonic Regression using the Pool Adjacent Violators Algorithm (PAVA).
/// It maps raw model outputs to calibrated probabilities by finding a non-decreasing
/// step function that minimizes the squared error relative to the true labels.
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
    /// Creates a new `IsotonicCalibrator` by fitting it to the provided predictions and true labels.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - The raw predictions (e.g., scores or probabilities from the base model).
    /// * `y_true` - The true binary targets (0.0 or 1.0).
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

    /// Transforms raw predictions into calibrated probabilities using the fitted isotonic function.
    ///
    /// It uses linear interpolation between the computed thresholds.
    /// Values outside the calibration range are clamped to the boundaries.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotonic_simple() {
        let y_pred = vec![0.1, 0.2, 0.3, 0.4];
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let calibrator = IsotonicCalibrator::new(&y_pred, &y_true);

        assert_eq!(calibrator.thresholds.len(), 4);
        // Blocks are not merged since y_true is monotonic
        // Thresholds: 0.1, 0.2, 0.3, 0.4
        // Values: 0.0, 0.0, 1.0, 1.0
        assert_eq!(calibrator.thresholds[0], 0.1);
        assert_eq!(calibrator.values[0], 0.0);
        assert_eq!(calibrator.thresholds[1], 0.2);
        assert_eq!(calibrator.values[1], 0.0);
        assert_eq!(calibrator.thresholds[2], 0.3);
        assert_eq!(calibrator.values[2], 1.0);
        assert_eq!(calibrator.thresholds[3], 0.4);
        assert_eq!(calibrator.values[3], 1.0);

        let test_pred = vec![0.1, 0.25, 0.4];
        let transformed = calibrator.transform(&test_pred);
        assert_eq!(transformed[0], 0.0); // <= 0.1
        assert_eq!(transformed[2], 1.0); // >= 0.4
        // 0.25 is between thresholds 0.2 and 0.3
        // y0 = 0.0, y1 = 1.0, x0 = 0.2, x1 = 0.3
        // slope = (1-0)/(0.3-0.2) = 1/0.1 = 10
        // y = 0 + 10 * (0.25 - 0.2) = 10 * 0.05 = 0.5
        assert!((transformed[1] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_isotonic_decreasing() {
        // Non-monotonic data should be pooled
        let y_pred = vec![0.1, 0.2, 0.3];
        let y_true = vec![1.0, 0.0, 1.0];
        let calibrator = IsotonicCalibrator::new(&y_pred, &y_true);
        // Initially: (1,1,1), (0,1,0), (1,1,1)
        // (1,1,1) > (0,1,0) -> Merge -> (1,2,0.5)
        // (0.5, 2) < (1,1,1) -> OK
        // Result: 0.5, 0.5, 1.0 (after merging)
        assert_eq!(calibrator.values.len(), 2);
        assert_eq!(calibrator.values[0], 0.5);
        assert_eq!(calibrator.values[1], 1.0);
    }

    #[test]
    fn test_isotonic_empty() {
        let calibrator = IsotonicCalibrator::new(&[], &[]);
        assert!(calibrator.thresholds.is_empty());
        let transformed = calibrator.transform(&[0.5]);
        assert_eq!(transformed[0], 0.5);
    }
}
