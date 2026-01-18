use crate::objective_functions::objective::Objective;
use crate::{errors::PerpetualError, utils::percentiles, Matrix, PerpetualBooster};
use std::collections::HashMap;

pub type CalData<'a> = (Matrix<'a, f64>, &'a [f64], &'a [f64]); // (x_flat_data, rows, cols), y, alpha

impl PerpetualBooster {
    /// Calibrate models to get prediction intervals
    /// * `alpha` - Alpha list to train calibration models for
    pub fn calibrate(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        data_cal: CalData,
    ) -> Result<(), PerpetualError> {
        let (x_cal, y_cal, alpha) = data_cal;

        for alpha_ in alpha {
            let lower_quantile = Some(alpha_ / 2.0);
            let mut model_lower = PerpetualBooster::default().set_objective(Objective::QuantileLoss {
                quantile: lower_quantile,
            });
            model_lower.fit(data, y, sample_weight, group)?;

            let upper_quantile = Some(1.0 - alpha_ / 2.0);
            let mut model_upper = PerpetualBooster::default().set_objective(Objective::QuantileLoss {
                quantile: upper_quantile,
            });

            model_upper.fit(data, y, sample_weight, group)?;

            let y_cal_pred_lower = model_lower.predict(&x_cal, true);
            let y_cal_pred_upper = model_upper.predict(&x_cal, true);
            let mut scores: Vec<f64> = Vec::with_capacity(y_cal.len());
            for i in 0..y_cal.len() {
                scores.push(f64::max(y_cal_pred_lower[i] - y_cal[i], y_cal[i] - y_cal_pred_upper[i]));
            }
            let perc = (1.0 - *alpha_) * (1.0 + 1.0 * (1.0 / (scores.len() as f64)));
            let score = percentiles(&scores, &vec![1.0; scores.len()], &[perc])[0];
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

// Unit-tests
#[cfg(test)]
mod tests {

    use crate::objective_functions::objective::Objective;
    use crate::Matrix;
    use crate::PerpetualBooster;
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;

    fn read_data(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ];
        let target_name = "MedHouseVal";

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

        let headers = csv_reader.headers()?.clone();
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|&name| headers.iter().position(|h| h == name).unwrap())
            .collect();
        let target_index = headers.iter().position(|h| h == target_name).unwrap();

        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
        let mut y = Vec::new();

        for result in csv_reader.records() {
            let record = result?;

            // Parse target
            let target_str = &record[target_index];
            let target_val = if target_str.is_empty() {
                f64::NAN
            } else {
                target_str.parse::<f64>().unwrap_or(f64::NAN)
            };
            y.push(target_val);

            // Parse features
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                let val = if val_str.is_empty() {
                    f64::NAN
                } else {
                    val_str.parse::<f64>().unwrap_or(f64::NAN)
                };
                data_columns[i].push(val);
            }
        }

        let data: Vec<f64> = data_columns.into_iter().flatten().collect();
        Ok((data, y))
    }

    #[test]
    fn test_cqr() -> Result<(), Box<dyn Error>> {
        let (data_train, y_train) = read_data("resources/cal_housing_train.csv")?;
        let (data_test, y_test) = read_data("resources/cal_housing_test.csv")?;

        let _n_train_subset = 1000.min(y_train.len());
        let _n_test_subset = 500.min(y_test.len());

        let rows_full = y_train.len();
        let limit_train = 1000.min(rows_full);
        let mut data_train_sub = Vec::new();
        // Extract 8 columns
        for c in 0..8 {
            let col_start = c * rows_full;
            data_train_sub.extend_from_slice(&data_train[col_start..col_start + limit_train]);
        }
        let y_train_sub = y_train[0..limit_train].to_vec();

        let rows_test_full = y_test.len();
        let limit_test = 500.min(rows_test_full);
        let mut data_test_sub = Vec::new();
        for c in 0..8 {
            let col_start = c * rows_test_full;
            data_test_sub.extend_from_slice(&data_test[col_start..col_start + limit_test]);
        }
        let y_test_sub = y_test[0..limit_test].to_vec();

        // Create Matrix from ndarray.
        let matrix_train = Matrix::new(&data_train_sub, y_train_sub.len(), 8);
        let matrix_test = Matrix::new(&data_test_sub, y_test_sub.len(), 8);

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(5)
            .set_budget(0.1)
            .set_iteration_limit(Some(5))
            .set_memory_limit(Some(0.0001));

        model.fit(&matrix_train, &y_train_sub, None, None)?;

        let alpha = vec![0.2];
        let data_cal = (matrix_test, y_test_sub.as_slice(), alpha.as_slice());

        model.calibrate(&matrix_train, &y_train_sub, None, None, data_cal)?;

        let matrix_test_eval = Matrix::new(&data_test_sub, y_test_sub.len(), 8);
        let _intervals = model.predict_intervals(&matrix_test_eval, true);

        Ok(())
    }
}
