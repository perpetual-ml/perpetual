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
    use polars::io::SerReader;
    use polars::prelude::{CsvReadOptions, DataType};
    use std::error::Error;
    use std::sync::Arc;

    #[test]
    fn test_cqr() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_train = Arc::new(all_names.clone());
        let column_names_test = Arc::new(all_names.clone());

        let df_train = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_train))
            .try_into_reader_with_file_path(Some("resources/cal_housing_train.csv".into()))?
            .finish()
            .unwrap();

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...
        let id_vars_train: Vec<&str> = Vec::new();
        let mdf_train = df_train.unpivot(feature_names.clone(), &id_vars_train)?;
        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_train = Vec::from_iter(
            mdf_train
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_train = Vec::from_iter(
            df_train
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_train = Matrix::new(&data_train, y_train.len(), 8);
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_train, &y_train, None, None)?;

        let alpha = vec![0.1];
        let data_cal = (matrix_test, y_test.as_slice(), alpha.as_slice());

        model.calibrate(&matrix_train, &y_train, None, None, data_cal)?;

        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);
        let _intervals = model.predict_intervals(&matrix_test, true);

        Ok(())
    }
}
