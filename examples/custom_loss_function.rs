//! Example: Training a Perpetual Boosting Machine with custom objective functions
//! ------------------------------------------------------------------------------
//! This example shows how to supply your own loss to the gradient-boosting
//! machine at runtime. We define a minimal **squared-error** objective that
//! implements the `ObjectiveFunction` trait, inject it through
//! `BoosterConfig::with_custom_objective`, fit on the California-housing test
//! set, and print a few diagnostics.
//!
//! ```bash
//! # From the crate root of *perpetual*
//! cargo run --example custom_loss_function
//! ```
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use perpetual::metrics::evaluation::Metric;
use perpetual::objective_functions::Objective;
use perpetual::{Matrix, PerpetualBooster};

//-----------------//
// Define function //
//-----------------//
#[derive(Clone)]
struct CustomSquaredLoss;
impl perpetual::objective_functions::ObjectiveFunction for CustomSquaredLoss {
    #[inline]
    fn loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> Vec<f32> {
        match sample_weight {
            Some(sample_weight) => y
                .iter()
                .zip(yhat)
                .zip(sample_weight)
                .map(|((y_, yhat_), w_)| {
                    let s = *y_ - *yhat_;
                    (s * s * *w_) as f32
                })
                .collect(),
            None => y
                .iter()
                .zip(yhat)
                .map(|(y_, yhat_)| {
                    let s = *y_ - *yhat_;
                    (s * s) as f32
                })
                .collect(),
        }
    }

    #[inline]
    fn gradient(
        &self,
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        _group: Option<&[u64]>,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        match sample_weight {
            Some(sample_weight) => {
                let (g, h) = y
                    .iter()
                    .zip(yhat)
                    .zip(sample_weight)
                    .map(|((y_, yhat_), w_)| (((yhat_ - *y_) * *w_) as f32, *w_ as f32))
                    .unzip();
                (g, Some(h))
            }
            None => (
                y.iter().zip(yhat).map(|(y_, yhat_)| (yhat_ - *y_) as f32).collect(),
                None,
            ),
        }
    }

    fn initial_value(&self, y: &[f64], sample_weight: Option<&[f64]>, _group: Option<&[u64]>) -> f64 {
        if let Some(w) = sample_weight {
            let sum_w: f64 = w.iter().sum();
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sum_w
        } else {
            y.iter().sum::<f64>() / y.len() as f64
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredLogError
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    //----------------------//
    // 1. Build dataset     //
    //----------------------//

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

    let file = File::open("resources/cal_housing_test.csv")?;
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

    let matrix = Matrix::new(&data, y.len(), 8);

    //--------------------------------------//
    // 2. Build booster w/ custom objective //
    //--------------------------------------//

    // Explicitly annotate type for CustomSquaredLoss to match Objective enum variant if needed,
    // but Objective::Custom takes Arc<dyn ObjectiveFunction>.
    // The previous code was: Objective::Custom(Arc::new(CustomSquaredLoss))
    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::Custom(Arc::new(CustomSquaredLoss)))
        .set_max_bin(10)
        .set_budget(0.1);

    //-------------------//
    // 3. Fit and report //
    //-------------------//
    booster.fit(&matrix, &y, None, None)?;

    let n_trees = booster.get_prediction_trees().len();
    println!("Model trained with {n_trees} trees (budget = {})", booster.cfg.budget);

    let preds = booster.predict(&matrix, true);
    println!("First 5 predictions: {:?}", &preds[..5]);

    Ok(())
}
