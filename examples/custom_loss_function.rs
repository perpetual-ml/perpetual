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
use std::sync::Arc;

use polars::io::SerReader;
use polars::prelude::{CsvReadOptions, DataType};

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

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(Arc::new(all_names.clone())))
        .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
        .finish()?;

    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.unpivot(feature_names.to_vec(), &id_vars)?;

    let data: Vec<f64> = mdf
        .select_at_idx(1)
        .expect("Invalid column")
        .f64()? // Returns Result<Float64Chunked>
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let y: Vec<f64> = df
        .column("MedHouseVal")?
        .cast(&DataType::Float64)?
        .f64()? // Returns Result<Float64Chunked>
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let matrix = Matrix::new(&data, y.len(), 8);

    //--------------------------------------//
    // 2. Build booster w/ custom objective //
    //--------------------------------------//
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
