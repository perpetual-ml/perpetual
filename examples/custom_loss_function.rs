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

use perpetual::{Matrix, UnivariateBooster};
use perpetual::metrics::Metric;
use perpetual::objective_functions::ObjectiveFunction;

//-----------------//
// Define function //
//-----------------//
#[derive(Clone)]
struct CustomSquaredLoss;
impl ObjectiveFunction for CustomSquaredLoss {

    fn hessian_is_constant(&self) -> bool {
        true
    }

    fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> Vec<f32> {
        y.iter()
            .zip(yhat)
            .enumerate()
            .map(|(idx, (y_i, yhat_i))| {
                let diff = yhat_i - y_i;
                let l = diff * diff;
                match sample_weight {
                    Some(w) => (l * w[idx]) as f32,
                    None => l as f32,
                }
            })
            .collect()
    }

    fn calc_grad_hess(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>,) -> (Vec<f32>, Option<Vec<f32>>) {
        let grad: Vec<f32> = y
            .iter()
            .zip(yhat)
            .enumerate()
            .map(|(idx, (y_i, yhat_i))| {
                let g = 2.0 * (yhat_i - y_i);
                match sample_weight {
                    Some(w) => (g * w[idx]) as f32,
                    None => g as f32,
                }
            })
            .collect();
        let hess = vec![2.0_f32; y.len()];
        (grad, Some(hess))
    }

    fn calc_init(&self, y: &[f64], sample_weight: Option<&[f64]>) -> f64 {
        match sample_weight {
            Some(w) => {
                let sw: f64 = w.iter().sum();
                y.iter().enumerate().map(|(i, y_i)| y_i * w[i]).sum::<f64>() / sw
            }
            None => y.iter().sum::<f64>() / y.len() as f64,
        }
    }

    fn default_metric(&self) -> Metric {
        Metric::RootMeanSquaredError
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
    let mut booster = UnivariateBooster::default();
    booster.cfg = booster.cfg.clone().with_custom_objective(CustomSquaredLoss);
    booster.cfg.max_bin = 10;
    booster.cfg.budget = 0.1;

    //-------------------//
    // 3. Fit and report //
    //-------------------//
    booster.fit(&matrix, &y, None)?;

    let n_trees = booster.get_prediction_trees().len();
    println!("Model trained with {n_trees} trees (budget = {})", booster.cfg.budget);

    let preds = booster.predict(&matrix, true);
    println!("First 5 predictions: {:?}", &preds[..5]);

    Ok(())
    
}
