//! Abalone Age Prediction – Regression with Model Save / Load
//! ===========================================================
//! Predict the age of abalone (number of rings + 1.5) from eight
//! physical measurements.  After training, the model is **saved to a
//! JSON file**, then **loaded back** before computing predictions — a
//! round-trip demonstration of model serialization.
//!
//! The dataset is downloaded from the UCI ML Repository at runtime.
//!
//! ```bash
//! cargo run --release --example abalone
//! ```

use perpetual::booster::config::BoosterIO;
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;

const ABALONE_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";

const FEATURE_NAMES: [&str; 8] = [
    "Sex(encoded)",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
];

fn rmse(y: &[f64], yhat: &[f64]) -> f64 {
    let mse: f64 = y.iter().zip(yhat).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64;
    mse.sqrt()
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Download & parse
    //    Format:  Sex(M/F/I), 7 continuous features, Rings (target)
    //    No header row.
    // ------------------------------------------------------------------
    println!("Downloading Abalone dataset from UCI …");
    let body = reqwest::blocking::get(ABALONE_URL)?.text()?;

    let n_features = 8;
    let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9 {
            continue;
        }

        // Encode Sex: M=0, F=1, I=2 (infant)
        let sex = match fields[0] {
            "M" => 0.0,
            "F" => 1.0,
            "I" => 2.0,
            _ => f64::NAN,
        };
        data_columns[0].push(sex);

        // 7 continuous features (columns 1-7)
        for col in 1..n_features {
            let val = fields[col].parse::<f64>().unwrap_or(f64::NAN);
            data_columns[col].push(val);
        }

        // Target: Rings (age ≈ rings + 1.5)
        let rings = fields[8].parse::<f64>().unwrap_or(f64::NAN);
        y.push(rings);
    }

    let n_rows = y.len();
    println!("Loaded {n_rows} samples, {n_features} features ({FEATURE_NAMES:?}).");

    // ------------------------------------------------------------------
    // 2. 80/20 train-test split
    // ------------------------------------------------------------------
    let split = (n_rows as f64 * 0.8) as usize;

    let train_cols: Vec<Vec<f64>> = data_columns.iter().map(|c| c[..split].to_vec()).collect();
    let test_cols: Vec<Vec<f64>> = data_columns.iter().map(|c| c[split..].to_vec()).collect();
    let y_train = &y[..split];
    let y_test = &y[split..];

    let flat_train: Vec<f64> = train_cols.into_iter().flatten().collect();
    let flat_test: Vec<f64> = test_cols.into_iter().flatten().collect();

    let matrix_train = Matrix::new(&flat_train, split, n_features);
    let matrix_test = Matrix::new(&flat_test, n_rows - split, n_features);

    // ------------------------------------------------------------------
    // 3. Train
    // ------------------------------------------------------------------
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(0.5);

    model.fit(&matrix_train, y_train, None, None)?;

    let n_trees = model.get_prediction_trees().len();
    println!("Trained {n_trees} trees.");

    let pred_test = model.predict(&matrix_test, true);
    println!("RMSE test (before save): {:.4}", rmse(y_test, &pred_test));

    // ------------------------------------------------------------------
    // 4. Save model to JSON
    // ------------------------------------------------------------------
    let model_path = "abalone_model.json";
    model.save_booster(model_path)?;
    println!("Model saved to {model_path}");

    // ------------------------------------------------------------------
    // 5. Load model and predict again
    // ------------------------------------------------------------------
    let loaded = PerpetualBooster::load_booster(model_path)?;
    let pred_loaded = loaded.predict(&matrix_test, true);
    println!("RMSE test (after load):  {:.4}", rmse(y_test, &pred_loaded));

    // Quick sanity check: predictions must be identical
    assert_eq!(pred_test, pred_loaded, "Predictions differ after round-trip!");
    println!("Round-trip save/load verified ✓");

    // Clean up the temp file
    std::fs::remove_file(model_path)?;

    Ok(())
}
