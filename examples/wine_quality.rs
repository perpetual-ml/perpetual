//! Wine Quality Regression
//! =======================
//! Predict the quality score (0–10) of red wines from eleven
//! physicochemical measurements.  The dataset is downloaded from the
//! UCI Machine-Learning Repository at runtime.
//!
//! ```bash
//! cargo run --release --example wine_quality
//! ```

use perpetual::objective_functions::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;

const WINE_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv";

fn rmse(y: &[f64], yhat: &[f64]) -> f64 {
    let mse: f64 = y.iter().zip(yhat).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64;
    mse.sqrt()
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Fetch the dataset (semicolon-delimited, 1599 rows, 12 columns)
    // ------------------------------------------------------------------
    println!("Downloading Wine-Quality (red) dataset from UCI …");
    let body = reqwest::blocking::get(WINE_URL)?.text()?;

    let mut lines = body.lines();
    let header = lines.next().expect("empty file");
    let col_names: Vec<&str> = header.split(';').map(|s| s.trim_matches('"')).collect();
    let n_features = col_names.len() - 1; // last column is "quality"
    println!("Columns: {col_names:?}");

    let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(';').collect();
        if fields.len() != col_names.len() {
            continue;
        }
        for (col, val) in fields[..n_features].iter().enumerate() {
            data_columns[col].push(val.parse::<f64>()?);
        }
        y.push(fields[n_features].parse::<f64>()?);
    }

    let n_rows = y.len();
    println!("Loaded {n_rows} samples, {n_features} features.");

    // ------------------------------------------------------------------
    // 2. 80 / 20 train-test split
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

    let trees = model.get_prediction_trees();
    println!("Trees: {}, budget used: {}", trees.len(), model.cfg.budget);

    // ------------------------------------------------------------------
    // 4. Evaluate
    // ------------------------------------------------------------------
    let pred_train = model.predict(&matrix_train, true);
    let pred_test = model.predict(&matrix_test, true);

    println!("RMSE train: {:.4}", rmse(y_train, &pred_train));
    println!("RMSE test:  {:.4}", rmse(y_test, &pred_test));
    println!(
        "Sample predictions (test): {:?}",
        pred_test[..5].iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
    );
    println!("Sample actuals     (test): {:?}", &y_test[..5]);

    Ok(())
}
