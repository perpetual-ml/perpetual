//! Iris Flower Classification
//! ==========================
//! Classic machine-learning example: predict whether an Iris flower is
//! *setosa* (label 1) vs *non-setosa* (label 0) using the four petal /
//! sepal measurements.  The dataset is downloaded at runtime from the
//! UCI Machine-Learning Repository.
//!
//! ```bash
//! cargo run --release --example iris
//! ```

use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;

const IRIS_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

/// Accuracy = correct / total
fn accuracy(y_true: &[f64], y_prob: &[f64], threshold: f64) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_prob)
        .filter(|&(&t, &p)| (p >= threshold) as u8 as f64 == t)
        .count();
    correct as f64 / y_true.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Download the Iris dataset (150 rows, 4 features + class label)
    // ------------------------------------------------------------------
    println!("Downloading Iris dataset from UCI …");
    let body = reqwest::blocking::get(IRIS_URL)?.text()?;

    let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); 4];
    let mut y = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }

        // Features: sepal_length, sepal_width, petal_length, petal_width
        for (col, field) in fields[..4].iter().enumerate() {
            data_columns[col].push(field.parse::<f64>()?);
        }

        // Binary target: 1.0 = Iris-setosa, 0.0 = everything else
        let label = if fields[4] == "Iris-setosa" { 1.0 } else { 0.0 };
        y.push(label);
    }

    let n_rows = y.len();
    let n_features = 4;
    println!("Loaded {n_rows} samples with {n_features} features.");

    // ------------------------------------------------------------------
    // 2. Train / test split (first 120 train, last 30 test)
    // ------------------------------------------------------------------
    let split = 120;
    let train_cols: Vec<Vec<f64>> = data_columns.iter().map(|c| c[..split].to_vec()).collect();
    let test_cols: Vec<Vec<f64>> = data_columns.iter().map(|c| c[split..].to_vec()).collect();
    let y_train = &y[..split];
    let y_test = &y[split..];

    let flat_train: Vec<f64> = train_cols.into_iter().flatten().collect();
    let flat_test: Vec<f64> = test_cols.into_iter().flatten().collect();

    let matrix_train = Matrix::new(&flat_train, split, n_features);
    let matrix_test = Matrix::new(&flat_test, n_rows - split, n_features);

    // ------------------------------------------------------------------
    // 3. Fit a PerpetualBooster for binary classification
    // ------------------------------------------------------------------
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(0.3);

    model.fit(&matrix_train, y_train, None, None)?;

    let n_trees = model.get_prediction_trees().len();
    println!("Trained model with {n_trees} trees.");

    // ------------------------------------------------------------------
    // 4. Evaluate
    // ------------------------------------------------------------------
    let proba_train = model.predict_proba(&matrix_train, true, false);
    let proba_test = model.predict_proba(&matrix_test, true, false);

    println!("Train accuracy: {:.2}%", accuracy(y_train, &proba_train, 0.5) * 100.0);
    println!("Test  accuracy: {:.2}%", accuracy(y_test, &proba_test, 0.5) * 100.0);
    println!(
        "First 10 test probabilities: {:?}",
        &proba_test[..10.min(proba_test.len())]
    );

    Ok(())
}
