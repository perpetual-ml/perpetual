//! Breast Cancer Wisconsin – Binary Classification with Feature Importance
//! =======================================================================
//! Predict whether a breast tumour is malignant (1) or benign (0) using
//! 30 numeric features derived from digitised images of fine-needle
//! aspirates.  After fitting the model we print **gain-based feature
//! importance** so we can see which measurements matter most.
//!
//! The dataset is downloaded at runtime from the UCI ML Repository.
//!
//! ```bash
//! cargo run --release --example breast_cancer
//! ```

use perpetual::booster::config::ImportanceMethod;
use perpetual::objective_functions::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;

const DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";

/// Feature names for the 30 real-valued input features of the WDBC dataset.
const FEATURE_NAMES: [&str; 30] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dim_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dim_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dim_worst",
];

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
    // 1. Download & parse
    //    Format: id, diagnosis(M/B), 30 features  (no header row)
    // ------------------------------------------------------------------
    println!("Downloading Breast Cancer Wisconsin (WDBC) dataset …");
    let body = reqwest::blocking::get(DATA_URL)?.text()?;

    let n_features = 30;
    let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 32 {
            continue;
        }

        // Column 1 is "M" (malignant → 1) or "B" (benign → 0)
        let label = if fields[1] == "M" { 1.0 } else { 0.0 };
        y.push(label);

        // Columns 2..32 are the 30 numeric features
        for col in 0..n_features {
            let val = fields[col + 2].parse::<f64>().unwrap_or(f64::NAN);
            data_columns[col].push(val);
        }
    }

    let n_rows = y.len();
    println!("Loaded {n_rows} samples, {n_features} features.");

    // ------------------------------------------------------------------
    // 2. 80/20 split
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
        .set_objective(Objective::LogLoss)
        .set_budget(0.4);

    model.fit(&matrix_train, y_train, None, None)?;

    let n_trees = model.get_prediction_trees().len();
    println!("Trained {n_trees} trees.");

    // ------------------------------------------------------------------
    // 4. Evaluate
    // ------------------------------------------------------------------
    let proba_test = model.predict_proba(&matrix_test, true);
    println!("Test accuracy: {:.2}%", accuracy(y_test, &proba_test, 0.5) * 100.0);

    // ------------------------------------------------------------------
    // 5. Feature importance (gain)
    // ------------------------------------------------------------------
    let importance = model.calculate_feature_importance(ImportanceMethod::Gain, true);

    // Sort descending by importance value
    let mut imp_vec: Vec<(usize, f32)> = importance.into_iter().collect();
    imp_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop-10 features by gain:");
    for (rank, (idx, score)) in imp_vec.iter().take(10).enumerate() {
        let name = if *idx < FEATURE_NAMES.len() {
            FEATURE_NAMES[*idx]
        } else {
            "unknown"
        };
        println!("  {:>2}. {:<24} {:.4}", rank + 1, name, score);
    }

    Ok(())
}
