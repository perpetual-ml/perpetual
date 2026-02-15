//! Goodreads Book-Rating Regression
//! =================================
//! Predict the average rating of Goodreads "best-of" books using
//! features such as year, page count, and star-rating breakdowns.
//! Uses the bundled `resources/goodreads.csv` dataset.
//!
//! ```bash
//! cargo run --release --example goodreads
//! ```

use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn rmse(y: &[f64], yhat: &[f64]) -> f64 {
    let mse: f64 = y.iter().zip(yhat).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64;
    mse.sqrt()
}

fn mae(y: &[f64], yhat: &[f64]) -> f64 {
    y.iter().zip(yhat).map(|(a, b)| (a - b).abs()).sum::<f64>() / y.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Load goodreads.csv
    //    Columns: rank, category, year, avg_rating, pages, published,
    //             publisher, 5stars, 4stars, 3stars, 2stars, 1stars, ratings
    // ------------------------------------------------------------------
    let feature_cols = [
        "year", "pages", "5stars", "4stars", "3stars", "2stars", "1stars", "ratings",
    ];
    let target_col = "avg_rating";

    let file = File::open("resources/goodreads.csv")?;
    let reader = BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

    let headers = rdr.headers()?.clone();
    let feat_idx: Vec<usize> = feature_cols
        .iter()
        .map(|&n| headers.iter().position(|h| h == n).unwrap())
        .collect();
    let target_idx = headers.iter().position(|h| h == target_col).unwrap();

    let n_features = feature_cols.len();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y = Vec::new();

    for rec in rdr.records() {
        let rec = rec?;
        let target_val = rec[target_idx].parse::<f64>().unwrap_or(f64::NAN);
        if target_val.is_nan() {
            continue; // skip rows without a valid target
        }
        y.push(target_val);
        for (i, &idx) in feat_idx.iter().enumerate() {
            let val = rec[idx].parse::<f64>().unwrap_or(f64::NAN);
            cols[i].push(val);
        }
    }

    let n_rows = y.len();
    println!("Goodreads dataset: {n_rows} books, {n_features} features.");

    // ------------------------------------------------------------------
    // 2. 80/20 split
    // ------------------------------------------------------------------
    let split = (n_rows as f64 * 0.8) as usize;

    let train_cols: Vec<Vec<f64>> = cols.iter().map(|c| c[..split].to_vec()).collect();
    let test_cols: Vec<Vec<f64>> = cols.iter().map(|c| c[split..].to_vec()).collect();
    let y_train = &y[..split];
    let y_test = &y[split..];

    let flat_train: Vec<f64> = train_cols.into_iter().flatten().collect();
    let flat_test: Vec<f64> = test_cols.into_iter().flatten().collect();

    let matrix_train = Matrix::new(&flat_train, split, n_features);
    let matrix_test = Matrix::new(&flat_test, n_rows - split, n_features);

    // ------------------------------------------------------------------
    // 3. Train with SquaredLoss
    // ------------------------------------------------------------------
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(0.4);

    model.fit(&matrix_train, y_train, None, None)?;
    println!("Trained {} trees.", model.get_prediction_trees().len());

    // ------------------------------------------------------------------
    // 4. Evaluate
    // ------------------------------------------------------------------
    let pred_train = model.predict(&matrix_train, true);
    let pred_test = model.predict(&matrix_test, true);

    println!(
        "\nTrain  RMSE: {:.4}  MAE: {:.4}",
        rmse(y_train, &pred_train),
        mae(y_train, &pred_train)
    );
    println!(
        "Test   RMSE: {:.4}  MAE: {:.4}",
        rmse(y_test, &pred_test),
        mae(y_test, &pred_test)
    );

    // Show a few predictions
    println!("\nSample predictions vs actuals:");
    for i in 0..5.min(y_test.len()) {
        println!(
            "  Book {:>3}: predicted {:.2}, actual {:.2}",
            i + 1,
            pred_test[i],
            y_test[i]
        );
    }

    Ok(())
}
