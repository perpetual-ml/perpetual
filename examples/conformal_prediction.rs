//! Conformal Prediction Intervals
//! ===============================
//! Train a model on California Housing, then use the built-in
//! **Conformal Quantile Regression (CQR)** calibration to produce
//! prediction intervals that come with valid coverage guarantees.
//!
//! ```bash
//! cargo run --release --example conformal_prediction
//! ```

use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

const FEATURE_NAMES: [&str; 8] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
];

fn read_cal_housing(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

    let headers = rdr.headers()?.clone();
    let feat_idx: Vec<usize> = FEATURE_NAMES
        .iter()
        .map(|&n| headers.iter().position(|h| h == n).unwrap())
        .collect();
    let target_idx = headers.iter().position(|h| h == "MedHouseVal").unwrap();

    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 8];
    let mut y = Vec::new();

    for rec in rdr.records() {
        let rec = rec?;
        y.push(rec[target_idx].parse::<f64>().unwrap_or(f64::NAN));
        for (i, &idx) in feat_idx.iter().enumerate() {
            cols[i].push(rec[idx].parse::<f64>().unwrap_or(f64::NAN));
        }
    }

    let flat: Vec<f64> = cols.into_iter().flatten().collect();
    Ok((flat, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Load the data
    // ------------------------------------------------------------------
    let (data_train_all, y_train_all) = read_cal_housing("resources/cal_housing_train.csv")?;
    let (data_test, y_test) = read_cal_housing("resources/cal_housing_test.csv")?;

    let n_all = y_train_all.len();
    let n_test = y_test.len();
    let n_features = 8;

    // Split training into fit (70 %) and calibration (30 %)
    let n_fit = (n_all as f64 * 0.7) as usize;
    let n_cal = n_all - n_fit;

    // Column-major split: each column has n_all values
    let mut fit_cols: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut cal_cols: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    for col in 0..n_features {
        let start = col * n_all;
        fit_cols[col] = data_train_all[start..start + n_fit].to_vec();
        cal_cols[col] = data_train_all[start + n_fit..start + n_all].to_vec();
    }

    let flat_fit: Vec<f64> = fit_cols.into_iter().flatten().collect();
    let flat_cal: Vec<f64> = cal_cols.into_iter().flatten().collect();

    let y_fit = &y_train_all[..n_fit];
    let y_cal = &y_train_all[n_fit..];

    // Calibration quantiles
    let quantiles: Vec<f64> = vec![0.05, 0.95];

    let matrix_fit = Matrix::new(&flat_fit, n_fit, n_features);
    let matrix_cal = Matrix::new(&flat_cal, n_cal, n_features);
    let matrix_test = Matrix::new(&data_test, n_test, n_features);

    // ------------------------------------------------------------------
    // 2. Fit the main model
    // ------------------------------------------------------------------
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(0.5);

    model.fit(&matrix_fit, y_fit, None, None)?;
    println!(
        "Trained {} trees on {n_fit} samples.",
        model.get_prediction_trees().len()
    );

    // 3. Calibrate with conformal quantile regression
    // ------------------------------------------------------------------
    let quantiles_slice = quantiles.as_slice();
    let cal_data = (&matrix_cal, y_cal, quantiles_slice);
    model.calibrate_conformal(&matrix_fit, y_fit, None, None, cal_data)?;
    println!("Calibrated conformal model on {n_cal} held-out samples.");

    // ------------------------------------------------------------------
    // 4. Predict intervals on the test set
    // ------------------------------------------------------------------
    let intervals = model.predict_intervals(&matrix_test, true);

    // intervals is HashMap<String, Vec<Vec<f64>>>, e.g. "0.05" -> [[lo], ...], "0.95" -> [[hi], ...]
    println!("\nInterval keys: {:?}", intervals.keys().collect::<Vec<_>>());

    // Show first 10 samples
    let point_preds = model.predict(&matrix_test, true);

    println!(
        "\n{:<6} {:>10} {:>10} {:>10} {:>10}",
        "Row", "Lower", "Pred", "Upper", "Actual"
    );
    println!("{}", "-".repeat(50));

    let lo_key = intervals.keys().find(|k| k.contains("0.05")).cloned();
    let hi_key = intervals.keys().find(|k| k.contains("0.95")).cloned();

    if let (Some(lo_key), Some(hi_key)) = (lo_key, hi_key) {
        let lo_vals = &intervals[&lo_key];
        let hi_vals = &intervals[&hi_key];

        let mut covered = 0usize;
        for i in 0..n_test {
            let lo = lo_vals[i][0];
            let hi = hi_vals[i][0];
            if y_test[i] >= lo && y_test[i] <= hi {
                covered += 1;
            }
            if i < 10 {
                println!(
                    "{:<6} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                    i, lo, point_preds[i], hi, y_test[i]
                );
            }
        }
        let coverage = covered as f64 / n_test as f64 * 100.0;
        println!(
            "\nEmpirical coverage (90 % target): {:.1}% ({covered}/{n_test})",
            coverage
        );
    } else {
        println!("Could not find interval keys for 0.05 / 0.95");
    }

    Ok(())
}
