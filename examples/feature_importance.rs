//! Feature Importance & SHAP Contributions
//! ========================================
//! Train a model on the bundled California-Housing dataset and then
//! demonstrate **multiple importance methods** (Gain, Weight, Cover) as
//! well as **SHAP-style feature contributions** for individual
//! predictions.
//!
//! ```bash
//! cargo run --release --example feature_importance
//! ```

use perpetual::booster::config::{ContributionsMethod, ImportanceMethod};
use perpetual::objective_functions::Objective;
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

fn print_importance(model: &PerpetualBooster, method: ImportanceMethod, label: &str) {
    let imp = model.calculate_feature_importance(method, true);
    let mut v: Vec<(usize, f32)> = imp.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n--- {label} ---");
    for (idx, score) in &v {
        let name = FEATURE_NAMES.get(*idx).unwrap_or(&"?");
        println!("  {:<12} {:.4}", name, score);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Load data
    // ------------------------------------------------------------------
    let (data_train, y_train) = read_cal_housing("resources/cal_housing_train.csv")?;
    let (data_test, y_test) = read_cal_housing("resources/cal_housing_test.csv")?;

    let n_train = y_train.len();
    let n_test = y_test.len();

    let matrix_train = Matrix::new(&data_train, n_train, 8);
    let matrix_test = Matrix::new(&data_test, n_test, 8);

    // ------------------------------------------------------------------
    // 2. Train
    // ------------------------------------------------------------------
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(0.5);

    model.fit(&matrix_train, &y_train, None, None)?;
    println!(
        "Trained {} trees on California-Housing.",
        model.get_prediction_trees().len()
    );

    // ------------------------------------------------------------------
    // 3. Global feature importance (three methods)
    // ------------------------------------------------------------------
    print_importance(&model, ImportanceMethod::Gain, "Feature Importance: Gain");
    print_importance(
        &model,
        ImportanceMethod::Weight,
        "Feature Importance: Weight (split count)",
    );
    print_importance(&model, ImportanceMethod::Cover, "Feature Importance: Cover");

    // ------------------------------------------------------------------
    // 4. SHAP-style feature contributions for a single sample
    // ------------------------------------------------------------------
    let contribs = model.predict_contributions(&matrix_test, ContributionsMethod::Weight, true);

    // contribs is a flat vec of shape [n_test × (n_features + 1)]
    let stride = FEATURE_NAMES.len() + 1; // 9
    let row = 0; // explain the first test sample
    let offset = row * stride;

    println!("\n--- SHAP-like contributions (Weight method) for test row {row} ---");
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        println!("  {:<12} {:+.4}", name, contribs[offset + i]);
    }
    println!("  {:<12} {:+.4}", "bias", contribs[offset + FEATURE_NAMES.len()]);

    let pred = model.predict(&matrix_test, true);
    println!("\n  Prediction: {:.4}  |  Actual: {:.4}", pred[row], y_test[row]);

    Ok(())
}
