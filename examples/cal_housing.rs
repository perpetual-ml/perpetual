//! An example using the `california housing` dataset

// cargo run --release --example cal_housing 0.1 0.3 2

// cargo build --release --example cal_housing
// hyperfine --runs 3 ./target/release/examples/cal_housing
// hyperfine --runs 3 .\target\release\examples\cal_housing
// hyperfine --runs 11 'cargo run --release --example cal_housing 0.1 0.3 2'

// cargo flamegraph --example cal_housing

use perpetual::{objective::Objective, Matrix, PerpetualBooster};
use polars::prelude::*;
use std::env;
use std::error::Error;

pub fn mse(y_test: &[f64], y_pred: &[f64]) -> f32 {
    let mut error = 0.0;
    for i in 0..y_test.len() {
        error += (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i]);
    }
    let e = error / y_test.len() as f64;
    e as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let growth_control = &args[1].parse::<f32>().unwrap();
    let budget = &args[2].parse::<f32>().unwrap();
    let es = &args[3].parse::<usize>().unwrap_or(1);

    let _all_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
    ];

    let _feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];

    let df_train = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(Arc::new(_all_names.iter().map(|&s| s.to_string()).collect())))
        .try_into_reader_with_file_path(Some("resources/cal_housing_train.csv".into()))?
        .finish()
        .unwrap();

    let df_test = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(Arc::new(_all_names.iter().map(|&s| s.to_string()).collect())))
        .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
        .finish()
        .unwrap();

    // Get data in column major format...
    let id_vars_train: Vec<&str> = Vec::new();
    let mdf_train = df_train.melt(&id_vars_train, _feature_names)?;
    let id_vars_test: Vec<&str> = Vec::new();
    let mdf_test = df_test.melt(&id_vars_test, _feature_names)?;

    let data_train = Vec::from_iter(
        mdf_train
            .select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    let data_test = Vec::from_iter(
        mdf_test
            .select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    let y_train = Vec::from_iter(
        df_train
            .column("MedHouseVal")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    let y_test = Vec::from_iter(
        df_test
            .column("MedHouseVal")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // Create Matrix from ndarray.
    let matrix_train = Matrix::new(&data_train, y_train.len(), 8);
    let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_growth_control(*growth_control)
        .set_early_stopping_rounds(*es);
    model.fit(&matrix_train, &y_train, None, None, *budget, None, None)?;

    let trees = model.get_prediction_trees();
    println!("n_rounds: {:?}", trees.len());

    let n_leaves: usize = trees.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
    println!("n_leaves: {:?}", n_leaves);

    let y_pred = model.predict(&matrix_train, true, None);
    let error = mse(&y_train, &y_pred);
    println!("mse_train: {:?}", error);

    let y_pred = model.predict(&matrix_test, true, None);
    let error = mse(&y_test, &y_pred);
    println!("mse_test: {:?}", error);

    println!("tree:");
    println!("{}", trees[0]);

    Ok(())
}
