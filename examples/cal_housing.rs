//! An example using the `california housing` dataset

// cargo run --release --example cal_housing 0.5

// cargo build --release --example cal_housing
// hyperfine --runs 3 ./target/release/examples/cal_housing
// hyperfine --runs 3 .\target\release\examples\cal_housing
// hyperfine --runs 11 'cargo run --release --example cal_housing 0.5'

// cargo flamegraph --example cal_housing

use perpetual::{Matrix, PerpetualBooster, objective_functions::Objective};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::time::SystemTime;

pub fn mse(y_test: &[f64], y_pred: &[f64]) -> f32 {
    let mut error = 0.0;
    for i in 0..y_test.len() {
        error += (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i]);
    }
    let e = error / y_test.len() as f64;
    e as f32
}

fn read_data(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];
    let target_name = "MedHouseVal";

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

    let headers = csv_reader.headers()?.clone();
    let feature_indices: Vec<usize> = feature_names
        .iter()
        .map(|&name| headers.iter().position(|h| h == name).unwrap())
        .collect();
    let target_index = headers.iter().position(|h| h == target_name).unwrap();

    let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
    let mut y = Vec::new();

    for result in csv_reader.records() {
        let record = result?;

        // Parse target
        let target_str = &record[target_index];
        let target_val = if target_str.is_empty() {
            f64::NAN
        } else {
            target_str.parse::<f64>().unwrap_or(f64::NAN)
        };
        y.push(target_val);

        // Parse features
        for (i, &idx) in feature_indices.iter().enumerate() {
            let val_str = &record[idx];
            let val = if val_str.is_empty() {
                f64::NAN
            } else {
                val_str.parse::<f64>().unwrap_or(f64::NAN)
            };
            data_columns[i].push(val);
        }
    }

    let data: Vec<f64> = data_columns.into_iter().flatten().collect();
    Ok((data, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let budget = &args[1].parse::<f32>().unwrap_or(0.5);

    let (data_train, y_train) = read_data("resources/cal_housing_train.csv")?;
    let (data_test, y_test) = read_data("resources/cal_housing_test.csv")?;

    // Create Matrix from ndarray.
    let matrix_train = Matrix::new(&data_train, y_train.len(), 8);
    let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(*budget);

    let now = SystemTime::now();
    model.fit(&matrix_train, &y_train, None, None)?;
    println!("now.elapsed: {:?}", now.elapsed().unwrap().as_secs_f32());

    let trees = model.get_prediction_trees();
    println!("n_rounds: {:?}", trees.len());

    let n_leaves: usize = trees.iter().map(|t| t.nodes.len().div_ceil(2)).sum();
    println!("n_leaves: {:?}", n_leaves);

    let y_pred = model.predict(&matrix_train, true);
    let error = mse(&y_train, &y_pred);
    println!("mse_train: {:?}", error);

    let y_pred = model.predict(&matrix_test, true);
    let error = mse(&y_test, &y_pred);
    println!("mse_test: {:?}", error);

    println!("tree:");
    for t in trees {
        println!("{}", t);
    }

    Ok(())
}
