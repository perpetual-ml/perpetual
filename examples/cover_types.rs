//! An example using the `cover types` dataset

// cargo run --release --example cover_types 1.0

// cargo build --release --example cover_types
// hyperfine --runs 3 ./target/release/examples/cover_types
// hyperfine --runs 3 .\target\release\examples\cover_types 1.0
// hyperfine --runs 3 'cargo run --release --example cover_types 1.0'

// cargo flamegraph --example cover_types

//! An example using the `cover types` dataset

// cargo run --release --example cover_types 1.0

// cargo build --release --example cover_types
// hyperfine --runs 3 ./target/release/examples/cover_types
// hyperfine --runs 3 .\target\release\examples\cover_types 1.0
// hyperfine --runs 3 'cargo run --release --example cover_types 1.0'

// cargo flamegraph --example cover_types

use perpetual::{Matrix, PerpetualBooster, objective::Objective};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

pub fn mse(y_test: &[f64], y_pred: &[f64]) -> f32 {
    let mut error = 0.0;
    for i in 0..y_test.len() {
        error += (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i]);
    }
    let e = error / y_test.len() as f64;
    e as f32
}

pub fn multiclass_log_loss(y_true: &[f64], y_pred: &[Vec<f64>]) -> f64 {
    let mut losses = vec![0.0; y_true.len()];
    let eps = 1e-11;
    for (i, y_p) in y_pred.iter().enumerate() {
        let y_p_exp = y_p.iter().map(|e| e.exp()).collect::<Vec<f64>>();
        let y_p_exp_sum = y_p_exp.iter().sum::<f64>();
        let probabilities = y_p_exp.iter().map(|e| e / y_p_exp_sum).collect::<Vec<f64>>();
        let cls_idx = (y_true[i] - 1.0) as usize;
        let p = f64::max(eps, f64::min(1.0 - eps, probabilities[cls_idx]));
        losses[i] = -p.ln();
    }
    losses.iter().sum::<f64>() / losses.len() as f64
}

fn read_data(path: &str, feature_names: &[&str]) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let target_name = "Cover_Type";

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
    let budget = &args[1].parse::<f32>().unwrap_or(1.0);

    let mut features: Vec<&str> = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area_0",
        "Wilderness_Area_1",
        "Wilderness_Area_2",
        "Wilderness_Area_3",
    ]
    .to_vec();

    let soil_types = (0..40).map(|i| format!("{}_{}", "Soil_Type", i)).collect::<Vec<_>>();
    let s_types = soil_types.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    features.extend(s_types);

    let (data_train, y_train) = read_data("resources/cover_types_train.csv", &features)?;
    let (data_test, y_test) = read_data("resources/cover_types_test.csv", &features)?;

    // Create Matrix from ndarray.
    let matrix_train = Matrix::new(&data_train, y_train.len(), 54);
    let matrix_test = Matrix::new(&data_test, y_test.len(), 54);

    let mut raw_train_array = vec![vec![0.0; 7]; y_train.len()];
    let mut raw_test_array = vec![vec![0.0; 7]; y_test.len()];
    for i in 1..8 {
        println!();

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::LogLoss)
            .set_budget(*budget);

        let y_tr: Vec<f64> = y_train
            .iter()
            .map(|y| if (*y as i32) == i { 1.0 } else { 0.0 })
            .collect();

        model.fit(&matrix_train, &y_tr, None, None)?;
        println!("Completed fitting model number: {}", i);

        let trees = model.get_prediction_trees();
        println!("n_rounds: {:?}", trees.len());

        let n_leaves: usize = trees.iter().map(|t| t.nodes.len().div_ceil(2)).sum();
        println!("n_leaves: {:?}", n_leaves);

        let y_pred_train = model.predict(&matrix_train, true);
        let y_pred_test = model.predict(&matrix_test, true);

        raw_train_array
            .iter_mut()
            .enumerate()
            .for_each(|(idx, raw)| raw[(i - 1) as usize] = y_pred_train[idx]);
        raw_test_array
            .iter_mut()
            .enumerate()
            .for_each(|(idx, raw)| raw[(i - 1) as usize] = y_pred_test[idx]);
    }

    let loss_train = multiclass_log_loss(&y_train, &raw_train_array);
    let loss_test = multiclass_log_loss(&y_test, &raw_test_array);

    println!("loss_train: {}", loss_train);
    println!("loss_test: {}", loss_test);

    Ok(())
}
