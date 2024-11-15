//! An example using the `cover types` dataset

// cargo run --release --example cover_types 1.0

// cargo build --release --example cover_types
// hyperfine --runs 3 ./target/release/examples/cover_types
// hyperfine --runs 3 .\target\release\examples\cover_types 1.0
// hyperfine --runs 3 'cargo run --release --example cover_types 1.0'

// cargo flamegraph --example cover_types

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

pub fn multiclass_log_loss(y_true: &[f64], y_pred: &[Vec<f64>]) -> f64 {
    let mut losses = vec![0.0; y_true.len()];
    let eps = 1e-11;
    for (i, y_p) in y_pred.iter().enumerate() {
        let y_p_exp = y_p.iter().map(|e| e.exp()).collect::<Vec<f64>>();
        let y_p_exp_sum = y_p_exp.iter().sum::<f64>();
        let probabilities = y_p_exp.iter().map(|e| e / y_p_exp_sum).collect::<Vec<f64>>();
        let cls_idx = (y_true[i] - 1.0) as usize;
        let p = f64::max(eps, f64::min(1.0 - eps, probabilities[cls_idx]));
        losses[i] = -1.0 * p.ln();
    }
    losses.iter().sum::<f64>() / losses.len() as f64
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

    let mut features_and_target = features.clone();
    features_and_target.push("Cover_Type");

    let features_and_target_arc1 = features_and_target
        .iter()
        .map(|s| String::from(s.to_owned()))
        .collect::<Vec<String>>()
        .into();

    let features_and_target_arc2 = features_and_target
        .iter()
        .map(|s| String::from(s.to_owned()))
        .collect::<Vec<String>>()
        .into();

    let df_train = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(features_and_target_arc1))
        .try_into_reader_with_file_path(Some("resources/cover_types_train.csv".into()))?
        .finish()
        .unwrap();

    let df_test = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(features_and_target_arc2))
        .try_into_reader_with_file_path(Some("resources/cover_types_test.csv".into()))?
        .finish()
        .unwrap();

    // Get data in column major format...
    let id_vars_train: Vec<&str> = Vec::new();
    let mdf_train = df_train.unpivot(&features, &id_vars_train)?;
    let id_vars_test: Vec<&str> = Vec::new();
    let mdf_test = df_test.unpivot(&features, &id_vars_test)?;

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
            .column("Cover_Type")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    let y_test = Vec::from_iter(
        df_test
            .column("Cover_Type")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // Create Matrix from ndarray.
    let matrix_train = Matrix::new(&data_train, y_train.len(), 54);
    let matrix_test = Matrix::new(&data_test, y_test.len(), 54);

    let mut raw_train_array = vec![vec![0.0; 7]; y_train.len()];
    let mut raw_test_array = vec![vec![0.0; 7]; y_test.len()];
    for i in 1..8 {
        println!();

        let mut model = PerpetualBooster::default().set_objective(Objective::LogLoss);

        let y_tr: Vec<f64> = y_train
            .iter()
            .map(|y| if (*y as i32) == i { 1.0 } else { 0.0 })
            .collect();

        model.fit(
            &matrix_train,
            &y_tr,
            *budget,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        println!("Completed fitting model number: {}", i);

        let trees = model.get_prediction_trees();
        println!("n_rounds: {:?}", trees.len());

        let n_leaves: usize = trees.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
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
