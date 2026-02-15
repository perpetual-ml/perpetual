use crate::errors::PerpetualError;
use crate::objective::Objective;
use crate::{CalibrationMethod, Matrix, PerpetualBooster};
use csv;
use std::fs::File;
use std::io::BufReader;

#[allow(clippy::type_complexity)]
fn read_calibration_data(
    train_path: &str,
    test_path: &str,
    n_features: usize,
    train_frac: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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

    let load_raw = |path: &str| -> (Vec<Vec<f64>>, Vec<f64>) {
        let file = File::open(path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);
        let headers = csv_reader.headers().unwrap().clone();
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|&name| headers.iter().position(|h| h == name).expect("Feature not found"))
            .collect();
        let target_index = headers.iter().position(|h| h == target_name).expect("Target not found");
        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
        let mut y = Vec::new();
        for result in csv_reader.records() {
            let record = result.expect("CSV record error");
            let target_str = &record[target_index];
            y.push(target_str.parse::<f64>().unwrap_or(f64::NAN));
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                data_columns[i].push(val_str.parse::<f64>().unwrap_or(f64::NAN));
            }
        }
        (data_columns, y)
    };

    // Load and split train
    let (train_cols_raw, train_y_raw) = load_raw(train_path);
    let n_rows_train_raw = train_y_raw.len();
    let n_train = (n_rows_train_raw as f64 * train_frac).floor() as usize;

    let mut train_data = Vec::with_capacity(n_train * n_features);
    let mut cal_data = Vec::with_capacity((n_rows_train_raw - n_train) * n_features);

    for col in &train_cols_raw {
        train_data.extend_from_slice(&col[..n_train]);
        cal_data.extend_from_slice(&col[n_train..]);
    }

    let train_y = train_y_raw[..n_train].to_vec();
    let cal_y = train_y_raw[n_train..].to_vec();

    // Load test
    let (test_cols_raw, test_y) = load_raw(test_path);
    let test_data: Vec<f64> = test_cols_raw.into_iter().flatten().collect();

    (train_data, train_y, cal_data, cal_y, test_data, test_y)
}

#[test]
fn test_calibration_min_max() {
    let n_features = 8;
    let (train_vec, y_train, cal_vec, y_cal, test_vec, y_test) = read_calibration_data(
        "resources/cal_housing_train.csv",
        "resources/cal_housing_test.csv",
        n_features,
        0.75,
    );

    let train_data = Matrix::<f64>::new(&train_vec, y_train.len(), n_features);
    let cal_data = Matrix::<f64>::new(&cal_vec, y_cal.len(), n_features);
    let test_data = Matrix::<f64>::new(&test_vec, y_test.len(), n_features);

    let alpha_val = 0.1;
    let alpha = vec![alpha_val];

    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_save_node_stats(true);
    booster.fit(&train_data, &y_train, None, None).unwrap();

    booster
        .calibrate(CalibrationMethod::MinMax, (&cal_data, &y_cal, &alpha))
        .unwrap();

    let intervals = booster.predict_intervals(&test_data, true);
    assert!(intervals.contains_key("0.1"));
    let bounds = &intervals["0.1"];
    assert_eq!(bounds.len(), test_data.rows);

    // Coverage check
    let mut covered = 0usize;
    if !y_test.is_empty() {
        println!(
            "test_calibration_min_max: Example y_test[0]: {}, interval: [{}, {}]",
            y_test[0], bounds[0][0], bounds[0][1]
        );
    }
    for i in 0..y_test.len() {
        let low = bounds[i][0];
        let high = bounds[i][1];
        if y_test[i].is_finite() && y_test[i] >= low && y_test[i] <= high {
            covered += 1;
        }
    }
    let coverage = covered as f64 / (y_test.len() as f64);
    println!("test_calibration_min_max: coverage = {}", coverage);
    assert!(
        (coverage - (1.0 - alpha_val)).abs() < 0.01,
        "coverage {} not within tolerance of {}",
        coverage,
        1.0 - alpha_val
    );
}

#[test]
fn test_calibration_grp() {
    let n_features = 8;
    let (train_vec, y_train, cal_vec, y_cal, test_vec, y_test) = read_calibration_data(
        "resources/cal_housing_train.csv",
        "resources/cal_housing_test.csv",
        n_features,
        0.75,
    );

    let train_data = Matrix::<f64>::new(&train_vec, y_train.len(), n_features);
    let cal_data = Matrix::<f64>::new(&cal_vec, y_cal.len(), n_features);
    let test_data = Matrix::<f64>::new(&test_vec, y_test.len(), n_features);

    let alpha_val = 0.1;
    let alpha = vec![alpha_val];

    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_save_node_stats(true);
    booster.fit(&train_data, &y_train, None, None).unwrap();

    booster
        .calibrate(CalibrationMethod::GRP, (&cal_data, &y_cal, &alpha))
        .expect("Calibration failed for GRP");

    let intervals = booster.predict_intervals(&test_data, true);
    let bounds = &intervals["0.1"];

    // Coverage check
    let mut covered = 0usize;
    if !y_test.is_empty() {
        println!(
            "test_calibration_grp: Example y_test[0]: {}, interval: [{}, {}]",
            y_test[0], bounds[0][0], bounds[0][1]
        );
    }
    for i in 0..y_test.len() {
        let low = bounds[i][0];
        let high = bounds[i][1];
        if y_test[i].is_finite() && y_test[i] >= low && y_test[i] <= high {
            covered += 1;
        }
    }
    let coverage = covered as f64 / (y_test.len() as f64);
    println!("test_calibration_grp: coverage = {}", coverage);
    assert!(
        (coverage - (1.0 - alpha_val)).abs() < 0.01,
        "coverage {} not within tolerance of {}",
        coverage,
        1.0 - alpha_val
    );
}

#[test]
fn test_calibration_weight_variance() {
    let n_features = 8;
    let (train_vec, y_train, cal_vec, y_cal, test_vec, y_test) = read_calibration_data(
        "resources/cal_housing_train.csv",
        "resources/cal_housing_test.csv",
        n_features,
        0.75,
    );

    let train_data = Matrix::<f64>::new(&train_vec, y_train.len(), n_features);
    let cal_data = Matrix::<f64>::new(&cal_vec, y_cal.len(), n_features);
    let test_data = Matrix::<f64>::new(&test_vec, y_test.len(), n_features);

    let alpha_val = 0.1;
    let alpha = vec![alpha_val];

    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_save_node_stats(true);
    booster.fit(&train_data, &y_train, None, None).unwrap();

    booster
        .calibrate(CalibrationMethod::WeightVariance, (&cal_data, &y_cal, &alpha))
        .expect("Calibration failed for WeightVariance");

    let intervals = booster.predict_intervals(&test_data, true);
    let bounds = &intervals["0.1"];

    // Coverage check
    let mut covered = 0usize;
    if !y_test.is_empty() {
        println!(
            "test_calibration_weight_variance: Example y_test[0]: {}, interval: [{}, {}]",
            y_test[0], bounds[0][0], bounds[0][1]
        );
    }
    for i in 0..y_test.len() {
        let low = bounds[i][0];
        let high = bounds[i][1];
        if y_test[i].is_finite() && y_test[i] >= low && y_test[i] <= high {
            covered += 1;
        }
    }
    let coverage = covered as f64 / (y_test.len() as f64);
    println!("test_calibration_weight_variance: coverage = {}", coverage);
    assert!(
        (coverage - (1.0 - alpha_val)).abs() < 0.01,
        "coverage {} not within tolerance of {}",
        coverage,
        1.0 - alpha_val
    );
}

#[test]
fn test_calibrate_without_save_node_stats() {
    // Small dummy dataset
    let train_vec = vec![1.0, 2.0, 3.0, 4.0];
    let y_train = vec![1.0, 2.0];
    let cal_vec = vec![1.0, 2.0];
    let y_cal = vec![1.0];
    let alpha = vec![0.1];

    let train_data = Matrix::<f64>::new(&train_vec, 2, 2);
    let cal_data = Matrix::<f64>::new(&cal_vec, 1, 2);

    // Create booster with save_node_stats = false (default)
    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_save_node_stats(false); // Explicitly False, though default is false

    booster.fit(&train_data, &y_train, None, None).unwrap();

    // calibrate should fail
    let result = booster.calibrate(CalibrationMethod::MinMax, (&cal_data, &y_cal, &alpha));

    match result {
        Ok(_) => panic!("Should have returned error because save_node_stats is false"),
        Err(e) => match e {
            PerpetualError::InvalidParameter(param, expected, actual) => {
                assert_eq!(param, "save_node_stats");
                assert_eq!(expected, "true");
                assert_eq!(actual, "false");
            }
            _ => panic!("Returned wrong error type: {:?}", e),
        },
    }
}
