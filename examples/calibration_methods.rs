use perpetual::booster::config::CalibrationMethod;
use perpetual::objective_functions::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn read_data(path: &str, feature_names: &[&str]) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
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
        y.push(record[target_index].parse::<f64>()?);
        for (i, &idx) in feature_indices.iter().enumerate() {
            data_columns[i].push(record[idx].parse::<f64>()?);
        }
    }

    let data: Vec<f64> = data_columns.into_iter().flatten().collect();
    Ok((data, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    let feature_names = [
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];

    // 1. Read data
    let (data_vec, y_vec) = read_data("resources/cal_housing_test.csv", &feature_names)?;
    let rows = y_vec.len();
    let cols = feature_names.len();

    // Split into train (80%) and calibration (20%)
    let train_size = (rows as f64 * 0.8) as usize;
    let x_train_vec = data_vec
        .chunks(rows)
        .flat_map(|c| c[..train_size].to_vec())
        .collect::<Vec<_>>();
    let y_train = y_vec[..train_size].to_vec();

    let x_cal_vec = data_vec
        .chunks(rows)
        .flat_map(|c| c[train_size..].to_vec())
        .collect::<Vec<_>>();
    let y_cal = y_vec[train_size..].to_vec();

    let x_train = Matrix::new(&x_train_vec, train_size, cols);
    let x_cal = Matrix::new(&x_cal_vec, rows - train_size, cols);

    let alpha = vec![0.1]; // 90% confidence

    // 2. Demonstrate different calibration methods
    let methods = [
        CalibrationMethod::Conformal,
        CalibrationMethod::MinMax,
        CalibrationMethod::GRP,
        CalibrationMethod::WeightVariance,
    ];

    for method in methods {
        println!("\nTesting Method: {:?}", method);

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_budget(1.0)
            .set_calibration_method(method);

        // Fit the model
        model.fit(&x_train, &y_train, None, None)?;

        // Calibrate
        // Note: For Conformal, it needs full data to fit quantile models internally.
        // For M1, M2, M3, it uses x_cal to find thresholds.
        let cal_data = (&x_cal, y_cal.as_slice(), alpha.as_slice());
        match method {
            CalibrationMethod::Conformal => {
                model.calibrate_conformal(&x_train, &y_train, None, None, cal_data)?;
            }
            _ => {
                model.calibrate(method, cal_data)?;
            }
        }

        // Predict intervals
        let intervals = model.predict_intervals(&x_cal, true);

        // Basic check on coverage
        if let Some(alpha_intervals) = intervals.get("0.1") {
            let lower = &alpha_intervals[0];
            let upper = &alpha_intervals[1];
            let mut captured = 0;
            let mut avg_width = 0.0;
            for i in 0..y_cal.len() {
                if y_cal[i] >= lower[i] && y_cal[i] <= upper[i] {
                    captured += 1;
                }
                avg_width += upper[i] - lower[i];
            }
            let coverage = captured as f64 / y_cal.len() as f64;
            println!("  Coverage: {:.2}%", coverage * 100.0);
            println!("  Average Width: {:.4}", avg_width / y_cal.len() as f64);
        }
    }

    Ok(())
}
