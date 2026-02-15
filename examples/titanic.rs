//! An example using the `titanic` dataset
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let budget = &args[1].parse::<f32>().unwrap();

    let feature_names = ["pclass", "age", "sibsp", "parch", "fare"];
    let target_name = "survived";

    let file = File::open("resources/titanic.csv")?;
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

    // Flatten columns to create column-major data
    let data: Vec<f64> = data_columns.into_iter().flatten().collect();

    // Create Matrix from ndarray.
    let matrix = Matrix::new(&data, y.len(), 5);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(*budget);
    model.fit(&matrix, &y, None, None)?;

    println!("Model prediction: {:?} ...", &model.predict(&matrix, true)[0..10]);

    Ok(())
}
