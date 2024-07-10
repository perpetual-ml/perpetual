//! An example using the `titanic` dataset
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use polars::prelude::*;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let budget = &args[1].parse::<f32>().unwrap();

    let features_and_target = ["survived", "pclass", "age", "sibsp", "parch", "fare"];

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(Arc::new(
            features_and_target.iter().map(|&s| s.to_string()).collect(),
        )))
        .try_into_reader_with_file_path(Some("resources/titanic.csv".into()))?
        .finish()
        .unwrap();

    // Get data in column major format...
    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.melt(id_vars, ["pclass", "age", "sibsp", "parch", "fare"])?;

    let data = Vec::from_iter(
        mdf.select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    let y = Vec::from_iter(
        df.column("survived")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // Create Matrix from ndarray.
    let matrix = Matrix::new(&data, y.len(), 5);

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default().set_objective(Objective::LogLoss);
    model.fit(&matrix, &y, None, None, *budget, None, None)?;

    println!("Model prediction: {:?} ...", &model.predict(&matrix, true, None)[0..10]);

    Ok(())
}
