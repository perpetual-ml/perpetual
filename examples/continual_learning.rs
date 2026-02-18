use perpetual::data::ColumnarMatrix;
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::error::Error;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let (data_full, y_full) = read_data("resources/cal_housing_train.csv")?;
    let n_features = data_full.len();
    let n_samples = y_full.len();

    // Configuration
    let initial_batch_size = 6512;
    let batch_size = 500;

    let strategies = vec!["No Pruning", "Bottom-Up", "Top-Down", "Statistical", "Retrain"];

    println!("Total samples: {}, Features: {}", n_samples, n_features);

    // Prepare CSV for results
    println!("Strategy,Batch,Time_Fit_ms,Time_Prune_ms,Nodes,MSE,Dataset_Fraction");

    for strategy in strategies {
        // DATA SPLITTING
        // Initial batch
        let initial_end = initial_batch_size.min(n_samples);
        let y_initial = &y_full[0..initial_end];

        let data_initial_slices: Vec<&[f64]> = data_full.iter().map(|c| &c[0..initial_end]).collect();
        let matrix_initial = ColumnarMatrix::new(data_initial_slices.clone(), None, initial_end);

        // Train initial model
        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_budget(1.0);

        // For incremental learning strategies, we must set reset=false after first fit
        // Actually, we can just set it once here and it will affect subsequent fit calls
        if strategy != "Retrain" {
            model = model.set_reset(Some(false));
        }

        let start = Instant::now();
        model.fit_columnar(&matrix_initial, y_initial, None, None)?;
        let initial_fit_time = start.elapsed().as_millis();

        // Initial Pruning (on initial batch)
        let initial_flat: Vec<f64> = data_initial_slices.iter().flat_map(|c| c.iter().copied()).collect();
        let matrix_initial_flat = Matrix::new(&initial_flat, initial_end, n_features);

        let prune_start = Instant::now();
        let res = match strategy {
            "Bottom-Up" => model.prune(&matrix_initial_flat, y_initial, None, None),
            "Top-Down" => model.prune_top_down(&matrix_initial_flat, y_initial, None, None),
            "Statistical" => model.prune_statistical(&matrix_initial_flat, y_initial, None, None),
            _ => Ok(()),
        };
        if let Err(e) = res {
            eprintln!("Error pruning initial batch for {}: {}", strategy, e);
        }
        let initial_prune_time = prune_start.elapsed().as_millis();

        let nodes: usize = model.get_prediction_trees().iter().map(|t| t.nodes.len()).sum();

        // Since it's the first batch, we evaluate on it (warmup)
        let preds = model.predict_columnar(&matrix_initial, false);
        let mse = calculate_mse(&preds, y_initial);

        println!(
            "{},{},{},{},{},{:.4},{:.2}",
            strategy,
            0,
            initial_fit_time,
            initial_prune_time,
            nodes,
            mse,
            initial_end as f64 / n_samples as f64
        );

        // INCREMENTAL LEARNING
        let mut current_idx = initial_end;
        let mut batch_idx = 1;

        while current_idx < n_samples {
            let end_idx = (current_idx + batch_size).min(n_samples);
            let y_batch = &y_full[current_idx..end_idx];

            // 1. EVALUATE on next batch (before training/updating)
            let data_batch_slices: Vec<&[f64]> = data_full.iter().map(|c| &c[current_idx..end_idx]).collect();
            let matrix_batch_columnar = ColumnarMatrix::new(data_batch_slices.clone(), None, end_idx - current_idx);
            let preds = model.predict_columnar(&matrix_batch_columnar, false);
            let mse = calculate_mse(&preds, y_batch);

            // 2. PREPARE for pruning (flattened batch)
            let batch_flat: Vec<f64> = data_batch_slices.iter().flat_map(|c| c.iter().copied()).collect();
            let matrix_batch_flat = Matrix::new(&batch_flat, end_idx - current_idx, n_features);

            // 3. PREPARE for training (cumulative)
            let y_cumulative = &y_full[0..end_idx];
            let data_cumulative_slices: Vec<&[f64]> = data_full.iter().map(|c| &c[0..end_idx]).collect();
            let matrix_cumulative = ColumnarMatrix::new(data_cumulative_slices, None, end_idx);

            // 4. UPDATE / RETRAIN
            let fit_start = Instant::now();
            if strategy == "Retrain" {
                model = PerpetualBooster::default()
                    .set_objective(Objective::SquaredLoss)
                    .set_budget(1.0);
                model.fit_columnar(&matrix_cumulative, y_cumulative, None, None)?;
            } else {
                // reset is already false, so this adds trees based on cumulative data
                model.fit_columnar(&matrix_cumulative, y_cumulative, None, None)?;
            }
            let fit_time = fit_start.elapsed().as_millis();

            // 5. PRUNE (on latest batch)
            let prune_start = Instant::now();
            let res = match strategy {
                "Bottom-Up" => model.prune(&matrix_batch_flat, y_batch, None, None),
                "Top-Down" => model.prune_top_down(&matrix_batch_flat, y_batch, None, None),
                "Statistical" => model.prune_statistical(&matrix_batch_flat, y_batch, None, None),
                _ => Ok(()),
            };
            if let Err(e) = res {
                eprintln!("Error pruning batch {} for {}: {}", batch_idx, strategy, e);
            }
            let prune_time = prune_start.elapsed().as_millis();

            let nodes: usize = model.get_prediction_trees().iter().map(|t| t.nodes.len()).sum();

            // PRINT (MSE is from the eval step at start of loop)
            println!(
                "{},{},{},{},{},{:.4},{:.2}",
                strategy,
                batch_idx,
                fit_time,
                prune_time,
                nodes,
                mse,
                end_idx as f64 / n_samples as f64
            );

            current_idx = end_idx;
            batch_idx += 1;
        }
    }

    Ok(())
}

fn calculate_mse(preds: &[f64], y: &[f64]) -> f64 {
    preds.iter().zip(y.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / preds.len() as f64
}

fn read_data(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let file = fs::read_to_string(path)?;
    let mut y = Vec::new();
    let mut data_columns: Vec<Vec<f64>> = Vec::new();

    let mut lines = file.lines();
    lines.next(); // Skip header
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| x.parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()?;

        if !values.is_empty() {
            let n_cols = values.len() - 1;
            if data_columns.is_empty() {
                data_columns = vec![Vec::new(); n_cols];
            }
            for i in 0..n_cols {
                data_columns[i].push(values[i]);
            }
            y.push(values[n_cols]);
        }
    }
    Ok((data_columns, y))
}
