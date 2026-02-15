/// Quick timing of cal_housing fit to understand baseline.
use perpetual::{Matrix, PerpetualBooster, objective_functions::Objective};
use std::time::Instant;

fn main() {
    let file = std::fs::read_to_string("resources/cal_housing_train.csv").expect("Failed to read CSV");
    let mut lines = file.lines();
    lines.next(); // skip header

    let n_features = 8;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();

    for line in lines {
        let vals: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
        y.push(vals[n_features]);
        for j in 0..n_features {
            columns[j].push(vals[j]);
        }
    }
    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);

    // Timed runs for both budget values and datasets
    for budget in [1.0_f32] {
        let n_runs = 6;
        let mut times = Vec::new();
        for _ in 0..n_runs {
            let start = Instant::now();
            let mut booster = PerpetualBooster::default()
                .set_objective(Objective::SquaredLoss)
                .set_budget(budget);
            booster.fit(&data, &y, None, None).unwrap();
            let elapsed = start.elapsed();
            let trees = booster.get_prediction_trees().len();
            // Count total nodes across all trees
            times.push(elapsed);
            eprintln!("  budget={} trees={} time={:.3}s", budget, trees, elapsed.as_secs_f64());
        }
        let avg = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / n_runs as f64;
        let min = times.iter().map(|t| t.as_secs_f64()).fold(f64::MAX, f64::min);
        eprintln!("budget={} avg={:.3}s min={:.3}s", budget, avg, min);
    }
}
