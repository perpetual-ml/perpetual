/// Instrument the tree.fit loop with timing.
/// Adds thread-local timing accumulators to measure split_node phase breakdown.
use perpetual::{objective_functions::Objective, Matrix, PerpetualBooster};
use std::time::Instant;

fn main() {
    let file = std::fs::read_to_string("resources/cal_housing_train.csv").expect("read");
    let mut lines = file.lines();
    lines.next();
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
    eprintln!("N={} F={}", y.len(), n_features);

    // Run 5 times and take best
    let mut best = f64::MAX;
    for run in 0..5 {
        let t0 = Instant::now();
        let mut booster = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_budget(1.0);
        booster.fit(&data, &y, None, None).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best {
            best = elapsed;
        }
        let trees = booster.get_prediction_trees().len();
        eprintln!("run={} trees={} time={:.3}s", run, trees, elapsed);
    }
    eprintln!("best={:.3}s", best);

    // Now cover_types
    eprintln!("\n--- cover_types ---");
    let file = std::fs::read_to_string("resources/cover_types_train.csv").expect("read");
    let lines_iter = file.lines().skip(1).take(50000);
    let n_features = 20;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y2: Vec<f64> = Vec::new();
    for line in lines_iter {
        let vals: Vec<f64> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        y2.push(vals[vals.len() - 1]);
        for j in 0..n_features {
            columns[j].push(vals[j]);
        }
    }
    let mut class_counts = std::collections::HashMap::new();
    for &v in &y2 {
        *class_counts.entry(v as i64).or_insert(0usize) += 1;
    }
    let majority_class = *class_counts.iter().max_by_key(|(_, c)| **c).unwrap().0;
    for v in y2.iter_mut() {
        *v = if *v as i64 == majority_class { 0.0 } else { 1.0 };
    }
    let data_vec2: Vec<f64> = columns.into_iter().flatten().collect();
    let data2 = Matrix::new(&data_vec2, y2.len(), n_features);
    eprintln!("N={} F={}", y2.len(), n_features);

    let mut best2 = f64::MAX;
    for run in 0..5 {
        let t0 = Instant::now();
        let mut booster = PerpetualBooster::default().set_objective(Objective::LogLoss);
        booster.fit(&data2, &y2, None, None).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best2 {
            best2 = elapsed;
        }
        let trees = booster.get_prediction_trees().len();
        eprintln!("run={} trees={} time={:.3}s", run, trees, elapsed);
    }
    eprintln!("best={:.3}s", best2);
}
