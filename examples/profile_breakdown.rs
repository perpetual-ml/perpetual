/// Profile the time breakdown of PerpetualBooster::fit.
use perpetual::{
    Matrix, PerpetualBooster,
    objective::{Objective, ObjectiveFunction},
};
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

    // Warm up
    {
        let mut b = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_budget(1.0);
        b.fit(&data, &y, None, None).unwrap();
    }

    // Timed run
    let t0 = Instant::now();
    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(1.0);
    booster.fit(&data, &y, None, None).unwrap();
    let total = t0.elapsed();
    let trees = booster.get_prediction_trees().len();
    let total_nodes: usize = booster.get_prediction_trees().iter().map(|t| t.nodes.len()).sum();
    eprintln!(
        "trees={} total_nodes={} total_time={:.3}s avg_tree_time={:.3}ms",
        trees,
        total_nodes,
        total.as_secs_f64(),
        total.as_secs_f64() * 1000.0 / trees as f64
    );

    // Now measure individual component costs
    // 1. Gradient computation time
    let n_iters = 1000;
    let yhat = vec![0.5_f64; y.len()];
    let obj = Objective::SquaredLoss;
    let t = Instant::now();
    for _ in 0..n_iters {
        let _ = std::hint::black_box(obj.gradient(&y, &yhat, None, None));
    }
    let grad_time = t.elapsed().as_secs_f64() / n_iters as f64;
    eprintln!("gradient_time={:.3}ms (per call)", grad_time * 1000.0);

    // 2. Index copy time
    let mut buf = data.index.to_owned();
    let t = Instant::now();
    for _ in 0..n_iters {
        buf.copy_from_slice(&data.index);
        std::hint::black_box(&buf);
    }
    let copy_time = t.elapsed().as_secs_f64() / n_iters as f64;
    eprintln!("index_copy_time={:.3}ms (per call)", copy_time * 1000.0);

    // Summary
    let estimated_overhead = trees as f64 * (grad_time + copy_time);
    eprintln!(
        "estimated_per_iter_overhead(grad+copy)={:.3}ms",
        (grad_time + copy_time) * 1000.0
    );
    eprintln!("estimated_total_overhead={:.3}s", estimated_overhead);
    eprintln!("implied_tree_fit_time={:.3}s", total.as_secs_f64() - estimated_overhead);
}
