#![warn(unused_imports)]
// Benchmarking calculation
// of objective functions
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perpetual::objective_functions::{self, *};
use std::time::Duration;

// data generating
// functions
mod utils;
use utils::prediction_pair;

// logloss
pub fn benchmark_logloss(c: &mut Criterion) {

    // generate prediction pairs
    let (y, y_hat, sample_weights) = prediction_pair(1_000_000usize);

    // prepare benchmark
    let mut calculate_objective = c.benchmark_group("Objective Functions");

    // warm up
    calculate_objective.warm_up_time(Duration::from_secs(120));
    calculate_objective.bench_function(
        "Unweighted objective function", |b| {
            b.iter(||{

                // instantiate objective function
                let objective_function = Objective::LogLoss.as_function();

                // calculate gradients
                objective_function.calc_grad_hess(black_box(&y), black_box(&y_hat), black_box(None));

            });
        });

    // warm up
    calculate_objective.warm_up_time(Duration::from_secs(120));
    calculate_objective.bench_function(
    "Weighted objective function", |b| {
        b.iter(||{

            // instantiate objective function
            let objective_function = Objective::LogLoss.as_function();

            // calculate gradients
            objective_function.calc_grad_hess(black_box(&y), black_box(&y_hat), black_box(Some(&sample_weights)));

        });
    });

}

criterion_group!(benches, benchmark_logloss);
criterion_main!(benches);