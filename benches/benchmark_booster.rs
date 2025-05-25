// Benchmarking training time
// for Multivariate and Univariate
// boosters
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perpetual::{UnivariateBooster};
use perpetual::data::Matrix;
use std::time::Duration;

// data generating
// functions
mod utils;
use utils::create_data;

// default booster
pub fn default_univariate_booster(c: &mut Criterion) {

    // sample size
    let n_samples= 1_000usize;
    let n_features= 10usize;

    // prepare data
    let (data, y) = create_data(n_samples, n_features);
    let matrix = Matrix::new(&data, n_samples, n_features);

    // prepare function
    // for benchmark
    let mut booster_train = c.benchmark_group("train_booster");
    
    // warm up
    booster_train.warm_up_time(Duration::from_secs(120));

    println!("\nBenchmarking on a {} x {} matrix:\n", n_features, n_samples);
    booster_train.bench_function(
        "train_booster_default", |b| {
            b.iter(
                || {

                    // setup booster
                    let mut booster = UnivariateBooster::default()
                    .set_budget(0.3);
                    
                    // fit the booster
                    booster.fit(black_box(&matrix), black_box(&y),black_box(None)).unwrap();
                }
            )
        }
    );
}

criterion_group!(benches, default_univariate_booster);
criterion_main!(benches);