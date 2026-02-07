use criterion::{criterion_group, criterion_main, Criterion};
use perpetual::data::Matrix;
use perpetual::objective_functions::Objective;
use perpetual::PerpetualBooster;
use std::fs;
use std::hint::black_box;
use std::time::Duration;

pub fn training_benchmark(c: &mut Criterion) {
    let file_content =
        fs::read_to_string("resources/cal_housing_train.csv").expect("Something went wrong reading the file");

    // Skip header
    let mut lines = file_content.lines();
    lines.next(); // Skip header line

    let mut data_vec: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().expect("Parse error"))
            .collect();
        // Last column is target
        y.push(values[8]);
        // First 8 columns are features
        data_vec.extend_from_slice(&values[0..8]);
    }

    let data = Matrix::new(&data_vec, y.len(), 8);

    let mut group = c.benchmark_group("training_benchmark");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(60)); // Increased for stability
    group.sample_size(40); // Increased from 10 to 40 to reduce outlier impact

    group.bench_function("train_booster_cal_housing", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default()
                .set_objective(Objective::SquaredLoss)
                .set_budget(2.0);
            // Using default parameters as baseline
            booster
                .fit(black_box(&data), black_box(&y), black_box(None), black_box(None))
                .unwrap();
        })
    });
    group.finish();
}

pub fn training_benchmark_cover_types(c: &mut Criterion) {
    let file_content =
        fs::read_to_string("resources/cover_types_train.csv").expect("Something went wrong reading the file");

    // Skip header and limit rows
    let lines = file_content.lines().skip(1).take(10000);

    let mut data_vec: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>().expect("Parse error"))
            .collect();
        // Last column is target
        y.push(values[values.len() - 1]);
        // Rest are features
        data_vec.extend_from_slice(&values[0..10]);
    }

    let n_cols = data_vec.len() / y.len();
    let data = Matrix::new(&data_vec, y.len(), n_cols);

    let mut group = c.benchmark_group("training_benchmark_cover");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(40);

    group.bench_function("train_booster_cover_types", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default().set_objective(Objective::LogLoss);
            booster
                .fit(black_box(&data), black_box(&y), black_box(None), black_box(None))
                .unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, training_benchmark, training_benchmark_cover_types);
criterion_main!(benches);
