use criterion::{Criterion, criterion_group, criterion_main};
use perpetual::PerpetualBooster;
use perpetual::data::Matrix;
use perpetual::objective::Objective;
use std::fs;
use std::hint::black_box;
use std::time::Duration;

// Baseline results:
// cal_housing: 5.66s -> 1.83s -> 1.37s
// cover_types: 9.07s -> 3.98s -> 3.33s

pub fn training_benchmark_cal_housing(c: &mut Criterion) {
    let file_content =
        fs::read_to_string("resources/cal_housing_train.csv").expect("Something went wrong reading the file");

    // Skip header
    let mut lines = file_content.lines();
    lines.next(); // Skip header line

    let n_features = 8;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();

    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().expect("Parse error"))
            .collect();
        // Last column is target
        y.push(values[n_features]);
        // First 8 columns are features (column-major order)
        for j in 0..n_features {
            columns[j].push(values[j]);
        }
    }

    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);

    let mut group = c.benchmark_group("training_benchmark_cal_housing");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(120)); // Increased for stability
    group.sample_size(20); // Increased from 10 to 40 to reduce outlier impact

    group.bench_function("training_benchmark_cal_housing", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default()
                .set_objective(Objective::SquaredLoss)
                .set_budget(1.0);
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
    let lines = file_content.lines().skip(1).take(50000);

    let n_features = 20;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();

    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>().expect("Parse error"))
            .collect();
        // Last column is target
        y.push(values[values.len() - 1]);
        // First columns are features (column-major order)
        for j in 0..n_features {
            columns[j].push(values[j]);
        }
    }

    // Convert to binary: find majority class → 0, rest → 1
    let mut class_counts = std::collections::HashMap::new();
    for &v in &y {
        *class_counts.entry(v as i64).or_insert(0usize) += 1;
    }
    let majority_class = *class_counts.iter().max_by_key(|(_, c)| **c).unwrap().0;
    for v in y.iter_mut() {
        *v = if *v as i64 == majority_class { 0.0 } else { 1.0 };
    }

    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);

    let mut group = c.benchmark_group("training_benchmark_cover_types");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(120));
    group.sample_size(20);

    group.bench_function("training_benchmark_cover_types", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default().set_objective(Objective::LogLoss);
            booster
                .fit(black_box(&data), black_box(&y), black_box(None), black_box(None))
                .unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, training_benchmark_cal_housing, training_benchmark_cover_types);
criterion_main!(benches);
