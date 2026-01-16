use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perpetual::data::Matrix;
use perpetual::PerpetualBooster;
use std::fs;
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
    group.measurement_time(Duration::from_secs(20)); // Give it more time for stable results
    group.sample_size(10); // Reduce sample size as training might be slow

    group.bench_function("train_booster_cal_housing", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default();
            // Using default parameters as baseline
            booster.fit(black_box(&data), black_box(&y), black_box(None)).unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, training_benchmark);
criterion_main!(benches);
