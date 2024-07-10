use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use perpetual::binning::bin_matrix;
use perpetual::booster::PerpetualBooster;
use perpetual::constants::N_NODES_ALLOCATED;
use perpetual::constraints::ConstraintMap;
use perpetual::data::Matrix;
use perpetual::histogram::HistogramMatrix;
use perpetual::objective::{LogLoss, ObjectiveFunction};
use perpetual::splitter::MissingImputerSplitter;
use perpetual::tree::Tree;
use perpetual::utils::{fast_f64_sum, fast_sum, naive_sum};
use std::fs;
use std::time::Duration;

pub fn tree_benchmarks(c: &mut Criterion) {
    let file = fs::read_to_string("resources/contiguous_no_missing_100k_samp_seed0.csv")
        .expect("Something went wrong reading the file");
    let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let file =
        fs::read_to_string("resources/performance_100k_samp_seed0.csv").expect("Something went wrong reading the file");
    let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
    let yhat = vec![0.5; y.len()];
    let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, None, None);
    let loss = LogLoss::calc_loss(&y, &yhat, None, None);

    let v: Vec<f32> = vec![10.; 300000];
    c.bench_function("Niave Sum", |b| b.iter(|| naive_sum(black_box(&v))));
    c.bench_function("fast sum", |b| b.iter(|| fast_sum(black_box(&v))));
    c.bench_function("fast f64 sum", |b| b.iter(|| fast_f64_sum(black_box(&v))));

    c.bench_function("calc_grad_hess", |b| {
        b.iter(|| LogLoss::calc_grad_hess(black_box(&y), black_box(&yhat), black_box(None), black_box(None)))
    });

    let data = Matrix::new(&data_vec, y.len(), 5);
    let splitter = MissingImputerSplitter {
        eta: 0.3,
        allow_missing_splits: true,
        constraints_map: ConstraintMap::new(),
    };
    let mut tree = Tree::new();

    let bindata = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
    let bdata = Matrix::new(&bindata.binned_data, data.rows, data.cols);
    let col_index: Vec<usize> = (0..data.cols).collect();

    let hist_init = HistogramMatrix::empty(&bdata, &bindata.cuts, &col_index, true, false);
    let mut hist_map: HashMap<usize, HistogramMatrix> = HashMap::with_capacity(N_NODES_ALLOCATED);
    for i in 0..N_NODES_ALLOCATED {
        hist_map.insert(i, hist_init.clone());
    }

    tree.fit(
        &bdata,
        data.index.to_owned(),
        &col_index,
        &bindata.cuts,
        &g,
        h.as_deref(),
        &splitter,
        true,
        Some(f32::MAX),
        &loss,
        &y,
        LogLoss::calc_loss,
        &yhat,
        None,
        None,
        false,
        &mut hist_map,
        None,
    );

    println!("{}", tree.nodes.len());
    c.bench_function("Train Tree", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();

            train_tree.fit(
                black_box(&bdata),
                black_box(data.index.to_owned()),
                black_box(&col_index),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(h.as_deref()),
                black_box(&splitter),
                black_box(false),
                Some(f32::MAX),
                black_box(&loss),
                black_box(&y),
                black_box(LogLoss::calc_loss),
                black_box(&yhat),
                None,
                None,
                false,
                black_box(&mut hist_map),
                None,
            );
        })
    });
    c.bench_function("Train Tree - column subset", |b| {
        b.iter(|| {
            let mut train_tree: Tree = Tree::new();

            train_tree.fit(
                black_box(&bdata),
                black_box(data.index.to_owned()),
                black_box(&[1, 3, 4]),
                black_box(&bindata.cuts),
                black_box(&g),
                black_box(h.as_deref()),
                black_box(&splitter),
                black_box(false),
                Some(f32::MAX),
                black_box(&loss),
                black_box(&y),
                black_box(LogLoss::calc_loss),
                black_box(&yhat),
                None,
                None,
                false,
                black_box(&mut hist_map),
                None,
            );
        })
    });
    c.bench_function("Tree Predict (Single Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(false), black_box(&f64::NAN)))
    });
    c.bench_function("Tree Predict (Multi Threaded)", |b| {
        b.iter(|| tree.predict(black_box(&data), black_box(true), black_box(&f64::NAN)))
    });

    // Gradient Booster
    // Bench building
    let mut booster_train = c.benchmark_group("train_booster");
    booster_train.warm_up_time(Duration::from_secs(10));
    booster_train.sample_size(50);
    // booster_train.sampling_mode(SamplingMode::Linear);
    booster_train.bench_function("train_booster_default", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default().set_parallel(false);
            booster
                .fit(
                    black_box(&data),
                    black_box(&y),
                    black_box(None),
                    black_box(None),
                    black_box(0.3),
                    black_box(None),
                    black_box(None),
                )
                .unwrap();
        })
    });
    booster_train.bench_function("train_booster_with_column_sampling", |b| {
        b.iter(|| {
            let mut booster = PerpetualBooster::default().set_parallel(false);
            booster
                .fit(
                    black_box(&data),
                    black_box(&y),
                    black_box(None),
                    black_box(None),
                    black_box(0.3),
                    black_box(None),
                    black_box(None),
                )
                .unwrap();
        })
    });
    let mut booster = PerpetualBooster::default();
    booster.fit(&data, &y, None, None, 0.3, None, None).unwrap();
    booster_train.bench_function("Predict Booster", |b| {
        b.iter(|| booster.predict(black_box(&data), false, None))
    });
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
