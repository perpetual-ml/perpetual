/// Micro-benchmarks of individual operations to identify bottlenecks
use perpetual::{Matrix, objective_functions::Objective};
use std::hint::black_box;
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
    let _data = Matrix::new(&data_vec, y.len(), n_features);
    let n = y.len();
    eprintln!("N={} F={}", n, n_features);

    // Benchmark loss_single computation (the loss_decr inner loop)
    let obj = Objective::SquaredLoss;
    let yhat_val = 2.0_f64;
    let y_val = 1.5_f64;
    let n_iters = 1_000_000;
    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..n_iters {
        acc += black_box(obj.loss_single(black_box(y_val), black_box(yhat_val), None));
    }
    let loss_single_ns = t.elapsed().as_nanos() as f64 / n_iters as f64;
    eprintln!("loss_single: {:.1}ns (acc={})", loss_single_ns, acc);

    // Benchmark loss_decr loop for a realistic node (1000 samples)
    let node_size = 1000;
    let yhat = vec![2.0_f64; n];
    let loss = vec![0.5_f32; n];
    let mut loss_decr = vec![0.0_f32; n];
    let mut loss_decr_avg;
    let index_length = n as f32;
    let weight = 0.1_f64;
    let indices: Vec<usize> = (0..node_size).collect();

    let n_reps = 10000;
    let t = Instant::now();
    for _rep in 0..n_reps {
        loss_decr_avg = 0.0;
        for i in indices.iter() {
            let _i = *i;
            let yhat_new = yhat[_i] + weight;
            let loss_new = obj.loss_single(y[_i], yhat_new, None);
            loss_decr_avg -= loss_decr[_i] / index_length;
            loss_decr[_i] = loss[_i] - loss_new;
            loss_decr_avg += loss_decr[_i] / index_length;
        }
        black_box(loss_decr_avg);
    }
    let loss_loop_us = t.elapsed().as_micros() as f64 / n_reps as f64;
    eprintln!(
        "loss_decr loop (n={}): {:.1}μs ({:.1}ns/sample)",
        node_size,
        loss_loop_us,
        loss_loop_us * 1000.0 / node_size as f64
    );

    // Benchmark pivot (using small array to represent operation)
    let n_pivot = 16000;
    let mut idx: Vec<usize> = (0..n_pivot).collect();
    let feature_col: Vec<u16> = (0..n_pivot as u16).map(|x| x % 256).collect();
    let split_bin = 128_u16;
    let n_reps = 10000;
    let t = Instant::now();
    for _rep in 0..n_reps {
        // Simulate a simple pivot: move < split_bin to left, >= to right
        let mut lo = 0;
        let mut hi = n_pivot - 1;
        while lo < hi {
            while lo < hi && feature_col[idx[lo]] < split_bin {
                lo += 1;
            }
            while lo < hi && feature_col[idx[hi]] >= split_bin {
                hi -= 1;
            }
            if lo < hi {
                idx.swap(lo, hi);
                lo += 1;
                hi -= 1;
            }
        }
        black_box(&idx);
        // Reset
        for (i, item) in idx.iter_mut().enumerate() {
            *item = i;
        }
    }
    let pivot_us = t.elapsed().as_micros() as f64 / n_reps as f64;
    eprintln!(
        "pivot (n={}): {:.1}μs ({:.1}ns/sample)",
        n_pivot,
        pivot_us,
        pivot_us * 1000.0 / n_pivot as f64
    );

    // Benchmark f32 scatter-add (simulating histogram accumulation)
    let n_accum = 16000;
    let index_arr: Vec<usize> = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..n_accum)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                (h.finish() as usize) % n_accum
            })
            .collect()
    };
    let bins: Vec<u16> = (0..n).map(|i| (i % 256) as u16).collect();
    let grads: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let mut flat = vec![0.0_f32; 256 * 5];
    let mut flat_c = vec![0_u32; 256 * 5];

    let n_reps = 100000;
    let t = Instant::now();
    for _rep in 0..n_reps {
        for &i in &index_arr {
            let bin = bins[i] as usize;
            let fold = i % 5;
            let slot = bin * 5 + fold;
            flat[slot] += grads[i];
            flat_c[slot] += 1;
        }
        black_box(&flat);
    }
    let accum_us = t.elapsed().as_micros() as f64 / n_reps as f64;
    eprintln!(
        "scatter-add (n={}): {:.1}μs ({:.1}ns/sample)",
        n_accum,
        accum_us,
        accum_us * 1000.0 / n_accum as f64
    );

    // Benchmark Rayon scope overhead
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let n_reps = 100000;
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    for _rep in 0..n_reps {
        pool.scope(|s| {
            for _i in 0..8 {
                s.spawn(|_| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                });
            }
        });
    }
    let scope_us = t.elapsed().as_micros() as f64 / n_reps as f64;
    eprintln!(
        "rayon scope (8 tasks): {:.1}μs (counter={})",
        scope_us,
        counter.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Benchmark empty Rayon scope
    let t = Instant::now();
    for _rep in 0..n_reps {
        pool.scope(|_s| {});
    }
    let empty_scope_us = t.elapsed().as_micros() as f64 / n_reps as f64;
    eprintln!("rayon scope (empty): {:.1}μs", empty_scope_us);

    // Estimate total time breakdown for 1 tree (198 splits, N=16512, F=8)
    let splits = 198;
    let total_hist_samples = 16512 * 8; // rough estimate: N * ~8 levels, halving each time
    let total_pivot_samples = 16512 * 9; // sum of parent sizes across all splits
    let total_loss_samples = 16512 * 9; // sum across all new children

    let estimated_hist_ms = total_hist_samples as f64 * accum_us / n_accum as f64 / 1000.0 / 8.0; // /8 for 8 threads
    let estimated_pivot_ms = total_pivot_samples as f64 * pivot_us / n_pivot as f64 / 1000.0;
    let estimated_loss_ms = total_loss_samples as f64 * loss_loop_us / node_size as f64 / 1000.0;
    let estimated_rayon_ms = splits as f64 * 2.0 * scope_us / 1000.0;
    let estimated_subtraction_ms = splits as f64 * 8.0 * 256.0 * 15.0 * 1.0 / 1_000_000.0; // ~1ns per op

    eprintln!("\n--- Estimated per-tree breakdown (198 splits) ---");
    eprintln!("histogram build: {:.2}ms", estimated_hist_ms);
    eprintln!("pivot:           {:.2}ms", estimated_pivot_ms);
    eprintln!("loss_decr:       {:.2}ms", estimated_loss_ms);
    eprintln!(
        "rayon overhead:  {:.2}ms (2 scopes/split × {:.1}μs)",
        estimated_rayon_ms, scope_us
    );
    eprintln!("subtraction:     {:.2}ms", estimated_subtraction_ms);
    let total_estimated =
        estimated_hist_ms + estimated_pivot_ms + estimated_loss_ms + estimated_rayon_ms + estimated_subtraction_ms;
    eprintln!("TOTAL estimated: {:.2}ms (actual ~42ms)", total_estimated);
    eprintln!(
        "unaccounted:     {:.2}ms ({:.0}%)",
        42.0 - total_estimated,
        (42.0 - total_estimated) / 42.0 * 100.0
    );
}
