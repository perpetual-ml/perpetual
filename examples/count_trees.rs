use perpetual::PerpetualBooster;
use perpetual::data::Matrix;
use perpetual::objective::Objective;
use std::fs;
use std::time::Instant;

fn main() {
    let file_content = fs::read_to_string("resources/cal_housing_train.csv").unwrap();
    let mut lines = file_content.lines();
    lines.next();
    let n_features = 8;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();
    for line in lines {
        let values: Vec<f64> = line.split(',').map(|x| x.parse::<f64>().unwrap()).collect();
        y.push(values[n_features]);
        for j in 0..n_features {
            columns[j].push(values[j]);
        }
    }
    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);
    let start = Instant::now();
    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(1.0);
    booster.fit(&data, &y, None, None).unwrap();
    let elapsed = start.elapsed();
    println!(
        "cal_housing budget=1.0: Trees={}, Time={:?}",
        booster.trees.len(),
        elapsed
    );
    let tree_sizes: Vec<usize> = booster.trees.iter().map(|t| t.nodes.len()).collect();
    let avg_nodes = tree_sizes.iter().sum::<usize>() as f64 / tree_sizes.len() as f64;
    println!(
        "  avg_nodes_per_tree={:.1}, min={}, max={}",
        avg_nodes,
        tree_sizes.iter().min().unwrap(),
        tree_sizes.iter().max().unwrap()
    );

    // Cover types benchmark config
    let file_content2 = fs::read_to_string("resources/cover_types_train.csv").unwrap();
    let lines2 = file_content2.lines().skip(1).take(50000);
    let n_features2 = 20;
    let mut columns2: Vec<Vec<f64>> = vec![Vec::new(); n_features2];
    let mut y2: Vec<f64> = Vec::new();
    for line in lines2 {
        let values: Vec<f64> = line.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
        y2.push(values[values.len() - 1]);
        for j in 0..n_features2 {
            columns2[j].push(values[j]);
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
    let data_vec2: Vec<f64> = columns2.into_iter().flatten().collect();
    let data2 = Matrix::new(&data_vec2, y2.len(), n_features2);
    let start2 = Instant::now();
    let mut booster2 = PerpetualBooster::default().set_objective(Objective::LogLoss);
    booster2.fit(&data2, &y2, None, None).unwrap();
    let elapsed2 = start2.elapsed();
    println!(
        "cover_types budget=default: Trees={}, Time={:?}",
        booster2.trees.len(),
        elapsed2
    );
    let tree_sizes2: Vec<usize> = booster2.trees.iter().map(|t| t.nodes.len()).collect();
    let avg_nodes2 = tree_sizes2.iter().sum::<usize>() as f64 / tree_sizes2.len() as f64;
    println!(
        "  avg_nodes_per_tree={:.1}, min={}, max={}",
        avg_nodes2,
        tree_sizes2.iter().min().unwrap(),
        tree_sizes2.iter().max().unwrap()
    );
}
