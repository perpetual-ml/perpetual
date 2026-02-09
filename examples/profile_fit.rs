use perpetual::data::Matrix;
use perpetual::objective_functions::Objective;
use perpetual::PerpetualBooster;
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

    // Budget 2.0, same as benchmark
    let mut booster = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(2.0)
        .set_log_iterations(1);
    let start = Instant::now();
    booster.fit(&data, &y, None, None).unwrap();
    let elapsed = start.elapsed();
    eprintln!("Trees: {}, Time: {:?}", booster.trees.len(), elapsed);

    // Print tree stats
    let mut total_nodes = 0;
    for t in booster.trees.iter() {
        total_nodes += t.nodes.len();
    }
    eprintln!("Total nodes across all trees: {}", total_nodes);
    eprintln!(
        "Avg nodes per tree: {:.1}",
        total_nodes as f64 / booster.trees.len() as f64
    );
}
