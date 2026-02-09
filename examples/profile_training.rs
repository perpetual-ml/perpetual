use perpetual::data::Matrix;
use perpetual::objective_functions::Objective;
use perpetual::PerpetualBooster;
use std::fs;

fn main() {
    // Cal housing benchmark
    let file_content =
        fs::read_to_string("resources/cal_housing_train.csv").expect("Something went wrong reading the file");
    let mut lines = file_content.lines();
    lines.next();
    let n_features = 8;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();
    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().expect("Parse error"))
            .collect();
        y.push(values[n_features]);
        for j in 0..n_features {
            columns[j].push(values[j]);
        }
    }
    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);

    println!("Data: {} rows x {} cols", y.len(), 8);

    // Run once with log_iterations to see tree count
    {
        let mut booster = PerpetualBooster::default().set_objective(Objective::SquaredLoss);
        booster.fit(&data, &y, None, None).unwrap();
        println!("Trees: {}", booster.get_prediction_trees().len());
        let tree_sizes: Vec<usize> = booster.get_prediction_trees().iter().map(|t| t.nodes.len()).collect();
        println!("Tree sizes: {:?}", tree_sizes);
    }
    let file_content =
        fs::read_to_string("resources/cover_types_train.csv").expect("Something went wrong reading the file");
    let lines = file_content.lines().skip(1).take(2000);
    let n_features = 10;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_features];
    let mut y: Vec<f64> = Vec::new();
    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>().expect("Parse error"))
            .collect();
        y.push(values[values.len() - 1]);
        for j in 0..n_features {
            columns[j].push(values[j]);
        }
    }
    let data_vec: Vec<f64> = columns.into_iter().flatten().collect();
    let data = Matrix::new(&data_vec, y.len(), n_features);

    println!("\nData: {} rows x {} cols", y.len(), n_features);

    {
        let mut booster = PerpetualBooster::default().set_objective(Objective::LogLoss);
        booster.fit(&data, &y, None, None).unwrap();
        println!("Trees: {}", booster.get_prediction_trees().len());
        let tree_sizes: Vec<usize> = booster.get_prediction_trees().iter().map(|t| t.nodes.len()).collect();
        println!("Tree sizes: {:?}", tree_sizes);
    }
}
