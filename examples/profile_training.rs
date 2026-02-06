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
    let mut data_vec: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().expect("Parse error"))
            .collect();
        y.push(values[8]);
        data_vec.extend_from_slice(&values[0..8]);
    }
    let data = Matrix::new(&data_vec, y.len(), 8);

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
    let mut data_vec: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for line in lines {
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>().expect("Parse error"))
            .collect();
        y.push(values[values.len() - 1]);
        data_vec.extend_from_slice(&values[0..10]);
    }
    let n_cols = data_vec.len() / y.len();
    let data = Matrix::new(&data_vec, y.len(), n_cols);

    println!("\nData: {} rows x {} cols", y.len(), n_cols);

    {
        let mut booster = PerpetualBooster::default().set_objective(Objective::LogLoss);
        booster.fit(&data, &y, None, None).unwrap();
        println!("Trees: {}", booster.get_prediction_trees().len());
        let tree_sizes: Vec<usize> = booster.get_prediction_trees().iter().map(|t| t.nodes.len()).collect();
        println!("Tree sizes: {:?}", tree_sizes);
    }
}
