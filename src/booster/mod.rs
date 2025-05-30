// public modules
pub mod univariate_booster;
pub mod multivariate_booster;
pub mod predict;
pub mod config;

// private modules
mod setters;

// unit-testing for boosters
//
// univariate booster
// multivariate booster
#[cfg(test)]
mod univariate_booster_test {
    
    use std::error::Error;
    use std::sync::Arc;
    use polars::io::SerReader;
    use polars::prelude::{CsvReadOptions, DataType};
    use crate::{Matrix, UnivariateBooster};
    use crate::objective_functions::Objective;
    use crate::booster::config::*;
    use std::collections::HashSet;
    use crate::utils::between;
    use approx::{assert_relative_eq};
    use std::fs;
    use crate::objective_functions::ObjectiveFunction;
    use crate::metrics::Metric;


     #[test]
    fn test_booster_fit() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);

        let mut booster = UnivariateBooster::default().set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_booster_fit_no_fitted_base_score() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance-fare.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);

        let mut booster = UnivariateBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(300)
            .set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

    #[test]
    fn test_tree_save() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, 891, 5);

        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        //let data = Matrix::new(data.get_col(1), 891, 1);
        let mut booster = UnivariateBooster::default()
            .set_max_bin(300)
            .set_base_score(0.5)
            .set_budget(0.3);

        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, true);

        booster.save_booster("resources/model64.json").unwrap();
        let booster2 = UnivariateBooster::load_booster("resources/model64.json").unwrap();
        assert_eq!(booster2.predict(&data, true)[0..10], preds[0..10]);

        // Test with non-NAN missing.
        booster.cfg.missing = 0.0;
        booster.save_booster("resources/modelmissing.json").unwrap();
        let booster3 = UnivariateBooster::load_booster("resources/modelmissing.json").unwrap();
        assert_eq!(booster3.cfg.missing, 0.);
        assert_eq!(booster3.cfg.missing, booster.cfg.missing);
    }

    #[test]
    fn test_gbm_categorical() -> Result<(), Box<dyn Error>> {
        let n_columns = 13;

        let file = fs::read_to_string("resources/titanic_test_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file =
            fs::read_to_string("resources/titanic_test_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let mut booster = UnivariateBooster::default()
            .set_budget(0.1)
            .set_categorical_features(Some(cat_index));

        booster.fit(&data, &y, None).unwrap();

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let probabilities = booster.predict_proba(&data, true);

        let accuracy = probabilities
            .iter()
            .zip(y.iter())
            .map(|(p, y)| if p.round() == *y { 1 } else { 0 })
            .sum::<usize>() as f32
            / y.len() as f32;

        println!("accuracy: {}", accuracy);
        assert!(between(0.76, 0.78, accuracy));

        Ok(())
    }

    #[test]
    fn test_gbm_parallel() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_train = Arc::new(all_names.clone());
        let column_names_test = Arc::new(all_names.clone());

        let df_train = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_train))
            .try_into_reader_with_file_path(Some("resources/cal_housing_train.csv".into()))?
            .finish()
            .unwrap();

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...
        let id_vars_train: Vec<&str> = Vec::new();
        let mdf_train = df_train.unpivot(feature_names.clone(), &id_vars_train)?;
        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_train = Vec::from_iter(
            mdf_train
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_train = Vec::from_iter(
            df_train
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_train = Matrix::new(&data_train, y_train.len(), 8);
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model1 = UnivariateBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_num_threads(Some(1))
            .set_budget(0.1);
        let mut model2 = UnivariateBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_num_threads(Some(2))
            .set_budget(0.1);

        model1.fit(&matrix_test, &y_test, None)?;
        model2.fit(&matrix_test, &y_test, None)?;

        let trees1 = model1.get_prediction_trees();
        let trees2 = model2.get_prediction_trees();
        assert_eq!(trees1.len(), trees2.len());

        let n_leaves1: usize = trees1.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
        let n_leaves2: usize = trees2.iter().map(|t| (t.nodes.len() + 1) / 2).sum();
        assert_eq!(n_leaves1, n_leaves2);

        println!("{}", trees1.last().unwrap());
        println!("{}", trees2.last().unwrap());

        let y_pred1 = model1.predict(&matrix_train, true);
        let y_pred2 = model2.predict(&matrix_train, true);

        let mse1 = y_pred1
            .iter()
            .zip(y_train.iter())
            .map(|(y1, y2)| (y1 - y2) * (y1 - y2))
            .sum::<f64>()
            / y_train.len() as f64;
        let mse2 = y_pred2
            .iter()
            .zip(y_train.iter())
            .map(|(y1, y2)| (y1 - y2) * (y1 - y2))
            .sum::<f64>()
            / y_train.len() as f64;
        assert_relative_eq!(mse1, mse2, max_relative = 0.99);

        Ok(())
    }

    #[test]
    fn test_gbm_sensory() -> Result<(), Box<dyn Error>> {
        let n_columns = 11;
        let iter_limit = 10;

        let file = fs::read_to_string("resources/sensory_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/sensory_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_columns);

        let cat_index = HashSet::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let mut booster = UnivariateBooster::default()
            .set_log_iterations(1)
            .set_objective(Objective::SquaredLoss)
            .set_categorical_features(Some(cat_index))
            .set_iteration_limit(Some(iter_limit))
            .set_memory_limit(Some(0.00003))
            .set_budget(1.0);

        booster.fit(&data, &y, None).unwrap();

        let split_features_test = vec![6, 6, 6, 1, 6, 1, 6, 9, 1, 6];
        let split_gains_test = vec![
            31.172100067138672,
            25.249399185180664,
            20.45199966430664,
            17.50349998474121,
            16.566099166870117,
            14.345199584960938,
            13.418600082397461,
            12.505200386047363,
            12.23270034790039,
            10.869000434875488,
        ];
        for (i, tree) in booster.get_prediction_trees().iter().enumerate() {
            let nodes = &tree.nodes;
            let root_node = nodes.get(&0).unwrap();
            println!("i: {}", i);
            println!("nodes.len: {}", nodes.len());
            println!("root_node.split_feature: {}", root_node.split_feature);
            println!("root_node.split_gain: {}", root_node.split_gain);
            assert_eq!(3, nodes.len());
            assert_eq!(root_node.split_feature, split_features_test[i]);
            assert_relative_eq!(root_node.split_gain, split_gains_test[i], max_relative = 0.99);
        }
        assert_eq!(iter_limit, booster.get_prediction_trees().len());

        let pred_nodes = booster.predict_nodes(&data, true);
        println!("pred_nodes.len: {}", pred_nodes.len());
        assert_eq!(booster.get_prediction_trees().len(), pred_nodes.len());
        assert_eq!(data.rows, pred_nodes[0].len());

        Ok(())
    }

    #[test]
    fn test_booster_fit_subsample() {
        let file =
            fs::read_to_string("resources/contiguous_with_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let data = Matrix::new(&data_vec, 891, 5);
        let mut booster = UnivariateBooster::default()
            .set_max_bin(300)
            .set_base_score(0.5)
            .set_budget(0.3);
        booster.fit(&data, &y, None).unwrap();
        let preds = booster.predict(&data, false);
        let contribs = booster.predict_contributions(&data, ContributionsMethod::Average, false);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        println!("{}", booster.trees[0]);
        println!("{}", booster.trees[0].nodes.len());
        println!("{}", booster.trees.last().unwrap().nodes.len());
        println!("{:?}", &preds[0..10]);
    }

     #[test]
    fn test_huber_loss() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_test = Arc::new(all_names.clone());

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...

        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = UnivariateBooster::default()
            .set_objective(Objective::HuberLoss { delta: Some(1.0) })
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 45);

        Ok(())
    }

     #[test]
    fn test_adaptive_huber_loss() -> Result<(), Box<dyn Error>> {
        let all_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
            "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let column_names_test = Arc::new(all_names.clone());

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(column_names_test))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        // Get data in column major format...

        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(feature_names, &id_vars_test)?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_test = Vec::from_iter(
            df_test
                .column("MedHouseVal")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let matrix_test = Matrix::new(&data_test, y_test.len(), 8);

        // Create booster.
        // To provide parameters generate a default booster, and then use
        // the relevant `set_` methods for any parameters you would like to
        // adjust.
        let mut model = UnivariateBooster::default()
            .set_objective(Objective::AdaptiveHuberLoss { quantile: Some(0.5) } )
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_test, &y_test, None)?;

        let trees = model.get_prediction_trees();
        println!("trees = {}", trees.len());
        assert_eq!(trees.len(), 31);

        Ok(())
    }

    #[test]
    fn test_custom_objective_function() -> Result<(), Box<dyn Error>> {

        // define objective function
        #[derive(Clone)]
        struct CustomSquaredLoss;
        impl ObjectiveFunction for CustomSquaredLoss {

            fn hessian_is_constant(&self) -> bool {
                false // fails if true
            }

            fn calc_loss(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>) -> Vec<f32> {
                y.iter()
                    .zip(yhat)
                    .enumerate()
                    .map(|(idx, (y_i, yhat_i))| {
                        let diff = y_i - yhat_i;
                        let l = diff * diff;
                        match sample_weight {
                            Some(w) => (l * w[idx]) as f32,
                            None => l as f32,
                        }
                    })
                    .collect()
            }

            fn calc_grad_hess(&self, y: &[f64], yhat: &[f64], sample_weight: Option<&[f64]>,) -> (Vec<f32>, Option<Vec<f32>>) {
                let grad: Vec<f32> = y
                    .iter()
                    .zip(yhat)
                    .enumerate()
                    .map(|(idx, (y_i, yhat_i))| {
                        let g = yhat_i - y_i;
                        match sample_weight {
                            Some(w) => (g * w[idx]) as f32,
                            None => g as f32,
                        }
                    })
                    .collect();
                let hess = vec![2.0_f32; y.len()];
                (grad, Some(hess))
            }

            fn calc_init(&self, y: &[f64], sample_weight: Option<&[f64]>) -> f64 {
                match sample_weight {
                    Some(w) => {
                        let sw: f64 = w.iter().sum();
                        y.iter().enumerate().map(|(i, y_i)| y_i * w[i]).sum::<f64>() / sw
                    }
                    None => y.iter().sum::<f64>() / y.len() as f64,
                }
            }

            fn default_metric(&self) -> Metric {
                Metric::RootMeanSquaredError
            }

            fn constant_hessian(&self, weights_flag: bool) -> bool {
                if weights_flag {
                    false
                } else {
                    true
                }
            }
        }


         let all_names = [
        "MedInc".to_string(),
        "HouseAge".to_string(),
        "AveRooms".to_string(),
        "AveBedrms".to_string(),
        "Population".to_string(),
        "AveOccup".to_string(),
        "Latitude".to_string(),
        "Longitude".to_string(),
        "MedHouseVal".to_string(),
        ];

        let feature_names = [
            "MedInc".to_string(),
            "HouseAge".to_string(),
            "AveRooms".to_string(),
            "AveBedrms".to_string(),
            "Population".to_string(),
            "AveOccup".to_string(),
            "Latitude".to_string(),
            "Longitude".to_string(),
        ];

        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(Arc::new(all_names.clone())))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()?;

        
        let id_vars: Vec<&str> = Vec::new();
        let mdf = df.unpivot(feature_names.to_vec(), &id_vars)?;

        let data: Vec<f64> = mdf
            .select_at_idx(1)
            .expect("Invalid column")
            .f64()? // Returns Result<Float64Chunked>
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();

        let y: Vec<f64> = df
            .column("MedHouseVal")?
            .cast(&DataType::Float64)?
            .f64()? // Returns Result<Float64Chunked>
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();

        let matrix = Matrix::new(&data, y.len(), 8);

        // define booster with custom loss 
        // function
        let mut custom_booster = UnivariateBooster::default()
        .set_objective(Objective::function(CustomSquaredLoss))
        .set_max_bin(1)
        .set_budget(0.1)
        .set_stopping_rounds(Some(1));

        // define booster with builting
        // squared loss
        let mut booster = UnivariateBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_max_bin(1)
        .set_budget(0.1)
        .set_stopping_rounds(Some(1));

        // fit
        booster.fit(&matrix, &y, None)?;
        custom_booster.fit(&matrix, &y, None)?;

        // // predict values
        let custom_prediction = custom_booster.predict(&matrix, false);
        let booster_prediction = booster.predict(&matrix, false);

        assert_relative_eq!(custom_prediction[..5], booster_prediction[..5], max_relative = 1e-6);

        Ok(())
    }

}



#[cfg(test)]
mod multivariate_booster_test {
    
    use crate::{utils::between, MultivariateBooster};
    use polars::{
        io::SerReader,
        prelude::{CsvReadOptions, DataType},
    };
    use std::error::Error;
    use crate::Matrix;
    use crate::objective_functions::Objective;

    #[test]
    fn test_multi_output_booster() -> Result<(), Box<dyn Error>> {
        let n_classes = 7;
        let n_columns = 54;
        let n_rows = 1000;
        let max_bin = 10;

        let mut features: Vec<&str> = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area_0",
            "Wilderness_Area_1",
            "Wilderness_Area_2",
            "Wilderness_Area_3",
        ]
        .to_vec();

        let soil_types = (0..40).map(|i| format!("{}_{}", "Soil_Type", i)).collect::<Vec<_>>();
        let s_types = soil_types.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        features.extend(s_types);

        let mut features_and_target = features.clone();
        features_and_target.push("Cover_Type");

        let features_and_target_arc = features_and_target
            .iter()
            .map(|s| String::from(s.to_owned()))
            .collect::<Vec<String>>()
            .into();

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(features_and_target_arc))
            .try_into_reader_with_file_path(Some("resources/cover_types_test.csv".into()))?
            .finish()
            .unwrap()
            .head(Some(n_rows));

        // Get data in column major format...
        let id_vars_test: Vec<&str> = Vec::new();
        let mdf_test = df_test.unpivot(&features, &id_vars_test)?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        let y_test = Vec::from_iter(
            df_test
                .column("Cover_Type")?
                .cast(&DataType::Float64)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );

        // Create Matrix from ndarray.
        let data = Matrix::new(&data_test, y_test.len(), n_columns);

        let mut y_vec: Vec<Vec<f64>> = Vec::new();
        for i in 0..n_classes {
            y_vec.push(
                y_test
                    .iter()
                    .map(|y| if (*y as usize) == (i + 1) { 1.0 } else { 0.0 })
                    .collect(),
            );
        }
        let y_data = y_vec.into_iter().flatten().collect::<Vec<f64>>();
        let y = Matrix::new(&y_data, y_test.len(), n_classes);

        let mut booster = MultivariateBooster::default()
            .set_objective(Objective::LogLoss)
            .set_max_bin(max_bin)
            .set_n_boosters(n_classes)
            .set_budget(0.1)
            .set_timeout(Some(60.0));

        println!("The number of boosters: {:?}", booster.get_boosters().len());
        assert!(booster.get_boosters().len() == n_classes);

        booster.fit(&data, &y, None).unwrap();

        let probas = booster.predict_proba(&data, true);

        assert!(between(0.999, 1.001, probas[0..n_classes].iter().sum::<f64>() as f32));

        Ok(())
    }
}