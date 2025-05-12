use crate::{
    errors::PerpetualError,
    node::{Node, NodeType},
    objective_functions::{calc_init_callables, loss_callables},
    tree::tree::Tree,
    Matrix, PerpetualBooster,
};
use std::collections::HashMap;

impl PerpetualBooster {
    /// Remove trees which don't generalize with new data.
    ///
    /// * `data` -  Either a pandas DataFrame, or a 2 dimensional numpy array.
    /// * `y` - Either a pandas Series, or a 1 dimensional numpy array.
    /// * `sample_weight` - Instance weights to use when training the model.
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {
        let calc_loss = loss_callables(&self.objective);

        let old_length = self.trees.len();
        let old_n_nodes: usize = self.trees.iter().map(|t| t.nodes.len()).sum();

        let base_score = calc_init_callables(&self.objective)(y, sample_weight, self.quantile);
        let yhat = vec![base_score; y.len()];
        let init_losses = calc_loss(y, &yhat, sample_weight, self.quantile);
        let init_loss = init_losses.iter().sum::<f32>() / init_losses.len() as f32;

        self.trees.iter_mut().for_each(|t| {
            t.prune_bottom_up(
                data,
                &self.missing,
                calc_loss,
                init_loss,
                y,
                sample_weight,
                self.quantile,
                self.base_score,
            )
        });

        self.trees.retain(|t| t.nodes.len() > 1);

        let new_length = self.trees.len();
        let new_n_nodes: usize = self.trees.iter().map(|t| t.nodes.len()).sum();

        println!(
            "pruning: n_trees: {} -> {}, n_nodes: {} -> {}",
            old_length, new_length, old_n_nodes, new_n_nodes
        );

        Ok(())
    }
}

type LossFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f64>) -> Vec<f32>;

impl Tree {
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        calc_loss: LossFn,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f64>,
        base_score: f64,
    ) {
        let old_length = self.nodes.len();
        // loss values for each node
        let mut node_losses: HashMap<usize, Vec<f32>> =
            self.nodes.iter().map(|(k, _v)| (k.clone(), Vec::new())).collect();

        match sample_weight {
            None => {
                data.index.iter().for_each(|i| {
                    self.predict_loss(
                        data,
                        *i,
                        missing,
                        calc_loss,
                        base_score,
                        quantile,
                        &mut node_losses,
                        y,
                        None,
                    );
                });
            }
            Some(sw) => {
                data.index.iter().for_each(|i| {
                    self.predict_loss(
                        data,
                        *i,
                        missing,
                        calc_loss,
                        base_score,
                        quantile,
                        &mut node_losses,
                        y,
                        Some(&[sw[*i]]),
                    );
                });
            }
        }

        let node_losses_tuple: HashMap<usize, (f32, usize)> = node_losses
            .into_iter()
            .map(|(k, v)| (k, (v.iter().sum::<f32>(), v.len())))
            .collect();

        let mut node_idx = 0;
        let mut unchecked_nodes = Vec::new();
        unchecked_nodes.push(node_idx);
        while !unchecked_nodes.is_empty() {
            node_idx = unchecked_nodes.pop().unwrap();

            let node = self.nodes.get(&node_idx).unwrap();
            let (loss_sum, loss_count) = node_losses_tuple.get(&node_idx).unwrap();
            let loss_improvement = init_loss - loss_sum / *loss_count as f32;

            if loss_improvement > 0.0 && !node.is_leaf {
                unchecked_nodes.push(node.left_child);
                unchecked_nodes.push(node.right_child);
            } else {
                self.remove_children(node_idx);
            }
        }

        let new_length = self.nodes.len();
        println!("Pruned nodes: {} -> {}", old_length, new_length);
    }

    pub fn predict_loss(
        &self,
        data: &Matrix<f64>,
        row: usize,
        missing: &f64,
        calc_loss: LossFn,
        base_score: f64,
        quantile: Option<f64>,
        node_losses: &mut HashMap<usize, Vec<f32>>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) {
        let mut node_idx = 0;
        loop {
            let node = self.nodes.get(&node_idx).unwrap();
            let loss = calc_loss(
                &[y[row]],
                &[node.weight_value as f64 + base_score],
                sample_weight,
                quantile,
            )[0];
            let nl = node_losses.get_mut(&node_idx).unwrap();
            nl.push(loss);
            if !node.is_leaf {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
            } else {
                break;
            }
        }
    }

    pub fn prune_bottom_up(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        calc_loss: LossFn,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        quantile: Option<f64>,
        base_score: f64,
    ) {
        let old_length = self.nodes.len();
        // loss values for each node
        let mut node_losses: HashMap<usize, Vec<f32>> =
            self.nodes.iter().map(|(k, _v)| (k.clone(), Vec::new())).collect();
        let mut parent_nodes = self
            .nodes
            .values()
            .filter(
                |x| match (self.nodes.get(&x.left_child), self.nodes.get(&x.right_child)) {
                    (Some(left_child), Some(right_child)) => left_child.is_leaf && right_child.is_leaf,
                    _ => false,
                },
            )
            .map(|n| (n.num, n.node_type))
            .collect::<Vec<_>>();

        match sample_weight {
            None => {
                data.index.iter().for_each(|i| {
                    let i_ = *i;
                    let (pred, node_idx) = self.predict_row_and_node_idx(data, *i, missing);
                    let loss = calc_loss(&[y[i_]], &[pred + base_score], None, quantile)[0];
                    let nl = node_losses.get_mut(&node_idx).unwrap();
                    nl.push(loss);
                });
            }
            Some(sw) => {
                data.index.iter().for_each(|i| {
                    let i_ = *i;
                    let (pred, node_idx) = self.predict_row_and_node_idx(data, *i, missing);
                    let loss = calc_loss(&[y[i_]], &[pred + base_score], Some(&[sw[i_]]), quantile)[0];
                    let nl = node_losses.get_mut(&node_idx).unwrap();
                    nl.push(loss);
                });
            }
        }

        let mut node_losses_tuple: HashMap<usize, (f32, usize)> = node_losses
            .into_iter()
            .map(|(k, v)| (k, (v.iter().sum::<f32>(), v.len())))
            .collect();

        while !parent_nodes.is_empty() {
            let (node_num, node_type) = parent_nodes.pop().unwrap();
            let node: &mut Node;
            let sibling_node: &mut Node;
            if node_num != 0 {
                let sibling_num = if node_type == NodeType::Left {
                    node_num + 1
                } else {
                    node_num - 1
                };
                let [node_opt, sibling_node_opt] = self.nodes.get_disjoint_mut([&node_num, &sibling_num]);
                node = node_opt.unwrap();
                sibling_node = sibling_node_opt.unwrap();
            } else {
                let [node_opt, sibling_node_opt] = self.nodes.get_disjoint_mut([&node_num, &1]);
                node = node_opt.unwrap();
                sibling_node = sibling_node_opt.unwrap();
            }
            let left_child = node.left_child;
            let right_child = node.right_child;
            let parent_node = node.parent_node;
            let loss_sum = node_losses_tuple[&left_child].0 + node_losses_tuple[&right_child].0;
            let loss_length = node_losses_tuple[&left_child].1 + node_losses_tuple[&right_child].1;
            let child_loss_avg = loss_sum / loss_length as f32;
            let loss_improvement = init_loss - child_loss_avg;
            if loss_improvement < 0.0 {
                node.is_leaf = true;
                if sibling_node.is_leaf && node.num != 0 {
                    parent_nodes.push((node.parent_node, self.nodes[&parent_node].node_type));
                };
                self.nodes.remove(&left_child);
                self.nodes.remove(&right_child);
                node_losses_tuple.insert(node_num, (loss_sum, loss_length));
            }
        }

        let new_length = self.nodes.len();
        println!("Pruned nodes: {} -> {}", old_length, new_length);
    }

    pub fn predict_row_and_node_idx(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> (f64, usize) {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                return (node.weight_value as f64, node_idx);
            } else {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective_functions::Objective;
    use polars::io::SerReader;
    use polars::prelude::{CsvReadOptions, DataType};
    use std::error::Error;
    use std::sync::Arc;

    #[test]
    fn test_pruning() -> Result<(), Box<dyn Error>> {
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

        let mut model = PerpetualBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_train, &y_train, None)?;

        model.prune(&matrix_test, &y_test, None)?;

        Ok(())
    }
}
