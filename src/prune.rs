use crate::{
    errors::PerpetualError,
    node::{Node, NodeType},
    tree::tree::Tree,
    Matrix, UnivariateBooster,
};
use std::collections::HashMap;
use crate::objective_functions::{ObjectiveFunction, Objective, CustomObjective};
use crate::objective_functions::{
    gradient_hessian_callables,
    loss_callables,
    calc_init_callables,
    LossFn,
};
use std::sync::Arc;

impl UnivariateBooster {
    /// Remove trees which don't generalize with new data.
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {
        // Pull objective callables once
        let (_gh, calc_loss, calc_init, _hess_const, _metric) =
            if let Some(custom) = &self.cfg.custom_objective {
                (
                    Arc::clone(&custom.grad_hess),
                    Arc::clone(&custom.loss),
                    Arc::clone(&custom.init),
                    custom.hessian_constant,
                    custom.metric,
                )
            } else {
                let inst = self.cfg.objective.instantiate();
                (
                    gradient_hessian_callables(inst.clone()),
                    loss_callables(inst.clone()),
                    calc_init_callables(inst.clone()),
                    inst.hessian_is_constant(),
                    inst.default_metric(),
                )
            };

        let old_length = self.trees.len();
        let old_n_nodes: usize = self.trees.iter().map(|t| t.nodes.len()).sum();

        let base_score = calc_init(y, sample_weight);
        let yhat = vec![base_score; y.len()];
        let init_losses = calc_loss(y, &yhat, sample_weight);
        let init_loss = init_losses.iter().sum::<f32>() / init_losses.len() as f32;

        for tree in &mut self.trees {
            tree.prune_bottom_up(
                data,
                &self.cfg.missing,
                calc_loss.clone(),
                init_loss,
                y,
                sample_weight,
                self.base_score,
            );
        }

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

//type LossFn = fn(&[f64], &[f64], Option<&[f64]>, Option<f64>) -> Vec<f32>;
//use crate::objective_functions::LossFn;

impl Tree {
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        calc_loss: LossFn,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
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

/// Bottom-up prune a single tree using loss evaluations.
    pub fn prune_bottom_up(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        calc_loss: LossFn,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        base_score: f64,
    ) {
        let old_length = self.nodes.len();
        let mut node_losses: HashMap<usize, Vec<f32>> =
            self.nodes.keys().map(|&k| (k, Vec::new())).collect();

        // accumulate losses per node
        for &i in &data.index {
            let (pred, node_idx) = self.predict_row_and_node_idx(data, i, missing);
            let loss = calc_loss(&[y[i]], &[pred + base_score], sample_weight)[0];
            node_losses.get_mut(&node_idx).unwrap().push(loss);
        }

        let mut parent_nodes: Vec<usize> = self
            .nodes
            .values()
            .filter(|n| {
                let left = self.nodes.get(&n.left_child);
                let right = self.nodes.get(&n.right_child);
                matches!((left, right), (Some(l), Some(r)) if l.is_leaf && r.is_leaf)
            })
            .map(|n| n.num)
            .collect();

        let mut node_stats: HashMap<usize, (f32, usize)> = node_losses
            .into_iter()
            .map(|(k, v)| (k, (v.iter().sum::<f32>(), v.len())))
            .collect();

        // iterative bottom-up removal
        while let Some(node_num) = parent_nodes.pop() {
            let node = self.nodes.get(&node_num).unwrap();
            let left = node.left_child;
            let right = node.right_child;
            let &(sum_left, cnt_left) = &node_stats[&left];
            let &(sum_right, cnt_right) = &node_stats[&right];
            let loss_sum = sum_left + sum_right;
            let loss_count = cnt_left + cnt_right;
            let child_avg = loss_sum / loss_count as f32;
            let improvement = init_loss - child_avg;

            if improvement < 0.0 {
                // prune
                let parent_num = node.parent_node;
                self.nodes.get_mut(&node_num).unwrap().is_leaf = true;
                self.nodes.remove(&left);
                self.nodes.remove(&right);
                node_stats.insert(node_num, (loss_sum, loss_count));

                // if parent becomes eligible, re-add
                if node_num != 0 {
                    let par = &self.nodes[&parent_num];
                    if par.is_leaf {
                        parent_nodes.push(parent_num);
                    }
                }
            }
        }

        let new_length = self.nodes.len();
        println!("Pruned nodes: {} -> {}", old_length, new_length);
    }

    fn predict_row_and_node_idx(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> (f64, usize) {
        let mut idx = 0;
        loop {
            let node = &self.nodes[&idx];
            if node.is_leaf {
                return (node.weight_value as f64, idx);
            }
            idx = node.get_child_idx(data.get(row, node.split_feature), missing);
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

        let mut model = UnivariateBooster::default()
            .set_objective(Objective::SquaredLoss)
            .set_max_bin(10)
            .set_budget(0.1);

        model.fit(&matrix_train, &y_train, None)?;

        model.prune(&matrix_test, &y_test, None)?;

        Ok(())
    }
}
