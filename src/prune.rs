use crate::objective_functions::objective::ObjectiveFunction;
use crate::objective_functions::Objective;
use crate::{decision_tree::tree::Tree, errors::PerpetualError, Matrix, PerpetualBooster};
use std::collections::HashMap;

impl PerpetualBooster {
    /// Remove trees which don't generalize with new data.
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        // initialize objective function
        let objective_fn = &self.cfg.objective;

        let old_length = self.trees.len();
        let old_n_nodes: usize = self.trees.iter().map(|t| t.nodes.len()).sum();

        let base_score = objective_fn.initial_value(y, sample_weight, group);
        let yhat = vec![base_score; y.len()];
        let init_losses = objective_fn.loss(y, &yhat, sample_weight, group);
        let init_loss = init_losses.iter().sum::<f32>() / init_losses.len() as f32;

        for tree in &mut self.trees {
            tree.prune_bottom_up(
                data,
                &self.cfg.missing,
                objective_fn,
                init_loss,
                y,
                sample_weight,
                group,
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

impl Tree {
    /// Top-down prune using per-sample loss
    #[allow(clippy::too_many_arguments)]
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        objective_fn: &Objective,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        base_score: f64,
    ) {
        let old_length = self.nodes.len();
        let mut node_losses: HashMap<usize, Vec<f32>> = self.nodes.keys().map(|&k| (k, Vec::new())).collect();

        for &i in &data.index {
            self.predict_loss(
                data,
                i,
                missing,
                objective_fn,
                base_score,
                &mut node_losses,
                y,
                sample_weight,
                group,
            );
        }

        let node_losses_tuple: HashMap<usize, (f32, usize)> = node_losses
            .into_iter()
            .map(|(k, v)| (k, (v.iter().sum::<f32>(), v.len())))
            .collect();

        let mut unchecked = Vec::new();
        unchecked.push(0);
        while let Some(idx) = unchecked.pop() {
            let node = &self.nodes[&idx];
            let &(sum, cnt) = &node_losses_tuple[&idx];
            let improvement = init_loss - sum / cnt as f32;
            if improvement > 0.0 && !node.is_leaf {
                unchecked.push(node.left_child);
                unchecked.push(node.right_child);
            } else {
                self.remove_children(idx);
            }
        }

        let new_length = self.nodes.len();
        println!("Pruned nodes: {} -> {}", old_length, new_length);
    }

    /// Compute loss at the leaf reached by row
    #[allow(clippy::too_many_arguments)]
    pub fn predict_loss(
        &self,
        data: &Matrix<f64>,
        row: usize,
        missing: &f64,
        objective_fn: &Objective,
        base_score: f64,
        node_losses: &mut HashMap<usize, Vec<f32>>,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) {
        let mut idx = 0;
        loop {
            let node = &self.nodes[&idx];
            let loss = objective_fn.loss(
                &[y[row]],
                &[node.weight_value as f64 + base_score],
                sample_weight,
                group,
            )[0];
            node_losses.get_mut(&idx).unwrap().push(loss);
            if node.is_leaf {
                break;
            }
            idx = node.get_child_idx(data.get(row, node.split_feature), missing);
        }
    }

    /// Bottom-up prune a single tree using loss evaluations
    #[allow(clippy::too_many_arguments)]
    pub fn prune_bottom_up(
        &mut self,
        data: &Matrix<f64>,
        missing: &f64,
        objective_fn: &Objective,
        init_loss: f32,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        base_score: f64,
    ) {
        let old_length = self.nodes.len();
        let mut node_losses: HashMap<usize, Vec<f32>> = self.nodes.keys().map(|&k| (k, Vec::new())).collect();

        for &i in &data.index {
            let (pred, nid) = self.predict_row_and_node_idx(data, i, missing);
            let loss = objective_fn.loss(&[y[i]], &[pred + base_score], sample_weight, group)[0];
            node_losses.get_mut(&nid).unwrap().push(loss);
        }

        let mut parents: Vec<usize> = self
            .nodes
            .values()
            .filter(|n| {
                let l = self.nodes.get(&n.left_child);
                let r = self.nodes.get(&n.right_child);
                matches!((l, r), (Some(lc), Some(rc)) if lc.is_leaf && rc.is_leaf)
            })
            .map(|n| n.num)
            .collect();

        let mut stats: HashMap<usize, (f32, usize)> = node_losses
            .into_iter()
            .map(|(k, v)| (k, (v.iter().sum::<f32>(), v.len())))
            .collect();

        while let Some(num) = parents.pop() {
            let node = &self.nodes[&num];
            let left = node.left_child;
            let right = node.right_child;
            let (sl, cl) = stats[&left];
            let (sr, cr) = stats[&right];
            let sum = sl + sr;
            let cnt = cl + cr;
            let avg = sum / cnt as f32;
            let imp = init_loss - avg;
            if imp < 0.0 {
                let pnum = node.parent_node;
                self.nodes.get_mut(&num).unwrap().is_leaf = true;
                self.nodes.remove(&left);
                self.nodes.remove(&right);
                stats.insert(num, (sum, cnt));
                if num != 0 && self.nodes[&pnum].is_leaf {
                    parents.push(pnum);
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
    use crate::objective_functions::objective::Objective;
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

        model.fit(&matrix_train, &y_train, None, None)?;

        model.prune(&matrix_test, &y_test, None, None)?;

        Ok(())
    }
}
