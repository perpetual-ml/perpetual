use super::tree::Tree;
use crate::{utils::odds, Matrix};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

impl Tree {
    pub fn predict_contributions_row_probability_change(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        missing: &f64,
        current_logodds: f64,
    ) -> f64 {
        contribs[contribs.len() - 1] +=
            odds(current_logodds + self.nodes.get(&0).unwrap().weight_value as f64) - odds(current_logodds);
        let mut node_idx = 0;
        let mut lo = current_logodds;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            let node_odds = odds(node.weight_value as f64 + current_logodds);
            if node.is_leaf {
                lo += node.weight_value as f64;
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let child_odds = odds(self.nodes.get(&child_idx).unwrap().weight_value as f64 + current_logodds);
            let delta = child_odds - node_odds;
            contribs[node.split_feature] += delta;
            node_idx = child_idx;
        }
        lo
    }

    // Branch average difference predictions
    pub fn predict_contributions_row_midpoint_difference(&self, row: &[f64], contribs: &mut [f64], missing: &f64) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let child = &self.nodes.get(&child_idx).unwrap();
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[&node.right_child]
            } else {
                &self.nodes[&node.left_child]
            };
            let mid = (child.weight_value * child.hessian_sum + other_child.weight_value * other_child.hessian_sum)
                / (child.hessian_sum + other_child.hessian_sum);
            let delta = child.weight_value - mid;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // Branch difference predictions.
    pub fn predict_contributions_row_branch_difference(&self, row: &[f64], contribs: &mut [f64], missing: &f64) {
        // Bias term is left as 0.

        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            //       p
            //    / | \
            //   l  m  r
            //
            // where l < r and we are going down r
            // The contribution for a would be r - l.

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let other_child = if child_idx == node.left_child {
                &self.nodes[&node.right_child]
            } else {
                &self.nodes[&node.left_child]
            };
            let delta = self.nodes.get(&child_idx).unwrap().weight_value - other_child.weight_value;
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    // How does the travelled childs weight change relative to the
    // mode branch.
    pub fn predict_contributions_row_mode_difference(&self, row: &[f64], contribs: &mut [f64], missing: &f64) {
        // Bias term is left as 0.
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            }

            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            // If we are going down the missing branch, do nothing and leave
            // it at zero.
            if node.has_missing_branch() && child_idx == node.missing_node {
                node_idx = child_idx;
                continue;
            }
            let left_node = &self.nodes.get(&node.left_child).unwrap();
            let right_node = &self.nodes.get(&node.right_child).unwrap();
            let child_weight = self.nodes.get(&child_idx).unwrap().weight_value;

            let delta = if left_node.hessian_sum == right_node.hessian_sum {
                0.
            } else if left_node.hessian_sum > right_node.hessian_sum {
                child_weight - left_node.weight_value
            } else {
                child_weight - right_node.weight_value
            };
            contribs[node.split_feature] += delta as f64;
            node_idx = child_idx;
        }
    }

    pub fn predict_contributions_row_weight(&self, row: &[f64], contribs: &mut [f64], missing: &f64) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += self.nodes.get(&0).unwrap().weight_value as f64;
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = self.nodes.get(&node_idx).unwrap().weight_value as f64;
            let child_weight = self.nodes.get(&child_idx).unwrap().weight_value as f64;
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_weight(&self, data: &Matrix<f64>, contribs: &mut [f64], missing: &f64) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| self.predict_contributions_row_weight(&data.get_row(*row), contribs, missing))
    }

    pub fn predict_contributions_row_average(
        &self,
        row: &[f64],
        contribs: &mut [f64],
        weights: &HashMap<usize, f64>,
        missing: &f64,
    ) {
        // Add the bias term first...
        contribs[contribs.len() - 1] += weights[&0];
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            }
            // Get change of weight given child's weight.
            let child_idx = node.get_child_idx(&row[node.split_feature], missing);
            let node_weight = weights[&node_idx];
            let child_weight = weights[&child_idx];
            let delta = child_weight - node_weight;
            contribs[node.split_feature] += delta;
            node_idx = child_idx
        }
    }

    pub fn predict_contributions_average(
        &self,
        data: &Matrix<f64>,
        contribs: &mut [f64],
        weights: &HashMap<usize, f64>,
        missing: &f64,
    ) {
        // There needs to always be at least 2 trees
        data.index
            .par_iter()
            .zip(contribs.par_chunks_mut(data.cols + 1))
            .for_each(|(row, contribs)| {
                self.predict_contributions_row_average(&data.get_row(*row), contribs, weights, missing)
            })
    }

    fn predict_row(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                return node.weight_value as f64;
            } else {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
            }
        }
    }

    pub fn predict_row_from_row_slice(&self, row: &[f64], missing: &f64) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                return node.weight_value as f64;
            } else {
                node_idx = node.get_child_idx(&row[node.split_feature], missing);
            }
        }
    }

    fn predict_single_threaded(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index.iter().map(|i| self.predict_row(data, *i, missing)).collect()
    }

    fn predict_parallel(&self, data: &Matrix<f64>, missing: &f64) -> Vec<f64> {
        data.index
            .par_iter()
            .map(|i| self.predict_row(data, *i, missing))
            .collect()
    }

    pub fn predict(&self, data: &Matrix<f64>, parallel: bool, missing: &f64) -> Vec<f64> {
        if parallel {
            self.predict_parallel(data, missing)
        } else {
            self.predict_single_threaded(data, missing)
        }
    }

    fn predict_nodes_row(&self, data: &Matrix<f64>, row: usize, missing: &f64) -> HashSet<usize> {
        let mut node_idx = 0;
        let mut v = HashSet::new();
        v.insert(node_idx);
        loop {
            let node = &self.nodes.get(&node_idx).unwrap();
            if node.is_leaf {
                break;
            } else {
                node_idx = node.get_child_idx(data.get(row, node.split_feature), missing);
                v.insert(node_idx);
            }
        }
        v
    }

    fn predict_nodes_single_threaded(&self, data: &Matrix<f64>, missing: &f64) -> Vec<HashSet<usize>> {
        data.index
            .iter()
            .map(|i| self.predict_nodes_row(data, *i, missing))
            .collect()
    }

    fn predict_nodes_parallel(&self, data: &Matrix<f64>, missing: &f64) -> Vec<HashSet<usize>> {
        data.index
            .par_iter()
            .map(|i| self.predict_nodes_row(data, *i, missing))
            .collect()
    }

    pub fn predict_nodes(&self, data: &Matrix<f64>, parallel: bool, missing: &f64) -> Vec<HashSet<usize>> {
        if parallel {
            self.predict_nodes_parallel(data, missing)
        } else {
            self.predict_nodes_single_threaded(data, missing)
        }
    }
}
