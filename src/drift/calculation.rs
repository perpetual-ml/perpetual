use crate::booster::core::PerpetualBooster;
use crate::data::Matrix;
use crate::drift::stats::chi2_contingency_2x2;
use std::collections::HashMap;

/// Calculate drift metrics based on the model's node distributions.
///
/// # Arguments
///
/// * `booster` - The trained booster model.
/// * `data` - The new data to evaluate for drift.
/// * `drift_type` - Type of drift: "data" (multivariate) or "concept".
/// * `parallel` - Whether to use parallel processing for predictions.
pub fn calculate_drift(booster: &PerpetualBooster, data: &Matrix<f64>, drift_type: &str, parallel: bool) -> f32 {
    let trees = booster.get_prediction_trees();
    if trees.is_empty() {
        return 0.0;
    }

    // 1. Get node predictions for the new data
    let node_preds = booster.predict_nodes(data, parallel);

    calculate_drift_from_nodes(trees, &node_preds, drift_type)
}

/// Calculate drift metrics from columnar data.
pub fn calculate_drift_columnar(
    booster: &PerpetualBooster,
    data: &crate::data::ColumnarMatrix<f64>,
    drift_type: &str,
    parallel: bool,
) -> f32 {
    let trees = booster.get_prediction_trees();
    if trees.is_empty() {
        return 0.0;
    }

    // 1. Get node predictions for the new data
    let node_preds = booster.predict_nodes_columnar(data, parallel);

    calculate_drift_from_nodes(trees, &node_preds, drift_type)
}

fn calculate_drift_from_nodes(
    trees: &[crate::tree::core::Tree],
    node_preds: &[Vec<std::collections::HashSet<usize>>],
    drift_type: &str,
) -> f32 {
    // 2. Aggregate node counts for the new data
    // node_preds is [tree][sample][nodes]
    let mut new_node_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); trees.len()];
    for (tree_idx, tree_results) in node_preds.iter().enumerate() {
        if tree_idx >= trees.len() {
            continue;
        }
        for sample_nodes in tree_results {
            for &node_idx in sample_nodes {
                *new_node_counts[tree_idx].entry(node_idx).or_insert(0) += 1;
            }
        }
    }

    let mut drift_stats = Vec::new();

    // 3. Compare with training counts stored in nodes
    for (tree_idx, tree) in trees.iter().enumerate() {
        for (&_node_idx, node) in &tree.nodes {
            if node.is_leaf {
                continue;
            }

            // For drift detection, we look at the children of this node
            let left_idx = node.left_child;
            let right_idx = node.right_child;

            let left_node = match tree.nodes.get(&left_idx) {
                Some(n) => n,
                None => continue,
            };
            let right_node = match tree.nodes.get(&right_idx) {
                Some(n) => n,
                None => continue,
            };

            let left_stats = left_node.stats.as_ref();
            let right_stats = right_node.stats.as_ref();

            if let (Some(l_s), Some(r_s)) = (left_stats, right_stats) {
                // Determine if we should include this node based on drift_type
                let should_include = match drift_type {
                    "data" => true,                                       // multivariate data drift: all nodes
                    "concept" => left_node.is_leaf || right_node.is_leaf, // concept drift: parents of leaves
                    _ => false,
                };

                if should_include {
                    let train_l = l_s.count as f64;
                    let train_r = r_s.count as f64;
                    let new_l = *new_node_counts[tree_idx].get(&left_idx).unwrap_or(&0) as f64;
                    let new_r = *new_node_counts[tree_idx].get(&right_idx).unwrap_or(&0) as f64;

                    if (train_l > 0.0 || train_r > 0.0) && (new_l > 0.0 || new_r > 0.0) {
                        let stat = chi2_contingency_2x2(train_l, new_l, train_r, new_r);
                        if !stat.is_nan() {
                            drift_stats.push(stat);
                        }
                    }
                }
            }
        }
    }

    if drift_stats.is_empty() {
        0.0
    } else {
        (drift_stats.iter().sum::<f64>() / drift_stats.len() as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_drift() {
        use crate::booster::core::PerpetualBooster;
        use crate::objective::Objective;
        let mut booster = PerpetualBooster::default();
        booster.cfg.objective = Objective::SquaredLoss;
        booster.cfg.save_node_stats = true;

        let data_vec = vec![1.0, 2.0, 3.0, 4.0];
        let data = Matrix::new(&data_vec, 4, 1);
        let target = vec![0.0, 1.0, 0.0, 1.0];

        booster.fit(&data, &target, None, None).unwrap();

        let drift = calculate_drift(&booster, &data, "data", false);
        assert!(drift >= 0.0);
    }
}
