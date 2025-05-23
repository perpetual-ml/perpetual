use crate::{tree::tree::Tree, utils::is_missing};

/// Partial Dependence Calculator
// struct PDCalculator {
//     partial_dependence: f32,
//     base_score: f64,
//     tree_prediction: f64,

// }

fn get_node_cover(tree: &Tree, node_idx: usize) -> f32 {
    tree.nodes[&node_idx].hessian_sum
}

pub fn tree_partial_dependence(
    tree: &Tree,
    node_idx: usize,
    feature: usize,
    value: f64,
    proportion: f32,
    missing: &f64,
) -> f64 {
    let n = &tree.nodes[&node_idx];
    if n.is_leaf {
        f64::from(proportion * n.weight_value)
    } else if n.split_feature == feature {
        let child = if is_missing(&value, missing) {
            n.missing_node
        } else if value < n.split_value {
            n.left_child
        } else {
            n.right_child
        };
        tree_partial_dependence(tree, child, feature, value, proportion, missing)
    } else {
        let left_cover = get_node_cover(tree, n.left_child);
        let right_cover = get_node_cover(tree, n.right_child);
        let missing_cover = if n.has_missing_branch() {
            get_node_cover(tree, n.missing_node)
        } else {
            0.0
        };
        let total_cover = left_cover + right_cover + missing_cover;
        let missing_pd = if n.has_missing_branch() {
            tree_partial_dependence(
                tree,
                n.missing_node,
                feature,
                value,
                proportion * (missing_cover / total_cover),
                missing,
            )
        } else {
            0.
        };
        tree_partial_dependence(
            tree,
            n.left_child,
            feature,
            value,
            proportion * (left_cover / total_cover),
            missing,
        ) + tree_partial_dependence(
            tree,
            n.right_child,
            feature,
            value,
            proportion * (right_cover / total_cover),
            missing,
        ) + missing_pd
    }
}