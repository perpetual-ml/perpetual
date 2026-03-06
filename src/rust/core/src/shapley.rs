//! Shapley
//!
//! Implementation of the TreeSHAP algorithm for explaining model predictions
//! by calculating feature contributions.
use crate::{node::Node, tree::core::Tree};

#[derive(Debug, Clone, Copy)]
struct PathElement {
    feature_index: usize,
    zero_fraction: f32,
    one_fraction: f32,
    pweight: f32,
}

impl Default for PathElement {
    fn default() -> Self {
        Self {
            feature_index: 0,
            zero_fraction: 0.,
            one_fraction: 0.,
            pweight: 0.,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PathList {
    paths: Vec<PathElement>,
}

impl PathList {
    fn get_element(&mut self, i: usize) -> &PathElement {
        if i == self.paths.len() {
            self.paths.push(PathElement::default());
            &self.paths[i]
        } else {
            // This will panic for us, if we are out of bounds.
            &self.paths[i]
        }
    }
    fn get_element_mut(&mut self, i: usize) -> &mut PathElement {
        if i == self.paths.len() {
            self.paths.push(PathElement::default());
            &mut self.paths[i]
        } else {
            // This will panic for us, if we are out of bounds.
            &mut self.paths[i]
        }
    }
    // fn with_capacity(capacity: usize) -> PathList {
    //     PathList {
    //         paths: Vec::with_capacity(capacity),
    //     }
    // }
    // fn with_empty(l: usize) -> PathList {
    //     PathList {
    //         paths: vec![PathElement::default(); l],
    //     }
    // }
}

fn extend_path(
    unique_path: &mut PathList,
    unique_depth: usize,
    zero_fraction: f32,
    one_fraction: f32,
    feature_index: usize,
) {
    unique_path.get_element_mut(unique_depth).feature_index = feature_index;
    unique_path.get_element_mut(unique_depth).zero_fraction = zero_fraction;
    unique_path.get_element_mut(unique_depth).one_fraction = one_fraction;
    unique_path.get_element_mut(unique_depth).pweight = if unique_depth == 0 { 1.0 } else { 0.0 };
    for i in (0..unique_depth).rev() {
        unique_path.get_element_mut(i + 1).pweight +=
            (one_fraction * unique_path.get_element(i).pweight * (i + 1) as f32) / (unique_depth + 1) as f32;
        unique_path.get_element_mut(i).pweight =
            (zero_fraction * unique_path.get_element(i).pweight * (unique_depth - i) as f32)
                / (unique_depth + 1) as f32;
    }
}

fn unwind_path(unique_path: &mut PathList, unique_depth: usize, path_index: usize) {
    let one_fraction = unique_path.get_element(path_index).one_fraction;
    let zero_fraction = unique_path.get_element(path_index).zero_fraction;
    let mut next_one_portion = unique_path.get_element(unique_depth).pweight;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0. {
            let tmp = unique_path.get_element(i).pweight;
            unique_path.get_element_mut(i).pweight =
                (next_one_portion * (unique_depth + 1) as f32) / ((i + 1) as f32 * one_fraction);
            next_one_portion = tmp
                - (unique_path.get_element(i).pweight * zero_fraction * (unique_depth - i) as f32)
                    / (unique_depth + 1) as f32;
        } else {
            unique_path.get_element_mut(i).pweight = (unique_path.get_element(i).pweight * (unique_depth + 1) as f32)
                / (zero_fraction * (unique_depth - i) as f32);
        }
    }
    for i in path_index..unique_depth {
        unique_path.get_element_mut(i).feature_index = unique_path.get_element(i + 1).feature_index;
        unique_path.get_element_mut(i).zero_fraction = unique_path.get_element(i + 1).zero_fraction;
        unique_path.get_element_mut(i).one_fraction = unique_path.get_element(i + 1).one_fraction;
    }
}

fn unwound_path_sum(unique_path: &mut PathList, unique_depth: usize, path_index: usize) -> f32 {
    let one_fraction = unique_path.get_element(path_index).one_fraction;
    let zero_fraction = unique_path.get_element(path_index).zero_fraction;
    let mut next_one_portion = unique_path.get_element(unique_depth).pweight;
    let mut total = 0.0;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = (next_one_portion * (unique_depth + 1) as f32) / ((i + 1) as f32 * one_fraction);
            total += tmp;
            next_one_portion = unique_path.get_element(i).pweight
                - tmp * zero_fraction * ((unique_depth - i) as f32 / (unique_depth + 1) as f32);
        } else if zero_fraction != 0.0 {
            total += (unique_path.get_element(i).pweight / zero_fraction)
                / ((unique_depth - i) as f32 / (unique_depth + 1) as f32);
        } else if unique_path.get_element(i).pweight != 0.0 {
            panic!("Unique path {} must have zero weight", i);
        }
    }
    total
}

fn get_hot_cold_children(next_node_idx: usize, node: &Node) -> Vec<usize> {
    if node.has_missing_branch() {
        // we know there will be 3 children if there is a missing branch.
        if next_node_idx == node.right_child {
            vec![node.right_child, node.left_child, node.missing_node]
        } else if next_node_idx == node.left_child {
            vec![node.left_child, node.right_child, node.missing_node]
        } else {
            vec![node.missing_node, node.left_child, node.right_child]
        }
    } else if next_node_idx == node.right_child {
        vec![node.right_child, node.left_child]
    } else {
        vec![node.left_child, node.right_child]
    }
}

#[allow(clippy::too_many_arguments)]
fn tree_shap(
    tree: &Tree,
    row: &[f64],
    contribs: &mut [f64],
    node_index: usize,
    mut unique_depth: usize,
    mut unique_path: PathList,
    parent_zero_fraction: f32,
    parent_one_fraction: f32,
    parent_feature_index: usize,
    missing: &f64,
) {
    let node = &tree.nodes[&node_index];
    extend_path(
        &mut unique_path,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    );
    if node.is_leaf {
        for i in 1..(unique_depth + 1) {
            let w = unwound_path_sum(&mut unique_path, unique_depth, i);
            let el = unique_path.get_element(i);
            contribs[el.feature_index] += f64::from(w * (el.one_fraction - el.zero_fraction) * node.weight_value);
        }
    } else {
        let next_node_idx = node.get_child_idx(&row[node.split_feature], missing);
        let hot_cold_children = get_hot_cold_children(next_node_idx, node);
        let mut incoming_zero_fraction = 1.0;
        let mut incoming_one_fraction = 1.0;

        let mut path_index = 0;
        while path_index <= unique_depth {
            if unique_path.get_element(path_index).feature_index == node.split_feature {
                break;
            }
            path_index += 1;
        }

        if path_index != (unique_depth + 1) {
            incoming_zero_fraction = unique_path.get_element(path_index).zero_fraction;
            incoming_one_fraction = unique_path.get_element(path_index).one_fraction;
            unwind_path(&mut unique_path, unique_depth, path_index);
            unique_depth -= 1;
        }

        for (i, n_idx) in hot_cold_children.into_iter().enumerate() {
            let zero_fraction = (tree.nodes[&n_idx].hessian_sum / node.hessian_sum) * incoming_zero_fraction;
            let onf = if i == 0 { incoming_one_fraction } else { 0. };
            tree_shap(
                tree,
                row,
                contribs,
                n_idx,
                unique_depth + 1,
                unique_path.clone(),
                zero_fraction,
                onf,
                node.split_feature,
                missing,
            )
        }
    }
}

pub fn predict_contributions_row_shapley(tree: &Tree, row: &[f64], contribs: &mut [f64], missing: &f64) {
    contribs[contribs.len() - 1] += tree.get_average_leaf_weights(0);
    tree_shap(
        tree,
        row,
        contribs,
        0,
        0,
        PathList::default(),
        1.,
        1.,
        row.len() + 100,
        missing,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::core::{Tree, TreeStopper};
    use std::collections::HashMap;

    #[test]
    fn test_tree_shapley_simple() {
        let mut nodes = HashMap::new();
        // Root node: split on feature 0 at value 5.0
        nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: 0.0,
                hessian_sum: 10.0,
                split_value: 5.0,
                split_feature: 0,
                split_gain: 1.0,
                missing_node: 1,
                left_child: 1,
                right_child: 2,
                is_leaf: false,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        // Left child: leaf with weight 1.0
        nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 1.0,
                hessian_sum: 5.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 1,
                left_child: 1,
                right_child: 1,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        // Right child: leaf with weight 2.0
        nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: 2.0,
                hessian_sum: 5.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 2,
                left_child: 2,
                right_child: 2,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );

        let tree = Tree {
            nodes,
            stopper: TreeStopper::MaxNodes,
            depth: 1,
            n_leaves: 2,
            leaf_bounds: Vec::new(),
            train_index: Vec::new(),
        };

        let row = vec![1.0]; // Should go left (1.0 < 5.0)
        let mut contribs = vec![0.0; 2]; // feature 0 + bias
        predict_contributions_row_shapley(&tree, &row, &mut contribs, &f64::NAN);

        // Prediction should be 1.0
        // Bias should be average weight: (1.0 * 5.0 + 2.0 * 5.0) / 10.0 = 1.5
        // Contrib 0 should be 1.0 - 1.5 = -0.5
        assert_eq!(contribs[1], 1.5); // Bias
        assert_eq!(contribs[0], -0.5); // Feature 0
        assert_eq!(contribs[0] + contribs[1], 1.0);

        let row2 = vec![6.0]; // Should go right (6.0 > 5.0)
        let mut contribs2 = vec![0.0; 2];
        predict_contributions_row_shapley(&tree, &row2, &mut contribs2, &f64::NAN);
        assert_eq!(contribs2[1], 1.5); // Bias
        assert_eq!(contribs2[0], 0.5); // Feature 0
        assert_eq!(contribs2[0] + contribs2[1], 2.0);
    }

    #[test]
    fn test_tree_shapley_complex() {
        let mut nodes = HashMap::new();
        // Root: split on feature 0 at 5.0
        nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: 0.0,
                hessian_sum: 10.0,
                split_value: 5.0,
                split_feature: 0,
                split_gain: 1.0,
                missing_node: 2,
                left_child: 1,
                right_child: 2,
                is_leaf: false,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        // Left: split on feature 1 at 10.0
        nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 0.0,
                hessian_sum: 6.0,
                split_value: 10.0,
                split_feature: 1,
                split_gain: 1.0,
                missing_node: 4,
                left_child: 3,
                right_child: 4,
                is_leaf: false,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        // Right: leaf weight 3.0
        nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: 3.0,
                hessian_sum: 4.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 2,
                left_child: 2,
                right_child: 2,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        // Node 3: leaf weight 1.0 (on feature 1 left)
        nodes.insert(
            3,
            Node {
                num: 3,
                weight_value: 1.0,
                hessian_sum: 2.0,
                split_value: 0.0,
                split_feature: 1,
                split_gain: 0.0,
                missing_node: 3,
                left_child: 3,
                right_child: 3,
                is_leaf: true,
                parent_node: 1,
                left_cats: None,
                stats: None,
            },
        );
        // Node 4: leaf weight 2.0 (on feature 1 right)
        nodes.insert(
            4,
            Node {
                num: 4,
                weight_value: 2.0,
                hessian_sum: 4.0,
                split_value: 0.0,
                split_feature: 1,
                split_gain: 0.0,
                missing_node: 4,
                left_child: 4,
                right_child: 4,
                is_leaf: true,
                parent_node: 1,
                left_cats: None,
                stats: None,
            },
        );

        let tree = Tree {
            nodes,
            stopper: TreeStopper::MaxNodes,
            depth: 2,
            n_leaves: 3,
            leaf_bounds: Vec::new(),
            train_index: Vec::new(),
        };

        let row = vec![1.0, 1.0]; // Goes to Node 3 (weight 1.0)
        let mut contribs = vec![0.0; 3]; // f0, f1, bias
        predict_contributions_row_shapley(&tree, &row, &mut contribs, &f64::NAN);

        // Bias = average leaf weight
        // Node 3: 1.0 * 2.0 = 2.0
        // Node 4: 2.0 * 4.0 = 8.0
        // Node 2: 3.0 * 4.0 = 12.0
        // Total = (2 + 8 + 12) / 10 = 22 / 10 = 2.2
        assert!((contribs[2] - 2.2).abs() < 1e-7);
        assert!((contribs[0] + contribs[1] + contribs[2] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_tree_shapley_missing() {
        let mut nodes = HashMap::new();
        // Root: split on feature 0 at 5.0, missing branch to node 3
        nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: 0.0,
                hessian_sum: 10.0,
                split_value: 5.0,
                split_feature: 0,
                split_gain: 1.0,
                missing_node: 3,
                left_child: 1,
                right_child: 2,
                is_leaf: false,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 1.0,
                hessian_sum: 4.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 1,
                left_child: 1,
                right_child: 1,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: 2.0,
                hessian_sum: 4.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 2,
                left_child: 2,
                right_child: 2,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            3,
            Node {
                num: 3,
                weight_value: 3.0,
                hessian_sum: 2.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 3,
                left_child: 3,
                right_child: 3,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );

        let tree = Tree {
            nodes,
            stopper: TreeStopper::MaxNodes,
            depth: 1,
            n_leaves: 3,
            leaf_bounds: Vec::new(),
            train_index: Vec::new(),
        };

        let row = vec![f64::NAN]; // Goes to missing node 3
        let mut contribs = vec![0.0; 2];
        predict_contributions_row_shapley(&tree, &row, &mut contribs, &f64::NAN);

        // Bias = (1.0*4 + 2.0*4 + 3.0*2)/10 = (4 + 8 + 6)/10 = 1.8
        assert!((contribs[1] - 1.8).abs() < 1e-7);
        assert!((contribs[0] + contribs[1] - 3.0).abs() < 1e-7);
    }
}
