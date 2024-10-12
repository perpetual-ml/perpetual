use crate::{tree::Tree, utils::is_missing};

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::binning::bin_matrix;
    use crate::constraints::ConstraintMap;
    use crate::data::Matrix;
    use crate::histogram::{NodeHistogram, NodeHistogramOwned};
    use crate::objective::{LogLoss, ObjectiveFunction};
    use crate::splitter::{MissingImputerSplitter, SplitInfo, SplitInfoSlice};
    use crate::tree::Tree;
    use std::fs;

    #[test]
    fn test_partial_dependence() {
        let is_const_hess = false;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = LogLoss::calc_grad_hess(&y, &yhat, None, None);
        let loss = LogLoss::calc_loss(&y, &yhat, None, None);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut g,
            h.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            LogLoss::calc_loss,
            &yhat,
            None,
            None,
            false,
            &mut hist_tree,
            None,
            &split_info_slice,
            n_nodes_alloc,
        );

        let pdp1 = tree_partial_dependence(&tree, 0, 0, 1.0, 1.0, &f64::NAN);
        let pdp2 = tree_partial_dependence(&tree, 0, 0, 2.0, 1.0, &f64::NAN);
        let pdp3 = tree_partial_dependence(&tree, 0, 0, 3.0, 1.0, &f64::NAN);
        println!("{}, {}, {}", pdp1, pdp2, pdp3);
    }
}
