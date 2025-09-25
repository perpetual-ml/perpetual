use crate::data::Matrix;
use crate::grower::Grower;
use crate::histogram::{update_histogram, NodeHistogram};
use crate::node::{Node, NodeType, SplittableNode};
use crate::objective_functions::objective::ObjectiveFunction;
use crate::objective_functions::Objective;
use crate::partial_dependence::tree_partial_dependence;
use crate::splitter::{SplitInfoSlice, Splitter};
use crate::utils::{fast_f64_sum, gain, gain_const_hess, weight, weight_const_hess};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{self, Display};

#[derive(Deserialize, Serialize, Clone, PartialEq, Debug)]
pub enum TreeStopper {
    Generalization,
    StepSize,
    MaxNodes,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Tree {
    pub nodes: HashMap<usize, Node>,
    pub stopper: TreeStopper,
    pub depth: usize,
    pub n_leaves: usize,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Tree {
            nodes: HashMap::new(),
            stopper: TreeStopper::Generalization,
            depth: 0,
            n_leaves: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit<T: Splitter>(
        &mut self,
        objective_function: &Objective,
        data: &Matrix<u16>,
        mut index: Vec<usize>,
        col_index: &[usize],
        grad: &mut [f32],
        mut hess: Option<&mut [f32]>,
        splitter: &T,
        pool: &ThreadPool,
        target_loss_decrement: Option<f32>,
        loss: &[f32],
        y: &[f64],
        //loss: crate::objective_functions::LossFn,
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        is_const_hess: bool,
        hist_tree: &mut [NodeHistogram],
        cat_index: Option<&HashSet<usize>>,
        split_info_slice: &mut SplitInfoSlice,
        n_nodes_alloc: usize,
    ) {
        let mut n_nodes = 1;
        self.n_leaves = 1;

        let root_hist = unsafe { hist_tree.get_unchecked_mut(0) };
        update_histogram(
            root_hist,
            0,
            index.len(),
            data,
            grad,
            hess.as_deref(),
            &index,
            col_index,
            pool,
            false,
        );

        let root_node = create_root_node(&index, grad, hess.as_deref());
        self.nodes.insert(root_node.num, root_node.as_node(splitter.get_eta()));

        let mut growable = BinaryHeap::<SplittableNode>::default();

        let mut loss_decr = vec![0.0_f32; index.len()];
        let mut loss_decr_avg = 0.0_f32;
        let index_length = index.len() as f32;

        growable.add_node(root_node);
        while !growable.is_empty() {
            // If this will push us over the number of allocated nodes, break.
            if self.nodes.len() > (n_nodes_alloc - 3) {
                self.stopper = TreeStopper::MaxNodes;
                break;
            }

            if let Some(tld) = target_loss_decrement {
                if loss_decr_avg > tld {
                    self.stopper = TreeStopper::StepSize;
                    break;
                }
            }
            // We know there is a value here, because of how the
            // while loop is setup.
            // Grab a splitable node from the stack
            // If we can split it, and update the corresponding
            // tree nodes children.
            let mut node = growable.get_next_node();
            let n_idx = node.num;

            // For max_leaves, subtract 1 from the n_leaves
            // every time we pop from the growable stack
            // then, if we can add two children, add two to
            // n_leaves. If we can't split the node any
            // more, then just add 1 back to n_leaves
            self.n_leaves -= 1;

            let new_nodes = splitter.split_node(
                &n_nodes,
                &mut node,
                &mut index,
                col_index,
                data,
                grad,
                hess.as_deref_mut(),
                pool,
                is_const_hess,
                hist_tree,
                cat_index,
                split_info_slice,
            );

            let n_new_nodes = new_nodes.len();
            if n_new_nodes == 0 {
                self.n_leaves += 1;
            } else {
                // self.nodes[n_idx].make_parent_node(node);
                if let Some(x) = self.nodes.get_mut(&n_idx) {
                    x.make_parent_node(node);
                }
                self.n_leaves += n_new_nodes;
                n_nodes += n_new_nodes;

                let mut y_buffer = None;

                for n in new_nodes {
                    let node = n.as_node(splitter.get_eta());
                    let node_indices = &index[n.start_idx..n.stop_idx];

                    if let Some(_tld) = target_loss_decrement {
                        if group.is_some() {
                            // TODO: this could be more efficient. e.g. if all the nodes indices are
                            // in the same group. Currently we compute the loss for all the indices
                            let y_hat_new = y_buffer.get_or_insert_with(|| vec![0.0; y.len()]);
                            y_hat_new.copy_from_slice(yhat);

                            for &i in node_indices {
                                y_hat_new[i] += node.weight_value as f64;
                            }

                            let loss_new = objective_function.loss(y, y_hat_new, sample_weight, group);
                            for &i in node_indices {
                                loss_decr_avg -= loss_decr[i] / index_length;
                                loss_decr[i] = loss[i] - loss_new[i];
                                loss_decr_avg += loss_decr[i] / index_length;
                            }
                        } else {
                            for i in node_indices.iter() {
                                let _i = *i;
                                let s_weight: Vec<f64>;
                                let s_w = match sample_weight {
                                    Some(sample_weight) => {
                                        s_weight = vec![sample_weight[_i]];
                                        Some(&s_weight[..])
                                    }
                                    None => None,
                                };

                                let yhat_new = yhat[_i] + node.weight_value as f64;
                                let loss_new = objective_function.loss(&[y[_i]], &[yhat_new], s_w, None)[0];
                                loss_decr_avg -= loss_decr[_i] / index_length;
                                loss_decr[_i] = loss[_i] - loss_new;
                                loss_decr_avg += loss_decr[_i] / index_length;
                            }
                        }
                    }

                    self.depth = max(self.depth, node.depth);
                    self.nodes.insert(node.num, node);

                    if !n.is_missing_leaf {
                        growable.add_node(n)
                    }
                }
            }
        }

        // Any final post processing required.
        splitter.clean_up_splits(self);
    }

    pub fn remove_children(&mut self, node_idx: usize) {
        let (_removed_node_idx, removed_node) = self.nodes.remove_entry(&node_idx).unwrap();
        if !removed_node.is_leaf {
            self.remove_children(removed_node.left_child);
            self.remove_children(removed_node.right_child);
        }
    }

    pub fn value_partial_dependence(&self, feature: usize, value: f64, missing: &f64) -> f64 {
        tree_partial_dependence(self, 0, feature, value, 1.0, missing)
    }

    fn distribute_node_leaf_weights(&self, i: usize, weights: &mut HashMap<usize, f64>) -> f64 {
        let node = &self.nodes[&i];
        let mut w = node.weight_value as f64;
        if !node.is_leaf {
            let left_node = &self.nodes[&node.left_child];
            let right_node = &self.nodes[&node.right_child];
            w = left_node.hessian_sum as f64 * self.distribute_node_leaf_weights(node.left_child, weights);
            w += right_node.hessian_sum as f64 * self.distribute_node_leaf_weights(node.right_child, weights);
            // If this a tree with a missing branch.
            if node.has_missing_branch() {
                let missing_node = &self.nodes[&node.missing_node];
                w += missing_node.hessian_sum as f64 * self.distribute_node_leaf_weights(node.missing_node, weights);
            }
            w /= node.hessian_sum as f64;
        }
        weights.insert(i, w);
        w
    }

    pub fn distribute_leaf_weights(&self) -> HashMap<usize, f64> {
        let mut weights = HashMap::new();
        self.distribute_node_leaf_weights(0, &mut weights);
        weights
    }

    pub fn get_average_leaf_weights(&self, i: usize) -> f64 {
        let node = &self.nodes[&i];
        let mut w = node.weight_value as f64;
        if node.is_leaf {
            w
        } else {
            let left_node = &self.nodes[&node.left_child];
            let right_node = &self.nodes[&node.right_child];
            w = left_node.hessian_sum as f64 * self.get_average_leaf_weights(node.left_child);
            w += right_node.hessian_sum as f64 * self.get_average_leaf_weights(node.right_child);
            // If this a tree with a missing branch.
            if node.has_missing_branch() {
                let missing_node = &self.nodes[&node.missing_node];
                w += missing_node.hessian_sum as f64 * self.get_average_leaf_weights(node.missing_node);
            }
            w /= node.hessian_sum as f64;
            w
        }
    }

    fn calc_feature_node_stats<F>(&self, calc_stat: &F, node: &Node, stats: &mut HashMap<usize, (f32, usize)>)
    where
        F: Fn(&Node) -> f32,
    {
        if node.is_leaf {
            return;
        }
        stats
            .entry(node.split_feature)
            .and_modify(|(v, c)| {
                *v += calc_stat(node);
                *c += 1;
            })
            .or_insert((calc_stat(node), 1));
        self.calc_feature_node_stats(calc_stat, &self.nodes[&node.left_child], stats);
        self.calc_feature_node_stats(calc_stat, &self.nodes[&node.right_child], stats);
        if node.has_missing_branch() {
            self.calc_feature_node_stats(calc_stat, &self.nodes[&node.missing_node], stats);
        }
    }

    fn get_node_stats<F>(&self, calc_stat: &F, stats: &mut HashMap<usize, (f32, usize)>)
    where
        F: Fn(&Node) -> f32,
    {
        self.calc_feature_node_stats(calc_stat, &self.nodes[&0], stats);
    }

    pub fn calculate_importance_weight(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|_: &Node| 1., stats);
    }

    pub fn calculate_importance_gain(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.split_gain, stats);
    }

    pub fn calculate_importance_cover(&self, stats: &mut HashMap<usize, (f32, usize)>) {
        self.get_node_stats(&|n: &Node| n.hessian_sum, stats);
    }
}

impl Display for Tree {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut print_buffer: Vec<usize> = vec![0];
        let mut r = String::new();
        while let Some(idx) = print_buffer.pop() {
            let node = &self.nodes[&idx];
            if node.is_leaf {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
            } else {
                r += format!("{}{}\n", "      ".repeat(node.depth).as_str(), node).as_str();
                print_buffer.push(node.right_child);
                print_buffer.push(node.left_child);
                if node.has_missing_branch() {
                    print_buffer.push(node.missing_node);
                }
            }
        }
        write!(f, "{}", r)
    }
}

pub fn create_root_node(index: &[usize], grad: &[f32], hess: Option<&[f32]>) -> SplittableNode {
    let (gradient_sum, hessian_sum) = match hess {
        Some(hess) => (fast_f64_sum(grad), fast_f64_sum(hess)),
        None => (fast_f64_sum(grad), grad.len() as f32),
    };

    let (root_gain, root_weight) = match hess {
        Some(_hess) => (gain(gradient_sum, hessian_sum), weight(gradient_sum, hessian_sum)),
        None => (
            gain_const_hess(gradient_sum, grad.len()),
            weight_const_hess(gradient_sum, grad.len()),
        ),
    };

    SplittableNode::new(
        0,
        root_weight,
        root_gain,
        gradient_sum,
        hessian_sum,
        index.len(),
        0,
        0,
        index.len(),
        f32::NEG_INFINITY,
        f32::INFINITY,
        NodeType::Root,
        HashSet::new(),
        HashSet::new(),
    )
}

// Unit-testing
#[cfg(test)]
mod tests {

    use crate::binning::bin_matrix;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::histogram::NodeHistogramOwned;
    use crate::objective_functions::objective::{Objective, ObjectiveFunction};
    use crate::splitter::{MissingImputerSplitter, SplitInfo};
    use crate::utils::precision_round;

    use std::error::Error;
    use std::fs;

    use crate::decision_tree::tree::Tree;
    use crate::histogram::NodeHistogram;
    use crate::splitter::SplitInfoSlice;
    use crate::Matrix;
    use std::collections::HashSet;

    #[test]
    fn test_tree_fit() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None, None);
        let loss = objective_function.loss(&y, &yhat, None, None);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
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
            //loss_fn.clone(),
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &mut split_info_slice,
            n_nodes_alloc,
        );

        println!("{}", tree);
        let preds = tree.predict(&data, false, &f64::NAN);
        println!("{:?}", &preds[0..10]);
        assert_eq!(27, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_monotone() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None, None);
        let loss = objective_function.loss(&y, &yhat, None, None);
        let is_const_hess = h.is_none();
        println!("GRADIENT -- {:?}", g);

        let data_ = Matrix::new(&data_vec, 891, 5);
        let data = Matrix::new(data_.get_col(1), 891, 1);
        let map = ConstraintMap::from([(0, Constraint::Negative)]);
        let splitter = MissingImputerSplitter::new(0.3, true, map);
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 100, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        //let is_const_hess = false;

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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
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
            //loss_fn.clone(),
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &mut split_info_slice,
            n_nodes_alloc,
        );

        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pred_data_vec.dedup();
        let pred_data = Matrix::new(&pred_data_vec, pred_data_vec.len(), 1);

        let preds = tree.predict(&pred_data, false, &f64::NAN);
        let increasing = preds.windows(2).all(|a| a[0] >= a[1]);
        assert!(increasing);

        let weights = tree.distribute_leaf_weights();

        // Average contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 3), precision_round(j, 3));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_lossguide() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None, None);
        let is_const_hess = h.is_none();
        let loss = objective_function.loss(&y, &yhat, None, None);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        //let is_const_hess = false;

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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
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
            //loss_fn.clone(),
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &mut split_info_slice,
            n_nodes_alloc,
        );

        println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        // assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_categorical() -> Result<(), Box<dyn Error>> {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let n_bins = 256;
        let n_rows = 712;
        let n_columns = 13;

        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (mut grad, mut hess) = objective_function.gradient(&y, &yhat, None, None);
        let is_const_hess = hess.is_none();
        let loss = objective_function.loss(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        //let is_const_hess = false;

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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        let mut tree = Tree::new();
        tree.fit(
            &objective_function,
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut grad,
            hess.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            //loss_fn.clone(),
            &yhat,
            None,
            None,
            false,
            &mut hist_tree,
            Some(&cat_index),
            &mut split_info_slice,
            n_nodes_alloc,
        );
        println!("{}", tree);
        println!("tree.nodes.len: {}", tree.nodes.len());
        println!("data.first: {:?}", data.get_row(10));
        println!("tree.predict: {}", tree.predict(&data, true, &f64::NAN)[10]);
        println!("hist_tree[0]:");
        for (i, item) in hist_tree[10].data.get(3).iter().enumerate() {
            println!("The {}th item is {:?}", i, item);
        }

        let pred_nodes = tree.predict_nodes(&data, true, &f64::NAN);
        println!("pred_nodes.len: {}", pred_nodes.len());
        println!("pred_nodes[0].len: {}", pred_nodes[0].len());
        println!("pred_nodes[0]: {:?}", pred_nodes[0]);
        println!("data.rows: {}", data.rows);
        assert_eq!(data.rows, pred_nodes.len());

        Ok(())
    }

    // TODO: add test_tree_ranking
}
