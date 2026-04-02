//! Tree
//!
//! The core [`Tree`] struct that holds a trained decision tree and the logic
//! for growing, fitting, and evaluating it.
use crate::data::Matrix;
use crate::grower::Grower;
use crate::histogram::{NodeHistogram, update_histogram};
use crate::node::{Node, NodeType, SplittableNode};
use crate::objective::{Objective, ObjectiveFunction};
use crate::partial_dependence::tree_partial_dependence;
use crate::splitter::{SplitInfoSlice, Splitter};
use crate::utils::{gain_const_hess_reg, gain_reg, weight_const_hess_reg, weight_reg};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{self, Display};

/// Reason a tree stopped growing.
#[derive(Deserialize, Serialize, Clone, PartialEq, Debug)]
pub enum TreeStopper {
    /// The generalization check signaled to stop.
    Generalization,
    /// The step-size (learning rate) became too small.
    StepSize,
    /// The maximum number of nodes was reached.
    MaxNodes,
}

/// A single decision tree in the ensemble.
///
/// Stores nodes in a `HashMap<usize, Node>` keyed by node index
/// (root = 0), along with tree-level metadata.
#[derive(Deserialize, Serialize, Clone)]
pub struct Tree {
    /// Map of node index to [`Node`].
    pub nodes: HashMap<usize, Node>,
    /// Reason this tree stopped growing.
    pub stopper: TreeStopper,
    /// Maximum depth of the tree.
    pub depth: usize,
    /// Number of leaf nodes.
    pub n_leaves: usize,
    /// Training-time leaf info: `(weight_value, start_idx, stop_idx)` for each leaf.
    /// Used for fast yhat updates without re-traversing the tree.
    #[serde(skip)]
    pub leaf_bounds: Vec<(f64, usize, usize)>,
    /// Training-time leaf node ids paired with their contiguous index bounds.
    #[serde(skip)]
    pub leaf_node_assignments: Vec<(usize, usize, usize)>,
    /// The rearranged index from the most recent `fit()` call.
    /// Samples belonging to the same leaf are contiguous.
    #[serde(skip)]
    pub train_index: Vec<usize>,
    /// Best fold-generalization signal observed while growing this tree.
    #[serde(
        default = "crate::booster::config::default_nan_f32",
        deserialize_with = "crate::booster::config::parse_f32"
    )]
    pub generalization_score: f32,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Tree {
            nodes: HashMap::with_capacity(16),
            stopper: TreeStopper::Generalization,
            depth: 0,
            n_leaves: 0,
            leaf_bounds: Vec::new(),
            leaf_node_assignments: Vec::new(),
            train_index: Vec::new(),
            generalization_score: 0.0,
        }
    }

    fn fold_weight_stability(weights: &[f32; 5]) -> f32 {
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        let mean_abs = weights.iter().map(|value| value.abs()).sum::<f32>() / weights.len() as f32;
        if mean_abs <= f32::EPSILON {
            return 1.0;
        }

        let variance = weights.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_abs;
        let positive_share = weights.iter().filter(|&&value| value >= 0.0).count() as f32 / weights.len() as f32;
        let sign_consistency = positive_share.max(1.0 - positive_share);

        (sign_consistency / (1.0 + cv)).clamp(0.5, 1.0)
    }

    fn update_generalization_score(&mut self, node: &SplittableNode) {
        let Some(stats) = node.stats.as_ref() else {
            return;
        };
        let Some(generalization) = stats.generalization else {
            return;
        };

        let stability = Self::fold_weight_stability(&stats.weights);
        let node_score = generalization * (0.99 + 0.01 * stability);
        self.generalization_score = self.generalization_score.max(node_score);
    }

    pub fn rescale_outputs(&mut self, factor: f32) {
        if (factor - 1.0).abs() <= f32::EPSILON {
            return;
        }

        for node in self.nodes.values_mut() {
            node.weight_value *= factor;
            if let Some(weights) = &mut node.leaf_weights {
                *weights = weights.map(|value| value * factor);
            }
            if let Some(stats) = &mut node.stats {
                stats.weights = stats.weights.map(|value| value * factor);
            }
        }

        for leaf in &mut self.leaf_bounds {
            leaf.0 *= f64::from(factor);
        }
    }

    fn get_interaction_constraints_allowed_features(
        &self,
        node: &SplittableNode,
        interaction_constraints: &Vec<Vec<usize>>,
    ) -> Option<HashSet<usize>> {
        // If Root, no constraints
        if node.num == 0 {
            return None;
        }

        let mut ancestor_features = HashSet::new();
        let mut curr_node_idx = node.parent_node;

        // Traverse up
        while let Some(curr_node) = self.nodes.get(&curr_node_idx) {
            ancestor_features.insert(curr_node.split_feature);
            if curr_node.num == 0 || curr_node.num == curr_node.parent_node {
                break;
            }
            curr_node_idx = curr_node.parent_node;
        }

        if ancestor_features.is_empty() {
            return None;
        }

        // Find allowed features
        let mut final_allowed: HashSet<usize> = HashSet::new();
        let mut found_valid_group = false;

        // 1. Explicit groups
        for group in interaction_constraints {
            let group_set: HashSet<usize> = group.iter().cloned().collect();
            if ancestor_features.is_subset(&group_set) {
                final_allowed.extend(group.iter());
                found_valid_group = true;
            }
        }

        // 2. Implicit singletons
        let mut constrained_universe: HashSet<usize> = HashSet::new();
        for g in interaction_constraints {
            constrained_universe.extend(g.iter().cloned());
        }

        for &anc in &ancestor_features {
            if !constrained_universe.contains(&anc) {
                // Ancestor is unlisted. Its group is {anc}.
                // Check if `ancestor_features` is subset of {anc} -> means `ancestor_features` == {anc}.
                if ancestor_features.len() == 1 && ancestor_features.contains(&anc) {
                    final_allowed.insert(anc);
                    found_valid_group = true;
                }
            }
        }

        if !found_valid_group {
            return Some(HashSet::new());
        }

        Some(final_allowed)
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
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        is_const_hess: bool,
        hist_tree: &mut [NodeHistogram],
        cat_index: Option<&HashSet<usize>>,
        use_randomized_folds: bool,
        split_info_slice: &mut SplitInfoSlice,
        n_nodes_alloc: usize,
        save_node_stats: bool,
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
            use_randomized_folds,
            pool,
            false,
        );

        let root_node = create_root_node(&index, grad, hess.as_deref(), splitter.get_leaf_regularization());
        self.update_generalization_score(&root_node);
        self.nodes
            .insert(root_node.num, root_node.as_node(splitter.get_eta(), save_node_stats));

        let mut growable = BinaryHeap::<SplittableNode>::default();

        let mut loss_decr = vec![0.0_f32; y.len()];
        let mut loss_decr_avg = 0.0_f32;
        let index_length = index.len() as f32;

        let mut leaf_bounds: Vec<(f64, usize, usize)> = Vec::new();
        let mut leaf_node_assignments: Vec<(usize, usize, usize)> = Vec::new();

        growable.add_node(root_node);
        while !growable.is_empty() {
            // If this will push us over the number of allocated nodes, break.
            if self.nodes.len() + 2 > n_nodes_alloc {
                self.stopper = TreeStopper::MaxNodes;
                break;
            }

            if target_loss_decrement.is_some_and(|tld| loss_decr_avg > tld) {
                self.stopper = TreeStopper::StepSize;
                break;
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

            // Get allowed features from interaction constraints (if configured)
            let allowed = if let Some(constraints) = splitter.get_interaction_constraints() {
                self.get_interaction_constraints_allowed_features(&node, constraints)
            } else {
                None
            };

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
                use_randomized_folds,
                split_info_slice,
                allowed.as_ref(),
            );

            let n_new_nodes = new_nodes.len();
            if n_new_nodes == 0 {
                self.n_leaves += 1;
                // Node couldn't be split — record as leaf
                let weight = self.nodes.get(&node.num).map(|n| n.weight_value as f64).unwrap_or(0.0);
                leaf_bounds.push((weight, node.start_idx, node.stop_idx));
                leaf_node_assignments.push((node.num, node.start_idx, node.stop_idx));
            } else {
                // self.nodes[n_idx].make_parent_node(node);
                if let Some(x) = self.nodes.get_mut(&n_idx) {
                    x.make_parent_node(node, splitter.get_eta());
                }
                self.n_leaves += n_new_nodes;
                n_nodes += n_new_nodes;

                let mut y_buffer = None;

                for n in new_nodes {
                    self.update_generalization_score(&n);
                    let node = n.as_node(splitter.get_eta(), save_node_stats);
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
                            let inv_n = 1.0 / index_length;
                            let mut delta_sum = 0.0f32;
                            for &i in node_indices {
                                let new_decr = loss[i] - loss_new[i];
                                delta_sum += new_decr - loss_decr[i];
                                loss_decr[i] = new_decr;
                            }
                            loss_decr_avg += delta_sum * inv_n;
                        } else {
                            // Hoist invariants: cast weight once, precompute reciprocal.
                            // Split sample_weight branch outside the loop to avoid
                            // per-iteration branching. Batch delta accumulation.
                            let weight_f64 = node.weight_value as f64;
                            let inv_n = 1.0 / index_length;
                            let mut delta_sum = 0.0f32;

                            if objective_function.requires_batch_evaluation() {
                                delta_sum = self.update_layer_loss_batch(
                                    objective_function,
                                    node_indices,
                                    y,
                                    yhat,
                                    sample_weight,
                                    loss,
                                    &mut loss_decr,
                                    weight_f64,
                                );
                            } else {
                                match sample_weight {
                                    Some(sw) => {
                                        for &_i in node_indices {
                                            let yhat_new = yhat[_i] + weight_f64;
                                            let loss_new =
                                                objective_function.loss_single(y[_i], yhat_new, Some(sw[_i]));
                                            let new_decr = loss[_i] - loss_new;
                                            delta_sum += new_decr - loss_decr[_i];
                                            loss_decr[_i] = new_decr;
                                        }
                                    }
                                    None => {
                                        for &_i in node_indices {
                                            let yhat_new = yhat[_i] + weight_f64;
                                            let loss_new = objective_function.loss_single(y[_i], yhat_new, None);
                                            let new_decr = loss[_i] - loss_new;
                                            delta_sum += new_decr - loss_decr[_i];
                                            loss_decr[_i] = new_decr;
                                        }
                                    }
                                }
                            }
                            loss_decr_avg += delta_sum * inv_n;
                        }
                    }

                    self.depth = max(self.depth, n.stats.as_ref().map_or(0, |s| s.depth));
                    let node_weight = node.weight_value as f64;
                    self.nodes.insert(node.num, node);

                    if !n.is_missing_leaf {
                        growable.add_node(n)
                    } else {
                        // Missing-leaf nodes are terminal — record leaf bounds
                        leaf_bounds.push((node_weight, n.start_idx, n.stop_idx));
                        leaf_node_assignments.push((n.num, n.start_idx, n.stop_idx));
                    }
                }
            }
        }

        // Drain remaining growable nodes — they are all leaves (early stop)
        while !growable.is_empty() {
            let remaining = growable.get_next_node();
            let weight = self
                .nodes
                .get(&remaining.num)
                .map(|n| n.weight_value as f64)
                .unwrap_or(0.0);
            leaf_bounds.push((weight, remaining.start_idx, remaining.stop_idx));
            leaf_node_assignments.push((remaining.num, remaining.start_idx, remaining.stop_idx));
        }

        self.leaf_bounds = leaf_bounds;
        self.leaf_node_assignments = leaf_node_assignments;
        self.train_index = index;

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

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn update_layer_loss_batch(
        &self,
        objective_function: &Objective,
        node_indices: &[usize],
        y: &[f64],
        yhat: &[f64],
        sample_weight: Option<&[f64]>,
        loss: &[f32],
        loss_decr: &mut [f32],
        weight_value: f64,
    ) -> f32 {
        let mut yhat_new = yhat.to_vec();
        for &i in node_indices {
            yhat_new[i] += weight_value;
        }
        let loss_new = objective_function.loss(y, &yhat_new, sample_weight, None);
        let mut delta_sum = 0.0f32;
        for &i in node_indices {
            let new_decr = loss[i] - loss_new[i];
            delta_sum += new_decr - loss_decr[i];
            loss_decr[i] = new_decr;
        }
        delta_sum
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
                r += format!(
                    "{}{}\n",
                    "      ".repeat(node.stats.as_ref().map_or(0, |s| s.depth)).as_str(),
                    node
                )
                .as_str();
            } else {
                r += format!(
                    "{}{}\n",
                    "      ".repeat(node.stats.as_ref().map_or(0, |s| s.depth)).as_str(),
                    node
                )
                .as_str();
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

pub fn create_root_node(
    index: &[usize],
    grad: &[f32],
    hess: Option<&[f32]>,
    leaf_regularization: f32,
) -> SplittableNode {
    let (gradient_sum, hessian_sum) = match hess {
        Some(hess) => (
            index.iter().map(|&row| grad[row]).sum::<f32>(),
            index.iter().map(|&row| hess[row]).sum::<f32>(),
        ),
        None => (index.iter().map(|&row| grad[row]).sum::<f32>(), index.len() as f32),
    };

    let (root_gain, root_weight) = match hess {
        Some(_hess) => (
            gain_reg(gradient_sum, hessian_sum, leaf_regularization),
            weight_reg(gradient_sum, hessian_sum, leaf_regularization),
        ),
        None => (
            gain_const_hess_reg(gradient_sum, grad.len(), leaf_regularization),
            weight_const_hess_reg(gradient_sum, grad.len(), leaf_regularization),
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
        None,
        [root_weight; 5],
    )
}

// Unit-testing
#[cfg(test)]
mod tests {
    use super::create_root_node;

    use crate::binning::bin_matrix;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::histogram::NodeHistogramOwned;
    use crate::node::{Node, NodeStats, SplittableNode};
    use crate::objective::{Objective, ObjectiveFunction};
    use crate::splitter::{MissingImputerSplitter, SplitInfo};
    use crate::utils::precision_round;

    use std::error::Error;
    use std::fs;

    use crate::Matrix;
    use crate::histogram::NodeHistogram;
    use crate::splitter::SplitInfoSlice;
    use crate::tree::Tree;
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
        let splitter = MissingImputerSplitter::new(0.3, 0.0, true, ConstraintMap::new(), None);
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

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
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            false,
            &mut split_info_slice,
            n_nodes_alloc,
            false,
        );

        println!("{}", tree);
        let preds = tree.predict(&data, false, &f64::NAN);
        println!("{:?}", &preds[0..10]);
        assert!(tree.nodes.len() > 1);
        assert!(tree.nodes.len() <= n_nodes_alloc);
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
    fn test_create_root_node_respects_sampled_index() {
        let index = vec![1, 3];
        let grad = vec![1.0, 2.0, 3.0, 4.0];
        let hess = vec![0.5, 1.5, 2.5, 3.5];

        let node = create_root_node(&index, &grad, Some(&hess), 0.0);

        assert_eq!(node.stats.as_ref().unwrap().count, 2);
        assert!((node.gradient_sum - 6.0).abs() < 1e-6);
        assert!((node.hessian_sum - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_tree_rescale_outputs() {
        let mut tree = Tree::new();
        tree.nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: 2.0,
                leaf_weights: Some([1.0, 2.0, 3.0, 4.0, 5.0]),
                hessian_sum: 1.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: Some(Box::new(NodeStats {
                    depth: 0,
                    node_type: crate::node::NodeType::Root,
                    count: 3,
                    generalization: Some(1.0),
                    weights: [1.0, 2.0, 3.0, 4.0, 5.0],
                })),
            },
        );
        tree.leaf_bounds = vec![(2.0, 0, 3)];
        tree.generalization_score = 0.9;

        tree.rescale_outputs(0.5);

        let node = tree.nodes.get(&0).unwrap();
        assert!((node.weight_value - 1.0).abs() < 1e-6);
        assert_eq!(node.leaf_weights.unwrap(), [0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(node.stats.as_ref().unwrap().weights, [0.5, 1.0, 1.5, 2.0, 2.5]);
        assert!((tree.leaf_bounds[0].0 - 1.0).abs() < 1e-9);
        assert!((tree.generalization_score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_update_generalization_score_does_not_depend_on_saved_node_stats() {
        let mut tree = Tree::new();
        let node = SplittableNode::new(
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            10,
            0,
            0,
            10,
            f32::NEG_INFINITY,
            f32::INFINITY,
            crate::node::NodeType::Root,
            None,
            [0.5, 0.52, 0.48, 0.51, 0.49],
        );
        let mut node = node;
        node.stats.as_mut().unwrap().generalization = Some(1.01);

        tree.update_generalization_score(&node);
        let stored = node.as_node(1.0, false);
        assert!(stored.stats.is_none());
        assert!(tree.generalization_score > 1.0);
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
        let splitter = MissingImputerSplitter::new(0.3, 0.0, true, map, None);
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 100, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        //let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

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
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            false,
            &mut split_info_slice,
            n_nodes_alloc,
            false,
        );

        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.total_cmp(b));
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
        let splitter = MissingImputerSplitter::new(0.3, 0.0, true, ConstraintMap::new(), None);
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

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
            &yhat,
            None,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            false,
            &mut split_info_slice,
            n_nodes_alloc,
            false,
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

        let splitter = MissingImputerSplitter::new(0.3, 0.0, true, ConstraintMap::new(), None);

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

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
            &yhat,
            None,
            None,
            false,
            &mut hist_tree,
            Some(&cat_index),
            false,
            &mut split_info_slice,
            n_nodes_alloc,
            false,
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
