use crate::data::Matrix;
use crate::grower::Grower;
use crate::histogram::{update_histogram, NodeHistogram};
use crate::node::{Node, NodeType, SplittableNode};
use crate::partial_dependence::tree_partial_dependence;
use crate::splitter::{SplitInfoSlice, Splitter};
use crate::utils::{fast_f64_sum, gain, gain_const_hess, weight, weight_const_hess};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{self, Display};
use crate::objective_functions::{ObjectiveFunction};
use std::sync::Arc;

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
        objective_function: &Arc<dyn ObjectiveFunction>,
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
        is_const_hess: bool,
        mut hist_tree: &mut [NodeHistogram],
        cat_index: Option<&HashSet<usize>>,
        split_info_slice: &SplitInfoSlice,
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
                &mut hist_tree,
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

                for n in new_nodes {
                    let node = n.as_node(splitter.get_eta());

                    if let Some(_tld) = target_loss_decrement {
                        for i in index[n.start_idx..n.stop_idx].iter() {
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
                            let loss_new = objective_function.loss(&[y[_i]], &[yhat_new], s_w)[0];
                            loss_decr_avg -= loss_decr[_i] / index_length;
                            loss_decr[_i] = loss[_i] - loss_new;
                            loss_decr_avg += loss_decr[_i] / index_length;
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
        index.len() as usize,
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