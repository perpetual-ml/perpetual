use crate::booster::MissingNodeTreatment;
use crate::constraints::{Constraint, ConstraintMap};
use crate::data::{FloatData, JaggedMatrix, Matrix};
use crate::histogram::{reorder_cat_bins, sort_cat_bins, BinF32, HistogramMatrix};
use crate::node::{NodeType, SplittableNode};
use crate::tree::Tree;
use crate::utils::{
    between, bound_to_parent, constrained_weight, constrained_weight_const_hess, cull_gain, gain_given_weight,
    gain_given_weight_const_hess, pivot_on_split, pivot_on_split_exclude_missing,
};
use hashbrown::HashMap;
use std::collections::HashSet;

#[inline]
fn average(numbers: &[f32]) -> f32 {
    numbers.iter().sum::<f32>() / numbers.len() as f32
}

#[inline]
fn sum_two_slice(x: &[f32; 5], y: &[f32; 5]) -> [f32; 5] {
    let mut sum = [0.0_f32; 5];
    for i in 0..5 {
        sum[i] = x[i] + y[i]
    }
    sum
}

#[inline]
fn sum_two_slice_usize(x: &[usize; 5], y: &[usize; 5]) -> [usize; 5] {
    let mut sum = [0_usize; 5];
    for i in 0..5 {
        sum[i] = x[i] + y[i]
    }
    sum
}

#[derive(Debug)]
pub struct SplitInfo {
    pub split_gain: f32,
    pub split_feature: usize,
    pub split_value: f64,
    pub split_bin: u16,
    pub left_node: NodeInfo,
    pub right_node: NodeInfo,
    pub missing_node: MissingInfo,
    pub generalization: Option<f32>,
}

#[derive(Debug)]
pub struct NodeInfo {
    pub gain: f32,
    pub grad: f32,     // used as gradient_sum in SplittableNode.from_node_info
    pub cover: f32,    // used as hessian_sum in SplittableNode.from_node_info
    pub counts: usize, // used as hessian_sum in SplittableNode.from_node_info
    pub weight: f32,   // used as weight_value in SplittableNode.from_node_info
    pub bounds: (f32, f32),
}

#[derive(Debug)]
pub enum MissingInfo {
    Left,
    Right,
    Leaf(NodeInfo),
    Branch(NodeInfo),
}

pub trait Splitter {
    /// When a split happens, how many leaves will the tree increase by?
    /// For example, if a binary split happens, the split will increase the
    /// number of leaves by 1, if a ternary split happens, the number of leaves will
    /// increase by 2.
    fn new_leaves_added(&self) -> usize {
        1
    }
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint>;
    // fn get_allow_missing_splits(&self) -> bool;
    fn get_eta(&self) -> f32;

    /// Perform any post processing on the tree that is
    /// relevant for the specific splitter, empty default
    /// implementation so that it can be called even if it's
    /// not used.
    fn clean_up_splits(&self, _tree: &mut Tree) {}

    /// Find the best possible split, considering all feature histograms.
    /// If we wanted to add Column sampling, this is probably where
    /// we would need to do it, otherwise, it would be at the tree level.
    fn best_split(
        &self,
        node: &SplittableNode,
        col_index: &[usize],
        is_const_hess: bool,
        hist_tree: &HashMap<usize, HistogramMatrix>,
    ) -> Option<SplitInfo> {
        let mut best_split_info = None;
        let mut best_gain = 0.0;
        for (idx, feature) in col_index.iter().enumerate() {
            let split_info = if is_const_hess {
                self.best_feature_split_const_hess(node, *feature, idx, hist_tree)
            } else {
                self.best_feature_split(node, *feature, idx, hist_tree)
            };

            match split_info {
                Some(info) => {
                    if info.split_gain > best_gain {
                        best_gain = info.split_gain;
                        best_split_info = Some(info);
                    }
                }
                None => continue,
            }
        }
        best_split_info
    }

    /// Evaluate a split, returning the node info for the left, and right splits,
    /// as well as the node info the missing data of a feature.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        left_counts: usize,
        right_gradient: f32,
        right_hessian: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_hessian: f32,
        missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)>;

    #[allow(clippy::too_many_arguments)]
    fn evaluate_split_const_hess(
        &self,
        left_gradient: f32,
        left_counts: usize,
        right_gradient: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)>;

    /// The idx is the index of the feature in the histogram data, whereas feature
    /// is the index of the actual feature in the data.
    fn best_feature_split(
        &self,
        node: &SplittableNode,
        feature: usize,
        idx: usize,
        hist_tree: &HashMap<usize, HistogramMatrix>,
    ) -> Option<SplitInfo> {
        let mut split_info: Option<SplitInfo> = None;
        let mut max_gain: Option<f32> = None;
        let mut generalization: Option<f32>;

        let histogram = hist_tree.get(&node.num).unwrap().0.get_col(idx);
        let hist: Vec<BinF32> = histogram[1..]
            .iter()
            .cloned()
            .filter(|b| b.counts.iter().sum::<usize>() > 0)
            .collect();

        // We also know we will have a missing bin.
        let miss_bin = &histogram[0];

        let constraint = self.get_constraint(&feature);

        let node_grad_sum = hist
            .iter()
            .fold([f32::ZERO; 5], |acc, e| sum_two_slice(&acc, &e.g_folded));
        let node_hess_sum = hist
            .iter()
            .fold([f32::ZERO; 5], |acc, e| sum_two_slice(&acc, &e.h_folded.unwrap()));
        let node_coun_sum = hist.iter().fold([0; 5], |acc, e| sum_two_slice_usize(&acc, &e.counts));

        let node_g_sum = node_grad_sum.iter().sum::<f32>();
        let node_h_sum = node_hess_sum.iter().sum::<f32>();
        let node_c_sum = node_coun_sum.iter().sum::<usize>();

        let mut right_gradient_train = [f32::ZERO; 5];
        let mut right_hessian_train = [f32::ZERO; 5];
        let mut right_counts_train = [0_usize; 5];
        let mut right_gradient_valid = [f32::ZERO; 5];
        let mut right_hessian_valid = [f32::ZERO; 5];
        let mut right_counts_valid = [0_usize; 5];

        let mut cuml_gradient_train = [f32::ZERO; 5];
        let mut cuml_hessian_train = [f32::ZERO; 5];
        let mut cuml_counts_train = [0_usize; 5];
        let mut cuml_gradient_valid = [f32::ZERO; 5];
        let mut cuml_hessian_valid = [f32::ZERO; 5];
        let mut cuml_counts_valid = [0_usize; 5];

        for bin in hist {
            let left_gradient_train = cuml_gradient_train;
            let left_hessian_train = cuml_hessian_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_hessian_valid = cuml_hessian_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut left_objs = [0.0; 5];
            let mut right_objs = [0.0; 5];
            let mut train_scores = [0.0; 5];
            let mut valid_scores = [0.0; 5];
            let mut n_folds: u8 = 0;
            for j in 0..5 {
                right_gradient_train[j] = node_g_sum - node_grad_sum[j] - cuml_gradient_train[j];
                right_hessian_train[j] = node_h_sum - node_hess_sum[j] - cuml_hessian_train[j];
                right_counts_train[j] = node_c_sum - node_coun_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_hessian_valid[j] = node_hess_sum[j] - cuml_hessian_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];
                cuml_gradient_train[j] += bin.g_folded.iter().sum::<f32>() - bin.g_folded[j];
                cuml_hessian_train[j] += bin.h_folded.unwrap().iter().sum::<f32>() - bin.h_folded.unwrap()[j];
                cuml_counts_train[j] += bin.counts.iter().sum::<usize>() - bin.counts[j];
                cuml_gradient_valid[j] += bin.g_folded[j];
                cuml_hessian_valid[j] += bin.h_folded.unwrap()[j];
                cuml_counts_valid[j] += bin.counts[j];

                if right_counts_train[j] == 0
                    || right_counts_valid[j] == 0
                    || left_counts_train[j] == 0
                    || left_counts_valid[j] == 0
                {
                    continue;
                }

                // gain and weight (leaf value, predicted value) are calculated here.
                let (left_node, right_node, _missing_info) = match self.evaluate_split(
                    left_gradient_train[j],
                    left_hessian_train[j],
                    left_counts_train[j],
                    right_gradient_train[j],
                    right_hessian_train[j],
                    right_counts_train[j],
                    miss_bin.g_folded.iter().sum(),
                    miss_bin.h_folded.unwrap().iter().sum::<f32>(),
                    miss_bin.counts.iter().sum(),
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                ) {
                    None => {
                        continue;
                    }
                    Some(v) => v,
                };

                let left_obj = left_gradient_valid[j] * left_node.weight
                    + 0.5 * left_hessian_valid[j] * left_node.weight * left_node.weight;
                let right_obj = right_gradient_valid[j] * right_node.weight
                    + 0.5 * right_hessian_valid[j] * right_node.weight * right_node.weight;
                left_objs[j] = left_obj / left_counts_train[j] as f32;
                right_objs[j] = right_obj / right_counts_train[j] as f32;
                valid_scores[j] = (left_obj + right_obj) / (left_counts_valid[j] + right_counts_valid[j]) as f32;
                train_scores[j] =
                    -0.5 * (left_node.gain + right_node.gain) / (left_counts_train[j] + right_counts_train[j]) as f32;

                n_folds += 1;
            }

            if n_folds >= 5 || node.num == 0 {
                let train_score = average(&train_scores);
                let valid_score = average(&valid_scores);
                let parent_score = -0.5 * node.gain_value / node.counts_sum as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                generalization = Some(delta_score_train / delta_score_valid);

                if generalization < Some(1.0) && node.num != 0 {
                    continue;
                }
            } else {
                continue;
            }

            // gain and weight (leaf value, predicted value) are calculated here. line: 939
            let (mut left_node_info, mut right_node_info, missing_info) = match self.evaluate_split(
                left_gradient_valid.iter().sum(),
                left_hessian_valid.iter().sum::<f32>(),
                left_counts_valid.iter().sum::<usize>(),
                right_gradient_valid.iter().sum(),
                right_hessian_valid.iter().sum::<f32>(),
                right_counts_valid.iter().sum::<usize>(),
                miss_bin.g_folded.iter().sum(),
                miss_bin.h_folded.unwrap().iter().sum::<f32>(),
                miss_bin.counts.iter().sum(),
                node.lower_bound,
                node.upper_bound,
                node.weight_value,
                constraint,
            ) {
                None => {
                    continue;
                }
                Some(v) => v,
            };

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);

            // Check monotonicity holds
            let split_gain = cull_gain(split_gain, left_node_info.weight, right_node_info.weight, constraint);

            if split_gain <= 0.0 {
                continue;
            }

            let mid = (left_node_info.weight + right_node_info.weight) / 2.0;
            let (left_bounds, right_bounds) = match constraint {
                None | Some(Constraint::Unconstrained) => (
                    (node.lower_bound, node.upper_bound),
                    (node.lower_bound, node.upper_bound),
                ),
                Some(Constraint::Negative) => ((mid, node.upper_bound), (node.lower_bound, mid)),
                Some(Constraint::Positive) => ((node.lower_bound, mid), (mid, node.upper_bound)),
            };
            left_node_info.bounds = left_bounds;
            right_node_info.bounds = right_bounds;

            // If split gain is NaN, one of the sides is empty, do not allow this split.
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (!generalization.is_none() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info = Some(SplitInfo {
                    split_gain,
                    split_feature: feature,
                    split_value: bin.cut_value,
                    split_bin: bin.num,
                    left_node: left_node_info,
                    right_node: right_node_info,
                    missing_node: missing_info,
                    generalization,
                });
            }
        }

        split_info
    }

    /// The idx is the index of the feature in the histogram data, whereas feature
    /// is the index of the actual feature in the data.

    fn best_feature_split_const_hess(
        &self,
        node: &SplittableNode,
        feature: usize,
        idx: usize,
        hist_tree: &HashMap<usize, HistogramMatrix>,
    ) -> Option<SplitInfo> {
        let mut split_info: Option<SplitInfo> = None;
        let mut max_gain: Option<f32> = None;
        let mut generalization: Option<f32>;

        let histogram = hist_tree.get(&node.num).unwrap().0.get_col(idx);
        let hist = histogram[1..]
            .iter()
            .filter(|b| b.counts.iter().sum::<usize>() > 0)
            .collect::<Vec<_>>();
        // let hist = &histogram[1..];

        // We also know we will have a missing bin.
        let miss_bin = &histogram[0];

        let constraint = self.get_constraint(&feature);

        let node_grad_sum = hist
            .iter()
            .fold([f32::ZERO; 5], |acc, e| sum_two_slice(&acc, &e.g_folded));
        let node_coun_sum = hist
            .iter()
            .fold([0_usize; 5], |acc, e| sum_two_slice_usize(&acc, &e.counts));

        let node_g_sum = node_grad_sum.iter().sum::<f32>();
        let node_c_sum = node_coun_sum.iter().sum::<usize>();

        let mut right_gradient_train = [f32::ZERO; 5];
        let mut right_counts_train = [0_usize; 5];
        let mut right_gradient_valid = [f32::ZERO; 5];
        let mut right_counts_valid = [0_usize; 5];

        let mut cuml_gradient_train = [f32::ZERO; 5];
        let mut cuml_counts_train = [0_usize; 5];
        let mut cuml_gradient_valid = [f32::ZERO; 5];
        let mut cuml_counts_valid = [0_usize; 5];

        for bin in hist {
            let left_gradient_train = cuml_gradient_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut left_objs = [0.0; 5];
            let mut right_objs = [0.0; 5];
            let mut train_scores = [0.0; 5];
            let mut valid_scores = [0.0; 5];
            let mut n_folds: u8 = 0;
            for j in 0..5 {
                right_gradient_train[j] = node_g_sum - node_grad_sum[j] - cuml_gradient_train[j];
                right_counts_train[j] = node_c_sum - node_coun_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += bin.g_folded.iter().sum::<f32>() - bin.g_folded[j];
                cuml_counts_train[j] += bin.counts.iter().sum::<usize>() - bin.counts[j];
                cuml_gradient_valid[j] += bin.g_folded[j];
                cuml_counts_valid[j] += bin.counts[j];

                if right_counts_train[j] == 0
                    || right_counts_valid[j] == 0
                    || left_counts_train[j] == 0
                    || left_counts_valid[j] == 0
                {
                    continue;
                }

                // gain and weight (leaf value, predicted value) are calculated here.
                let (left_node, right_node, _missing_info) = match self.evaluate_split_const_hess(
                    left_gradient_train[j],
                    left_counts_train[j],
                    right_gradient_train[j],
                    right_counts_train[j],
                    miss_bin.g_folded.iter().sum(),
                    miss_bin.counts.iter().sum(),
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                ) {
                    None => {
                        continue;
                    }
                    Some(v) => v,
                };
                // TODO: Handle missing info!
                let left_obj = left_gradient_valid[j] * left_node.weight
                    + 0.5 * (left_counts_valid[j] as f32) * left_node.weight * left_node.weight;
                let right_obj = right_gradient_valid[j] * right_node.weight
                    + 0.5 * (right_counts_valid[j] as f32) * right_node.weight * right_node.weight;
                left_objs[j] = left_obj / left_counts_train[j] as f32;
                right_objs[j] = right_obj / right_counts_train[j] as f32;
                valid_scores[j] = (left_obj + right_obj) / (left_counts_valid[j] + right_counts_valid[j]) as f32;
                train_scores[j] =
                    -0.5 * (left_node.gain + right_node.gain) / (left_counts_train[j] + right_counts_train[j]) as f32;

                n_folds += 1;
            }

            if n_folds >= 5 || node.num == 0 {
                let train_score = average(&train_scores);
                let valid_score = average(&valid_scores);
                let parent_score = -0.5 * node.gain_value / node.counts_sum as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                generalization = Some(delta_score_train / delta_score_valid);
                if generalization < Some(1.0) && node.num != 0 {
                    continue;
                }
            } else {
                continue;
            }

            // gain and weight (leaf value, predicted value) are calculated here. line: 939
            let (mut left_node_info, mut right_node_info, missing_info) = match self.evaluate_split_const_hess(
                left_gradient_valid.iter().sum(),
                left_counts_valid.iter().sum::<usize>(),
                right_gradient_valid.iter().sum(),
                right_counts_valid.iter().sum::<usize>(),
                miss_bin.g_folded.iter().sum(),
                miss_bin.counts.iter().sum(),
                node.lower_bound,
                node.upper_bound,
                node.weight_value,
                constraint,
            ) {
                None => {
                    continue;
                }
                Some(v) => v,
            };

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);

            // Check monotonicity holds
            let split_gain = cull_gain(split_gain, left_node_info.weight, right_node_info.weight, constraint);

            if split_gain <= 0.0 {
                continue;
            }

            let mid = (left_node_info.weight + right_node_info.weight) / 2.0;
            let (left_bounds, right_bounds) = match constraint {
                None | Some(Constraint::Unconstrained) => (
                    (node.lower_bound, node.upper_bound),
                    (node.lower_bound, node.upper_bound),
                ),
                Some(Constraint::Negative) => ((mid, node.upper_bound), (node.lower_bound, mid)),
                Some(Constraint::Positive) => ((node.lower_bound, mid), (mid, node.upper_bound)),
            };
            left_node_info.bounds = left_bounds;
            right_node_info.bounds = right_bounds;
            // If split gain is NaN, one of the sides is empty, do not allow this split.
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (!generalization.is_none() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info = Some(SplitInfo {
                    split_gain,
                    split_feature: feature,
                    split_value: bin.cut_value,
                    split_bin: bin.num,
                    left_node: left_node_info,
                    right_node: right_node_info,
                    missing_node: missing_info,
                    generalization,
                });
            }
        }
        split_info
    }

    /// Handle the split info, creating the children nodes, this function
    /// will return a vector of new splitable nodes, that can be added to the
    /// growable stack, and further split, or converted to leaf nodes.
    #[allow(clippy::too_many_arguments)]
    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: Option<&[f32]>,
        parallel: bool,
        hist_tree: &mut HashMap<usize, HistogramMatrix>,
        cat_index: Option<&[u64]>,
    ) -> Vec<SplittableNode>;

    /// Split the node, if we cant find a best split, we will need to
    /// return an empty vector, this node is a leaf.
    #[allow(clippy::too_many_arguments)]
    fn split_node(
        &self,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: Option<&[f32]>,
        parallel: bool,
        is_const_hess: bool,
        hist_tree: &mut HashMap<usize, HistogramMatrix>,
        cat_index: Option<&[u64]>,
    ) -> Vec<SplittableNode> {
        match self.best_split(node, col_index, is_const_hess, hist_tree) {
            Some(split_info) => self.handle_split_info(
                split_info, n_nodes, node, index, col_index, data, cuts, grad, hess, parallel, hist_tree, cat_index,
            ),
            None => Vec::new(),
        }
    }
}

/// Missing branch splitter
/// Always creates a separate branch for the missing values of a feature.
/// This results, in every node having a specific "missing", direction.
/// If this node is able, it will be split further, otherwise it will
/// a leaf node will be generated.
pub struct MissingBranchSplitter {
    pub eta: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
    pub terminate_missing_features: HashSet<usize>,
    pub missing_node_treatment: MissingNodeTreatment,
    pub force_children_to_bound_parent: bool,
}

impl MissingBranchSplitter {
    pub fn new_leaves_added(&self) -> usize {
        2
    }
    pub fn update_average_missing_nodes(tree: &mut Tree, node_idx: usize) -> f64 {
        let node = &tree.nodes[&node_idx];

        if node.is_leaf {
            return node.weight_value as f64;
        }

        let right = node.right_child;
        let left = node.left_child;
        let current_node = node.num;
        let missing = node.missing_node;

        let right_hessian = tree.nodes[&right].hessian_sum as f64;
        let right_avg_weight = Self::update_average_missing_nodes(tree, right);

        let left_hessian = tree.nodes[&left].hessian_sum as f64;
        let left_avg_weight = Self::update_average_missing_nodes(tree, left);

        // This way this process supports missing branches that terminate (and will neutralize)
        // and then if missing is split further those values will have non-zero contributions.
        let (missing_hessian, missing_avg_weight, missing_leaf) = if tree.nodes[&missing].is_leaf {
            (0., 0., true)
        } else {
            (
                tree.nodes[&missing].hessian_sum as f64,
                Self::update_average_missing_nodes(tree, missing),
                false,
            )
        };

        let update =
            (right_avg_weight * right_hessian + left_avg_weight * left_hessian + missing_avg_weight * missing_hessian)
                / (left_hessian + right_hessian + missing_hessian);

        // Update current node, and the missing value
        if let Some(n) = tree.nodes.get_mut(&current_node) {
            n.weight_value = update as f32;
        }
        // Only update the missing node if it's a leaf, otherwise we will auto-update
        // them via the recursion called earlier.
        if missing_leaf {
            if let Some(m) = tree.nodes.get_mut(&missing) {
                m.weight_value = update as f32;
            }
        }

        update
    }
}

impl Splitter for MissingBranchSplitter {
    fn clean_up_splits(&self, tree: &mut Tree) {
        if let MissingNodeTreatment::AverageLeafWeight = self.missing_node_treatment {
            MissingBranchSplitter::update_average_missing_nodes(tree, 0);
        }
    }

    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }

    fn get_eta(&self) -> f32 {
        self.eta
    }

    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        left_counts: usize,
        right_gradient: f32,
        right_hessian: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_hessian: f32,
        missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, there is nothing to split on,
        // and so we should continue.
        if (left_gradient == 0.0) && (left_hessian == 0.0) || (right_gradient == 0.0) && (right_hessian == 0.0) {
            return None;
        }

        let mut left_weight = constrained_weight(left_gradient, left_hessian, lower_bound, upper_bound, constraint);
        let mut right_weight = constrained_weight(right_gradient, right_hessian, lower_bound, upper_bound, constraint);

        if self.force_children_to_bound_parent {
            (left_weight, right_weight) = bound_to_parent(parent_weight, left_weight, right_weight);
            assert!(between(lower_bound, upper_bound, left_weight));
            assert!(between(lower_bound, upper_bound, right_weight));
        }

        let left_gain = gain_given_weight(left_gradient, left_hessian, left_weight);
        let right_gain = gain_given_weight(right_gradient, right_hessian, right_weight);

        // We have not considered missing at all up until this point, we could if we wanted
        // to give more predictive power probably to missing.
        // If we don't want to allow the missing branch to be split further,
        // we will default to creating an empty branch.

        // Set weight based on the missing node treatment.
        let missing_weight = match self.missing_node_treatment {
            MissingNodeTreatment::AssignToParent => constrained_weight(
                missing_gradient + left_gradient + right_gradient,
                missing_hessian + left_hessian + right_hessian,
                lower_bound,
                upper_bound,
                constraint,
            ),
            // Calculate the local leaf average for now, after training the tree.
            // Recursively assign to the leaf weights underneath.
            MissingNodeTreatment::AverageLeafWeight | MissingNodeTreatment::AverageNodeWeight => {
                (right_weight * right_hessian + left_weight * left_hessian) / (right_hessian + left_hessian)
            }
            MissingNodeTreatment::None => {
                // If there are no missing records, just default
                // to the parent weight.
                if missing_hessian == 0. || missing_gradient == 0. {
                    parent_weight
                } else {
                    constrained_weight(missing_gradient, missing_hessian, lower_bound, upper_bound, constraint)
                }
            }
        };

        let missing_gain = gain_given_weight(missing_gradient, missing_hessian, missing_weight);
        let missing_info = NodeInfo {
            gain: missing_gain,
            grad: missing_gradient,
            cover: missing_hessian,
            counts: missing_counts,
            weight: missing_weight,
            // Constrain to the same bounds as the parent.
            // This will ensure that splits further down in the missing only
            // branch are monotonic.
            bounds: (lower_bound, upper_bound),
        };
        let missing_node = // Check Missing direction
        if ((missing_gradient != 0.0) || (missing_hessian != 0.0)) && self.allow_missing_splits {
            MissingInfo::Branch(
                missing_info
            )
        } else {
            MissingInfo::Leaf(
                missing_info
            )
        };

        Some((
            NodeInfo {
                gain: left_gain,
                grad: left_gradient,
                cover: left_hessian,
                counts: left_counts,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                gain: right_gain,
                grad: right_gradient,
                cover: right_hessian,
                counts: right_counts,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_node,
        ))
    }

    fn evaluate_split_const_hess(
        &self,
        left_gradient: f32,
        left_counts: usize,
        right_gradient: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, there is nothing to split on,
        // and so we should continue.
        if (left_gradient == 0.0) && (left_counts == 0) || (right_gradient == 0.0) && (right_counts == 0) {
            return None;
        }

        let mut left_weight =
            constrained_weight_const_hess(left_gradient, left_counts, lower_bound, upper_bound, constraint);
        let mut right_weight =
            constrained_weight_const_hess(right_gradient, right_counts, lower_bound, upper_bound, constraint);

        if self.force_children_to_bound_parent {
            (left_weight, right_weight) = bound_to_parent(parent_weight, left_weight, right_weight);
            assert!(between(lower_bound, upper_bound, left_weight));
            assert!(between(lower_bound, upper_bound, right_weight));
        }

        let left_gain = gain_given_weight_const_hess(left_gradient, left_counts, left_weight);
        let right_gain = gain_given_weight_const_hess(right_gradient, right_counts, right_weight);

        // We have not considered missing at all up until this point, we could if we wanted
        // to give more predictive power probably to missing.
        // If we don't want to allow the missing branch to be split further,
        // we will default to creating an empty branch.

        // Set weight based on the missing node treatment.
        let missing_weight = match self.missing_node_treatment {
            MissingNodeTreatment::AssignToParent => constrained_weight_const_hess(
                missing_gradient + left_gradient + right_gradient,
                missing_counts + left_counts + right_counts,
                lower_bound,
                upper_bound,
                constraint,
            ),
            // Calculate the local leaf average for now, after training the tree.
            // Recursively assign to the leaf weights underneath.
            MissingNodeTreatment::AverageLeafWeight | MissingNodeTreatment::AverageNodeWeight => {
                (right_weight * right_counts as f32 + left_weight * left_counts as f32)
                    / (right_counts + left_counts) as f32
            }
            MissingNodeTreatment::None => {
                // If there are no missing records, just default
                // to the parent weight.
                if missing_counts == 0 || missing_gradient == 0. {
                    parent_weight
                } else {
                    constrained_weight_const_hess(
                        missing_gradient,
                        missing_counts,
                        lower_bound,
                        upper_bound,
                        constraint,
                    )
                }
            }
        };
        let missing_gain = gain_given_weight_const_hess(missing_gradient, missing_counts, missing_weight);
        let missing_info = NodeInfo {
            gain: missing_gain,
            grad: missing_gradient,
            cover: missing_counts as f32,
            counts: missing_counts,
            weight: missing_weight,
            // Constrain to the same bounds as the parent.
            // This will ensure that splits further down in the missing only
            // branch are monotonic.
            bounds: (lower_bound, upper_bound),
        };
        let missing_node = // Check Missing direction
        if ((missing_gradient != 0.0) || (missing_counts != 0)) && self.allow_missing_splits {
            MissingInfo::Branch(
                missing_info
            )
        } else {
            MissingInfo::Leaf(
                missing_info
            )
        };

        Some((
            NodeInfo {
                gain: left_gain,
                grad: left_gradient,
                cover: left_counts as f32,
                counts: left_counts,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                gain: right_gain,
                grad: right_gradient,
                cover: right_counts as f32,
                counts: right_counts,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_node,
        ))
    }

    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: Option<&[f32]>,
        parallel: bool,
        hist_map: &mut HashMap<usize, HistogramMatrix>,
        cat_index: Option<&[u64]>,
    ) -> Vec<SplittableNode> {
        let missing_child = *n_nodes;
        let left_child = missing_child + 1;
        let right_child = missing_child + 2;

        let (left_cat, right_cat) = get_categories(cat_index, &split_info, hist_map, node);

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Missing all falls to the bottom.
        let (mut missing_split_idx, mut split_idx) = pivot_on_split_exclude_missing(
            &mut index[node.start_idx..node.stop_idx],
            data.get_col(split_info.split_feature),
            split_info.split_bin,
            left_cat.as_deref(),
        );

        node.update_children(missing_child, left_child, right_child, &split_info, left_cat, right_cat);

        let (mut missing_is_leaf, mut missing_info) = match split_info.missing_node {
            MissingInfo::Branch(i) => {
                if self.terminate_missing_features.contains(&split_info.split_feature) {
                    (true, i)
                } else {
                    (false, i)
                }
            }
            MissingInfo::Leaf(i) => (true, i),
            _ => unreachable!(),
        };
        // Set missing weight to parent weight value...
        // This essentially neutralizes missing.
        // Manually calculating it, was leading to some small numeric
        // rounding differences...
        if let MissingNodeTreatment::AssignToParent = self.missing_node_treatment {
            missing_info.weight = node.weight_value;
        }

        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right - missing_split_idx;
        let n_missing = total_recs - (n_right + n_left);
        let max_ = match [n_missing, n_left, n_right]
            .iter()
            .enumerate()
            .max_by(|(_, i), (_, j)| i.cmp(j))
        {
            Some((i, _)) => i,
            // if we can't compare them, it doesn't
            // really matter, build the histogram on
            // any of them.
            None => 0,
        };

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;
        missing_split_idx += node.start_idx;

        // Build the histograms for the smaller node.
        if n_missing == 0 {
            // If there are no missing records, we know the missing value
            // will be a leaf, assign this node as a leaf.
            missing_is_leaf = true;
            if max_ == 1 {
                let right_hist = hist_map.get_mut(&right_child).unwrap();
                right_hist.update(
                    data,
                    cuts,
                    grad,
                    hess,
                    &index[split_idx..node.stop_idx],
                    col_index,
                    parallel,
                    true,
                );
                HistogramMatrix::from_parent_child(hist_map, node.num, right_child, left_child);
            } else {
                let left_hist = hist_map.get_mut(&left_child).unwrap();
                left_hist.update(
                    data,
                    cuts,
                    grad,
                    hess,
                    &index[missing_split_idx..split_idx],
                    col_index,
                    parallel,
                    true,
                );
                HistogramMatrix::from_parent_child(hist_map, node.num, left_child, right_child);
            }
        } else if max_ == 0 {
            // Max is missing, calculate the other two
            // levels histograms.
            let left_hist = hist_map.get_mut(&left_child).unwrap();
            left_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[missing_split_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            let right_hist = hist_map.get_mut(&right_child).unwrap();
            right_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            HistogramMatrix::from_parent_two_children(hist_map, node.num, left_child, right_child, missing_child);
        } else if max_ == 1 {
            let miss_hist = hist_map.get_mut(&missing_child).unwrap();
            miss_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..missing_split_idx],
                col_index,
                parallel,
                true,
            );
            let right_hist = hist_map.get_mut(&right_child).unwrap();
            right_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            HistogramMatrix::from_parent_two_children(hist_map, node.num, missing_child, right_child, left_child);
        } else {
            // right is the largest
            let miss_hist = hist_map.get_mut(&missing_child).unwrap();
            miss_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..missing_split_idx],
                col_index,
                parallel,
                true,
            );
            let left_hist = hist_map.get_mut(&left_child).unwrap();
            left_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[missing_split_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            HistogramMatrix::from_parent_two_children(hist_map, node.num, missing_child, left_child, right_child);
        }

        let mut missing_node = SplittableNode::from_node_info(
            missing_child,
            node.depth + 1,
            node.start_idx,
            missing_split_idx,
            missing_info,
            split_info.generalization,
            NodeType::Missing,
            node.num,
        );
        missing_node.is_missing_leaf = missing_is_leaf;
        let left_node = SplittableNode::from_node_info(
            left_child,
            node.depth + 1,
            missing_split_idx,
            split_idx,
            split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            split_info.right_node,
            split_info.generalization,
            NodeType::Right,
            node.num,
        );
        vec![missing_node, left_node, right_node]
    }
}

/// Missing imputer splitter
/// Splitter that imputes missing values, by sending
/// them down either the right or left branch, depending
/// on which results in a higher increase in gain.
pub struct MissingImputerSplitter {
    pub eta: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
}

impl MissingImputerSplitter {
    /// Generate a new missing imputer splitter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(eta: f32, allow_missing_splits: bool, constraints_map: ConstraintMap) -> Self {
        MissingImputerSplitter {
            eta,
            allow_missing_splits,
            constraints_map,
        }
    }
}

impl Splitter for MissingImputerSplitter {
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }

    fn get_eta(&self) -> f32 {
        self.eta
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_split(
        &self,
        left_gradient: f32,
        left_hessian: f32,
        left_counts: usize,
        right_gradient: f32,
        right_hessian: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_hessian: f32,
        _missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        _parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, we will possibly lead to a missing only
        // split, if we don't want this, bomb.
        if (left_counts == 0 || right_counts == 0) && !self.allow_missing_splits {
            return None;
        }

        // By default missing values will go into the right node.
        let mut missing_info = MissingInfo::Right;

        let mut left_gradient = left_gradient;
        let mut left_hessian = left_hessian;

        let mut right_gradient = right_gradient;
        let mut right_hessian = right_hessian;

        let mut left_weight = constrained_weight(left_gradient, left_hessian, lower_bound, upper_bound, constraint);
        let mut right_weight = constrained_weight(right_gradient, right_hessian, lower_bound, upper_bound, constraint);

        let mut left_gain = gain_given_weight(left_gradient, left_hessian, left_weight);
        let mut right_gain = gain_given_weight(right_gradient, right_hessian, right_weight);

        // Check Missing direction
        // Don't even worry about it, if there are no missing values
        // in this bin.
        if (missing_gradient != 0.0) || (missing_hessian != 0.0) {
            // If
            // TODO: Consider making this safer, by casting to f64, summing, and then
            // back to f32...
            // The weight if missing went left
            let missing_left_weight = constrained_weight(
                left_gradient + missing_gradient,
                left_hessian + missing_hessian,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain if missing went left
            let missing_left_gain = gain_given_weight(
                left_gradient + missing_gradient,
                left_hessian + missing_hessian,
                missing_left_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_left_gain = cull_gain(missing_left_gain, missing_left_weight, right_weight, constraint);

            // The gain if missing went right
            let missing_right_weight = constrained_weight(
                right_gradient + missing_gradient,
                right_hessian + missing_hessian,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain is missing went right
            let missing_right_gain = gain_given_weight(
                right_gradient + missing_gradient,
                right_hessian + missing_hessian,
                missing_right_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_right_gain = cull_gain(missing_right_gain, left_weight, missing_right_weight, constraint);

            if (missing_right_gain - right_gain) < (missing_left_gain - left_gain) {
                // Missing goes left
                left_gradient += missing_gradient;
                left_hessian += missing_hessian;
                left_gain = missing_left_gain;
                left_weight = missing_left_weight;
                missing_info = MissingInfo::Left;
            } else {
                // Missing goes right
                right_gradient += missing_gradient;
                right_hessian += missing_hessian;
                right_gain = missing_right_gain;
                right_weight = missing_right_weight;
                missing_info = MissingInfo::Right;
            }
        }

        Some((
            NodeInfo {
                grad: left_gradient,
                gain: left_gain,
                cover: left_hessian,
                counts: left_counts,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                grad: right_gradient,
                gain: right_gain,
                cover: right_hessian,
                counts: right_counts,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_info,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_split_const_hess(
        &self,
        left_gradient: f32,
        left_counts: usize,
        right_gradient: f32,
        right_counts: usize,
        missing_gradient: f32,
        missing_counts: usize,
        lower_bound: f32,
        upper_bound: f32,
        _parent_weight: f32,
        constraint: Option<&Constraint>,
    ) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
        // If there is no info right, or there is no
        // info left, we will possibly lead to a missing only
        // split, if we don't want this, bomb.
        if (left_counts == 0 || right_counts == 0) && !self.allow_missing_splits {
            return None;
        }

        // By default missing values will go into the right node.
        let mut missing_info = MissingInfo::Right;

        let mut left_gradient = left_gradient;
        let mut right_gradient = right_gradient;

        let mut left_counts = left_counts;
        let mut right_counts = right_counts;

        let mut left_weight =
            constrained_weight_const_hess(left_gradient, left_counts, lower_bound, upper_bound, constraint);
        let mut right_weight =
            constrained_weight_const_hess(right_gradient, right_counts, lower_bound, upper_bound, constraint);

        let mut left_gain = gain_given_weight_const_hess(left_gradient, left_counts, left_weight);
        let mut right_gain = gain_given_weight_const_hess(right_gradient, right_counts, right_weight);

        // Check Missing direction
        // Don't even worry about it, if there are no missing values
        // in this bin.
        if (missing_gradient != 0.0) || (missing_counts != 0) {
            // If
            // TODO: Consider making this safer, by casting to f64, summing, and then
            // back to f32...
            // The weight if missing went left
            let missing_left_weight = constrained_weight_const_hess(
                left_gradient + missing_gradient,
                left_counts + missing_counts,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain if missing went left
            let missing_left_gain = gain_given_weight_const_hess(
                left_gradient + missing_gradient,
                left_counts + missing_counts,
                missing_left_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_left_gain = cull_gain(missing_left_gain, missing_left_weight, right_weight, constraint);

            // The gain if missing went right
            let missing_right_weight = constrained_weight_const_hess(
                right_gradient + missing_gradient,
                right_counts + missing_counts,
                lower_bound,
                upper_bound,
                constraint,
            );
            // The gain is missing went right
            let missing_right_gain = gain_given_weight_const_hess(
                right_gradient + missing_gradient,
                right_counts + missing_counts,
                missing_right_weight,
            );
            // Confirm this wouldn't break monotonicity.
            let missing_right_gain = cull_gain(missing_right_gain, left_weight, missing_right_weight, constraint);

            if (missing_right_gain - right_gain) < (missing_left_gain - left_gain) {
                // Missing goes left
                left_gradient += missing_gradient;
                left_counts += missing_counts;
                left_gain = missing_left_gain;
                left_weight = missing_left_weight;
                missing_info = MissingInfo::Left;
            } else {
                // Missing goes right
                right_gradient += missing_gradient;
                right_counts += missing_counts;
                right_gain = missing_right_gain;
                right_weight = missing_right_weight;
                missing_info = MissingInfo::Right;
            }
        }

        Some((
            NodeInfo {
                grad: left_gradient,
                gain: left_gain,
                cover: left_counts as f32,
                counts: left_counts,
                weight: left_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            NodeInfo {
                grad: right_gradient,
                gain: right_gain,
                cover: right_counts as f32,
                counts: right_counts,
                weight: right_weight,
                bounds: (f32::NEG_INFINITY, f32::INFINITY),
            },
            missing_info,
        ))
    }

    fn handle_split_info(
        &self,
        split_info: SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        grad: &[f32],
        hess: Option<&[f32]>,
        parallel: bool,
        hist_map: &mut HashMap<usize, HistogramMatrix>,
        cat_index: Option<&[u64]>,
    ) -> Vec<SplittableNode> {
        let left_child = *n_nodes;
        let right_child = left_child + 1;

        let missing_right = match split_info.missing_node {
            MissingInfo::Left => false,
            MissingInfo::Right => true,
            _ => unreachable!(),
        };

        let (left_cat, right_cat) = get_categories(cat_index, &split_info, hist_map, node);

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Here we assign missing to a specific direction.
        // This will need to be refactored once we add a
        // separate missing branch.
        //
        // This function mutates index by swapping indices based on split bin
        let mut split_idx = pivot_on_split(
            &mut index[node.start_idx..node.stop_idx],
            data.get_col(split_info.split_feature),
            split_info.split_bin,
            missing_right,
            left_cat.as_deref(),
        );

        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right;

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;

        if let Some(c_index) = cat_index {
            let histograms = unsafe {
                hist_map
                    .get_many_unchecked_mut([&node.num, &left_child, &right_child])
                    .unwrap()
            };
            sort_cat_bins(histograms, c_index);
        }
        // Update the histograms inplace for the smaller node. Then for larger node.
        if n_left < n_right {
            let left_hist = hist_map.get_mut(&left_child).unwrap();
            left_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[node.start_idx..split_idx],
                col_index,
                parallel,
                true,
            );
            HistogramMatrix::from_parent_child(hist_map, node.num, left_child, right_child);
        } else {
            let right_hist = hist_map.get_mut(&right_child).unwrap();
            right_hist.update(
                data,
                cuts,
                grad,
                hess,
                &index[split_idx..node.stop_idx],
                col_index,
                parallel,
                true,
            );
            HistogramMatrix::from_parent_child(hist_map, node.num, right_child, left_child);
        }
        if let Some(c_index) = cat_index {
            let histograms = unsafe {
                hist_map
                    .get_many_unchecked_mut([&node.num, &left_child, &right_child])
                    .unwrap()
            };
            reorder_cat_bins(histograms, c_index, hess.is_none());
        }

        let missing_child = if missing_right { right_child } else { left_child };
        node.update_children(missing_child, left_child, right_child, &split_info, left_cat, right_cat);

        let left_node = SplittableNode::from_node_info(
            left_child,
            node.depth + 1,
            node.start_idx,
            split_idx,
            split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            split_info.right_node,
            split_info.generalization,
            NodeType::Right,
            node.num,
        );
        vec![left_node, right_node]
    }
}

#[inline]
fn get_categories(
    cat_index: Option<&[u64]>,
    s: &SplitInfo,
    hist_map: &HashMap<usize, HistogramMatrix>,
    n: &SplittableNode,
) -> (Option<Vec<u16>>, Option<Vec<u16>>) {
    let (left_cat, right_cat) = match cat_index {
        Some(c_index) => {
            if c_index.contains(&(s.split_feature as u64)) {
                let mut left_cat = Vec::new();
                let mut right_cat = Vec::new();
                let hist = hist_map.get(&n.num).unwrap().0.get_col(s.split_feature);
                let mut is_left = true;
                for bin in hist.iter() {
                    if bin.num == s.split_bin {
                        is_left = false;
                    } else if bin.num == 0 {
                        continue;
                    }
                    if is_left {
                        left_cat.push(bin.cut_value as u16);
                    } else {
                        right_cat.push(bin.cut_value as u16);
                    }
                }
                (Some(left_cat), Some(right_cat))
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    };
    (left_cat, right_cat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::data::Matrix;
    use crate::node::SplittableNode;
    use crate::objective::{LogLoss, ObjectiveFunction, SquaredLoss};
    use crate::utils::gain;
    use crate::utils::weight;
    use polars::prelude::*;
    use std::error::Error;
    use std::fs;

    #[test]
    fn test_data_split() {
        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];

        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter {
            eta: 0.3,
            allow_missing_splits: true,
            constraints_map: ConstraintMap::new(),
        };
        let gradient_sum = grad.iter().copied().sum();
        let hessian_sum = match hess {
            Some(ref hess) => hess.iter().copied().sum(),
            None => grad.len() as f32,
        };
        let root_weight = weight(gradient_sum, hessian_sum);
        let root_gain = gain(gradient_sum, hessian_sum);
        let data = Matrix::new(&data_vec, 891, 5);

        let b = bin_matrix(&data, None, 10, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let mut hist_init = HistogramMatrix::empty(&bdata, &b.cuts, &col_index, true, false);
        hist_init.update(&bdata, &b.cuts, &grad, hess.as_deref(), &index, &col_index, true, false);
        let hist_capacity = 100;
        let mut hist_map: HashMap<usize, HistogramMatrix> = HashMap::with_capacity(hist_capacity);
        for i in 0..hist_capacity {
            hist_map.insert(i, hist_init.clone());
        }

        let mut n = SplittableNode::new(
            0,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            grad.len() as usize,
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            None,
        );
        let s = splitter.best_split(&mut n, &col_index, false, &mut hist_map).unwrap();
        println!("{:?}", s);
        n.update_children(2, 1, 2, &s, None, None);
        assert_eq!(0, s.split_feature);
        assert_eq!(s.split_value, 3.0);
        assert!(between(93.0, 95.0, s.left_node.cover));
        assert!(between(114.0, 116.0, s.right_node.cover));
        assert!(between(7.0, 7.2, s.left_node.gain));
        assert!(between(298.0, 300.0, s.right_node.gain));
        assert!(between(88.0, 89.0, s.split_gain));
    }

    #[test]
    fn test_cal_housing() -> Result<(), Box<dyn Error>> {
        let n_bins = 256;
        let n_cols = 8;
        let feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "MedHouseVal",
        ];

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(Arc::new(feature_names.iter().map(|&s| s.to_string()).collect())))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        let id_vars: Vec<&str> = Vec::new();

        let mdf_test = df_test.melt(
            &id_vars,
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
        )?;

        let data_test = Vec::from_iter(
            mdf_test
                .select_at_idx(1)
                .expect("Invalid column")
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

        let y_test_avg = y_test.iter().sum::<f64>() / y_test.len() as f64;
        let yhat = vec![y_test_avg; y_test.len()];
        let (grad, hess) = SquaredLoss::calc_grad_hess(&y_test, &yhat, None, None);

        let splitter = MissingImputerSplitter {
            eta: 0.3,
            allow_missing_splits: false,
            constraints_map: ConstraintMap::new(),
        };
        let gradient_sum = grad.iter().copied().sum();
        let hessian_sum = match hess {
            Some(ref hess) => hess.iter().copied().sum(),
            None => grad.len() as f32,
        };
        let root_weight = weight(gradient_sum, hessian_sum);
        let root_gain = gain(gradient_sum, hessian_sum);
        let data = Matrix::new(&data_test, y_test.len(), n_cols);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let mut hist_init = HistogramMatrix::empty(&bdata, &b.cuts, &col_index, hess.is_some(), false);
        hist_init.update(&bdata, &b.cuts, &grad, hess.as_deref(), &index, &col_index, true, false);
        let hist_capacity = 10;
        let mut hist_map: HashMap<usize, HistogramMatrix> = HashMap::with_capacity(hist_capacity);
        for i in 0..hist_capacity {
            hist_map.insert(i, hist_init.clone());
        }

        let mut n = SplittableNode::new(
            0,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            grad.len() as usize,
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            None,
        );
        let s = splitter.best_split(&mut n, &col_index, true, &mut hist_map).unwrap();
        println!("{:?}", s);
        n.update_children(2, 1, 2, &s, None, None);
        assert_eq!(0, s.split_feature);
        assert!(between(4.8, 5.1, s.split_value as f32));
        Ok(())
    }

    #[test]
    fn test_categorical() -> Result<(), Box<dyn Error>> {
        let _n_bins = 256;

        let n_rows = 39073;
        let n_columns = 14;

        let file = fs::read_to_string("resources/adult_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);

        let file = fs::read_to_string("resources/adult_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter {
            eta: 0.3,
            allow_missing_splits: false,
            constraints_map: ConstraintMap::new(),
        };
        let gradient_sum = grad.iter().copied().sum();
        let hessian_sum = match hess {
            Some(ref hess) => hess.iter().copied().sum(),
            None => grad.len() as f32,
        };
        let root_weight = weight(gradient_sum, hessian_sum);
        let root_gain = gain(gradient_sum, hessian_sum);

        let cat_index = vec![1, 3, 5, 6, 7, 8, 13];

        let b = bin_matrix(&data, None, _n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let mut hist_init = HistogramMatrix::empty(&bdata, &b.cuts, &col_index, false, false);
        hist_init.update(&bdata, &b.cuts, &grad, hess.as_deref(), &index, &col_index, true, false);
        let hist_capacity = 10;
        let mut hist_map: HashMap<usize, HistogramMatrix> = HashMap::with_capacity(hist_capacity);
        for i in 0..hist_capacity {
            hist_map.insert(i, hist_init.clone());
        }

        let histograms = unsafe { hist_map.get_many_unchecked_mut([&0, &1, &2]).unwrap() };
        reorder_cat_bins(histograms, &cat_index, false);

        let mut n = SplittableNode::new(
            0,
            root_weight,
            root_gain,
            gradient_sum,
            hessian_sum,
            grad.len() as usize,
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            None,
        );

        let s = splitter.best_split(&mut n, &col_index, false, &mut hist_map).unwrap();
        println!("split info:");
        println!("{:?}", s);

        let (left_cat, right_cat) = get_categories(Some(&cat_index), &s, &hist_map, &n);

        println!("left_cat:");
        println!("{:?}", left_cat);
        println!("right_cat:");
        println!("{:?}", right_cat);

        println!("hist_map[0]: {:?}", hist_map[&0].0.get_col(7));

        assert!(left_cat.clone().unwrap().contains(&1));
        assert!(left_cat.clone().unwrap().contains(&6));

        n.update_children(1, 1, 2, &s, left_cat, right_cat);

        assert_eq!(7, s.split_feature);

        Ok(())
    }
}
