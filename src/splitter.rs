use crate::bin::sort_cat_bins_by_stat;
use crate::booster::MissingNodeTreatment;
use crate::constants::GENERALIZATION_THRESHOLD;
use crate::constraints::{Constraint, ConstraintMap};
use crate::data::{FloatData, Matrix};
use crate::histogram::{update_histogram, FeatureHistogram, NodeHistogram};
use crate::node::{NodeType, SplittableNode};
use crate::tree::Tree;
use crate::utils::{
    between, bound_to_parent, constrained_weight, constrained_weight_const_hess, cull_gain, gain_given_weight,
    gain_given_weight_const_hess, pivot_on_split, pivot_on_split_const_hess, pivot_on_split_exclude_missing,
    pivot_on_split_exclude_missing_const_hess,
};
use rayon::ThreadPool;
use std::borrow::BorrowMut;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};

#[inline]
fn average(numbers: &[f32]) -> f32 {
    numbers.iter().sum::<f32>() / numbers.len() as f32
}

#[inline]
fn sum_two_slice(x: &[f32; 5], y: &[f32; 5]) -> [f32; 5] {
    let mut sum = [0.0_f32; 5];
    x.iter().zip(y.iter()).enumerate().for_each(|(i, (x_, y_))| {
        sum[i] = x_ + y_;
    });
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
    pub left_cats: HashSet<usize>,
    pub right_cats: HashSet<usize>,
}

impl Default for SplitInfo {
    fn default() -> Self {
        SplitInfo {
            split_gain: -1.0,
            split_feature: 0,
            split_value: 0.0,
            split_bin: 0,
            left_node: NodeInfo::default(),
            right_node: NodeInfo::default(),
            missing_node: MissingInfo::None,
            generalization: None,
            left_cats: HashSet::new(),
            right_cats: HashSet::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SplitInfoSlice<'a> {
    pub data: &'a [UnsafeCell<SplitInfo>],
}

unsafe impl<'a> Send for SplitInfoSlice<'a> {}
unsafe impl<'a> Sync for SplitInfoSlice<'a> {}

impl<'a> SplitInfoSlice<'a> {
    pub fn new(data: &'a mut [SplitInfo]) -> Self {
        let ptr = data as *mut [SplitInfo] as *const [UnsafeCell<SplitInfo>];
        Self { data: unsafe { &*ptr } }
    }
    pub unsafe fn get_mut(&self, i: usize) -> &mut SplitInfo {
        let split_info = self.data[i].get().as_mut().unwrap();
        split_info
    }
    pub unsafe fn best_split_info(&self) -> &mut SplitInfo {
        let split_info = self
            .data
            .iter()
            .max_by(|s1, s2| {
                let g1 = s1.get().as_ref().unwrap().split_gain;
                let g2 = s2.get().as_ref().unwrap().split_gain;
                g1.partial_cmp(&g2).expect("Tried to compare a NaN")
            })
            .unwrap()
            .get()
            .as_mut()
            .unwrap();
        split_info
    }
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

impl Default for NodeInfo {
    fn default() -> Self {
        NodeInfo {
            gain: 0.0,
            grad: 0.0,
            cover: 0.0,
            counts: 0,
            weight: 0.0,
            bounds: (0.0, 0.0),
        }
    }
}

#[derive(Debug)]
pub enum MissingInfo {
    Left,
    Right,
    Leaf(NodeInfo),
    Branch(NodeInfo),
    None,
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
    fn get_constraint_map(&self) -> &ConstraintMap;
    fn get_allow_missing_splits(&self) -> bool;
    fn get_create_missing_branch(&self) -> bool;
    fn get_eta(&self) -> f32;
    fn get_missing_node_treatment(&self) -> MissingNodeTreatment;
    fn get_force_children_to_bound_parent(&self) -> bool;

    /// Perform any post processing on the tree that is
    /// relevant for the specific splitter, empty default
    /// implementation so that it can be called even if it's
    /// not used.
    fn clean_up_splits(&self, _tree: &mut Tree) {}

    /// Find the best possible split, considering all feature histograms.
    fn best_split(
        &self,
        node: &SplittableNode,
        col_index: &[usize],
        is_const_hess: bool,
        hist_tree: &[NodeHistogram],
        pool: &ThreadPool,
        cat_index: Option<&HashSet<usize>>,
        split_info_slice: &SplitInfoSlice,
    ) {
        let allow_missing_splits = self.get_allow_missing_splits();
        let constraint_map = self.get_constraint_map();
        let create_missing_branch = self.get_create_missing_branch();
        let missing_node_treatment = self.get_missing_node_treatment();
        let force_children_to_bound_parent = self.get_force_children_to_bound_parent();

        let hist_node = unsafe { hist_tree.get_unchecked(node.num) };

        let best_feature_split = best_feature_split_callables(is_const_hess);

        let feature_index = (0..col_index.len()).collect::<Vec<_>>();

        if pool.current_num_threads() > 1 {
            pool.scope(|s| {
                for (feature, feat_idx) in col_index.iter().zip(feature_index.iter()) {
                    s.spawn(|_| {
                        best_feature_split(
                            node,
                            *feat_idx,
                            *feature,
                            hist_node.data.get(*feat_idx).unwrap(),
                            cat_index,
                            constraint_map.get(feature),
                            force_children_to_bound_parent,
                            missing_node_treatment,
                            allow_missing_splits,
                            create_missing_branch,
                            split_info_slice,
                        );
                    });
                }
            });
        } else {
            for (feature, feat_idx) in col_index.iter().zip(feature_index.iter()) {
                let constraint = self.get_constraint(&feature);
                best_feature_split(
                    node,
                    *feat_idx,
                    *feature,
                    hist_node.data.get(*feat_idx).unwrap(),
                    cat_index,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                    create_missing_branch,
                    split_info_slice,
                );
            }
        }
    }

    /// Handle the split info, creating the children nodes, this function
    /// will return a vector of new splitable nodes, that can be added to the
    /// growable stack, and further split, or converted to leaf nodes.
    #[allow(clippy::too_many_arguments)]
    fn handle_split_info(
        &self,
        split_info: &mut SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        grad: &mut [f32],
        hess: Option<&mut [f32]>,
        pool: &ThreadPool,
        hist_tree: &[NodeHistogram],
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
        grad: &mut [f32],
        hess: Option<&mut [f32]>,
        pool: &ThreadPool,
        is_const_hess: bool,
        hist_tree: &[NodeHistogram],
        cat_index: Option<&HashSet<usize>>,
        split_info_slice: &SplitInfoSlice,
    ) -> Vec<SplittableNode> {
        self.best_split(
            node,
            col_index,
            is_const_hess,
            hist_tree,
            pool,
            cat_index,
            split_info_slice,
        );

        let split_info = unsafe { split_info_slice.best_split_info() };

        if split_info.split_gain > 0.0 {
            self.handle_split_info(
                split_info, n_nodes, node, index, col_index, data, grad, hess, pool, hist_tree,
            )
        } else {
            Vec::new()
        }
    }
}

/// Missing branch splitter
/// Always creates a separate branch for the missing values of a feature.
/// This results, in every node having a specific "missing", direction.
/// If this node is able, it will be split further, otherwise it will
/// a leaf node will be generated.
pub struct MissingBranchSplitter {
    pub create_missing_branch: bool,
    pub eta: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
    pub terminate_missing_features: HashSet<usize>,
    pub missing_node_treatment: MissingNodeTreatment,
    pub force_children_to_bound_parent: bool,
}

impl MissingBranchSplitter {
    /// Generate a new missing branch splitter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eta: f32,
        allow_missing_splits: bool,
        constraints_map: ConstraintMap,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: MissingNodeTreatment,
        force_children_to_bound_parent: bool,
    ) -> Self {
        MissingBranchSplitter {
            create_missing_branch: true,
            eta,
            allow_missing_splits,
            constraints_map,
            terminate_missing_features,
            missing_node_treatment,
            force_children_to_bound_parent,
        }
    }

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
    fn get_constraint_map(&self) -> &HashMap<usize, Constraint> {
        &self.constraints_map
    }
    fn get_allow_missing_splits(&self) -> bool {
        self.allow_missing_splits
    }
    fn get_create_missing_branch(&self) -> bool {
        self.create_missing_branch
    }
    fn get_eta(&self) -> f32 {
        self.eta
    }
    fn get_missing_node_treatment(&self) -> MissingNodeTreatment {
        self.missing_node_treatment
    }
    fn get_force_children_to_bound_parent(&self) -> bool {
        self.force_children_to_bound_parent
    }

    fn handle_split_info(
        &self,
        split_info: &mut SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        mut index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        grad: &mut [f32],
        mut hess: Option<&mut [f32]>,
        pool: &ThreadPool,
        hist_tree: &[NodeHistogram],
    ) -> Vec<SplittableNode> {
        let missing_child = *n_nodes;
        let left_child = missing_child + 1;
        let right_child = missing_child + 2;

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Missing all falls to the bottom.
        let mut missing_split_idx: usize;
        let mut split_idx: usize;
        if hess.is_some() {
            (missing_split_idx, split_idx) = pivot_on_split_exclude_missing(
                node.start_idx,
                node.stop_idx,
                &mut index,
                grad,
                &mut hess.as_mut().unwrap(),
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                &split_info.left_cats,
            );
        } else {
            (missing_split_idx, split_idx) = pivot_on_split_exclude_missing_const_hess(
                node.start_idx,
                node.stop_idx,
                &mut index,
                grad,
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                &split_info.left_cats,
            );
        }

        node.update_children(missing_child, left_child, right_child, &split_info);

        let (mut missing_is_leaf, missing_info) = match split_info.missing_node {
            MissingInfo::Branch(ref mut i) => {
                if self.terminate_missing_features.contains(&split_info.split_feature) {
                    (true, i.borrow_mut())
                } else {
                    (false, i.borrow_mut())
                }
            }
            MissingInfo::Leaf(ref mut i) => (true, i.borrow_mut()),
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
                let right_hist = unsafe { hist_tree.get_unchecked(right_child) };
                update_histogram(
                    right_hist,
                    split_idx,
                    node.stop_idx,
                    data,
                    grad,
                    hess.as_deref(),
                    &index,
                    col_index,
                    pool,
                    true,
                );
                NodeHistogram::from_parent_child(hist_tree, node.num, right_child, left_child);
            } else {
                let left_hist = unsafe { hist_tree.get_unchecked(left_child) };
                update_histogram(
                    left_hist,
                    missing_split_idx,
                    split_idx,
                    data,
                    grad,
                    hess.as_deref(),
                    &index,
                    col_index,
                    pool,
                    true,
                );
                NodeHistogram::from_parent_child(hist_tree, node.num, left_child, right_child);
            }
        } else if max_ == 0 {
            // Max is missing, calculate the other two
            // levels histograms.
            let left_hist = unsafe { hist_tree.get_unchecked(left_child) };
            update_histogram(
                left_hist,
                missing_split_idx,
                split_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            let right_hist = unsafe { hist_tree.get_unchecked(right_child) };
            update_histogram(
                right_hist,
                split_idx,
                node.stop_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            NodeHistogram::from_parent_two_children(hist_tree, node.num, left_child, right_child, missing_child);
        } else if max_ == 1 {
            let miss_hist = unsafe { hist_tree.get_unchecked(missing_child) };
            update_histogram(
                miss_hist,
                node.start_idx,
                missing_split_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            let right_hist = unsafe { hist_tree.get_unchecked(right_child) };
            update_histogram(
                right_hist,
                split_idx,
                node.stop_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            NodeHistogram::from_parent_two_children(hist_tree, node.num, missing_child, right_child, left_child);
        } else {
            // right is the largest
            let miss_hist = unsafe { hist_tree.get_unchecked(missing_child) };
            update_histogram(
                miss_hist,
                node.start_idx,
                missing_split_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            let left_hist = unsafe { hist_tree.get_unchecked(left_child) };
            update_histogram(
                left_hist,
                missing_split_idx,
                split_idx,
                data,
                grad,
                hess.as_deref(),
                &index,
                col_index,
                pool,
                true,
            );
            NodeHistogram::from_parent_two_children(hist_tree, node.num, missing_child, left_child, right_child);
        }

        let mut missing_node = SplittableNode::from_node_info(
            missing_child,
            node.depth + 1,
            node.start_idx,
            missing_split_idx,
            &missing_info,
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
            &split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            &split_info.right_node,
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
    pub create_missing_branch: bool,
    pub eta: f32,
    pub allow_missing_splits: bool,
    pub constraints_map: ConstraintMap,
    pub missing_node_treatment: MissingNodeTreatment,
    pub force_children_to_bound_parent: bool,
}

impl MissingImputerSplitter {
    /// Generate a new missing imputer splitter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(eta: f32, allow_missing_splits: bool, constraints_map: ConstraintMap) -> Self {
        MissingImputerSplitter {
            create_missing_branch: false,
            eta,
            allow_missing_splits,
            constraints_map,
            missing_node_treatment: MissingNodeTreatment::None,
            force_children_to_bound_parent: false,
        }
    }
}

impl Splitter for MissingImputerSplitter {
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }
    fn get_constraint_map(&self) -> &HashMap<usize, Constraint> {
        &self.constraints_map
    }
    fn get_allow_missing_splits(&self) -> bool {
        self.allow_missing_splits
    }
    fn get_create_missing_branch(&self) -> bool {
        self.create_missing_branch
    }
    fn get_eta(&self) -> f32 {
        self.eta
    }
    fn get_missing_node_treatment(&self) -> MissingNodeTreatment {
        self.missing_node_treatment
    }
    fn get_force_children_to_bound_parent(&self) -> bool {
        self.force_children_to_bound_parent
    }

    fn handle_split_info(
        &self,
        split_info: &mut SplitInfo,
        n_nodes: &usize,
        node: &mut SplittableNode,
        index: &mut [usize],
        col_index: &[usize],
        data: &Matrix<u16>,
        grad: &mut [f32],
        mut hess: Option<&mut [f32]>,
        pool: &ThreadPool,
        hist_tree: &[NodeHistogram],
    ) -> Vec<SplittableNode> {
        let left_child = *n_nodes;
        let right_child = left_child + 1;

        let missing_right = match split_info.missing_node {
            MissingInfo::Left => false,
            MissingInfo::Right => true,
            _ => unreachable!(),
        };

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Here we assign missing to a specific direction.
        // This will need to be refactored once we add a
        // separate missing branch.
        //
        // This function mutates index by swapping indices based on split bin
        let mut split_idx: usize;
        if hess.is_none() {
            split_idx = pivot_on_split_const_hess(
                node.start_idx,
                node.stop_idx,
                index,
                grad,
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                missing_right,
                &split_info.left_cats,
            );
        } else {
            split_idx = pivot_on_split(
                node.start_idx,
                node.stop_idx,
                index,
                grad,
                &mut hess.as_mut().unwrap(),
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                missing_right,
                &split_info.left_cats,
            );
        }

        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right;

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;

        // Update the histograms inplace for the smaller node. Then for larger node.
        if n_left < n_right {
            let left_hist = unsafe { hist_tree.get_unchecked(left_child) };
            update_histogram(
                left_hist,
                node.start_idx,
                split_idx,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
                true,
            );
            NodeHistogram::from_parent_child(hist_tree, node.num, left_child, right_child);
        } else {
            let right_hist = unsafe { hist_tree.get_unchecked(right_child) };
            update_histogram(
                right_hist,
                split_idx,
                node.stop_idx,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
                true,
            );
            NodeHistogram::from_parent_child(hist_tree, node.num, right_child, left_child);
        }

        let missing_child = if missing_right { right_child } else { left_child };
        node.update_children(missing_child, left_child, right_child, &split_info);

        let left_node = SplittableNode::from_node_info(
            left_child,
            node.depth + 1,
            node.start_idx,
            split_idx,
            &split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.depth + 1,
            split_idx,
            node.stop_idx,
            &split_info.right_node,
            split_info.generalization,
            NodeType::Right,
            node.num,
        );
        vec![left_node, right_node]
    }
}

type BestFeatureSplitFn = fn(
    &SplittableNode,
    usize,
    usize,
    &FeatureHistogram,
    Option<&HashSet<usize>>,
    Option<&Constraint>,
    bool,
    MissingNodeTreatment,
    bool,
    bool,
    &SplitInfoSlice,
);

#[inline]
pub fn best_feature_split_callables(is_const_hess: bool) -> BestFeatureSplitFn {
    match is_const_hess {
        true => best_feature_split_const_hess,
        false => best_feature_split_var_hess,
    }
}

/// The feat_idx is the index of the feature in the histogram data.
/// The feature is the index of the actual feature in the data.
#[inline]
fn best_feature_split_const_hess(
    node: &SplittableNode,
    feat_idx: usize,
    feature: usize,
    hist_feat: &FeatureHistogram,
    cat_index: Option<&HashSet<usize>>,
    constraint: Option<&Constraint>,
    force_children_to_bound_parent: bool,
    missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
    create_missing_branch: bool,
    split_info_slice: &SplitInfoSlice,
) {
    let split_info = unsafe { split_info_slice.get_mut(feat_idx) };
    split_info.split_gain = -1.0;
    split_info.left_cats = HashSet::new();
    split_info.right_cats = HashSet::new();

    let mut max_gain: Option<f32> = None;
    let mut generalization: Option<f32>;
    let mut all_cats: Vec<usize> = Vec::new();

    let evaluate_fn = eval_callables(true, create_missing_branch);

    let mut hist = hist_feat.data[1..]
        .iter()
        .filter(|b| {
            let bin = unsafe { b.get().as_ref().unwrap() };
            bin.counts.iter().sum::<usize>() > 0 && !bin.cut_value.is_nan()
        })
        .collect::<Vec<_>>();

    if let Some(c_index) = cat_index {
        if c_index.contains(&feature) {
            sort_cat_bins_by_stat(&mut hist, true);
            all_cats = hist
                .iter()
                .map(|b| unsafe { b.get().as_ref().unwrap().cut_value } as usize)
                .collect();
        }
    }

    // We also know we will have a missing bin.
    let miss_bin = unsafe { hist_feat.data.get_unchecked(0).get().as_ref().unwrap() };

    let node_grad_sum = hist.iter().fold([f32::ZERO; 5], |acc, e| {
        sum_two_slice(&acc, unsafe { &e.get().as_ref().unwrap().g_folded })
    });
    let node_coun_sum = hist.iter().fold([0_usize; 5], |acc, e| {
        sum_two_slice_usize(&acc, unsafe { &e.get().as_ref().unwrap().counts })
    });

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
        let b = unsafe { bin.get().as_ref().unwrap() };

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

            cuml_gradient_train[j] += b.g_folded.iter().sum::<f32>() - b.g_folded[j];
            cuml_counts_train[j] += b.counts.iter().sum::<usize>() - b.counts[j];
            cuml_gradient_valid[j] += b.g_folded[j];
            cuml_counts_valid[j] += b.counts[j];

            if right_counts_train[j] == 0
                || right_counts_valid[j] == 0
                || left_counts_train[j] == 0
                || left_counts_valid[j] == 0
            {
                continue;
            }

            // gain and weight (leaf value, predicted value) are calculated here.
            let (left_node, right_node, _missing_info) = match evaluate_fn(
                left_gradient_train[j],
                f32::NAN,
                left_counts_train[j],
                right_gradient_train[j],
                f32::NAN,
                right_counts_train[j],
                miss_bin.g_folded.iter().sum(),
                f32::NAN,
                miss_bin.counts.iter().sum(),
                node.lower_bound,
                node.upper_bound,
                node.weight_value,
                constraint,
                force_children_to_bound_parent,
                missing_node_treatment,
                allow_missing_splits,
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
            if generalization < Some(GENERALIZATION_THRESHOLD) && node.num != 0 {
                continue;
            }
        } else {
            continue;
        }

        // gain and weight (leaf value, predicted value) are calculated here. line: 939
        let (mut left_node_info, mut right_node_info, missing_info) = match evaluate_fn(
            left_gradient_valid.iter().sum(),
            f32::NAN,
            left_counts_valid.iter().sum::<usize>(),
            right_gradient_valid.iter().sum(),
            f32::NAN,
            right_counts_valid.iter().sum::<usize>(),
            miss_bin.g_folded.iter().sum(),
            f32::NAN,
            miss_bin.counts.iter().sum(),
            node.lower_bound,
            node.upper_bound,
            node.weight_value,
            constraint,
            force_children_to_bound_parent,
            missing_node_treatment,
            allow_missing_splits,
        ) {
            None => {
                continue;
            }
            Some(v) => v,
        };

        let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);

        // Check if monotonicity holds
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

            let mut left_cats: HashSet<usize> = HashSet::new();
            let mut right_cats: HashSet<usize> = all_cats.iter().copied().collect();

            for c in all_cats.iter() {
                if *c == b.cut_value as usize {
                    break;
                }
                left_cats.insert(*c);
                right_cats.remove(c);
            }

            split_info.split_gain = split_gain;
            split_info.split_feature = feature;
            split_info.split_value = b.cut_value;
            split_info.split_bin = b.num;
            split_info.left_node = left_node_info;
            split_info.right_node = right_node_info;
            split_info.missing_node = missing_info;
            split_info.generalization = generalization;
            split_info.left_cats = left_cats;
            split_info.right_cats = right_cats;
        }
    }
}

/// The feat_idx is the index of the feature in the histogram data.
/// The feature is the index of the actual feature in the data.
#[inline]
fn best_feature_split_var_hess(
    node: &SplittableNode,
    feat_idx: usize,
    feature: usize,
    hist_feat: &FeatureHistogram,
    cat_index: Option<&HashSet<usize>>,
    constraint: Option<&Constraint>,
    force_children_to_bound_parent: bool,
    missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
    create_missing_branch: bool,
    split_info_slice: &SplitInfoSlice,
) {
    let split_info = unsafe { split_info_slice.get_mut(feat_idx) };
    split_info.split_gain = -1.0;
    split_info.left_cats = HashSet::new();
    split_info.right_cats = HashSet::new();

    let mut max_gain: Option<f32> = None;
    let mut generalization: Option<f32>;
    let mut all_cats: Vec<usize> = Vec::new();

    let evaluate_fn = eval_callables(false, create_missing_branch);

    let mut hist = hist_feat.data[1..]
        .iter()
        .filter(|b| unsafe { b.get().as_ref().unwrap().counts.iter().sum::<usize>() } > 0)
        .collect::<Vec<_>>();

    if let Some(c_index) = cat_index {
        if c_index.contains(&feature) {
            sort_cat_bins_by_stat(&mut hist, false);
            all_cats = hist
                .iter()
                .map(|b| unsafe { b.get().as_ref().unwrap().cut_value } as usize)
                .collect();
        }
    }

    // We also know we will have a missing bin.
    let miss_bin = unsafe { hist_feat.data.get_unchecked(0).get().as_ref().unwrap() };

    let node_grad_sum = hist.iter().fold([f32::ZERO; 5], |acc, e| {
        sum_two_slice(&acc, unsafe { &e.get().as_ref().unwrap().g_folded })
    });
    let node_hess_sum = hist.iter().fold([f32::ZERO; 5], |acc, e| {
        sum_two_slice(&acc, unsafe { &e.get().as_ref().unwrap().h_folded.unwrap() })
    });
    let node_coun_sum = hist.iter().fold([0; 5], |acc, e| {
        sum_two_slice_usize(&acc, unsafe { &e.get().as_ref().unwrap().counts })
    });

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
        let b = unsafe { bin.get().as_ref().unwrap() };

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

            cuml_gradient_train[j] += b.g_folded.iter().sum::<f32>() - b.g_folded[j];
            cuml_hessian_train[j] += b.h_folded.unwrap().iter().sum::<f32>() - b.h_folded.unwrap()[j];
            cuml_counts_train[j] += b.counts.iter().sum::<usize>() - b.counts[j];
            cuml_gradient_valid[j] += b.g_folded[j];
            cuml_hessian_valid[j] += b.h_folded.unwrap()[j];
            cuml_counts_valid[j] += b.counts[j];

            if right_counts_train[j] == 0
                || right_counts_valid[j] == 0
                || left_counts_train[j] == 0
                || left_counts_valid[j] == 0
            {
                continue;
            }

            // gain and weight (leaf value, predicted value) are calculated here.
            let (left_node, right_node, _missing_info) = match evaluate_fn(
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
                force_children_to_bound_parent,
                missing_node_treatment,
                allow_missing_splits,
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

            if generalization < Some(GENERALIZATION_THRESHOLD) && node.num != 0 {
                continue;
            }
        } else {
            continue;
        }

        // gain and weight (leaf value, predicted value) are calculated here. line: 939
        let (mut left_node_info, mut right_node_info, missing_info) = match evaluate_fn(
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
            force_children_to_bound_parent,
            missing_node_treatment,
            allow_missing_splits,
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

        if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (generalization.is_some() || node.num == 0) {
            max_gain = Some(split_gain);

            let mut left_cats: HashSet<usize> = HashSet::new();
            let mut right_cats: HashSet<usize> = all_cats.iter().copied().collect();

            for c in all_cats.iter() {
                if *c == b.cut_value as usize {
                    break;
                }
                left_cats.insert(*c);
                right_cats.remove(c);
            }

            split_info.split_gain = split_gain;
            split_info.split_feature = feature;
            split_info.split_value = b.cut_value;
            split_info.split_bin = b.num;
            split_info.left_node = left_node_info;
            split_info.right_node = right_node_info;
            split_info.missing_node = missing_info;
            split_info.generalization = generalization;
            split_info.left_cats = left_cats;
            split_info.right_cats = right_cats;
        }
    }
}

type EvaluateFn = fn(
    f32,
    f32,
    usize,
    f32,
    f32,
    usize,
    f32,
    f32,
    usize,
    f32,
    f32,
    f32,
    Option<&Constraint>,
    bool,
    MissingNodeTreatment,
    bool,
) -> Option<(NodeInfo, NodeInfo, MissingInfo)>;

#[inline]
pub fn eval_callables(is_const_hess: bool, create_missing_branch: bool) -> EvaluateFn {
    match (is_const_hess, create_missing_branch) {
        (true, true) => evaluate_branch_split_const_hess,
        (true, false) => evaluate_impute_split_const_hess,
        (false, true) => evaluate_branch_split_var_hess,
        (false, false) => evaluate_impute_split_var_hess,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn evaluate_impute_split_const_hess(
    left_gradient: f32,
    _left_hessian: f32,
    left_counts: usize,
    right_gradient: f32,
    _right_hessian: f32,
    right_counts: usize,
    missing_gradient: f32,
    _missing_hessian: f32,
    missing_counts: usize,
    lower_bound: f32,
    upper_bound: f32,
    _parent_weight: f32,
    constraint: Option<&Constraint>,
    _force_children_to_bound_parent: bool,
    _missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
    // If there is no info right, or there is no
    // info left, we will possibly lead to a missing only
    // split, if we don't want this, bomb.
    if (left_counts == 0 || right_counts == 0) && !allow_missing_splits {
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn evaluate_impute_split_var_hess(
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
    _force_children_to_bound_parent: bool,
    _missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
    // If there is no info right, or there is no
    // info left, we will possibly lead to a missing only
    // split, if we don't want this, bomb.
    if (left_counts == 0 || right_counts == 0) && !allow_missing_splits {
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

#[inline]
fn evaluate_branch_split_const_hess(
    left_gradient: f32,
    _left_hessian: f32,
    left_counts: usize,
    right_gradient: f32,
    _right_hessian: f32,
    right_counts: usize,
    missing_gradient: f32,
    _missing_hessian: f32,
    missing_counts: usize,
    lower_bound: f32,
    upper_bound: f32,
    parent_weight: f32,
    constraint: Option<&Constraint>,
    force_children_to_bound_parent: bool,
    missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
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

    if force_children_to_bound_parent {
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
    let missing_weight = match missing_node_treatment {
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
                constrained_weight_const_hess(missing_gradient, missing_counts, lower_bound, upper_bound, constraint)
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
        if ((missing_gradient != 0.0) || (missing_counts != 0)) && allow_missing_splits {
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

#[inline]
fn evaluate_branch_split_var_hess(
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
    force_children_to_bound_parent: bool,
    missing_node_treatment: MissingNodeTreatment,
    allow_missing_splits: bool,
) -> Option<(NodeInfo, NodeInfo, MissingInfo)> {
    // If there is no info right, or there is no
    // info left, there is nothing to split on,
    // and so we should continue.
    if (left_gradient == 0.0) && (left_hessian == 0.0) || (right_gradient == 0.0) && (right_hessian == 0.0) {
        return None;
    }

    let mut left_weight = constrained_weight(left_gradient, left_hessian, lower_bound, upper_bound, constraint);
    let mut right_weight = constrained_weight(right_gradient, right_hessian, lower_bound, upper_bound, constraint);

    if force_children_to_bound_parent {
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
    let missing_weight = match missing_node_treatment {
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
        if ((missing_gradient != 0.0) || (missing_hessian != 0.0)) && allow_missing_splits {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::data::Matrix;
    use crate::histogram::NodeHistogramOwned;
    use crate::node::SplittableNode;
    use crate::objective::{LogLoss, ObjectiveFunction, SquaredLoss};
    use crate::tree::create_root_node;
    use crate::utils::gain;
    use crate::utils::weight;
    use polars::prelude::*;
    use std::error::Error;
    use std::fs;

    #[test]
    fn test_data_split() {
        let is_const_hess = false;
        let num_threads = 2;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];

        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let gradient_sum = grad.iter().sum();
        let hessian_sum = match hess {
            Some(ref hess) => hess.iter().sum(),
            None => grad.len() as f32,
        };
        let root_weight = weight(gradient_sum, hessian_sum);
        let root_gain = gain(gradient_sum, hessian_sum);
        let data = Matrix::new(&data_vec, 891, 5);

        let b = bin_matrix(&data, None, 10, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        for i in 0..n_nodes_alloc {
            update_histogram(
                unsafe { &mut hist_tree.get_unchecked(i) },
                0,
                index.len(),
                &bdata,
                &grad,
                hess.as_deref(),
                &index,
                &col_index,
                &pool,
                false,
            );
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
            HashSet::new(),
            HashSet::new(),
        );

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &mut n,
            &col_index,
            false,
            &mut hist_tree,
            &pool,
            None,
            &split_info_slice,
        );
        let s = unsafe { split_info_slice.best_split_info() };
        println!("{:?}", s);

        n.update_children(2, 1, 2, &s);

        assert_eq!(0, s.split_feature);
        assert_eq!(s.split_value, 3.0);
        assert!(between(93.0, 95.0, s.left_node.cover));
        assert!(between(114.0, 116.0, s.right_node.cover));
        assert!(between(7.0, 7.2, s.left_node.gain));
        assert!(between(298.0, 302.0, s.right_node.gain));
        assert!(between(88.0, 95.0, s.split_gain));
    }

    #[test]
    fn test_cal_housing() -> Result<(), Box<dyn Error>> {
        let n_bins = 256;
        let n_cols = 8;
        let is_const_hess = true;

        let feature_names = [
            "MedInc".to_owned(),
            "HouseAge".to_owned(),
            "AveRooms".to_owned(),
            "AveBedrms".to_owned(),
            "Population".to_owned(),
            "AveOccup".to_owned(),
            "Latitude".to_owned(),
            "Longitude".to_owned(),
            "MedHouseVal".to_owned(),
        ];

        let df_test = CsvReadOptions::default()
            .with_has_header(true)
            .with_columns(Some(Arc::new(feature_names)))
            .try_into_reader_with_file_path(Some("resources/cal_housing_test.csv".into()))?
            .finish()
            .unwrap();

        let id_vars: Vec<&str> = Vec::new();

        let mdf_test = df_test.unpivot(
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
            &id_vars,
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

        let splitter = MissingImputerSplitter::new(0.3, false, ConstraintMap::new());

        let data = Matrix::new(&data_test, y_test.len(), n_cols);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        for i in 0..n_nodes_alloc {
            update_histogram(
                unsafe { &mut hist_tree.get_unchecked(i) },
                0,
                index.len(),
                &bdata,
                &grad,
                hess.as_deref(),
                &index,
                &col_index,
                &pool,
                false,
            );
        }

        let mut n = create_root_node(&index, &grad, hess.as_deref());

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &mut n,
            &col_index,
            is_const_hess,
            &mut hist_tree,
            &pool,
            None,
            &split_info_slice,
        );
        let s = unsafe { split_info_slice.best_split_info() };
        println!("{:?}", s);

        n.update_children(2, 1, 2, &s);

        assert_eq!(0, s.split_feature);
        assert!(between(4.8, 5.1, s.split_value as f32));
        Ok(())
    }

    #[test]
    fn test_categorical() -> Result<(), Box<dyn Error>> {
        let n_bins = 256;
        let n_rows = 712;
        let n_cols = 9;
        let is_const_hess = false;
        let eta = 0.1;

        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data_vec_truncated = &data_vec[0..(n_cols * n_rows)];
        let data = Matrix::new(&data_vec_truncated, n_rows, n_cols);

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (grad, hess) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(eta, false, ConstraintMap::new());

        let gradient_sum = grad.iter().copied().sum();
        let hessian_sum = match hess {
            Some(ref hess) => hess.iter().copied().sum(),
            None => grad.len() as f32,
        };
        let root_weight = weight(gradient_sum, hessian_sum);
        let root_gain = gain(gradient_sum, hessian_sum);

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..10)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        for i in 0..10 {
            update_histogram(
                unsafe { &mut hist_tree.get_unchecked(i) },
                0,
                index.len(),
                &bdata,
                &grad,
                hess.as_deref(),
                &index,
                &col_index,
                &pool,
                false,
            );
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
            HashSet::new(),
            HashSet::new(),
        );

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &mut n,
            &col_index,
            is_const_hess,
            &mut hist_tree,
            &pool,
            Some(&cat_index),
            &split_info_slice,
        );
        let s = unsafe { split_info_slice.best_split_info() };

        n.update_children(1, 1, 2, &s);

        println!("split info:");
        println!("{:?}", s);

        println!("left_cats:");
        println!("{:?}", s.left_cats);
        println!("right_cats:");
        println!("{:?}", s.right_cats);

        println!("hist_tree[0]: {:?}", hist_tree_owned[0].data[7]);

        assert_eq!(8, s.split_feature);

        Ok(())
    }

    #[test]
    fn test_gbm_categorical_sensory() -> Result<(), Box<dyn Error>> {
        let n_bins = 256;
        let n_cols = 11;
        let is_const_hess = true;
        let eta = 0.1;

        let file = fs::read_to_string("resources/sensory_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/sensory_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, y.len(), n_cols);

        let cat_index = HashSet::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (grad, hess) = SquaredLoss::calc_grad_hess(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(eta, false, ConstraintMap::new());

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..10)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        for i in 0..10 {
            update_histogram(
                unsafe { &mut hist_tree.get_unchecked(i) },
                0,
                index.len(),
                &bdata,
                &grad,
                None,
                &index,
                &col_index,
                &pool,
                false,
            );
        }

        let mut n = create_root_node(&index, &grad, hess.as_deref());

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &mut n,
            &col_index,
            is_const_hess,
            &mut hist_tree,
            &pool,
            Some(&cat_index),
            &split_info_slice,
        );
        let s = unsafe { split_info_slice.best_split_info() };

        n.update_children(1, 1, 2, &s);

        println!("split_info_slice:");
        for s_data in split_info_slice.data.iter() {
            println!("{:?}", unsafe { &s_data.get().as_ref().unwrap() });
        }

        println!("split info:");
        println!("{:?}", s);

        println!("left_cats:");
        println!("{:?}", s.left_cats);
        println!("right_cats:");
        println!("{:?}", s.right_cats);

        println!("hist_tree.0.6: {:?}", hist_tree_owned[0].data[6]);

        assert_eq!(6, s.split_feature);

        Ok(())
    }
}
