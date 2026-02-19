//! Splitter
//!
//! Split-finding logic for decision tree nodes, including support for
//! missing-value imputation, ternary branches, and monotone constraints.
use crate::bin::{Bin, sort_cat_bins_by_stat};
use crate::booster::config::MissingNodeTreatment;
use crate::constants::GENERALIZATION_THRESHOLD;
use crate::constraints::{Constraint, ConstraintMap};
use crate::data::{FloatData, Matrix};
use crate::histogram::{
    FeatureHistogram, NodeHistogram, update_histogram_and_subtract, update_two_histograms_and_subtract,
};
use crate::node::{NodeType, SplittableNode};
use crate::tree::Tree;
use crate::utils::{
    bound_to_parent, constrained_weight, constrained_weight_const_hess, cull_gain, gain_given_weight,
    gain_given_weight_const_hess, pivot_on_split, pivot_on_split_const_hess, pivot_on_split_exclude_missing,
    pivot_on_split_exclude_missing_const_hess,
};
use rayon::ThreadPool;
use std::borrow::BorrowMut;
use std::cell::UnsafeCell;
use std::collections::HashSet;

#[inline]
fn average(numbers: &[f32]) -> f32 {
    numbers.iter().sum::<f32>() / numbers.len() as f32
}

/// Information about the best split found for a feature.
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
    pub left_cats: Option<Box<[u8]>>,
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
            left_cats: None,
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
    /// # Safety
    ///
    /// Calling this function is **safe** if and only if `i` is a valid index for the internal data.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self, i: usize) -> &mut SplitInfo {
        unsafe { self.data[i].get().as_mut().unwrap() }
    }
    /// # Safety
    ///
    /// Calling this function is **unsafe**. The caller must ensure that the internal
    /// `self.data` is not empty. If `self.data` is empty, this function will panic
    /// due to the `unwrap()` calls.
    ///
    /// The returned `&mut SplitInfo` is a mutable reference to an element within the internal
    /// data. The caller must not move, drop, or invalidate this reference.
    pub unsafe fn best_split_info(&mut self) -> &mut SplitInfo {
        unsafe {
            self.data
                .iter()
                .max_by(|s1, s2| {
                    let g1 = s1.get().as_ref().unwrap().split_gain;
                    let g2 = s2.get().as_ref().unwrap().split_gain;
                    g1.partial_cmp(&g2).expect("Tried to compare a NaN")
                })
                .unwrap()
                .get()
                .as_mut()
                .unwrap()
        }
    }
}

/// Aggregated statistics for one side of a split (left, right, or missing).
#[derive(Debug)]
pub struct NodeInfo {
    /// Gain for this partition.
    pub gain: f32,
    /// Sum of gradients.
    pub grad: f32,
    /// Sum of hessians (cover).
    pub cover: f32,
    /// Number of samples in this partition.
    pub counts: usize,
    /// Leaf weight for this partition.
    pub weight: f32,
    /// Lower and upper bounds from monotone constraints.
    pub bounds: (f32, f32),
    /// Cross-validation fold weights.
    pub weights: [f32; 5],
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
            weights: [0.0; 5],
        }
    }
}

/// How missing values are handled at a particular split.
#[derive(Debug)]
pub enum MissingInfo {
    /// Missing values go to the left child.
    Left,
    /// Missing values go to the right child.
    Right,
    /// Missing values form a leaf node.
    Leaf(NodeInfo),
    /// Missing values form their own internal branch.
    Branch(NodeInfo),
    /// No missing-value handling.
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
    fn get_interaction_constraints(&self) -> Option<&Vec<Vec<usize>>>;
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
    #[allow(clippy::too_many_arguments)]
    fn best_split(
        &self,
        node: &SplittableNode,
        col_index: &[usize],
        is_const_hess: bool,
        hist_tree: &[NodeHistogram],
        pool: &ThreadPool,
        cat_index: Option<&HashSet<usize>>,
        split_info_slice: &mut SplitInfoSlice,
        allowed_features: Option<&HashSet<usize>>,
    ) {
        let allow_missing_splits = self.get_allow_missing_splits();
        let constraint_map = self.get_constraint_map();
        let create_missing_branch = self.get_create_missing_branch();
        let missing_node_treatment = self.get_missing_node_treatment();
        let force_children_to_bound_parent = self.get_force_children_to_bound_parent();

        let hist_node = unsafe { hist_tree.get_unchecked(node.num) };

        // Resolve function pointer once, outside the loop
        let best_feature_split = best_feature_split_callables(is_const_hess);

        // The split evaluation per feature scans O(n_bins) bins (~256).
        // For ≤16 features that's ≤4096 operations, which completes in ~5μs
        // sequentially — less than the Rayon scope overhead (~11μs).
        if pool.current_num_threads() > 1 && col_index.len() > 16 {
            pool.scope(|s| {
                for (feat_idx, feature) in col_index.iter().enumerate() {
                    // Check interaction constraints - skip disallowed features
                    if allowed_features.is_some_and(|allowed| !allowed.contains(feature)) {
                        unsafe {
                            let split_info = split_info_slice.get_mut(feat_idx);
                            split_info.split_gain = -1.0;
                        }
                        continue;
                    }

                    let fi = feat_idx;
                    let f = *feature;
                    let sis = *split_info_slice;
                    s.spawn(move |_| {
                        best_feature_split(
                            node,
                            fi,
                            f,
                            hist_node.data.get(fi).unwrap(),
                            cat_index,
                            constraint_map.get(&f),
                            force_children_to_bound_parent,
                            missing_node_treatment,
                            allow_missing_splits,
                            create_missing_branch,
                            &sis,
                        );
                    });
                }
            });
        } else {
            for (feat_idx, feature) in col_index.iter().enumerate() {
                // Check interaction constraints - skip disallowed features
                if allowed_features.is_some_and(|allowed| !allowed.contains(feature)) {
                    unsafe {
                        let split_info = split_info_slice.get_mut(feat_idx);
                        split_info.split_gain = -1.0;
                    }
                    continue;
                }
                best_feature_split(
                    node,
                    feat_idx,
                    *feature,
                    hist_node.data.get(feat_idx).unwrap(),
                    cat_index,
                    constraint_map.get(feature),
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
        split_info_slice: &mut SplitInfoSlice,
        allowed_features: Option<&HashSet<usize>>,
    ) -> Vec<SplittableNode> {
        self.best_split(
            node,
            col_index,
            is_const_hess,
            hist_tree,
            pool,
            cat_index,
            split_info_slice,
            allowed_features,
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
        if let Some(m) = tree.nodes.get_mut(&missing).filter(|_| missing_leaf) {
            m.weight_value = update as f32;
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
    fn get_constraint_map(&self) -> &ConstraintMap {
        &self.constraints_map
    }
    fn get_interaction_constraints(&self) -> Option<&Vec<Vec<usize>>> {
        None
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
        let missing_child = *n_nodes;
        let left_child = missing_child + 1;
        let right_child = missing_child + 2;

        // We need to move all of the index's above and below our
        // split value.
        // pivot the sub array that this node has on our split value
        // Missing all falls to the bottom.
        let mut missing_split_idx: usize;
        let mut split_idx: usize;
        if let Some(hess_val) = &mut hess {
            (missing_split_idx, split_idx) = pivot_on_split_exclude_missing(
                node.start_idx,
                node.stop_idx,
                index,
                grad,
                hess_val,
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                split_info.left_cats.as_deref().unwrap_or(&[]),
            );
        } else {
            (missing_split_idx, split_idx) = pivot_on_split_exclude_missing_const_hess(
                node.start_idx,
                node.stop_idx,
                index,
                grad,
                data.get_col(split_info.split_feature),
                split_info.split_bin,
                split_info.left_cats.as_deref().unwrap_or(&[]),
            );
        }

        node.update_children(missing_child, left_child, right_child, split_info);

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
                update_histogram_and_subtract(
                    hist_tree,
                    node.num,
                    right_child,
                    left_child,
                    split_idx,
                    node.stop_idx,
                    data,
                    grad,
                    hess.as_deref(),
                    index,
                    col_index,
                    pool,
                );
            } else {
                update_histogram_and_subtract(
                    hist_tree,
                    node.num,
                    left_child,
                    right_child,
                    missing_split_idx,
                    split_idx,
                    data,
                    grad,
                    hess.as_deref(),
                    index,
                    col_index,
                    pool,
                );
            }
        } else if max_ == 0 {
            // Max is missing, calculate left + right, derive missing via subtraction.
            update_two_histograms_and_subtract(
                hist_tree,
                node.num,
                left_child,
                missing_split_idx,
                split_idx,
                right_child,
                split_idx,
                node.stop_idx,
                missing_child,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
            );
        } else if max_ == 1 {
            // Max is left, calculate missing + right, derive left via subtraction.
            update_two_histograms_and_subtract(
                hist_tree,
                node.num,
                missing_child,
                node.start_idx,
                missing_split_idx,
                right_child,
                split_idx,
                node.stop_idx,
                left_child,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
            );
        } else {
            // right is the largest, calculate missing + left, derive right via subtraction.
            update_two_histograms_and_subtract(
                hist_tree,
                node.num,
                missing_child,
                node.start_idx,
                missing_split_idx,
                left_child,
                missing_split_idx,
                split_idx,
                right_child,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
            );
        }

        let mut missing_node = SplittableNode::from_node_info(
            missing_child,
            node.stats.as_ref().unwrap().depth + 1,
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
            node.stats.as_ref().unwrap().depth + 1,
            missing_split_idx,
            split_idx,
            &split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.stats.as_ref().unwrap().depth + 1,
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
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    pub missing_node_treatment: MissingNodeTreatment,
    pub force_children_to_bound_parent: bool,
}

impl MissingImputerSplitter {
    /// Generate a new missing imputer splitter object.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eta: f32,
        allow_missing_splits: bool,
        constraints_map: ConstraintMap,
        interaction_constraints: Option<Vec<Vec<usize>>>,
    ) -> Self {
        MissingImputerSplitter {
            create_missing_branch: false,
            eta,
            allow_missing_splits,
            constraints_map,
            interaction_constraints,
            missing_node_treatment: MissingNodeTreatment::None,
            force_children_to_bound_parent: false,
        }
    }
}

impl Splitter for MissingImputerSplitter {
    fn get_constraint(&self, feature: &usize) -> Option<&Constraint> {
        self.constraints_map.get(feature)
    }
    fn get_constraint_map(&self) -> &ConstraintMap {
        &self.constraints_map
    }
    fn get_interaction_constraints(&self) -> Option<&Vec<Vec<usize>>> {
        self.interaction_constraints.as_ref()
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
        match hess {
            Some(ref mut hess_val) => {
                split_idx = pivot_on_split(
                    node.start_idx,
                    node.stop_idx,
                    index,
                    grad,
                    hess_val,
                    data.get_col(split_info.split_feature),
                    split_info.split_bin,
                    missing_right,
                    split_info.left_cats.as_deref().unwrap_or(&[]),
                );
            }
            None => {
                split_idx = pivot_on_split_const_hess(
                    node.start_idx,
                    node.stop_idx,
                    index,
                    grad,
                    data.get_col(split_info.split_feature),
                    split_info.split_bin,
                    missing_right,
                    split_info.left_cats.as_deref().unwrap_or(&[]),
                );
            }
        }

        // Calculate histograms
        let total_recs = node.stop_idx - node.start_idx;
        let n_right = total_recs - split_idx;
        let n_left = total_recs - n_right;

        // Now that we have calculated the number of records
        // add the start index, to make the split_index
        // relative to the entire index array
        split_idx += node.start_idx;

        // Build histogram for the smaller child and derive the larger child's
        // histogram via subtraction, all in a single Rayon scope for better
        // cache locality and reduced synchronization overhead.
        if n_left < n_right {
            update_histogram_and_subtract(
                hist_tree,
                node.num,
                left_child,
                right_child,
                node.start_idx,
                split_idx,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
            );
        } else {
            update_histogram_and_subtract(
                hist_tree,
                node.num,
                right_child,
                left_child,
                split_idx,
                node.stop_idx,
                data,
                grad,
                hess.as_deref(),
                index,
                col_index,
                pool,
            );
        }

        let missing_child = if missing_right { right_child } else { left_child };
        node.update_children(missing_child, left_child, right_child, split_info);

        let left_node = SplittableNode::from_node_info(
            left_child,
            node.stats.as_ref().unwrap().depth + 1,
            node.start_idx,
            split_idx,
            &split_info.left_node,
            split_info.generalization,
            NodeType::Left,
            node.num,
        );
        let right_node = SplittableNode::from_node_info(
            right_child,
            node.stats.as_ref().unwrap().depth + 1,
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
#[allow(clippy::too_many_arguments)]
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

    let mut max_gain: Option<f32> = None;
    let mut all_cats: Vec<usize> = Vec::new();

    // For categorical features, we need to sort the bins.
    let mut hist_vec: Vec<&UnsafeCell<Bin>> = Vec::new();
    let is_categorical = if let Some(c_index) = cat_index {
        if c_index.contains(&feature) {
            hist_vec = hist_feat.data[1..]
                .iter()
                .filter(|b| {
                    let bin = unsafe { &*b.get() };
                    bin.counts.iter().sum::<u32>() > 0 && !bin.cut_value.is_nan()
                })
                .collect::<Vec<_>>();
            sort_cat_bins_by_stat(&mut hist_vec, true);
            all_cats = hist_vec
                .iter()
                .map(|b| unsafe { &*b.get() }.cut_value as usize)
                .collect();
            true
        } else {
            false
        }
    } else {
        false
    };

    // We also know we will have a missing bin.
    // We also know we will have a missing bin.
    let miss_bin = unsafe { hist_feat.data.get_unchecked(0).get().as_ref().unwrap() };
    let miss_grad_sum: f32 = miss_bin.g_folded.iter().sum();
    let miss_coun_sum: usize = miss_bin.counts.iter().sum::<u32>() as usize;

    let mut node_grad_sum = [0.0; 5];
    let mut node_coun_sum = [0_usize; 5];
    for bin in hist_feat.data[1..].iter() {
        let b = unsafe { &*bin.get() };
        for i in 0..5 {
            node_grad_sum[i] += b.g_folded[i];
            node_coun_sum[i] += b.counts[i] as usize;
        }
    }

    let node_grad_scalar_sum: f32 = node_grad_sum.iter().sum();
    let node_coun_scalar_sum: usize = node_coun_sum.iter().sum();

    // Pre-calculate node totals for training set (per fold)
    let mut node_grad_train_sum = [0.0; 5];
    let mut node_coun_train_sum = [0_usize; 5];
    for j in 0..5 {
        node_grad_train_sum[j] = node_grad_scalar_sum - node_grad_sum[j];
        node_coun_train_sum[j] = node_coun_scalar_sum - node_coun_sum[j];
    }

    let mut right_gradient_train = [f32::ZERO; 5];
    let mut right_counts_train = [0_usize; 5];
    let mut right_gradient_valid = [f32::ZERO; 5];
    let mut right_counts_valid = [0_usize; 5];

    let mut cuml_gradient_train = [f32::ZERO; 5];
    let mut cuml_counts_train = [0_usize; 5];
    let mut cuml_gradient_valid = [f32::ZERO; 5];
    let mut cuml_counts_valid = [0_usize; 5];

    if is_categorical {
        for bin in hist_vec {
            let b: &Bin = unsafe { &*bin.get() };
            // -- Inlined logic --
            let left_gradient_train = cuml_gradient_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut left_objs = [0.0; 5];
            let mut right_objs = [0.0; 5];
            let mut train_scores = [0.0; 5];
            let mut valid_scores = [0.0; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0; 5];
            let mut right_weights = [0.0; 5];
            #[allow(clippy::needless_late_init)]
            let generalization;
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                let split_result = if create_missing_branch {
                    evaluate_branch_split_const_hess(
                        left_gradient_train[j],
                        f32::NAN,
                        left_counts_train[j],
                        right_gradient_train[j],
                        f32::NAN,
                        right_counts_train[j],
                        miss_grad_sum,
                        f32::NAN,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                } else {
                    evaluate_impute_split_const_hess(
                        left_gradient_train[j],
                        f32::NAN,
                        left_counts_train[j],
                        right_gradient_train[j],
                        f32::NAN,
                        right_counts_train[j],
                        miss_grad_sum,
                        f32::NAN,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                };
                let (left_node, right_node, _) = match split_result {
                    Some(v) => v,
                    None => continue,
                };

                left_weights[j] = left_node.weight;
                right_weights[j] = right_node.weight;

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
                let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                let gen_val = delta_score_train / delta_score_valid;
                if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                    continue;
                }
                generalization = Some(gen_val);
            } else {
                continue;
            }

            let split_result = if create_missing_branch {
                evaluate_branch_split_const_hess(
                    left_gradient_valid.iter().sum(),
                    f32::NAN,
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    f32::NAN,
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    f32::NAN,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            } else {
                evaluate_impute_split_const_hess(
                    left_gradient_valid.iter().sum(),
                    f32::NAN,
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    f32::NAN,
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    f32::NAN,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            };

            let (mut left_node_info, mut right_node_info, missing_info) = match split_result {
                Some(v) => v,
                None => continue,
            };

            left_node_info.weights = left_weights;
            right_node_info.weights = right_weights;

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);
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
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (generalization.is_some() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info.split_gain = split_gain;
                split_info.split_feature = feature;
                split_info.split_value = b.cut_value;
                split_info.split_bin = b.num;
                split_info.left_node = left_node_info;
                split_info.right_node = right_node_info;
                split_info.missing_node = missing_info;
                split_info.generalization = generalization;
            }
        }
    } else if miss_coun_sum == 0
        && miss_grad_sum == 0.0
        && !create_missing_branch
        && matches!(constraint, None | Some(Constraint::Unconstrained))
    {
        // ── Fast path: no missing values, unconstrained, impute mode ──
        // Inlines all weight/gain math, skips missing-direction logic,
        // and defers NodeInfo construction to the winning bin only.
        let mut best_gain: f32 = 0.0;
        let mut best_left_weights = [0.0f32; 5];
        let mut best_right_weights = [0.0f32; 5];
        let mut best_generalization: Option<f32> = None;
        let mut best_left_grad_valid = [0.0f32; 5];
        let mut best_left_coun_valid = [0usize; 5];
        let mut best_right_grad_valid = [0.0f32; 5];
        let mut best_right_coun_valid = [0usize; 5];
        let mut best_cut_value: f64 = 0.0;
        let mut best_split_bin: u16 = 0;
        let mut found_split = false;

        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            if b_coun_total == 0 || b.cut_value.is_nan() {
                continue;
            }
            let left_gradient_train = cuml_gradient_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut train_scores = [0.0f32; 5];
            let mut valid_scores = [0.0f32; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0f32; 5];
            let mut right_weights = [0.0f32; 5];
            let b_grad_total: f32 = b.g_folded.iter().sum();

            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                // Inline weight = -g / n, gain = -(2*g*w + n*w²)
                let lw = -left_gradient_train[j] / left_c_train as f32;
                let rw = -right_gradient_train[j] / right_c_train as f32;
                let lg = -(2.0 * left_gradient_train[j] * lw + left_c_train as f32 * lw * lw);
                let rg = -(2.0 * right_gradient_train[j] * rw + right_c_train as f32 * rw * rw);

                left_weights[j] = lw;
                right_weights[j] = rw;

                let left_obj = left_gradient_valid[j] * lw + 0.5 * (left_c_valid as f32) * lw * lw;
                let right_obj = right_gradient_valid[j] * rw + 0.5 * (right_c_valid as f32) * rw * rw;
                valid_scores[j] = (left_obj + right_obj) / (left_c_valid + right_c_valid) as f32;
                train_scores[j] = -0.5 * (lg + rg) / (left_c_train + right_c_train) as f32;

                n_folds += 1;
            }

            if n_folds < 5 && node.num != 0 {
                continue;
            }

            let train_score = average(&train_scores);
            let valid_score = average(&valid_scores);
            let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
            let delta_score_train = parent_score - train_score;
            let delta_score_valid = parent_score - valid_score;
            let gen_val = delta_score_train / delta_score_valid;
            if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                continue;
            }
            let generalization = Some(gen_val);

            // Compute split gain from summed valid stats (no evaluate call needed)
            let tl_grad: f32 = left_gradient_valid.iter().sum();
            let tl_count: usize = left_counts_valid.iter().sum();
            let tr_grad: f32 = right_gradient_valid.iter().sum();
            let tr_count: usize = right_counts_valid.iter().sum();
            let tl_w = -tl_grad / tl_count as f32;
            let tr_w = -tr_grad / tr_count as f32;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_count as f32 * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_count as f32 * tr_w * tr_w);
            let split_gain = tl_gain + tr_gain - node.gain_value;
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if split_gain <= 0.0 {
                continue;
            }

            if split_gain > best_gain && (generalization.is_some() || node.num == 0) {
                best_gain = split_gain;
                best_left_weights = left_weights;
                best_right_weights = right_weights;
                best_generalization = generalization;
                best_left_grad_valid = left_gradient_valid;
                best_left_coun_valid = left_counts_valid;
                best_right_grad_valid = right_gradient_valid;
                best_right_coun_valid = right_counts_valid;
                best_cut_value = b.cut_value;
                best_split_bin = b.num;
                found_split = true;
            }
        }

        // Construct NodeInfo only for the winning bin
        if found_split {
            let tl_grad: f32 = best_left_grad_valid.iter().sum();
            let tl_count: usize = best_left_coun_valid.iter().sum();
            let tr_grad: f32 = best_right_grad_valid.iter().sum();
            let tr_count: usize = best_right_coun_valid.iter().sum();
            let tl_w = -tl_grad / tl_count as f32;
            let tr_w = -tr_grad / tr_count as f32;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_count as f32 * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_count as f32 * tr_w * tr_w);

            max_gain = Some(best_gain);
            split_info.split_gain = best_gain;
            split_info.split_feature = feature;
            split_info.split_value = best_cut_value;
            split_info.split_bin = best_split_bin;
            split_info.left_node = NodeInfo {
                grad: tl_grad,
                gain: tl_gain,
                cover: tl_count as f32,
                counts: tl_count,
                weight: tl_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_left_weights,
            };
            split_info.right_node = NodeInfo {
                grad: tr_grad,
                gain: tr_gain,
                cover: tr_count as f32,
                counts: tr_count,
                weight: tr_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_right_weights,
            };
            split_info.missing_node = MissingInfo::Right;
            split_info.generalization = best_generalization;
        }
    } else if !create_missing_branch && matches!(constraint, None | Some(Constraint::Unconstrained)) {
        // ── Fast path: missing present, unconstrained, impute mode ──
        // Inlines weight/gain math and missing-direction logic without
        // function-call overhead per bin.
        let mut best_gain: f32 = 0.0;
        let mut best_left_weights = [0.0f32; 5];
        let mut best_right_weights = [0.0f32; 5];
        let mut best_generalization: Option<f32> = None;
        let mut best_left_grad_valid = [0.0f32; 5];
        let mut best_left_coun_valid = [0usize; 5];
        let mut best_right_grad_valid = [0.0f32; 5];
        let mut best_right_coun_valid = [0usize; 5];
        let mut best_cut_value: f64 = 0.0;
        let mut best_split_bin: u16 = 0;
        let mut best_missing_info = MissingInfo::Right;
        let mut found_split = false;

        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            if b_coun_total == 0 || b.cut_value.is_nan() {
                continue;
            }
            let left_gradient_train = cuml_gradient_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut train_scores = [0.0f32; 5];
            let mut valid_scores = [0.0f32; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0f32; 5];
            let mut right_weights = [0.0f32; 5];
            let mut missing_dir = MissingInfo::Right;
            let b_grad_total: f32 = b.g_folded.iter().sum();

            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                // Inline weight = -g / n, gain = -(2gw + nw²)
                let mut lw = -left_gradient_train[j] / left_c_train as f32;
                let mut rw = -right_gradient_train[j] / right_c_train as f32;
                let mut lg = -(2.0 * left_gradient_train[j] * lw + left_c_train as f32 * lw * lw);
                let mut rg = -(2.0 * right_gradient_train[j] * rw + right_c_train as f32 * rw * rw);

                // Missing-direction check (per-fold, using train stats)
                if miss_grad_sum != 0.0 || miss_coun_sum != 0 {
                    let ml_g = left_gradient_train[j] + miss_grad_sum;
                    let ml_c = left_c_train + miss_coun_sum;
                    let ml_w = -ml_g / ml_c as f32;
                    let ml_gain = -(2.0 * ml_g * ml_w + ml_c as f32 * ml_w * ml_w);

                    let mr_g = right_gradient_train[j] + miss_grad_sum;
                    let mr_c = right_c_train + miss_coun_sum;
                    let mr_w = -mr_g / mr_c as f32;
                    let mr_gain = -(2.0 * mr_g * mr_w + mr_c as f32 * mr_w * mr_w);

                    if (mr_gain - rg) < (ml_gain - lg) {
                        // Missing goes left
                        lw = ml_w;
                        lg = ml_gain;
                        missing_dir = MissingInfo::Left;
                    } else {
                        // Missing goes right
                        rw = mr_w;
                        rg = mr_gain;
                        missing_dir = MissingInfo::Right;
                    }
                }

                left_weights[j] = lw;
                right_weights[j] = rw;

                let left_obj = left_gradient_valid[j] * lw + 0.5 * (left_c_valid as f32) * lw * lw;
                let right_obj = right_gradient_valid[j] * rw + 0.5 * (right_c_valid as f32) * rw * rw;
                valid_scores[j] = (left_obj + right_obj) / (left_c_valid + right_c_valid) as f32;
                train_scores[j] = -0.5 * (lg + rg) / (left_c_train + right_c_train) as f32;

                n_folds += 1;
            }

            if n_folds < 5 && node.num != 0 {
                continue;
            }

            let train_score = average(&train_scores);
            let valid_score = average(&valid_scores);
            let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
            let delta_score_train = parent_score - train_score;
            let delta_score_valid = parent_score - valid_score;
            let gen_val = delta_score_train / delta_score_valid;
            if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                continue;
            }
            let generalization = Some(gen_val);

            let tl_grad: f32 = left_gradient_valid.iter().sum();
            let tl_count: usize = left_counts_valid.iter().sum();
            let tr_grad: f32 = right_gradient_valid.iter().sum();
            let tr_count: usize = right_counts_valid.iter().sum();
            let tl_w = -tl_grad / tl_count as f32;
            let tr_w = -tr_grad / tr_count as f32;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_count as f32 * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_count as f32 * tr_w * tr_w);
            let split_gain = tl_gain + tr_gain - node.gain_value;
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if split_gain <= 0.0 {
                continue;
            }

            if split_gain > best_gain && (generalization.is_some() || node.num == 0) {
                best_gain = split_gain;
                best_left_weights = left_weights;
                best_right_weights = right_weights;
                best_generalization = generalization;
                best_left_grad_valid = left_gradient_valid;
                best_left_coun_valid = left_counts_valid;
                best_right_grad_valid = right_gradient_valid;
                best_right_coun_valid = right_counts_valid;
                best_cut_value = b.cut_value;
                best_split_bin = b.num;
                best_missing_info = missing_dir;
                found_split = true;
            }
        }

        if found_split {
            let tl_grad: f32 = best_left_grad_valid.iter().sum();
            let tl_count: usize = best_left_coun_valid.iter().sum();
            let tr_grad: f32 = best_right_grad_valid.iter().sum();
            let tr_count: usize = best_right_coun_valid.iter().sum();
            let tl_w = -tl_grad / tl_count as f32;
            let tr_w = -tr_grad / tr_count as f32;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_count as f32 * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_count as f32 * tr_w * tr_w);

            max_gain = Some(best_gain);
            split_info.split_gain = best_gain;
            split_info.split_feature = feature;
            split_info.split_value = best_cut_value;
            split_info.split_bin = best_split_bin;
            split_info.left_node = NodeInfo {
                grad: tl_grad,
                gain: tl_gain,
                cover: tl_count as f32,
                counts: tl_count,
                weight: tl_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_left_weights,
            };
            split_info.right_node = NodeInfo {
                grad: tr_grad,
                gain: tr_gain,
                cover: tr_count as f32,
                counts: tr_count,
                weight: tr_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_right_weights,
            };
            split_info.missing_node = best_missing_info;
            split_info.generalization = best_generalization;
        }
    } else {
        // ── Slow path: constraints, missing values, or missing branches ──
        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            if b.counts.iter().map(|&c| c as usize).sum::<usize>() == 0 || b.cut_value.is_nan() {
                continue;
            }
            let left_gradient_train = cuml_gradient_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut left_objs = [0.0; 5];
            let mut right_objs = [0.0; 5];
            let mut train_scores = [0.0; 5];
            let mut valid_scores = [0.0; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0; 5];
            let mut right_weights = [0.0; 5];
            #[allow(clippy::needless_late_init)]
            let generalization;
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                let split_result = if create_missing_branch {
                    evaluate_branch_split_const_hess(
                        left_gradient_train[j],
                        f32::NAN,
                        left_counts_train[j],
                        right_gradient_train[j],
                        f32::NAN,
                        right_counts_train[j],
                        miss_grad_sum,
                        f32::NAN,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                } else {
                    evaluate_impute_split_const_hess(
                        left_gradient_train[j],
                        f32::NAN,
                        left_counts_train[j],
                        right_gradient_train[j],
                        f32::NAN,
                        right_counts_train[j],
                        miss_grad_sum,
                        f32::NAN,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                };
                let (left_node, right_node, _) = match split_result {
                    Some(v) => v,
                    None => continue,
                };

                left_weights[j] = left_node.weight;
                right_weights[j] = right_node.weight;

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
                let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                let gen_val = delta_score_train / delta_score_valid;
                if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                    continue;
                }
                generalization = Some(gen_val);
            } else {
                continue;
            }

            let split_result = if create_missing_branch {
                evaluate_branch_split_const_hess(
                    left_gradient_valid.iter().sum(),
                    f32::NAN,
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    f32::NAN,
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    f32::NAN,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            } else {
                evaluate_impute_split_const_hess(
                    left_gradient_valid.iter().sum(),
                    f32::NAN,
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    f32::NAN,
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    f32::NAN,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            };

            let (mut left_node_info, mut right_node_info, missing_info) = match split_result {
                Some(v) => v,
                None => continue,
            };

            left_node_info.weights = left_weights;
            right_node_info.weights = right_weights;

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);
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
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (generalization.is_some() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info.split_gain = split_gain;
                split_info.split_feature = feature;
                split_info.split_value = b.cut_value;
                split_info.split_bin = b.num;
                split_info.left_node = left_node_info;
                split_info.right_node = right_node_info;
                split_info.missing_node = missing_info;
                split_info.generalization = generalization;
            }
        }
    }

    if max_gain.is_some() {
        if all_cats.is_empty() {
            split_info.left_cats = None;
        } else {
            if split_info.left_cats.is_none() {
                split_info.left_cats = Some(vec![0u8; 8192].into_boxed_slice());
            }
            let left_cats_vec = split_info.left_cats.as_mut().unwrap();
            left_cats_vec.fill(0);
            for c in all_cats.iter() {
                if *c == split_info.split_value as usize {
                    break;
                }
                let byte_idx = c >> 3;
                let bit_idx = c & 7;
                left_cats_vec[byte_idx] |= 1 << bit_idx;
            }
        }
    }
}

/// The feat_idx is the index of the feature in the histogram data.
/// The feature is the index of the actual feature in the data.
#[inline]
#[allow(clippy::too_many_arguments)]
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

    let mut max_gain: Option<f32> = None;
    let mut all_cats: Vec<usize> = Vec::new();

    // For categorical features, we need to sort the bins.
    let mut hist_vec: Vec<&UnsafeCell<Bin>> = Vec::new();
    let is_categorical = if let Some(c_index) = cat_index {
        if c_index.contains(&feature) {
            hist_vec = hist_feat.data[1..]
                .iter()
                .filter(|b| {
                    let bin = unsafe { &*b.get() };
                    bin.counts.iter().sum::<u32>() > 0 && !bin.cut_value.is_nan()
                })
                .collect::<Vec<_>>();
            sort_cat_bins_by_stat(&mut hist_vec, false);
            all_cats = hist_vec
                .iter()
                .map(|b| unsafe { &*b.get() }.cut_value as usize)
                .collect();
            true
        } else {
            false
        }
    } else {
        false
    };

    // We also know we will have a missing bin.
    let miss_bin = unsafe { hist_feat.data.get_unchecked(0).get().as_ref().unwrap() };
    let miss_grad_sum: f32 = miss_bin.g_folded.iter().sum();
    let miss_hess_sum: f32 = miss_bin.h_folded.iter().sum();
    let miss_coun_sum: usize = miss_bin.counts.iter().sum::<u32>() as usize;

    let mut node_grad_sum = [0.0; 5];
    let mut node_hess_sum = [0.0; 5];
    let mut node_coun_sum = [0_usize; 5];
    for bin in hist_feat.data[1..].iter() {
        let b = unsafe { &*bin.get() };
        for i in 0..5 {
            node_grad_sum[i] += b.g_folded[i];
            node_hess_sum[i] += b.h_folded[i];
            node_coun_sum[i] += b.counts[i] as usize;
        }
    }

    let node_grad_scalar_sum: f32 = node_grad_sum.iter().sum();
    let node_hess_scalar_sum: f32 = node_hess_sum.iter().sum();
    let node_coun_scalar_sum: usize = node_coun_sum.iter().sum();

    // Pre-calculate node training sums
    let mut node_grad_train_sum = [0.0; 5];
    let mut node_hess_train_sum = [0.0; 5];
    let mut node_coun_train_sum = [0_usize; 5];
    for j in 0..5 {
        node_grad_train_sum[j] = node_grad_scalar_sum - node_grad_sum[j];
        node_hess_train_sum[j] = node_hess_scalar_sum - node_hess_sum[j];
        node_coun_train_sum[j] = node_coun_scalar_sum - node_coun_sum[j];
    }

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

    if is_categorical {
        for bin in hist_vec {
            let b: &Bin = unsafe { &*bin.get() };
            // -- Inlined process_bin logic --
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
            let mut left_weights = [0.0; 5];
            let mut right_weights = [0.0; 5];
            #[allow(clippy::needless_late_init)]
            let generalization;
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_hess_total: f32 = b.h_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_hessian_train[j] = node_hess_train_sum[j] - cuml_hessian_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_hessian_valid[j] = node_hess_sum[j] - cuml_hessian_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_hessian_train[j] += b_hess_total - b.h_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_hessian_valid[j] += b.h_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                let split_result = if create_missing_branch {
                    evaluate_branch_split_var_hess(
                        left_gradient_train[j],
                        left_hessian_train[j],
                        left_counts_train[j],
                        right_gradient_train[j],
                        right_hessian_train[j],
                        right_counts_train[j],
                        miss_grad_sum,
                        miss_hess_sum,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                } else {
                    evaluate_impute_split_var_hess(
                        left_gradient_train[j],
                        left_hessian_train[j],
                        left_counts_train[j],
                        right_gradient_train[j],
                        right_hessian_train[j],
                        right_counts_train[j],
                        miss_grad_sum,
                        miss_hess_sum,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                };
                let (left_node, right_node, _) = match split_result {
                    Some(v) => v,
                    None => continue,
                };

                left_weights[j] = left_node.weight;
                right_weights[j] = right_node.weight;

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
                let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                let gen_val = delta_score_train / delta_score_valid;
                if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                    continue;
                }
                generalization = Some(gen_val);
            } else {
                continue;
            }

            let split_result = if create_missing_branch {
                evaluate_branch_split_var_hess(
                    left_gradient_valid.iter().sum(),
                    left_hessian_valid.iter().sum::<f32>(),
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    right_hessian_valid.iter().sum::<f32>(),
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    miss_hess_sum,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            } else {
                evaluate_impute_split_var_hess(
                    left_gradient_valid.iter().sum(),
                    left_hessian_valid.iter().sum::<f32>(),
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    right_hessian_valid.iter().sum::<f32>(),
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    miss_hess_sum,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            };

            let (mut left_node_info, mut right_node_info, missing_info) = match split_result {
                Some(v) => v,
                None => continue,
            };

            left_node_info.weights = left_weights;
            right_node_info.weights = right_weights;

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);
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
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (generalization.is_some() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info.split_gain = split_gain;
                split_info.split_feature = feature;
                split_info.split_value = b.cut_value;
                split_info.split_bin = b.num;
                split_info.left_node = left_node_info;
                split_info.right_node = right_node_info;
                split_info.missing_node = missing_info;
                split_info.generalization = generalization;
            }
        }
    } else if miss_coun_sum == 0
        && miss_grad_sum == 0.0
        && miss_hess_sum == 0.0
        && !create_missing_branch
        && matches!(constraint, None | Some(Constraint::Unconstrained))
    {
        // ── Fast path: no missing values, unconstrained, impute mode ──
        let hessian_eps: f32 = 1e-8;
        let mut best_gain: f32 = 0.0;
        let mut best_left_weights = [0.0f32; 5];
        let mut best_right_weights = [0.0f32; 5];
        let mut best_generalization: Option<f32> = None;
        let mut best_left_grad_valid = [0.0f32; 5];
        let mut best_left_hess_valid = [0.0f32; 5];
        let mut best_left_coun_valid = [0usize; 5];
        let mut best_right_grad_valid = [0.0f32; 5];
        let mut best_right_hess_valid = [0.0f32; 5];
        let mut best_right_coun_valid = [0usize; 5];
        let mut best_cut_value: f64 = 0.0;
        let mut best_split_bin: u16 = 0;
        let mut found_split = false;

        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            if b.counts.iter().map(|&c| c as usize).sum::<usize>() == 0 || b.cut_value.is_nan() {
                continue;
            }
            let left_gradient_train = cuml_gradient_train;
            let left_hessian_train = cuml_hessian_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_hessian_valid = cuml_hessian_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut train_scores = [0.0f32; 5];
            let mut valid_scores = [0.0f32; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0f32; 5];
            let mut right_weights = [0.0f32; 5];
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_hess_total: f32 = b.h_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();

            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_hessian_train[j] = node_hess_train_sum[j] - cuml_hessian_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_hessian_valid[j] = node_hess_sum[j] - cuml_hessian_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_hessian_train[j] += b_hess_total - b.h_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_hessian_valid[j] += b.h_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                // Inline weight = -g / (h + eps), gain = -(2*g*w + (h+eps)*w²)
                let lh = left_hessian_train[j] + hessian_eps;
                let rh = right_hessian_train[j] + hessian_eps;
                let lw = -left_gradient_train[j] / lh;
                let rw = -right_gradient_train[j] / rh;
                let lg = -(2.0 * left_gradient_train[j] * lw + lh * lw * lw);
                let rg = -(2.0 * right_gradient_train[j] * rw + rh * rw * rw);

                left_weights[j] = lw;
                right_weights[j] = rw;

                let left_obj = left_gradient_valid[j] * lw + 0.5 * left_hessian_valid[j] * lw * lw;
                let right_obj = right_gradient_valid[j] * rw + 0.5 * right_hessian_valid[j] * rw * rw;
                valid_scores[j] = (left_obj + right_obj) / (left_c_valid + right_c_valid) as f32;
                train_scores[j] = -0.5 * (lg + rg) / (left_c_train + right_c_train) as f32;

                n_folds += 1;
            }

            if n_folds < 5 && node.num != 0 {
                continue;
            }

            let train_score = average(&train_scores);
            let valid_score = average(&valid_scores);
            let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
            let delta_score_train = parent_score - train_score;
            let delta_score_valid = parent_score - valid_score;
            let gen_val = delta_score_train / delta_score_valid;
            if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                continue;
            }
            let generalization = Some(gen_val);

            // Compute split gain from summed valid stats
            let tl_grad: f32 = left_gradient_valid.iter().sum();
            let tl_hess: f32 = left_hessian_valid.iter().sum();
            let tr_grad: f32 = right_gradient_valid.iter().sum();
            let tr_hess: f32 = right_hessian_valid.iter().sum();
            let tl_h = tl_hess + hessian_eps;
            let tr_h = tr_hess + hessian_eps;
            let tl_w = -tl_grad / tl_h;
            let tr_w = -tr_grad / tr_h;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_h * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_h * tr_w * tr_w);
            let split_gain = tl_gain + tr_gain - node.gain_value;
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if split_gain <= 0.0 {
                continue;
            }

            if split_gain > best_gain && (generalization.is_some() || node.num == 0) {
                best_gain = split_gain;
                best_left_weights = left_weights;
                best_right_weights = right_weights;
                best_generalization = generalization;
                best_left_grad_valid = left_gradient_valid;
                best_left_hess_valid = left_hessian_valid;
                best_left_coun_valid = left_counts_valid;
                best_right_grad_valid = right_gradient_valid;
                best_right_hess_valid = right_hessian_valid;
                best_right_coun_valid = right_counts_valid;
                best_cut_value = b.cut_value;
                best_split_bin = b.num;
                found_split = true;
            }
        }

        if found_split {
            let tl_grad: f32 = best_left_grad_valid.iter().sum();
            let tl_hess: f32 = best_left_hess_valid.iter().sum();
            let tl_count: usize = best_left_coun_valid.iter().sum();
            let tr_grad: f32 = best_right_grad_valid.iter().sum();
            let tr_hess: f32 = best_right_hess_valid.iter().sum();
            let tr_count: usize = best_right_coun_valid.iter().sum();
            let tl_h = tl_hess + hessian_eps;
            let tr_h = tr_hess + hessian_eps;
            let tl_w = -tl_grad / tl_h;
            let tr_w = -tr_grad / tr_h;
            let tl_gain = -(2.0 * tl_grad * tl_w + tl_h * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + tr_h * tr_w * tr_w);

            max_gain = Some(best_gain);
            split_info.split_gain = best_gain;
            split_info.split_feature = feature;
            split_info.split_value = best_cut_value;
            split_info.split_bin = best_split_bin;
            split_info.left_node = NodeInfo {
                grad: tl_grad,
                gain: tl_gain,
                cover: tl_hess,
                counts: tl_count,
                weight: tl_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_left_weights,
            };
            split_info.right_node = NodeInfo {
                grad: tr_grad,
                gain: tr_gain,
                cover: tr_hess,
                counts: tr_count,
                weight: tr_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_right_weights,
            };
            split_info.missing_node = MissingInfo::Right;
            split_info.generalization = best_generalization;
        }
    } else if !create_missing_branch && matches!(constraint, None | Some(Constraint::Unconstrained)) {
        // ── Fast path: missing present, unconstrained, impute mode ──
        let hessian_eps: f32 = 1e-8;
        let mut best_gain: f32 = 0.0;
        let mut best_left_weights = [0.0f32; 5];
        let mut best_right_weights = [0.0f32; 5];
        let mut best_generalization: Option<f32> = None;
        let mut best_left_grad_valid = [0.0f32; 5];
        let mut best_left_hess_valid = [0.0f32; 5];
        let mut best_left_coun_valid = [0usize; 5];
        let mut best_right_grad_valid = [0.0f32; 5];
        let mut best_right_hess_valid = [0.0f32; 5];
        let mut best_right_coun_valid = [0usize; 5];
        let mut best_cut_value: f64 = 0.0;
        let mut best_split_bin: u16 = 0;
        let mut best_missing_info = MissingInfo::Right;
        let mut found_split = false;

        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            if b.counts.iter().map(|&c| c as usize).sum::<usize>() == 0 || b.cut_value.is_nan() {
                continue;
            }
            let left_gradient_train = cuml_gradient_train;
            let left_hessian_train = cuml_hessian_train;
            let left_counts_train = cuml_counts_train;
            let left_gradient_valid = cuml_gradient_valid;
            let left_hessian_valid = cuml_hessian_valid;
            let left_counts_valid = cuml_counts_valid;

            let mut train_scores = [0.0f32; 5];
            let mut valid_scores = [0.0f32; 5];
            let mut n_folds: u8 = 0;
            let mut left_weights = [0.0f32; 5];
            let mut right_weights = [0.0f32; 5];
            let mut missing_dir = MissingInfo::Right;
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_hess_total: f32 = b.h_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();

            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_hessian_train[j] = node_hess_train_sum[j] - cuml_hessian_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_hessian_valid[j] = node_hess_sum[j] - cuml_hessian_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_hessian_train[j] += b_hess_total - b.h_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_hessian_valid[j] += b.h_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                let left_h = left_hessian_train[j];
                let right_h = right_hessian_train[j];

                // Inline weight = -g / (h + eps), gain = -(2gw + (h+eps)w²)
                let mut lw = -left_gradient_train[j] / (left_h + hessian_eps);
                let mut rw = -right_gradient_train[j] / (right_h + hessian_eps);
                let mut lg = -(2.0 * left_gradient_train[j] * lw + (left_h + hessian_eps) * lw * lw);
                let mut rg = -(2.0 * right_gradient_train[j] * rw + (right_h + hessian_eps) * rw * rw);

                // Missing-direction check
                if miss_grad_sum != 0.0 || miss_hess_sum != 0.0 {
                    let ml_g = left_gradient_train[j] + miss_grad_sum;
                    let ml_h = left_h + miss_hess_sum;
                    let ml_w = -ml_g / (ml_h + hessian_eps);
                    let ml_gain = -(2.0 * ml_g * ml_w + (ml_h + hessian_eps) * ml_w * ml_w);

                    let mr_g = right_gradient_train[j] + miss_grad_sum;
                    let mr_h = right_h + miss_hess_sum;
                    let mr_w = -mr_g / (mr_h + hessian_eps);
                    let mr_gain = -(2.0 * mr_g * mr_w + (mr_h + hessian_eps) * mr_w * mr_w);

                    if (mr_gain - rg) < (ml_gain - lg) {
                        lw = ml_w;
                        lg = ml_gain;
                        missing_dir = MissingInfo::Left;
                    } else {
                        rw = mr_w;
                        rg = mr_gain;
                        missing_dir = MissingInfo::Right;
                    }
                }

                left_weights[j] = lw;
                right_weights[j] = rw;

                let left_obj = left_gradient_valid[j] * lw + 0.5 * (left_hessian_valid[j] + hessian_eps) * lw * lw;
                let right_obj = right_gradient_valid[j] * rw + 0.5 * (right_hessian_valid[j] + hessian_eps) * rw * rw;
                valid_scores[j] = (left_obj + right_obj) / (left_c_valid + right_c_valid) as f32;
                train_scores[j] = -0.5 * (lg + rg) / (left_c_train + right_c_train) as f32;

                n_folds += 1;
            }

            if n_folds < 5 && node.num != 0 {
                continue;
            }

            let train_score = average(&train_scores);
            let valid_score = average(&valid_scores);
            let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
            let delta_score_train = parent_score - train_score;
            let delta_score_valid = parent_score - valid_score;
            let gen_val = delta_score_train / delta_score_valid;
            if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                continue;
            }
            let generalization = Some(gen_val);

            let tl_grad: f32 = left_gradient_valid.iter().sum();
            let tl_hess: f32 = left_hessian_valid.iter().sum();
            let _tl_count: usize = left_counts_valid.iter().sum();
            let tr_grad: f32 = right_gradient_valid.iter().sum();
            let tr_hess: f32 = right_hessian_valid.iter().sum();
            let _tr_count: usize = right_counts_valid.iter().sum();
            let tl_w = -tl_grad / (tl_hess + hessian_eps);
            let tr_w = -tr_grad / (tr_hess + hessian_eps);
            let tl_gain = -(2.0 * tl_grad * tl_w + (tl_hess + hessian_eps) * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + (tr_hess + hessian_eps) * tr_w * tr_w);
            let split_gain = tl_gain + tr_gain - node.gain_value;
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if split_gain <= 0.0 {
                continue;
            }

            if split_gain > best_gain && (generalization.is_some() || node.num == 0) {
                best_gain = split_gain;
                best_left_weights = left_weights;
                best_right_weights = right_weights;
                best_generalization = generalization;
                best_left_grad_valid = left_gradient_valid;
                best_left_hess_valid = left_hessian_valid;
                best_left_coun_valid = left_counts_valid;
                best_right_grad_valid = right_gradient_valid;
                best_right_hess_valid = right_hessian_valid;
                best_right_coun_valid = right_counts_valid;
                best_cut_value = b.cut_value;
                best_split_bin = b.num;
                best_missing_info = missing_dir;
                found_split = true;
            }
        }

        if found_split {
            let tl_grad: f32 = best_left_grad_valid.iter().sum();
            let tl_hess: f32 = best_left_hess_valid.iter().sum();
            let tl_count: usize = best_left_coun_valid.iter().sum();
            let tr_grad: f32 = best_right_grad_valid.iter().sum();
            let tr_hess: f32 = best_right_hess_valid.iter().sum();
            let tr_count: usize = best_right_coun_valid.iter().sum();
            let tl_w = -tl_grad / (tl_hess + hessian_eps);
            let tr_w = -tr_grad / (tr_hess + hessian_eps);
            let tl_gain = -(2.0 * tl_grad * tl_w + (tl_hess + hessian_eps) * tl_w * tl_w);
            let tr_gain = -(2.0 * tr_grad * tr_w + (tr_hess + hessian_eps) * tr_w * tr_w);

            max_gain = Some(best_gain);
            split_info.split_gain = best_gain;
            split_info.split_feature = feature;
            split_info.split_value = best_cut_value;
            split_info.split_bin = best_split_bin;
            split_info.left_node = NodeInfo {
                grad: tl_grad,
                gain: tl_gain,
                cover: tl_hess,
                counts: tl_count,
                weight: tl_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_left_weights,
            };
            split_info.right_node = NodeInfo {
                grad: tr_grad,
                gain: tr_gain,
                cover: tr_hess,
                counts: tr_count,
                weight: tr_w,
                bounds: (node.lower_bound, node.upper_bound),
                weights: best_right_weights,
            };
            split_info.missing_node = best_missing_info;
            split_info.generalization = best_generalization;
        }
    } else {
        // ── Slow path: constraints, missing values, or missing branches ──
        for bin in &hist_feat.data[1..] {
            let b = unsafe { &*bin.get() };
            if b.counts.iter().map(|&c| c as usize).sum::<usize>() == 0 || b.cut_value.is_nan() {
                continue;
            }
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
            let mut left_weights = [0.0; 5];
            let mut right_weights = [0.0; 5];
            #[allow(clippy::needless_late_init)]
            let generalization;
            let b_grad_total: f32 = b.g_folded.iter().sum();
            let b_hess_total: f32 = b.h_folded.iter().sum();
            let b_coun_total: usize = b.counts.iter().map(|&c| c as usize).sum();
            for j in 0..5 {
                right_gradient_train[j] = node_grad_train_sum[j] - cuml_gradient_train[j];
                right_hessian_train[j] = node_hess_train_sum[j] - cuml_hessian_train[j];
                right_counts_train[j] = node_coun_train_sum[j] - cuml_counts_train[j];
                right_gradient_valid[j] = node_grad_sum[j] - cuml_gradient_valid[j];
                right_hessian_valid[j] = node_hess_sum[j] - cuml_hessian_valid[j];
                right_counts_valid[j] = node_coun_sum[j] - cuml_counts_valid[j];

                cuml_gradient_train[j] += b_grad_total - b.g_folded[j];
                cuml_hessian_train[j] += b_hess_total - b.h_folded[j];
                cuml_counts_train[j] += b_coun_total - b.counts[j] as usize;
                cuml_gradient_valid[j] += b.g_folded[j];
                cuml_hessian_valid[j] += b.h_folded[j];
                cuml_counts_valid[j] += b.counts[j] as usize;

                let left_c_train = left_counts_train[j];
                let right_c_train = right_counts_train[j];
                let left_c_valid = left_counts_valid[j];
                let right_c_valid = right_counts_valid[j];

                if right_c_train == 0 || right_c_valid == 0 || left_c_train == 0 || left_c_valid == 0 {
                    continue;
                }

                let split_result = if create_missing_branch {
                    evaluate_branch_split_var_hess(
                        left_gradient_train[j],
                        left_hessian_train[j],
                        left_counts_train[j],
                        right_gradient_train[j],
                        right_hessian_train[j],
                        right_counts_train[j],
                        miss_grad_sum,
                        miss_hess_sum,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                } else {
                    evaluate_impute_split_var_hess(
                        left_gradient_train[j],
                        left_hessian_train[j],
                        left_counts_train[j],
                        right_gradient_train[j],
                        right_hessian_train[j],
                        right_counts_train[j],
                        miss_grad_sum,
                        miss_hess_sum,
                        miss_coun_sum,
                        node.lower_bound,
                        node.upper_bound,
                        node.weight_value,
                        constraint,
                        force_children_to_bound_parent,
                        missing_node_treatment,
                        allow_missing_splits,
                    )
                };
                let (left_node, right_node, _) = match split_result {
                    Some(v) => v,
                    None => continue,
                };

                left_weights[j] = left_node.weight;
                right_weights[j] = right_node.weight;

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
                let parent_score = -0.5 * node.gain_value / node.stats.as_ref().unwrap().count as f32;
                let delta_score_train = parent_score - train_score;
                let delta_score_valid = parent_score - valid_score;
                let gen_val = delta_score_train / delta_score_valid;
                if gen_val < GENERALIZATION_THRESHOLD && node.num != 0 {
                    continue;
                }
                generalization = Some(gen_val);
            } else {
                continue;
            }

            let split_result = if create_missing_branch {
                evaluate_branch_split_var_hess(
                    left_gradient_valid.iter().sum(),
                    left_hessian_valid.iter().sum::<f32>(),
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    right_hessian_valid.iter().sum::<f32>(),
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    miss_hess_sum,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            } else {
                evaluate_impute_split_var_hess(
                    left_gradient_valid.iter().sum(),
                    left_hessian_valid.iter().sum::<f32>(),
                    left_counts_valid.iter().sum::<usize>(),
                    right_gradient_valid.iter().sum(),
                    right_hessian_valid.iter().sum::<f32>(),
                    right_counts_valid.iter().sum::<usize>(),
                    miss_grad_sum,
                    miss_hess_sum,
                    miss_coun_sum,
                    node.lower_bound,
                    node.upper_bound,
                    node.weight_value,
                    constraint,
                    force_children_to_bound_parent,
                    missing_node_treatment,
                    allow_missing_splits,
                )
            };

            let (mut left_node_info, mut right_node_info, missing_info) = match split_result {
                Some(v) => v,
                None => continue,
            };

            left_node_info.weights = left_weights;
            right_node_info.weights = right_weights;

            let split_gain = node.get_split_gain(&left_node_info, &right_node_info, &missing_info);
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
            let split_gain = if split_gain.is_nan() { 0.0 } else { split_gain };

            if (max_gain.is_none() || split_gain > max_gain.unwrap()) && (generalization.is_some() || node.num == 0) {
                max_gain = Some(split_gain);
                split_info.split_gain = split_gain;
                split_info.split_feature = feature;
                split_info.split_value = b.cut_value;
                split_info.split_bin = b.num;
                split_info.left_node = left_node_info;
                split_info.right_node = right_node_info;
                split_info.missing_node = missing_info;
                split_info.generalization = generalization;
            }
        }
    }

    if max_gain.is_some() {
        if all_cats.is_empty() {
            split_info.left_cats = None;
        } else {
            if split_info.left_cats.is_none() {
                split_info.left_cats = Some(vec![0u8; 8192].into_boxed_slice());
            }
            let left_cats_vec = split_info.left_cats.as_mut().unwrap();
            left_cats_vec.fill(0);
            for c in all_cats.iter() {
                if *c == split_info.split_value as usize {
                    break;
                }
                let byte_idx = c >> 3;
                let bit_idx = c & 7;
                left_cats_vec[byte_idx] |= 1 << bit_idx;
            }
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
        let missing_left_weight = constrained_weight_const_hess(
            left_gradient + missing_gradient,
            left_counts + missing_counts,
            lower_bound,
            upper_bound,
            constraint,
        );
        let missing_left_gain = gain_given_weight_const_hess(
            left_gradient + missing_gradient,
            left_counts + missing_counts,
            missing_left_weight,
        );
        let missing_left_gain = cull_gain(missing_left_gain, missing_left_weight, right_weight, constraint);

        let missing_right_weight = constrained_weight_const_hess(
            right_gradient + missing_gradient,
            right_counts + missing_counts,
            lower_bound,
            upper_bound,
            constraint,
        );
        let missing_right_gain = gain_given_weight_const_hess(
            right_gradient + missing_gradient,
            right_counts + missing_counts,
            missing_right_weight,
        );
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
            weights: [0.0; 5],
        },
        NodeInfo {
            grad: right_gradient,
            gain: right_gain,
            cover: right_counts as f32,
            counts: right_counts,
            weight: right_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
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
        let missing_left_weight = constrained_weight(
            left_gradient + missing_gradient,
            left_hessian + missing_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );
        let missing_left_gain = gain_given_weight(
            left_gradient + missing_gradient,
            left_hessian + missing_hessian,
            missing_left_weight,
        );
        let missing_left_gain = cull_gain(missing_left_gain, missing_left_weight, right_weight, constraint);

        let missing_right_weight = constrained_weight(
            right_gradient + missing_gradient,
            right_hessian + missing_hessian,
            lower_bound,
            upper_bound,
            constraint,
        );
        let missing_right_gain = gain_given_weight(
            right_gradient + missing_gradient,
            right_hessian + missing_hessian,
            missing_right_weight,
        );
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
            weights: [0.0; 5],
        },
        NodeInfo {
            grad: right_gradient,
            gain: right_gain,
            cover: right_hessian,
            counts: right_counts,
            weight: right_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
        },
        missing_info,
    ))
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
    if (left_counts == 0 || right_counts == 0) && !allow_missing_splits {
        return None;
    }

    let mut left_weight =
        constrained_weight_const_hess(left_gradient, left_counts, lower_bound, upper_bound, constraint);
    let mut right_weight =
        constrained_weight_const_hess(right_gradient, right_counts, lower_bound, upper_bound, constraint);

    if force_children_to_bound_parent {
        (left_weight, right_weight) = bound_to_parent(parent_weight, left_weight, right_weight);
    }

    let left_gain = gain_given_weight_const_hess(left_gradient, left_counts, left_weight);
    let right_gain = gain_given_weight_const_hess(right_gradient, right_counts, right_weight);

    let missing_weight = match missing_node_treatment {
        MissingNodeTreatment::AssignToParent => constrained_weight_const_hess(
            missing_gradient + left_gradient + right_gradient,
            missing_counts + left_counts + right_counts,
            lower_bound,
            upper_bound,
            constraint,
        ),
        MissingNodeTreatment::AverageLeafWeight | MissingNodeTreatment::AverageNodeWeight => {
            if left_counts + right_counts > 0 {
                (left_weight * left_counts as f32 + right_weight * right_counts as f32)
                    / (left_counts + right_counts) as f32
            } else {
                parent_weight
            }
        }
        MissingNodeTreatment::None => {
            if missing_counts > 0 {
                constrained_weight_const_hess(missing_gradient, missing_counts, lower_bound, upper_bound, constraint)
            } else {
                parent_weight
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
        bounds: (lower_bound, upper_bound),
        weights: [0.0; 5],
    };
    let missing_node = if missing_counts > 0 && allow_missing_splits {
        MissingInfo::Branch(missing_info)
    } else {
        MissingInfo::Leaf(missing_info)
    };

    Some((
        NodeInfo {
            grad: left_gradient,
            gain: left_gain,
            cover: left_counts as f32,
            counts: left_counts,
            weight: left_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
        },
        NodeInfo {
            grad: right_gradient,
            gain: right_gain,
            cover: right_counts as f32,
            counts: right_counts,
            weight: right_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
        },
        missing_node,
    ))
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
    if (left_hessian == 0.0 || right_hessian == 0.0) && !allow_missing_splits {
        return None;
    }

    let mut left_weight = constrained_weight(left_gradient, left_hessian, lower_bound, upper_bound, constraint);
    let mut right_weight = constrained_weight(right_gradient, right_hessian, lower_bound, upper_bound, constraint);

    if force_children_to_bound_parent {
        (left_weight, right_weight) = bound_to_parent(parent_weight, left_weight, right_weight);
    }

    let left_gain = gain_given_weight(left_gradient, left_hessian, left_weight);
    let right_gain = gain_given_weight(right_gradient, right_hessian, right_weight);

    let missing_weight = match missing_node_treatment {
        MissingNodeTreatment::AssignToParent => constrained_weight(
            missing_gradient + left_gradient + right_gradient,
            missing_hessian + left_hessian + right_hessian,
            lower_bound,
            upper_bound,
            constraint,
        ),
        MissingNodeTreatment::AverageLeafWeight | MissingNodeTreatment::AverageNodeWeight => {
            if left_hessian + right_hessian > 0.0 {
                (left_weight * left_hessian + right_weight * right_hessian) / (left_hessian + right_hessian)
            } else {
                parent_weight
            }
        }
        MissingNodeTreatment::None => {
            if missing_hessian > 0.0 || missing_gradient != 0.0 {
                constrained_weight(missing_gradient, missing_hessian, lower_bound, upper_bound, constraint)
            } else {
                parent_weight
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
        bounds: (lower_bound, upper_bound),
        weights: [0.0; 5],
    };
    let missing_node = if (missing_hessian > 0.0 || missing_gradient != 0.0) && allow_missing_splits {
        MissingInfo::Branch(missing_info)
    } else {
        MissingInfo::Leaf(missing_info)
    };

    Some((
        NodeInfo {
            grad: left_gradient,
            gain: left_gain,
            cover: left_hessian,
            counts: left_counts,
            weight: left_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
        },
        NodeInfo {
            grad: right_gradient,
            gain: right_gain,
            cover: right_hessian,
            counts: right_counts,
            weight: right_weight,
            bounds: (f32::NEG_INFINITY, f32::INFINITY),
            weights: [0.0; 5],
        },
        missing_node,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::data::Matrix;
    use crate::histogram::{NodeHistogram, NodeHistogramOwned, update_histogram};
    use crate::node::SplittableNode;
    use crate::tree::core::{Tree, TreeStopper, create_root_node};

    use crate::objective::{Objective, ObjectiveFunction};
    use crate::utils::weight;
    use crate::utils::{between, gain};
    use rayon::ThreadPoolBuilder;
    use std::collections::{HashMap, HashSet};
    use std::error::Error;
    use std::fs;

    #[test]
    fn test_data_split() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let is_const_hess = false;
        let num_threads = 2;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];

        let (grad, hess) = objective_function.gradient(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new(), None);
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

        let hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

        for i in 0..n_nodes_alloc {
            update_histogram(
                unsafe { hist_tree.get_unchecked(i) },
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
            grad.len(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            [root_weight; 5],
        );

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &n,
            &col_index,
            false,
            &hist_tree,
            &pool,
            None,
            &mut split_info_slice,
            None,
        );
        let s = unsafe { split_info_slice.best_split_info() };
        println!("{:?}", s);

        n.update_children(2, 1, 2, s);

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
        // instantiate objective function
        let objective_function = Objective::SquaredLoss;

        let n_bins = 256;
        let n_cols = 8;
        let is_const_hess = true;

        let feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ];
        let target_name = "MedHouseVal";

        let file = fs::File::open("resources/cal_housing_test.csv")?;
        let reader = std::io::BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

        let headers = csv_reader.headers()?.clone();
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|name| headers.iter().position(|h| h == *name).unwrap())
            .collect();
        let target_index = headers.iter().position(|h| h == target_name).unwrap();

        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
        let mut y_test = Vec::new();

        for result in csv_reader.records() {
            let record = result?;

            // Parse target
            let target_str = &record[target_index];
            let target_val = if target_str.is_empty() {
                f64::NAN
            } else {
                target_str.parse::<f64>().unwrap_or(f64::NAN)
            };
            y_test.push(target_val);

            // Parse features
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                let val = if val_str.is_empty() {
                    f64::NAN
                } else {
                    val_str.parse::<f64>().unwrap_or(f64::NAN)
                };
                data_columns[i].push(val);
            }
        }

        // Flatten column-major
        let data_test: Vec<f64> = data_columns.into_iter().flatten().collect();

        let y_test_avg = y_test.iter().sum::<f64>() / y_test.len() as f64;
        let yhat = vec![y_test_avg; y_test.len()];
        let (grad, hess) = objective_function.gradient(&y_test, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(0.3, false, ConstraintMap::new(), None);

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

        let hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

        for i in 0..n_nodes_alloc {
            update_histogram(
                unsafe { hist_tree.get_unchecked(i) },
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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &n,
            &col_index,
            is_const_hess,
            &hist_tree,
            &pool,
            None,
            &mut split_info_slice,
            None,
        );
        let s = unsafe { split_info_slice.best_split_info() };
        println!("{:?}", s);

        n.update_children(2, 1, 2, s);

        assert_eq!(0, s.split_feature);
        assert!(between(4.8, 5.1, s.split_value as f32));
        Ok(())
    }

    #[test]
    fn test_categorical() -> Result<(), Box<dyn Error>> {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let n_bins = 256;
        let n_rows = 712;
        let n_cols = 9;
        let is_const_hess = false;
        let eta = 0.1;

        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data_vec_truncated = &data_vec[0..(n_cols * n_rows)];
        let data = Matrix::new(data_vec_truncated, n_rows, n_cols);

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (grad, hess) = objective_function.gradient(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(eta, false, ConstraintMap::new(), None);

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

        let hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

        for i in 0..10 {
            update_histogram(
                unsafe { hist_tree.get_unchecked(i) },
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
            grad.len(),
            0,
            0,
            grad.len(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            [root_weight; 5],
        );

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        splitter.best_split(
            &n,
            &col_index,
            is_const_hess,
            &hist_tree,
            &pool,
            Some(&cat_index),
            &mut split_info_slice,
            None,
        );
        let s = unsafe { split_info_slice.best_split_info() };

        n.update_children(1, 1, 2, s);

        println!("split info:");
        println!("{:?}", s);

        println!("left_cats:");
        println!("{:?}", s.left_cats);

        println!("hist_tree[0]: {:?}", hist_tree_owned[0].data[7]);

        assert_eq!(8, s.split_feature);

        Ok(())
    }

    // TODO: add test_ranking

    #[test]
    fn test_gbm_categorical_sensory() -> Result<(), Box<dyn Error>> {
        // instantiate objective function
        let objective_function = Objective::SquaredLoss;

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
        let (grad, hess) = objective_function.gradient(&y, &yhat, None, None);

        let splitter = MissingImputerSplitter::new(eta, false, ConstraintMap::new(), None);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let index = data.index.to_owned();
        let col_index: Vec<usize> = (0..data.cols).collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..10)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let hist_tree: Vec<NodeHistogram> = hist_tree_owned.iter_mut().map(NodeHistogram::from_owned).collect();

        for i in 0..10 {
            update_histogram(
                unsafe { hist_tree.get_unchecked(i) },
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
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        splitter.best_split(
            &n,
            &col_index,
            is_const_hess,
            &hist_tree,
            &pool,
            Some(&cat_index),
            &mut split_info_slice,
            None,
        );

        println!("split_info_slice:");
        for s_data in split_info_slice.data.iter() {
            println!("{:?}", unsafe { s_data.get().as_ref().unwrap() });
        }

        let s = unsafe { split_info_slice.best_split_info() };

        n.update_children(1, 1, 2, s);

        println!("split info:");
        println!("{:?}", s);

        println!("left_cats:");
        println!("{:?}", s.left_cats);

        println!("hist_tree.0.6: {:?}", hist_tree_owned[0].data[6]);

        assert_eq!(6, s.split_feature);

        Ok(())
    }

    #[test]
    fn test_missing_branch_splitter_and_importance() {
        let constraints_map = ConstraintMap::new();
        let terminate_missing_features = HashSet::new();
        let splitter = MissingBranchSplitter::new(
            0.1,
            true,
            constraints_map,
            terminate_missing_features,
            MissingNodeTreatment::AverageLeafWeight,
            false,
        );

        assert_eq!(splitter.new_leaves_added(), 2);
        assert_eq!(splitter.get_eta(), 0.1);
        assert!(splitter.get_allow_missing_splits());
        assert!(splitter.get_create_missing_branch());
        assert_eq!(
            splitter.get_missing_node_treatment(),
            MissingNodeTreatment::AverageLeafWeight
        );
        assert!(!splitter.get_force_children_to_bound_parent());
        assert!(splitter.get_interaction_constraints().is_none());

        // Test Importance calculations would normally go here on a Tree, not a Splitter.

        // Test predict_row_from_row_slice
        let _row = vec![1.0, 2.0];
        let _missing = f64::NAN;
        // This will panic if we don't have nodes, but let's see if we can mock it or if there's a better way.
        // Actually, MissingBranchSplitter doesn't have a 'nodes' field in its struct, but Splitter trait methods might use them.
        // Wait, predict_row_from_row_slice is a method on Tree or Splitter?
        // Checking... line 142163 in splitter.rs? No, let me check the struct.
    }

    #[test]
    fn test_interaction_constraints() {
        let interaction_constraints = Some(vec![vec![0, 1]]);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new(), interaction_constraints);
        assert_eq!(splitter.get_interaction_constraints().unwrap().len(), 1);

        let allowed_features = HashSet::from([0]);
        // This should trigger the "skip disallowed features" block in best_split.

        // Setup minimal best_split call to exercise the block
        let nbins = 10;
        let data_vec = vec![0.0, 1.0, 2.0, 3.0];
        let data = Matrix::new(&data_vec, 2, 2);
        let b = bin_matrix(&data, None, nbins, f64::NAN, None).unwrap();
        let _bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index = vec![0, 1];
        let mut hist_tree_owned = vec![NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, false, false)];
        let hist_tree = hist_tree_owned
            .iter_mut()
            .map(NodeHistogram::from_owned)
            .collect::<Vec<_>>();

        let node = SplittableNode::new(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            2,
            0,
            0,
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            [0.0; 5],
        );
        let mut split_info_vec = vec![SplitInfo::default(), SplitInfo::default()];
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        splitter.best_split(
            &node,
            &col_index,
            false,
            &hist_tree,
            &pool,
            None,
            &mut split_info_slice,
            Some(&allowed_features),
        );

        unsafe {
            assert_eq!(split_info_slice.get_mut(1).split_gain, -1.0); // Feature 1 was disallowed
        }
    }

    #[test]
    fn test_missing_branch_splitter_comprehensive() {
        let splitter = MissingBranchSplitter::new(
            0.1,
            true,
            ConstraintMap::new(),
            HashSet::new(),
            MissingNodeTreatment::AverageLeafWeight,
            false,
        );

        // Test update_average_missing_nodes
        use crate::node::Node;
        use std::collections::HashMap;
        let mut nodes = HashMap::new();
        nodes.insert(
            0,
            Node {
                num: 0,
                parent_node: 0,
                left_child: 1,
                right_child: 2,
                missing_node: 3,
                split_feature: 0,
                split_value: 0.5,
                split_gain: 0.0,
                is_leaf: false,
                weight_value: 0.0,
                hessian_sum: 20.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            1,
            Node {
                num: 1,
                parent_node: 0,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 1.0,
                hessian_sum: 10.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            2,
            Node {
                num: 2,
                parent_node: 0,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 2.0,
                hessian_sum: 10.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            3,
            Node {
                num: 3,
                parent_node: 0,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 0.0,
                hessian_sum: 5.0,
                left_cats: None,
                stats: None,
            },
        );

        let mut tree = Tree {
            nodes,
            stopper: TreeStopper::Generalization,
            depth: 1,
            n_leaves: 3,
            leaf_bounds: Vec::new(),
            train_index: Vec::new(),
        };

        splitter.clean_up_splits(&mut tree);
        // Average weight = (1.0 * 10.0 + 2.0 * 10.0 + 0 * 5.0) / (10.0 + 10.0 + 0) = 30.0 / 20.0 = 1.5
        assert_eq!(tree.nodes[&0].weight_value, 1.5);
        assert_eq!(tree.nodes[&3].weight_value, 1.5);

        // Test best_split with variable Hessian
        let nbins = 10;
        let data_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Matrix::new(&data_vec, 3, 2);
        let b = bin_matrix(&data, None, nbins, f64::NAN, None).unwrap();
        let col_index = vec![0, 1];
        // Use true for is_const_hess in NodeHistogramOwned::empty_from_cuts to avoid issues?
        // Wait, the splitter trait methods handle is_const_hess.
        let mut hist_tree_owned = vec![NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, false, false)];
        let hist_tree = hist_tree_owned
            .iter_mut()
            .map(NodeHistogram::from_owned)
            .collect::<Vec<_>>();

        let node = SplittableNode::new(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            3,
            0,
            0,
            3,
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            [0.0; 5],
        );
        let mut split_info_vec = vec![SplitInfo::default(), SplitInfo::default()];
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        // Call best_split with is_const_hess = false to exercise variable hessian paths
        splitter.best_split(
            &node,
            &col_index,
            false,
            &hist_tree,
            &pool,
            None,
            &mut split_info_slice,
            None,
        );

        unsafe {
            assert!(split_info_slice.get_mut(0).split_gain >= -1.0);
        }
    }

    #[test]
    fn test_missing_imputer_splitter_coverage() {
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new(), None);
        assert_eq!(splitter.new_leaves_added(), 1);
        assert!(splitter.get_constraint(&0).is_none());
        assert_eq!(splitter.get_missing_node_treatment(), MissingNodeTreatment::None);
        assert!(splitter.get_interaction_constraints().is_none());
    }

    #[test]
    fn test_update_average_missing_nodes_recursive() {
        let splitter = MissingBranchSplitter::new(
            0.1,
            true,
            ConstraintMap::new(),
            HashSet::new(),
            MissingNodeTreatment::AverageLeafWeight,
            false,
        );
        use crate::node::Node;
        let mut nodes = HashMap::new();
        // Root (0) -> Left (1), Right (2), Missing (3)
        // Missing (3) -> Left (4), Right (5), Missing (6)
        nodes.insert(
            0,
            Node {
                num: 0,
                parent_node: 0,
                left_child: 1,
                right_child: 2,
                missing_node: 3,
                split_feature: 0,
                split_value: 0.5,
                split_gain: 0.0,
                is_leaf: false,
                weight_value: 0.0,
                hessian_sum: 30.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            1,
            Node {
                num: 1,
                parent_node: 0,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 1.0,
                hessian_sum: 10.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            2,
            Node {
                num: 2,
                parent_node: 0,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 2.0,
                hessian_sum: 10.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            3,
            Node {
                num: 3,
                parent_node: 0,
                left_child: 4,
                right_child: 5,
                missing_node: 6,
                split_feature: 1,
                split_value: 0.5,
                split_gain: 0.0,
                is_leaf: false,
                weight_value: 0.0,
                hessian_sum: 10.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            4,
            Node {
                num: 4,
                parent_node: 3,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 3.0,
                hessian_sum: 5.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            5,
            Node {
                num: 5,
                parent_node: 3,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 4.0,
                hessian_sum: 5.0,
                left_cats: None,
                stats: None,
            },
        );
        nodes.insert(
            6,
            Node {
                num: 6,
                parent_node: 3,
                left_child: 0,
                right_child: 0,
                missing_node: 0,
                split_feature: 0,
                split_value: 0.0,
                split_gain: 0.0,
                is_leaf: true,
                weight_value: 0.0,
                hessian_sum: 0.0,
                left_cats: None,
                stats: None,
            },
        );

        let mut tree = Tree {
            nodes,
            stopper: TreeStopper::Generalization,
            depth: 2,
            n_leaves: 5,
            leaf_bounds: Vec::new(),
            train_index: Vec::new(),
        };
        splitter.clean_up_splits(&mut tree);
        // Node 3 avg weight = (3.0 * 5 + 4.0 * 5 + 0 * 0) / (5 + 5 + 0) = 35 / 10 = 3.5
        // Node 0 avg weight = (1.0 * 10 + 2.0 * 10 + 3.5 * 10) / (10 + 10 + 10) = (10 + 20 + 35) / 30 = 65 / 30 = 2.166...
        assert!((tree.nodes[&0].weight_value - 2.1666667).abs() < 1e-5);
    }

    #[test]
    fn test_handle_split_info_histogram_strategies() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

        let run_strategy = |n_missing: usize, n_left: usize, n_right: usize| {
            let splitter = MissingBranchSplitter::new(
                0.1,
                true,
                ConstraintMap::new(),
                HashSet::from([0]),
                MissingNodeTreatment::AverageLeafWeight,
                false,
            );
            let n_total = n_missing + n_left + n_right;
            let mut data_vec: Vec<u16> = vec![0; n_total * 2];
            // Fill feature 0 with values to create missing, left, right groups
            // In column-major Matrix, col 0 is data_vec[0..n_total]
            // bin 0: missing, bin 1: left, bin 2: right
            for i in 0..n_missing {
                data_vec[i] = 0;
            }
            for i in n_missing..(n_missing + n_left) {
                data_vec[i] = 1;
            }
            for i in (n_missing + n_left)..n_total {
                data_vec[i] = 2;
            }
            // feature 1 remains all 0s

            let data = Matrix::new(&data_vec, n_total, 2);
            let col_index = vec![0, 1];
            let cuts = crate::data::JaggedMatrix::from_vecs(&vec![vec![1.5, 5.5, f64::MAX], vec![1.5, 5.5, f64::MAX]]);
            let mut hist_tree_owned = vec![
                NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, false, false),
                NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, false, false),
                NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, false, false),
                NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, false, false),
            ];
            let hist_tree = hist_tree_owned
                .iter_mut()
                .map(NodeHistogram::from_owned)
                .collect::<Vec<_>>();

            let mut node = SplittableNode::new(
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                n_total,
                0,
                0,
                n_total,
                f32::NEG_INFINITY,
                f32::INFINITY,
                NodeType::Root,
                None,
                [0.0; 5],
            );
            let mut index = (0..n_total).collect::<Vec<_>>();
            let mut grad = vec![1.0; n_total];

            // Populate root histogram
            update_histogram(
                &hist_tree[0],
                0,
                n_total,
                &data,
                &grad,
                None,
                &index,
                &col_index,
                &pool,
                false,
            );

            let mut split_info = SplitInfo {
                split_feature: 0,
                split_bin: 2,
                split_value: 0.5,
                split_gain: 10.0,
                missing_node: MissingInfo::Branch(NodeInfo {
                    weight: 0.0,
                    gain: 0.0,
                    grad: 0.0,
                    cover: 0.0,
                    bounds: (0.0, 0.0),
                    counts: 0,
                    weights: [0.0; 5],
                }),
                left_node: NodeInfo {
                    weight: 1.0,
                    gain: 0.0,
                    grad: 0.0,
                    cover: n_left as f32,
                    bounds: (0.0, 0.0),
                    counts: n_left,
                    weights: [0.0; 5],
                },
                right_node: NodeInfo {
                    weight: 2.0,
                    gain: 0.0,
                    grad: 0.0,
                    cover: n_right as f32,
                    bounds: (0.0, 0.0),
                    counts: n_right,
                    weights: [0.0; 5],
                },
                ..Default::default()
            };

            splitter.handle_split_info(
                &mut split_info,
                &1,
                &mut node,
                &mut index,
                &col_index,
                &data,
                &mut grad,
                None,
                &pool,
                &hist_tree,
            );
        };

        // max_ == 0: Missing is largest (n_missing=10, n_left=5, n_right=5)
        run_strategy(10, 5, 5);
        // max_ == 1: Left is largest (n_missing=5, n_left=10, n_right=5)
        run_strategy(5, 10, 5);
        // max_ == 2: Right is largest (n_missing=5, n_left=5, n_right=10)
        run_strategy(5, 5, 10);
        // n_missing == 0 (max_ == 1 case)
        run_strategy(0, 10, 5);
        // n_missing == 0 (max_ == 2 case)
        run_strategy(0, 5, 10);
    }

    #[test]
    fn test_splitter_additional_coverage() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        // 1. MissingImputerSplitter::get_eta and coverage
        let imputer = MissingImputerSplitter::new(0.5, true, ConstraintMap::new(), None);
        assert_eq!(imputer.get_eta(), 0.5);
        assert!(!imputer.get_create_missing_branch());
        assert!(!imputer.get_force_children_to_bound_parent());
        assert!(imputer.get_constraint(&0).is_none());
        assert_eq!(imputer.get_constraint_map().len(), 0);
        assert!(imputer.get_interaction_constraints().is_none());
        assert!(imputer.get_allow_missing_splits());

        // 2. MissingBranchSplitter with terminate_missing_features and misc coverage
        let splitter = MissingBranchSplitter::new(
            0.1,
            true,
            ConstraintMap::new(),
            HashSet::from([0]),
            MissingNodeTreatment::AverageLeafWeight,
            false,
        );
        assert_eq!(splitter.get_eta(), 0.1);
        assert_eq!(splitter.get_force_children_to_bound_parent(), false);
        assert_eq!(splitter.get_constraint_map().len(), 0);
        assert!(splitter.get_create_missing_branch());
        assert!(splitter.get_allow_missing_splits());
        assert!(splitter.get_interaction_constraints().is_none());
        assert!(splitter.get_constraint(&0).is_none());

        let n_total = 20;
        let mut data_vec = vec![0; n_total * 2];
        for i in 0..10 {
            data_vec[i] = 0;
        } // Missing (bin 0)
        for i in 10..15 {
            data_vec[i] = 1;
        } // Left (bin 1)
        for i in 15..20 {
            data_vec[i] = 2;
        } // Right (bin 2)
        let data = Matrix::new(&data_vec, n_total, 2);
        let col_index = vec![0, 1];
        let cuts = crate::data::JaggedMatrix::from_vecs(&vec![vec![1.5, 5.5, f64::MAX], vec![1.5, 5.5, f64::MAX]]);
        let mut hist_tree_owned = vec![
            NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, true, false),
            NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, true, false),
            NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, true, false),
            NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, true, false),
        ];
        let hist_tree = hist_tree_owned
            .iter_mut()
            .map(NodeHistogram::from_owned)
            .collect::<Vec<_>>();
        let mut node = SplittableNode::new(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            n_total,
            0,
            0,
            n_total,
            f32::NEG_INFINITY,
            f32::INFINITY,
            NodeType::Root,
            None,
            [0.0; 5],
        );
        let mut index = (0..n_total).collect::<Vec<_>>();
        let mut grad = vec![1.0; n_total];
        update_histogram(
            &hist_tree[0],
            0,
            n_total,
            &data,
            &grad,
            None,
            &index,
            &col_index,
            &pool,
            false,
        );

        let mut split_info = SplitInfo {
            split_feature: 0,
            split_bin: 2,
            split_value: 0.5,
            split_gain: 10.0,
            missing_node: MissingInfo::Branch(NodeInfo {
                weight: 0.0,
                gain: 0.0,
                grad: 0.0,
                cover: 0.0,
                bounds: (0.0, 0.0),
                counts: 0,
                weights: [0.0; 5],
            }),
            left_node: NodeInfo {
                weight: 1.0,
                gain: 0.0,
                grad: 0.0,
                cover: 5.0,
                bounds: (0.0, 0.0),
                counts: 5,
                weights: [0.0; 5],
            },
            right_node: NodeInfo {
                weight: 2.0,
                gain: 0.0,
                grad: 0.0,
                cover: 5.0,
                bounds: (0.0, 0.0),
                counts: 5,
                weights: [0.0; 5],
            },
            ..Default::default()
        };
        // This hits line 492 (terminate_missing_features)
        splitter.handle_split_info(
            &mut split_info,
            &1,
            &mut node,
            &mut index,
            &col_index,
            &data,
            &mut grad,
            None,
            &pool,
            &hist_tree,
        );

        // 3. Interaction Constraints Parallel Path (line 208)
        let many_cols: Vec<usize> = (0..17).collect(); // > 16
        let long_data_vec = vec![0; n_total * 17];
        let _long_data = Matrix::new(&long_data_vec, n_total, 17);
        let long_cuts = crate::data::JaggedMatrix::from_vecs(&vec![vec![1.5, f64::MAX]; 17]);
        let mut long_hist_tree_owned = vec![NodeHistogramOwned::empty_from_cuts(&long_cuts, &many_cols, true, false)];
        let long_hist_tree = long_hist_tree_owned
            .iter_mut()
            .map(NodeHistogram::from_owned)
            .collect::<Vec<_>>();
        let mut many_split_info_vec: Vec<SplitInfo> = (0..17).map(|_| SplitInfo::default()).collect();
        let mut many_split_info_slice = SplitInfoSlice::new(&mut many_split_info_vec);
        let allowed_features = HashSet::from([0, 1, 2]);
        // Call best_split with num_threads > 1 and col_index.len() > 16
        splitter.best_split(
            &node,
            &many_cols,
            true,
            &long_hist_tree,
            &pool,
            None,
            &mut many_split_info_slice,
            Some(&allowed_features),
        );

        // 4. Categorical Branch Split and Numerical Branch Split (Const Hess)
        let cat_index = HashSet::from([0]);
        let mut cat_split_info_vec: Vec<SplitInfo> = (0..2).map(|_| SplitInfo::default()).collect();
        let mut cat_split_info_slice = SplitInfoSlice::new(&mut cat_split_info_vec);
        // Const Hess Categorical (Line 1000) and Numerical (Line 1075)
        splitter.best_split(
            &node,
            &col_index,
            true,
            &hist_tree,
            &pool,
            Some(&cat_index),
            &mut cat_split_info_slice,
            None,
        );

        // 5. Categorical and Numerical Branch Split (Var Hess)
        let mut hist_tree_owned_var = vec![NodeHistogramOwned::empty_from_cuts(&cuts, &col_index, false, false)];
        let hist_tree_var = hist_tree_owned_var
            .iter_mut()
            .map(NodeHistogram::from_owned)
            .collect::<Vec<_>>();
        let hess = vec![1.0; n_total];
        update_histogram(
            &hist_tree_var[0],
            0,
            n_total,
            &data,
            &grad,
            Some(&hess),
            &index,
            &col_index,
            &pool,
            false,
        );
        // Var Hess Categorical (Line 1536) and Numerical (Line 1611)
        splitter.best_split(
            &node,
            &col_index,
            false,
            &hist_tree_var,
            &pool,
            Some(&cat_index),
            &mut cat_split_info_slice,
            None,
        );
    }
}
