use crate::data::FloatData;
use crate::splitter::{MissingInfo, NodeInfo, SplitInfo};
use crate::utils::is_missing;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::{self, Debug};

#[derive(Debug, Deserialize, Serialize)]
pub struct SplittableNode {
    pub num: usize,
    pub weight_value: f32,
    pub gain_value: f32,
    pub gradient_sum: f32,
    pub hessian_sum: f32,
    pub counts_sum: usize,
    pub depth: usize,
    pub split_value: f64,
    pub split_feature: usize,
    pub split_gain: f32,
    pub missing_node: usize,
    pub left_child: usize,
    pub right_child: usize,
    pub start_idx: usize,
    pub stop_idx: usize,
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub is_leaf: bool,
    pub is_missing_leaf: bool,
    pub generalization: Option<f32>,
    pub node_type: NodeType,
    pub parent_node: usize,
    pub left_cats: HashSet<usize>,
    pub right_cats: HashSet<usize>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Node {
    pub num: usize,
    pub weight_value: f32,
    pub hessian_sum: f32,
    pub depth: usize,
    pub split_value: f64,
    pub split_feature: usize,
    pub split_gain: f32,
    pub missing_node: usize,
    pub left_child: usize,
    pub right_child: usize,
    pub is_leaf: bool,
    pub generalization: Option<f32>,
    pub node_type: NodeType,
    pub parent_node: usize,
    pub left_cats: HashSet<usize>,
    pub right_cats: HashSet<usize>,
}

impl Ord for SplittableNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain_value.total_cmp(&other.gain_value)
    }
}

impl PartialOrd for SplittableNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SplittableNode {
    fn eq(&self, other: &Self) -> bool {
        self.gain_value == other.gain_value
    }
}

impl Eq for SplittableNode {}

impl Node {
    /// Update all the info that is needed if this node is a
    /// parent node, this consumes the SplitableNode.
    pub fn make_parent_node(&mut self, split_node: SplittableNode) {
        self.is_leaf = false;
        self.missing_node = split_node.missing_node;
        self.split_value = split_node.split_value;
        self.split_feature = split_node.split_feature;
        self.split_gain = split_node.split_gain;
        self.left_child = split_node.left_child;
        self.right_child = split_node.right_child;
        self.parent_node = split_node.parent_node;
        self.left_cats = split_node.left_cats;
        self.right_cats = split_node.right_cats;
    }
    /// Get the path that should be traveled down, given a value.
    pub fn get_child_idx(&self, v: &f64, missing: &f64) -> usize {
        if !self.left_cats.is_empty() || !self.right_cats.is_empty() {
            if self.left_cats.contains(&(*v as usize)) {
                return self.left_child;
            } else if self.right_cats.contains(&(*v as usize)) {
                return self.right_child;
            } else {
                return self.missing_node;
            }
        }

        if is_missing(v, missing) {
            self.missing_node
        } else if v < &self.split_value {
            self.left_child
        } else {
            self.right_child
        }
    }

    pub fn has_missing_branch(&self) -> bool {
        (self.missing_node != self.right_child) && (self.missing_node != self.left_child)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum NodeType {
    Root,
    Left,
    Right,
    Missing,
}

impl SplittableNode {
    pub fn from_node_info(
        num: usize,
        depth: usize,
        start_idx: usize,
        stop_idx: usize,
        node_info: &NodeInfo,
        generalization: Option<f32>,
        node_type: NodeType,
        parent_node: usize,
    ) -> Self {
        SplittableNode {
            num,
            weight_value: node_info.weight,
            gain_value: node_info.gain,
            gradient_sum: node_info.grad,
            hessian_sum: node_info.cover,
            counts_sum: node_info.counts,
            depth,
            split_value: f64::ZERO,
            split_feature: 0,
            split_gain: f32::ZERO,
            missing_node: 0,
            left_child: 0,
            right_child: 0,
            start_idx,
            stop_idx,
            lower_bound: node_info.bounds.0,
            upper_bound: node_info.bounds.1,
            is_leaf: true,
            is_missing_leaf: false,
            generalization,
            node_type,
            parent_node,
            left_cats: HashSet::new(),
            right_cats: HashSet::new(),
        }
    }

    /// Create a default splitable node,
    /// we default to the node being a leaf.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num: usize,
        weight_value: f32,
        gain_value: f32,
        gradient_sum: f32,
        hessian_sum: f32,
        counts_sum: usize,
        depth: usize,
        start_idx: usize,
        stop_idx: usize,
        lower_bound: f32,
        upper_bound: f32,
        node_type: NodeType,
        left_cats: HashSet<usize>,
        right_cats: HashSet<usize>,
    ) -> Self {
        SplittableNode {
            num,
            weight_value,
            gain_value,
            gradient_sum,
            hessian_sum,
            counts_sum,
            depth,
            split_value: f64::ZERO,
            split_feature: 0,
            split_gain: f32::ZERO,
            missing_node: 0,
            left_child: 0,
            right_child: 0,
            start_idx,
            stop_idx,
            lower_bound,
            upper_bound,
            is_leaf: true,
            is_missing_leaf: false,
            generalization: None,
            node_type,
            parent_node: 0,
            left_cats,
            right_cats,
        }
    }

    pub fn update_children(
        &mut self,
        missing_child: usize,
        left_child: usize,
        right_child: usize,
        split_info: &SplitInfo,
    ) {
        self.left_child = left_child;
        self.right_child = right_child;
        self.split_feature = split_info.split_feature;
        self.split_gain = self.get_split_gain(&split_info.left_node, &split_info.right_node, &split_info.missing_node);
        self.split_value = split_info.split_value;
        self.missing_node = missing_child;
        self.is_leaf = false;
        self.left_cats = split_info.left_cats.clone();
        self.right_cats = split_info.right_cats.clone();
    }

    pub fn get_split_gain(
        &self,
        left_node_info: &NodeInfo,
        right_node_info: &NodeInfo,
        missing_node_info: &MissingInfo,
    ) -> f32 {
        let missing_split_gain = match &missing_node_info {
            MissingInfo::Branch(ni) | MissingInfo::Leaf(ni) => ni.gain,
            _ => 0.,
        };
        left_node_info.gain + right_node_info.gain + missing_split_gain - self.gain_value
    }

    pub fn as_node(&self, eta: f32) -> Node {
        Node {
            num: self.num,
            weight_value: self.weight_value * eta,
            hessian_sum: self.hessian_sum,
            depth: self.depth,
            missing_node: self.missing_node,
            split_value: self.split_value,
            split_feature: self.split_feature,
            split_gain: self.split_gain,
            left_child: self.left_child,
            right_child: self.right_child,
            is_leaf: self.is_leaf,
            generalization: self.generalization,
            node_type: self.node_type,
            parent_node: self.parent_node,
            left_cats: self.left_cats.clone(),
            right_cats: self.right_cats.clone(),
        }
    }
}

impl fmt::Display for Node {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_leaf {
            write!(f, "{}:leaf={},cover={}", self.num, self.weight_value, self.hessian_sum)
        } else {
            write!(
                f,
                "{}:[{} < {}] yes={},no={},missing={},gain={},cover={}",
                self.num,
                self.split_feature,
                self.split_value,
                self.left_child,
                self.right_child,
                self.missing_node,
                self.split_gain,
                self.hessian_sum
            )
        }
    }
}
