use crate::data::FloatData;
use crate::splitter::{MissingInfo, NodeInfo, SplitInfo};
use crate::utils::is_missing;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{self, Debug};

#[derive(Debug, Deserialize, Serialize)]
pub struct SplittableNode {
    pub num: usize,
    pub weight_value: f32,
    pub gain_value: f32,
    pub gradient_sum: f32,
    pub hessian_sum: f32,
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
    pub parent_node: usize,
    #[allow(clippy::box_collection)]
    pub left_cats: Option<Box<Vec<bool>>>,
    pub stats: Option<Box<NodeStats>>,
}

/// Statistics stored for each node when save_node_stats is enabled.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct NodeStats {
    pub depth: usize,
    pub node_type: NodeType,
    pub count: usize,
    pub generalization: Option<f32>,
    pub weights: [f32; 5],
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Node {
    pub num: usize,
    pub weight_value: f32,
    pub hessian_sum: f32,
    pub split_value: f64,
    pub split_feature: usize,
    pub split_gain: f32,
    pub missing_node: usize,
    pub left_child: usize,
    pub right_child: usize,
    pub is_leaf: bool,
    pub parent_node: usize,
    #[allow(clippy::box_collection)]
    pub left_cats: Option<Box<Vec<bool>>>,
    pub stats: Option<Box<NodeStats>>,
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
    pub fn make_parent_node(&mut self, split_node: SplittableNode, eta: f32) {
        self.is_leaf = false;
        self.missing_node = split_node.missing_node;
        self.split_value = split_node.split_value;
        self.split_feature = split_node.split_feature;
        self.split_gain = split_node.split_gain;
        self.left_child = split_node.left_child;
        self.right_child = split_node.right_child;
        self.parent_node = split_node.parent_node;
        self.left_cats = split_node.left_cats;
        // If we are keeping stats, update them from the split_node stats.
        if let Some(stats) = &mut self.stats {
            if let Some(sn_stats) = split_node.stats {
                stats.generalization = sn_stats.generalization;
                stats.weights = sn_stats.weights.map(|x| x * eta);
            }
        }
    }
    /// Get the path that should be traveled down, given a value.
    pub fn get_child_idx(&self, v: &f64, missing: &f64) -> usize {
        // Check for missing values FIRST
        if is_missing(v, missing) {
            return self.missing_node;
        }

        // Then check categorical splits
        if let Some(left_cats) = &self.left_cats {
            if *left_cats.get(*v as usize).unwrap_or(&false) {
                return self.left_child;
            } else {
                return self.right_child;
            }
        }

        // Finally numerical splits
        if v < &self.split_value {
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
    #[allow(clippy::too_many_arguments)]
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
            parent_node,
            left_cats: None,
            stats: Some(Box::new(NodeStats {
                depth,
                node_type,
                count: node_info.counts,
                generalization,
                weights: node_info.weights,
            })),
        }
    }

    /// Create a default splitable node,
    /// we default to the node being a leaf.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::box_collection)]
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
        left_cats: Option<Box<Vec<bool>>>,
        weights: [f32; 5],
    ) -> Self {
        SplittableNode {
            num,
            weight_value,
            gain_value,
            gradient_sum,
            hessian_sum,
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
            parent_node: 0,
            left_cats,
            stats: Some(Box::new(NodeStats {
                depth,
                node_type,
                count: counts_sum,
                generalization: None,
                weights,
            })),
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

    pub fn as_node(&self, eta: f32, save_node_stats: bool) -> Node {
        Node {
            num: self.num,
            weight_value: self.weight_value * eta,
            hessian_sum: self.hessian_sum,
            missing_node: self.missing_node,
            split_value: self.split_value,
            split_feature: self.split_feature,
            split_gain: self.split_gain,
            left_child: self.left_child,
            right_child: self.right_child,
            is_leaf: self.is_leaf,
            parent_node: self.parent_node,
            left_cats: self.left_cats.clone(),
            stats: if save_node_stats {
                if let Some(s) = &self.stats {
                    let mut stats = s.clone();
                    stats.weights = stats.weights.map(|x| x * eta);
                    Some(stats)
                } else {
                    None
                }
            } else {
                None
            },
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
