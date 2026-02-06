//! Grower
//!
//! This module manages the process of growing decision trees, supporting different
//! growth policies like depth-wise and loss-guided.
use serde::Deserialize;
use serde::Serialize;

use crate::node::SplittableNode;
use std::collections::BinaryHeap;
use std::collections::VecDeque;

/// Trait for handling the growth of the tree.
pub trait Grower {
    /// Add a node to the grower.
    fn add_node(&mut self, node: SplittableNode);
    /// Get the next node to split.
    fn get_next_node(&mut self) -> SplittableNode;
    /// Check if the grower is empty.
    fn is_empty(&self) -> bool;
}

impl Grower for BinaryHeap<SplittableNode> {
    fn add_node(&mut self, node: SplittableNode) {
        self.push(node);
    }

    fn get_next_node(&mut self) -> SplittableNode {
        self.pop().expect("Grower should not be empty")
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Grower for VecDeque<SplittableNode> {
    fn add_node(&mut self, node: SplittableNode) {
        self.push_front(node);
    }

    fn get_next_node(&mut self) -> SplittableNode {
        self.pop_back().expect("Grower should not be empty")
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Policy for growing the tree.
#[derive(Serialize, Deserialize)]
pub enum GrowPolicy {
    /// Depth-wise growth (level-wise).
    DepthWise,
    /// Loss-guided growth (leaf-wise).
    LossGuide,
}
