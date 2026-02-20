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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{NodeType, SplittableNode};

    fn create_node(num: usize, gain: f32) -> SplittableNode {
        SplittableNode::new(
            num,
            0.0,  // weight
            gain, // gain
            0.0,  // grad
            0.0,  // hess
            0,    // count
            0,    // depth
            0,    // start_idx
            0,    // stop_idx
            0.0,  // lower
            0.0,  // upper
            NodeType::Root,
            None,
            [0.0; 5],
        )
    }

    #[test]
    fn test_binary_heap_grower() {
        let mut grower: BinaryHeap<SplittableNode> = BinaryHeap::new();
        assert!(grower.is_empty());

        grower.add_node(create_node(1, 1.0));
        grower.add_node(create_node(2, 5.0));
        grower.add_node(create_node(3, 2.0));

        assert!(!grower.is_empty());

        // Max-heap based on gain
        let n1 = grower.get_next_node();
        assert_eq!(n1.num, 2);
        assert_eq!(n1.gain_value, 5.0);

        let n2 = grower.get_next_node();
        assert_eq!(n2.num, 3);
        assert_eq!(n2.gain_value, 2.0);

        let n3 = grower.get_next_node();
        assert_eq!(n3.num, 1);
        assert_eq!(n3.gain_value, 1.0);

        assert!(grower.is_empty());
    }

    #[test]
    fn test_vec_deque_grower() {
        let mut grower: VecDeque<SplittableNode> = VecDeque::new();
        assert!(grower.is_empty());

        grower.add_node(create_node(1, 1.0));
        grower.add_node(create_node(2, 5.0));
        grower.add_node(create_node(3, 2.0));

        assert!(!grower.is_empty());

        // FIFO behavior for DepthWise
        let n1 = grower.get_next_node();
        assert_eq!(n1.num, 1);

        let n2 = grower.get_next_node();
        assert_eq!(n2.num, 2);

        let n3 = grower.get_next_node();
        assert_eq!(n3.num, 3);

        assert!(grower.is_empty());
    }

    #[test]
    #[should_panic(expected = "Grower should not be empty")]
    fn test_binary_heap_panic() {
        let mut grower: BinaryHeap<SplittableNode> = BinaryHeap::new();
        grower.get_next_node();
    }

    #[test]
    #[should_panic(expected = "Grower should not be empty")]
    fn test_vec_deque_panic() {
        let mut grower: VecDeque<SplittableNode> = VecDeque::new();
        grower.get_next_node();
    }
}
