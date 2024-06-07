use serde::Deserialize;
use serde::Serialize;

use crate::node::SplittableNode;
use std::collections::BinaryHeap;
use std::collections::VecDeque;

pub trait Grower {
    fn add_node(&mut self, node: SplittableNode);
    fn get_next_node(&mut self) -> SplittableNode;
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
        self.is_empty()
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
        self.is_empty()
    }
}

#[derive(Serialize, Deserialize)]
pub enum GrowPolicy {
    DepthWise,
    LossGuide,
}
