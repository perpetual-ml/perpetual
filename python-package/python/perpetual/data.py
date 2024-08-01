from dataclasses import dataclass
from typing import Iterable


@dataclass
class Node:
    """Dataclass representation of a node, this represents all of the fields present in a tree node."""

    num: int
    weight_value: float
    hessian_sum: float
    depth: int
    split_value: float
    split_feature: int | str
    split_gain: float
    missing_node: int
    left_child: int
    right_child: int
    is_leaf: bool
    node_type: str
    parent_node: int
    generalization: float | None
    left_categories: Iterable | None
    right_categories: Iterable | None
