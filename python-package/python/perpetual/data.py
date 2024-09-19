from dataclasses import dataclass
from typing import Iterable, Optional, Union


@dataclass
class Node:
    """Dataclass representation of a node, this represents all of the fields present in a tree node."""

    num: int
    weight_value: float
    hessian_sum: float
    depth: int
    split_value: float
    split_feature: Union[str, int]
    split_gain: float
    missing_node: int
    left_child: int
    right_child: int
    is_leaf: bool
    node_type: str
    parent_node: int
    generalization: Optional[float]
    left_cats: Optional[Iterable]
    right_cats: Optional[Iterable]
