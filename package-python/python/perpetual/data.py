from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union


@dataclass
class Node:
    """Dataclass representation of a node, this represents all of the fields present in a tree node."""

    num: int
    weight_value: float
    hessian_sum: float
    depth: int = 0
    split_value: float = 0.0
    split_feature: Union[str, int] = ""
    split_gain: float = 0.0
    missing_node: int = 0
    left_child: int = 0
    right_child: int = 0
    is_leaf: bool = False
    node_type: str = "split"
    parent_node: int = 0
    generalization: Optional[float] = None
    left_cats: Optional[Iterable] = None
    right_cats: Optional[Iterable] = None
    count: int = 0
    weights: Optional[Iterable[float]] = None
    stats: Optional[Any] = None
