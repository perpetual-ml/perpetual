"""Data structures for tree inspection."""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union


@dataclass
class Node:
    """Dataclass representing a single node in a decision tree.

    Attributes
    ----------
    num : int
        Node index.
    weight_value : float
        Leaf weight (prediction contribution).
    hessian_sum : float
        Sum of hessians (sample coverage).
    depth : int
        Depth of the node in the tree.
    split_value : float
        Threshold used for splitting.
    split_feature : str or int
        Feature used for the split.
    split_gain : float
        Gain achieved by this split.
    missing_node : int
        Index of the child that receives missing values.
    left_child : int
        Index of the left child.
    right_child : int
        Index of the right child.
    is_leaf : bool
        Whether this node is a leaf.
    node_type : str
        One of ``"split"``, ``"leaf"``, or ``"missing"``.
    parent_node : int
        Index of the parent node.
    generalization : float or None
        Generalization score (if ``save_node_stats`` is enabled).
    left_cats : iterable or None
        Categorical values routed to the left child.
    right_cats : iterable or None
        Categorical values routed to the right child.
    count : int
        Number of samples reaching this node.
    weights : iterable of float or None
        Cross-validation fold weights.
    stats : any or None
        Additional node statistics.
    """

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
