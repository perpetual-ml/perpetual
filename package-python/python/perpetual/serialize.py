"""Serialization helpers for persisting model metadata.

Provides abstract and concrete serializers for scalars, JSON objects,
and NumPy arrays.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


class BaseSerializer(ABC, Generic[T]):
    """Abstract base for metadata serializers."""

    @abstractmethod
    def serialize(self, obj: T) -> str:
        """serialize method - should take an object and return a string"""

    @abstractmethod
    def deserialize(self, obj_repr: str) -> T:
        """deserialize method - should take a string and return original object"""


Scaler = Union[int, float, str]


class ScalerSerializer(BaseSerializer[Scaler]):
    """Serializer for scalar values (int, float, str)."""

    def serialize(self, obj: Scaler) -> str:
        if isinstance(obj, str):
            obj_ = f"'{obj}'"
        else:
            obj_ = str(obj)
        return obj_

    def deserialize(self, obj_repr: str) -> Scaler:
        return literal_eval(node_or_string=obj_repr)


ObjectItem = Union[
    List[Scaler],
    Dict[str, Scaler],
    Scaler,
]


class ObjectSerializer(BaseSerializer[ObjectItem]):
    """Serializer for JSON-compatible objects (lists, dicts, scalars)."""

    def serialize(self, obj: ObjectItem) -> str:
        return json.dumps(obj)

    def deserialize(self, obj_repr: str) -> ObjectItem:
        return json.loads(obj_repr)


@dataclass
class NumpyData:
    """Intermediate representation for a serialized NumPy array."""

    array: Union[List[float], List[int]]
    dtype: str
    shape: Tuple[int, ...]


class NumpySerializer(BaseSerializer[npt.NDArray]):
    """Serializer that round-trips NumPy arrays through JSON."""

    def serialize(self, obj: npt.NDArray) -> str:
        return json.dumps(
            {"array": obj.tolist(), "dtype": str(obj.dtype), "shape": obj.shape}
        )

    def deserialize(self, obj_repr: str) -> npt.NDArray:
        data = NumpyData(**json.loads(obj_repr))
        a = np.array(data.array, dtype=data.dtype)  # type: ignore
        if len(data.shape) == 1:
            return a
        else:
            return a.reshape(data.shape)
