from __future__ import annotations

import numpy as np
import pytest

from perpetual.serialize import (
    NumpySerializer,
    ObjectItem,
    ObjectSerializer,
    Scaler,
    ScalerSerializer,
)

scaler_values = [
    1,
    1.0,
    1.00101,
    "a string",
    True,
    False,
    None,
]


@pytest.mark.parametrize("value", scaler_values)
def test_scaler(value: Scaler):
    serializer = ScalerSerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert value == serializer.deserialize(r)


object_values = [
    [1, 2, 3],
    [1.0, 4.0],
    ["a", "b", "c"],
    {"a": 1.0, "b": 2.0},
    {"a": "test", "b": "what"},
    *scaler_values,
]


@pytest.mark.parametrize("value", object_values)
def test_object(value: ObjectItem):
    serializer = ObjectSerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert value == serializer.deserialize(r)


numpy_values = [
    np.array([1.0, 2.23]),
    np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3)),
    np.array([1, 2, 3, 4, 5, 6], dtype="int").reshape((2, 3)),
]


@pytest.mark.parametrize("value", numpy_values)
def test_numpy(value: np.ndarray):
    serializer = NumpySerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert np.array_equal(value, serializer.deserialize(r))
