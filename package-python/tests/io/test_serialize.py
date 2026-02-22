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


def test_serialization_module():
    # ScalerSerializer
    ss = ScalerSerializer()
    assert ss.serialize(1) == "1"
    assert ss.serialize("test") == "'test'"
    assert ss.deserialize("1") == 1
    assert ss.deserialize("'test'") == "test"

    # ObjectSerializer
    os_ser = ObjectSerializer()
    obj = {"a": [1, 2], "b": 3.0}
    ser = os_ser.serialize(obj)
    assert os_ser.deserialize(ser) == obj
    # Test numpy array handled in ObjectSerializer
    assert os_ser.serialize(np.array([1, 2])) == "[1, 2]"

    # NumpySerializer
    ns = NumpySerializer()
    a1 = np.array([1, 2, 3], dtype=np.float64)
    ser1 = ns.serialize(a1)
    assert np.allclose(ns.deserialize(ser1), a1)

    a2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ser2 = ns.serialize(a2)
    a2_deser = ns.deserialize(ser2)
    assert a2_deser.shape == (2, 2)
    assert np.array_equal(a2_deser, a2)
