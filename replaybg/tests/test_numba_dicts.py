"""Tests for ``utils.numba_dicts.to_typed_f64_dict``."""
import numpy as np
import pytest
from numba import float64, types
from numba.typed import Dict

from utils.numba_dicts import to_typed_f64_dict


def test_none_returns_empty_typed_dict():
    out = to_typed_f64_dict(None)
    assert isinstance(out, Dict)
    assert len(out) == 0


def test_basic_conversion():
    out = to_typed_f64_dict({"a": 1, "b": 2.5})
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(2.5)
    assert isinstance(out["a"], (float, np.floating))


def test_keys_are_stringified():
    out = to_typed_f64_dict({1: 10.0})
    assert "1" in out
    assert out["1"] == pytest.approx(10.0)


def test_value_type_is_float64():
    out = to_typed_f64_dict({"a": 1})
    empty = Dict.empty(key_type=types.unicode_type, value_type=float64)
    assert out._dict_type.value_type == empty._dict_type.value_type


def test_non_strict_skips_invalid_entries():
    out = to_typed_f64_dict({"a": 1.0, "bad": "not-a-number"}, strict=False)
    assert "a" in out
    assert "bad" not in out


def test_strict_raises_on_invalid_entry():
    with pytest.raises(TypeError):
        to_typed_f64_dict({"bad": "not-a-number"}, strict=True)
