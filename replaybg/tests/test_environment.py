"""Tests for ``environment`` — the ``Environment`` holder and the ``identity``
decorator used as the ``DEBUG`` fallback for Numba's ``njit``/``jitclass``.
"""
from environment import Environment, identity


def test_environment_defaults():
    e = Environment()
    assert e.ts == 1
    assert e.seed == 42
    assert e.plot_mode is True
    assert e.verbose is True


def test_environment_overrides_are_stored():
    e = Environment(ts=5, seed=7, plot_mode=False, verbose=False)
    assert e.ts == 5
    assert e.seed == 7
    assert e.plot_mode is False
    assert e.verbose is False
