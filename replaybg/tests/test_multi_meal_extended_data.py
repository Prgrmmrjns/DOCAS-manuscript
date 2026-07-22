"""Tests for ``data.multi_meal_extended_t1d_data.MultiMealExtendedT1DData``.

The extended variant adds second-occurrence meal channels (B2, L2, S2) on top
of the standard multi-meal channels, for simulations spanning more than one
day. The shared per-method contract is checked via ``dataclass_contract``; this
file adds the extended-specific routing (including the second-occurrence
channels) and the default / custom ``data_to_input`` map.
"""
import numpy as np
import pytest

from dataclass_contract import assert_meal_routing, assert_shared_contract
from data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData

BODY_WEIGHT = 100.0
CHANNELS = {
    "B": "meal_B",
    "L": "meal_L",
    "D": "meal_D",
    "S": "meal_S",
    "H": "meal_H",
    "B2": "meal_B2",
    "L2": "meal_L2",
    "S2": "meal_S2",
}


@pytest.fixture
def mme_data(multi_meal_extended_df, env):
    return MultiMealExtendedT1DData(
        data=multi_meal_extended_df, body_weight=BODY_WEIGHT, environment=env
    )


def test_shared_conversion_contract(mme_data, multi_meal_extended_df, env):
    assert_shared_contract(mme_data, multi_meal_extended_df, env, BODY_WEIGHT)


def test_meal_label_routing_including_second_occurrence(mme_data, multi_meal_extended_df):
    assert_meal_routing(mme_data, multi_meal_extended_df, CHANNELS, BODY_WEIGHT)


def test_default_data_to_input_map(mme_data):
    assert mme_data.data_to_input == {
        0: "meal_B",
        1: "meal_L",
        2: "meal_D",
        3: "meal_S",
        4: "meal_H",
        5: "meal_B2",
        6: "meal_L2",
        7: "meal_S2",
        8: "bolus",
        9: "basal",
        10: "t_hour",
        11: "forcing_ip",
        12: "forcing_ra",
    }


def test_custom_data_to_input_override(multi_meal_extended_df, env):
    mapping = {0: "meal_B2", 1: "meal_L2", 2: "meal_S2"}
    obj = MultiMealExtendedT1DData(
        data=multi_meal_extended_df,
        data_to_input=mapping,
        body_weight=BODY_WEIGHT,
        environment=env,
    )
    assert obj.data_to_input == mapping
    assert obj.u.shape == (obj.tsteps, 3)
    for col, attr in mapping.items():
        np.testing.assert_array_equal(obj.u[:, col], getattr(obj, attr))
