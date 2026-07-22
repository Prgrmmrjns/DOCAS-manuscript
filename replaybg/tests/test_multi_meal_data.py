"""Tests for ``data.multi_meal_t1d_data.MultiMealT1DData``.

This variant splits carbohydrate intake into per-label channels (breakfast,
lunch, dinner, snack, hypo-treatment) and adds a ``t_hour`` input. The full
per-method contract (time / insulin / meal-core / ``__setup_u``) is checked via
``dataclass_contract``; this file adds the multi-meal specifics: the default
``data_to_input`` map, label routing, and the custom-mapping override.
"""
import numpy as np
import pytest

from dataclass_contract import assert_meal_routing, assert_shared_contract
from data.multi_meal_t1d_data import MultiMealT1DData

BODY_WEIGHT = 100.0
CHANNELS = {
    "B": "meal_B",
    "L": "meal_L",
    "D": "meal_D",
    "S": "meal_S",
    "H": "meal_H",
}


@pytest.fixture
def mm_data(multi_meal_df, env):
    return MultiMealT1DData(
        data=multi_meal_df, body_weight=BODY_WEIGHT, environment=env
    )


def test_shared_conversion_contract(mm_data, multi_meal_df, env):
    assert_shared_contract(mm_data, multi_meal_df, env, BODY_WEIGHT)


def test_meal_label_routing(mm_data, multi_meal_df):
    assert_meal_routing(mm_data, multi_meal_df, CHANNELS, BODY_WEIGHT)


def test_default_data_to_input_map(mm_data):
    assert mm_data.data_to_input == {
        0: "meal_B",
        1: "meal_L",
        2: "meal_D",
        3: "meal_S",
        4: "meal_H",
        5: "bolus",
        6: "basal",
        7: "t_hour",
        8: "forcing_ip",
        9: "forcing_ra",
    }


def test_custom_data_to_input_override(multi_meal_df, env):
    mapping = {0: "basal", 1: "bolus"}
    obj = MultiMealT1DData(
        data=multi_meal_df,
        data_to_input=mapping,
        body_weight=BODY_WEIGHT,
        environment=env,
    )
    assert obj.data_to_input == mapping
    assert obj.u.shape == (obj.tsteps, 2)
    np.testing.assert_array_equal(obj.u[:, 0], obj.basal)
    np.testing.assert_array_equal(obj.u[:, 1], obj.bolus)
