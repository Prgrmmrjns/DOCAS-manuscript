"""Tests for ``data.single_meal_t1d_data.SingleMealT1DData``.

The class converts a raw dataframe into the regularly-sampled input arrays and
metadata consumed by the model/twinner. These tests pin down its deterministic
conversion contract (unit scaling, step expansion, the ``u`` matrix layout)
using the in-memory ``single_meal_df`` fixture.
"""
import numpy as np
import pytest

from data.single_meal_t1d_data import SingleMealT1DData

YTS = 5  # integration steps per data sample (hard-coded in the class)
BODY_WEIGHT = 100.0


@pytest.fixture
def sm_data(single_meal_df, env):
    return SingleMealT1DData(
        data=single_meal_df, body_weight=BODY_WEIGHT, environment=env
    )


def test_u2ss_is_bodyweight_normalized_mean_basal(sm_data, single_meal_df):
    expected = np.mean(single_meal_df.basal.values) * 1000 / BODY_WEIGHT
    assert sm_data.u2ss == pytest.approx(expected)


def test_step_counts_match_time_span(sm_data, single_meal_df, env):
    # 6 samples 5 min apart -> 25 minute span.
    minutes_span = (
        single_meal_df.t.iloc[-1] - single_meal_df.t.iloc[0]
    ).total_seconds() / 60
    expected_tsteps = int(minutes_span + YTS) * env.ts
    assert sm_data.yts == YTS
    assert sm_data.tsteps == expected_tsteps
    assert sm_data.tysteps == sm_data.tsteps // YTS
    assert sm_data.tysteps == single_meal_df.shape[0]


def test_glucose_and_nan_indices(sm_data, single_meal_df):
    np.testing.assert_array_equal(
        sm_data.y, single_meal_df.glucose.values.astype(float)
    )
    # Index 2 is NaN in the fixture, so it must be excluded from y_idxs.
    assert 2 not in sm_data.y_idxs
    np.testing.assert_array_equal(sm_data.y_idxs, np.array([0, 1, 3, 4, 5]))


def test_bolus_is_scaled_and_expanded(sm_data):
    # Fixture: bolus of 2.0 at sample index 1 -> steps [5, 10).
    expected = 2.0 * (1000 / BODY_WEIGHT)
    np.testing.assert_allclose(sm_data.bolus[5:10], expected)
    # Everything outside that block stays zero.
    assert sm_data.bolus[:5].sum() == 0.0
    assert sm_data.bolus[10:].sum() == 0.0


def test_meal_is_scaled_expanded_and_announced(sm_data):
    # Fixture: cho of 40.0 at sample index 2 -> steps [10, 15).
    expected_meal = 40.0 * (1000 / BODY_WEIGHT)
    np.testing.assert_allclose(sm_data.meal[10:15], expected_meal)
    assert sm_data.meal[:10].sum() == 0.0
    assert sm_data.meal[15:].sum() == 0.0

    # Announcement is placed only at the start of the block and equals cho * yts.
    assert sm_data.meal_announcement[10] == pytest.approx(40.0 * YTS)
    announcement = sm_data.meal_announcement.copy()
    announcement[10] = 0.0
    assert announcement.sum() == 0.0


def test_u_matrix_shape_and_column_mapping(sm_data):
    assert sm_data.u.shape == (sm_data.tsteps, len(sm_data.data_to_input))
    # Each column of u must equal the attribute named in data_to_input.
    for col, attr in sm_data.data_to_input.items():
        np.testing.assert_array_equal(sm_data.u[:, col], getattr(sm_data, attr))


def test_forcing_inputs_default_to_zero(sm_data):
    assert sm_data.forcing_ip.shape == (sm_data.tsteps,)
    assert sm_data.forcing_ra.shape == (sm_data.tsteps,)
    assert sm_data.forcing_ip.sum() == 0.0
    assert sm_data.forcing_ra.sum() == 0.0


def test_raw_time_and_hour_minute_expansion(sm_data, single_meal_df):
    # t_data keeps the original timestamps.
    np.testing.assert_array_equal(sm_data.t_data, single_meal_df.t.to_numpy())

    hours = single_meal_df.t.dt.hour.to_numpy().astype(int)
    minutes = single_meal_df.t.dt.minute.to_numpy().astype(int)
    for i in range(single_meal_df.shape[0]):
        block = slice(i * YTS, (i + 1) * YTS)
        np.testing.assert_array_equal(sm_data.t_hour[block], np.full(YTS, hours[i]))
        expected_min = np.arange(minutes[i], minutes[i] + YTS) % 60
        np.testing.assert_array_equal(sm_data.t_min[block], expected_min)


def test_raw_data_arrays_mirror_input_columns(sm_data, single_meal_df):
    np.testing.assert_array_equal(sm_data.basal_data, single_meal_df.basal.values)
    np.testing.assert_array_equal(sm_data.bolus_data, single_meal_df.bolus.values)
    np.testing.assert_array_equal(sm_data.meal_data, single_meal_df.cho.values)


def test_bolus_label_stored_at_block_start(sm_data):
    # Fixture bolus label 'B' at sample index 1 -> stored across steps [5, 10).
    assert sm_data.bolus_label[5] == "B"


def test_single_meal_type_is_placeholder(sm_data):
    # The single-meal variant does not label meals; every meal step is '-'.
    assert sm_data.meal_type[10] == "-"


def test_basal_is_scaled_and_expanded(sm_data, single_meal_df):
    scale = 1000 / BODY_WEIGHT
    for i in range(single_meal_df.shape[0]):
        block = slice(i * YTS, (i + 1) * YTS)
        np.testing.assert_allclose(
            sm_data.basal[block], single_meal_df.basal.values[i] * scale
        )


def test_custom_data_to_input_override(single_meal_df, env):
    mapping = {0: "basal", 1: "meal"}
    obj = SingleMealT1DData(
        data=single_meal_df,
        data_to_input=mapping,
        body_weight=BODY_WEIGHT,
        environment=env,
    )
    assert obj.data_to_input == mapping
    assert obj.u.shape == (obj.tsteps, 2)
    np.testing.assert_array_equal(obj.u[:, 0], obj.basal)
    np.testing.assert_array_equal(obj.u[:, 1], obj.meal)
