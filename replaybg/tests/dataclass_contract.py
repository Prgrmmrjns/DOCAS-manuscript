"""Shared assertion helpers for the ``*T1DData`` classes.

``MultiMealT1DData`` and ``MultiMealExtendedT1DData`` share an identical
time / insulin / meal-core / ``__setup_u`` pipeline and differ only in the set
of meal-label channels and the ``data_to_input`` map. These helpers pin down
the full per-method contract once so each class's test file only has to supply
its dataframe and its label→channel routing.

This module is intentionally NOT named ``test_*`` so pytest does not collect it.
"""
import numpy as np

YTS = 5  # integration steps per data sample (hard-coded in every data class)


def _blocks(idx):
    """Step-resolution slice for data-sample index ``idx``."""
    return slice(idx * YTS, (idx + 1) * YTS)


def assert_shared_contract(obj, df, env, body_weight):
    """Assert everything the data classes compute identically.

    Covers ``__time_setup`` (steps, ``t_data``, ``t_hour``, ``t_min``,
    ``t_start``), glucose/``y_idxs``, ``__insulin_setup`` (basal & bolus scaling,
    labels, raw ``*_data``), the aggregate meal / announcement / ``meal_type``
    from ``__meal_setup``, the zeroed forcing inputs, and the ``u`` matrix from
    ``__setup_u``.
    """
    n = df.shape[0]

    # ---- __time_setup: step counts -------------------------------------- #
    minutes_span = (df.t.iloc[-1] - df.t.iloc[0]).total_seconds() / 60
    assert obj.yts == YTS
    assert obj.tsteps == int(minutes_span + YTS) * env.ts
    assert obj.tysteps == obj.tsteps // YTS
    assert obj.tysteps == n

    # ---- __time_setup: raw times + hour/minute expansion ---------------- #
    np.testing.assert_array_equal(obj.t_data, df.t.to_numpy())
    hours = df.t.dt.hour.to_numpy().astype(int)
    minutes = df.t.dt.minute.to_numpy().astype(int)
    assert obj.t_start == hours[0] * 60 + minutes[0]
    for i in range(n):
        np.testing.assert_array_equal(obj.t_hour[_blocks(i)], np.full(YTS, hours[i]))
        expected_min = np.arange(minutes[i], minutes[i] + YTS) % 60
        np.testing.assert_array_equal(obj.t_min[_blocks(i)], expected_min)

    # ---- u2ss / body weight --------------------------------------------- #
    assert obj.u2ss == np.mean(df.basal.values) * 1000 / body_weight
    assert obj.body_weight == body_weight

    # ---- glucose + non-NaN indices -------------------------------------- #
    np.testing.assert_array_equal(obj.y, df.glucose.values.astype(float))
    np.testing.assert_array_equal(obj.y_idxs, np.where(~np.isnan(df.glucose.values))[0])

    # ---- __insulin_setup ------------------------------------------------ #
    np.testing.assert_array_equal(obj.basal_data, df.basal.values)
    np.testing.assert_array_equal(obj.bolus_data, df.bolus.values)
    scale = 1000.0 / body_weight
    for i in range(n):
        np.testing.assert_allclose(obj.basal[_blocks(i)], df.basal.values[i] * scale)
    for i in np.where(df.bolus.values)[0]:
        np.testing.assert_allclose(obj.bolus[_blocks(i)], df.bolus.values[i] * scale)
        assert obj.bolus_label[i * YTS] == df.bolus_label.values[i][:1]
    # Bolus is zero outside bolus blocks.
    expected_bolus = np.zeros(obj.tsteps)
    for i in np.where(df.bolus.values)[0]:
        expected_bolus[_blocks(i)] = df.bolus.values[i] * scale
    np.testing.assert_allclose(obj.bolus, expected_bolus)

    # ---- __meal_setup: aggregate meal / announcement / type ------------- #
    np.testing.assert_array_equal(obj.meal_data, df.cho.values)
    expected_meal = np.zeros(obj.tsteps)
    expected_ann = np.zeros(obj.tsteps)
    for i in np.where(df.cho.values)[0]:
        expected_meal[_blocks(i)] = df.cho.values[i] * scale
        expected_ann[i * YTS] = df.cho.values[i] * YTS
        # meal_type is a '<U1' array, so multi-char labels are truncated.
        assert obj.meal_type[i * YTS] == df.cho_label.values[i][:1]
    np.testing.assert_allclose(obj.meal, expected_meal)
    np.testing.assert_allclose(obj.meal_announcement, expected_ann)

    # ---- forcing inputs default to zero --------------------------------- #
    assert obj.forcing_ip.shape == (obj.tsteps,)
    assert obj.forcing_ra.shape == (obj.tsteps,)
    assert obj.forcing_ip.sum() == 0.0
    assert obj.forcing_ra.sum() == 0.0

    # ---- __setup_u ------------------------------------------------------ #
    assert obj.u.shape == (obj.tsteps, len(obj.data_to_input))
    for col, attr in obj.data_to_input.items():
        np.testing.assert_array_equal(obj.u[:, col], getattr(obj, attr))


def assert_meal_routing(obj, df, channel_by_label, body_weight):
    """Assert each labelled meal lands in exactly its channel and nowhere else.

    ``channel_by_label`` maps a cho label (e.g. ``'B'``) to the attribute name
    of its dedicated channel (e.g. ``'meal_B'``).
    """
    scale = 1000.0 / body_weight
    labels = df.cho_label.values
    cho = df.cho.values
    for label, attr in channel_by_label.items():
        expected = np.zeros(obj.tsteps)
        for i in np.where(cho)[0]:
            if labels[i] == label:
                expected[_blocks(i)] = cho[i] * scale
        np.testing.assert_allclose(getattr(obj, attr), expected)
