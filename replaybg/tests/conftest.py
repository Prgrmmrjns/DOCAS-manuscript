"""Shared pytest fixtures for the ReplayBG test suite.

The ``single_meal_df`` fixture builds a small, fully in-memory dataframe shaped
like the input ReplayBG consumes, so tests exercise the data-preparation and
model code without touching disk. It is deliberately tiny (6 samples, 5-minute
spacing) to keep the suite fast.
"""
import numpy as np
import pandas as pd
import pytest

from environment import Environment


@pytest.fixture
def env():
    """A default :class:`Environment` (ts=1, seed=42)."""
    return Environment()


@pytest.fixture
def single_meal_df():
    """A synthetic single-meal dataframe for ``SingleMealT1DData``.

    6 samples, 5 minutes apart, spanning 25 minutes. One glucose value is NaN
    (to exercise ``y_idxs``); there is a single bolus at index 1 and a single
    meal at index 2.
    """
    t = pd.date_range("2020-01-01 08:00:00", periods=6, freq="5min")

    glucose = np.array([120.0, 125.0, np.nan, 130.0, 128.0, 122.0])
    basal = np.full(6, 0.5)

    bolus = np.zeros(6)
    bolus[1] = 2.0
    bolus_label = np.array(["", "B", "", "", "", ""])

    cho = np.zeros(6)
    cho[2] = 40.0
    cho_label = np.array(["", "", "B", "", "", ""])

    return pd.DataFrame(
        {
            "t": t,
            "glucose": glucose,
            "basal": basal,
            "bolus": bolus,
            "bolus_label": bolus_label,
            "cho": cho,
            "cho_label": cho_label,
        }
    )


@pytest.fixture
def multi_meal_df():
    """A synthetic multi-meal dataframe for ``MultiMealT1DData``.

    10 samples, 5 minutes apart, starting at 07:58 so the first sample crosses
    both an hour boundary (07 -> 08) and a minute wrap (58 -> 02) — exercising
    the ``t_hour`` / ``t_min`` expansion. One meal of each supported label
    (B, L, D, S, H), two boluses, and one NaN glucose value.
    """
    t = pd.date_range("2020-01-01 07:58:00", periods=10, freq="5min")

    glucose = np.array(
        [120.0, 125.0, 130.0, 128.0, 122.0, 118.0, 124.0, np.nan, 133.0, 140.0]
    )
    basal = np.full(10, 0.5)

    bolus = np.zeros(10)
    bolus[1] = 4.0
    bolus[6] = 2.0
    bolus_label = np.array(["", "B", "", "", "", "", "C", "", "", ""])

    cho = np.zeros(10)
    cho[1] = 40.0  # B
    cho[2] = 50.0  # L
    cho[3] = 60.0  # D
    cho[4] = 20.0  # S
    cho[5] = 15.0  # H
    cho_label = np.array(["", "B", "L", "D", "S", "H", "", "", "", ""])

    return pd.DataFrame(
        {
            "t": t,
            "glucose": glucose,
            "basal": basal,
            "bolus": bolus,
            "bolus_label": bolus_label,
            "cho": cho,
            "cho_label": cho_label,
        }
    )


@pytest.fixture
def multi_meal_extended_df():
    """A synthetic dataframe for ``MultiMealExtendedT1DData``.

    12 samples, 5 minutes apart. Includes one meal of every supported label,
    including the second-occurrence channels (B2, L2, S2), plus a NaN glucose.
    """
    t = pd.date_range("2020-01-01 08:00:00", periods=12, freq="5min")

    glucose = np.array(
        [
            120.0, 125.0, 130.0, 128.0, 122.0, 118.0,
            124.0, 127.0, np.nan, 133.0, 140.0, 136.0,
        ]
    )
    basal = np.full(12, 0.5)

    bolus = np.zeros(12)
    bolus[1] = 4.0
    bolus_label = np.array([""] * 12)
    bolus_label[1] = "B"

    cho = np.zeros(12)
    cho[1] = 40.0   # B
    cho[2] = 50.0   # L
    cho[3] = 60.0   # D
    cho[4] = 20.0   # S
    cho[5] = 15.0   # H
    cho[6] = 45.0   # B2
    cho[7] = 55.0   # L2
    cho[8] = 25.0   # S2
    cho_label = np.array(
        ["", "B", "L", "D", "S", "H", "B2", "L2", "S2", "", "", ""]
    )

    return pd.DataFrame(
        {
            "t": t,
            "glucose": glucose,
            "basal": basal,
            "bolus": bolus,
            "bolus_label": bolus_label,
            "cho": cho,
            "cho_label": cho_label,
        }
    )
