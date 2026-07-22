"""Generate a synthetic single-day glucose trace for the single-meal model.

Forward-simulates the *single-meal* T1D model with its default parameters over a
fixed meal/bolus/basal scenario and writes the resulting interstitial-glucose
trace to ``example/data_single_meal.parquet`` in the same schema as
``example/data_day_1.parquet``.

The single-meal model does not distinguish meal types (breakfast/lunch/...) — all
carbohydrate events share one channel — so meals here carry only a generic label
("M") that is purely cosmetic for plotting.

The trace spans 06:00 to 19:00 on a single day (13 h, 5-minute sampling =
157 rows) and contains two bolused meals, at 07:00 and 13:00.

Run with::

    PYTHONPATH=. uv run python example/make_fake_single_meal_trace.py
"""

from multiprocessing import freeze_support

import os

import numpy as np
import pandas as pd

from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from replaybg import ReplayBG

# --- Scenario constants -----------------------------------------------------
YTS = 5                       # data sampling period (minutes) == data.yts
N_ROWS = 157                  # 06:00 -> 19:00 inclusive, 5-min grid
START = "11-May-2027 06:00:00"
BASAL = 0.025                 # U/min, constant
CARB_RATIO = 6.0              # g/U
OUT_NAME = "data_single_meal.parquet"

# Meals placed at fixed row indices on the 5-minute grid.
# (row, cho [g/min], cho_label, has_bolus)
MEALS = [
    (12, 10.0, "M", True),    # 11-May 07:00 breakfast
    (84, 14.0, "M", True),    # 11-May 13:00 lunch
]


def build_frame(meals):
    """Build the base dataframe (glucose left as NaN) from a meal list.

    Parameters
    ----------
    meals : list of (row, cho, cho_label, has_bolus)
        Meal events to place on the 5-minute grid.
    """
    t = pd.date_range(START, periods=N_ROWS, freq="5min")

    cho = np.zeros(N_ROWS)
    bolus = np.zeros(N_ROWS)
    cho_label = np.array([""] * N_ROWS, dtype=object)
    bolus_label = np.array([""] * N_ROWS, dtype=object)

    for row, grams, label, has_bolus in meals:
        cho[row] = grams
        cho_label[row] = label
        if has_bolus:
            bolus[row] = grams / CARB_RATIO
            bolus_label[row] = label

    return pd.DataFrame(
        {
            "t": t,
            "glucose": np.full(N_ROWS, np.nan),
            "cho": cho,
            "bolus": bolus,
            "basal": np.full(N_ROWS, BASAL),
            "bolus_label": bolus_label,
            "cho_label": cho_label,
        }
    )


def simulate(df):
    """Run a default-parameter replay and return the per-sample IG (length N_ROWS)."""
    rbg = ReplayBG(plot_mode=False, verbose=False)
    rbg_data = SingleMealT1DData(data=df, environment=rbg.environment)
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)
    out = rbg.replay(rbg_data=rbg_data, model=model)
    # out['output'] is per-minute IG; row i == minute i*YTS.
    ig = out["output"]
    return ig[np.arange(N_ROWS) * YTS]


def main():
    save_folder = os.path.abspath(os.path.dirname(__file__))

    df = build_frame(MEALS)
    ig = simulate(df)

    nadir_row = int(np.argmin(ig))
    print(f"nadir: {ig[nadir_row]:.1f} mg/dL at {df['t'].iloc[nadir_row]}")

    # Fill glucose from the simulation and format for saving.
    df["glucose"] = np.round(ig, 1)
    df["t"] = df["t"].dt.strftime("%d-%b-%Y %H:%M:%S")
    df["cho_label"] = df["cho_label"].replace("", np.nan)
    df["bolus_label"] = df["bolus_label"].replace("", np.nan)

    out_path = os.path.join(save_folder, OUT_NAME)
    df.to_parquet(out_path, index=False)

    print(f"glucose range: {df['glucose'].min():.1f}"
          f"–{df['glucose'].max():.1f} mg/dL")
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    freeze_support()
    main()
