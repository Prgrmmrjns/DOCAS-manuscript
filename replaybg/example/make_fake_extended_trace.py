"""Generate a synthetic two-day glucose trace for the test suite.

Forward-simulates the multi-meal *extended* T1D model with its default
parameters over a fixed meal/bolus/basal scenario and writes the resulting
interstitial-glucose trace to ``example/data_two_day_extended.parquet`` in the
same schema as ``example/data_day_1.parquet``.

The trace spans 06:00 on day 1 to 11:00 on day 2 (29 h, 5-minute sampling =
349 rows). Meals (labelled B/L/D/S/H on day 1 and B2/S2 on day 2) and boluses
(carb ratio 10 g/U for B/L/D only) drive the simulation. The hypotreatment (H)
is placed at the first day-1 hypoglycemia found by a first, H-free simulation,
then the trace is re-simulated with H included.

Run with::

    uv run python example/make_fake_extended_trace.py
"""

from multiprocessing import freeze_support

import os

import numpy as np
import pandas as pd

from data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData
from model.multi_meal_extended_t1d import MultiMealExtendedT1DModel
from replaybg import ReplayBG

# --- Scenario constants -----------------------------------------------------
YTS = 5                       # data sampling period (minutes) == data.yts
N_ROWS = 349                  # 06:00 day 1 -> 11:00 day 2 inclusive, 5-min grid
START = "11-May-2027 06:00:00"
BASAL = 0.025                 # U/min, constant
CARB_RATIO = 6.0             # g/U, applied to B/L/D only
HYPO_THRESHOLD = 70.0        # mg/dL
OUT_NAME = "data_two_day_extended.parquet"

# Meals placed at fixed row indices on the 5-minute grid (see plan table).
# (row, cho [g/min], cho_label, has_bolus)
BASE_MEALS = [
    (12, 10.0, "B", True),    # 11-May 07:00 breakfast
    (84, 14.0, "L", True),    # 11-May 13:00 lunch
    (114, 4.0, "S", False),   # 11-May 15:30 snack
    (168, 12.0, "D", True),   # 11-May 20:00 dinner
    (288, 6.0, "B2", False),  # 12-May 06:00 second-day breakfast
    (330, 5.0, "S2", False),  # 12-May 09:30 second-day snack
]

# Standalone boluses not attached to their meal's row: (row, units, label).
# 1.0 U pre-bolus 30 min (6 rows) before B2 at 06:00 (row 288 -> 282, 05:30),
# sized for the 6 g/min B2 meal at the current carb ratio.
EXTRA_BOLUSES = [(282, 6.0 / CARB_RATIO, "B2")]

# Rows belonging to calendar day 1 (06:00 -> 23:55 on 11-May): rows 0..215.
DAY1_LAST_ROW = 215


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

    for row, units, label in EXTRA_BOLUSES:
        bolus[row] = units
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
    rbg_data = MultiMealExtendedT1DData(data=df, environment=rbg.environment)
    model = MultiMealExtendedT1DModel(u2ss=rbg_data.u2ss,
                                      tsteps=rbg_data.tsteps,
                                      t_start=rbg_data.t_start)
    out = rbg.replay(rbg_data=rbg_data, model=model)
    # out['output'] is per-minute IG; row i == minute i*YTS.
    ig = out["output"]
    return ig[np.arange(N_ROWS) * YTS]


def main():
    save_folder = os.path.abspath(os.path.dirname(__file__))

    # --- Pass 1: simulate without the hypotreatment ------------------------
    df = build_frame(BASE_MEALS)
    ig1 = simulate(df)

    day1_ig = ig1[: DAY1_LAST_ROW + 1]
    day1_hypos = np.where(day1_ig < HYPO_THRESHOLD)[0]
    nadir_row = int(np.argmin(day1_ig))
    print(f"[pass 1] day-1 nadir: {day1_ig[nadir_row]:.1f} mg/dL "
          f"at {df['t'].iloc[nadir_row]}")

    if day1_hypos.size:
        h_row = int(day1_hypos[0])
        print(f"[pass 1] first day-1 hypo (<{HYPO_THRESHOLD:.0f}): "
              f"{df['t'].iloc[h_row]} -> placing H there")
    else:
        whole_hypos = np.where(ig1 < HYPO_THRESHOLD)[0]
        if whole_hypos.size == 0:
            raise SystemExit(
                f"No hypoglycemia (<{HYPO_THRESHOLD:.0f} mg/dL) anywhere in the "
                f"trace; overall nadir is {ig1.min():.1f} mg/dL at "
                f"{df['t'].iloc[int(np.argmin(ig1))]}. Adjust a bolus/meal so a "
                f"hypo occurs, then re-run — refusing to invent an H time."
            )
        h_row = int(whole_hypos[0])
        print(f"[pass 1] no day-1 hypo; falling back to first trace-wide hypo "
              f"at {df['t'].iloc[h_row]} -> placing H there")

    # --- Pass 2: simulate with the hypotreatment added ---------------------
    meals_with_h = BASE_MEALS + [(h_row, 3.0, "H", False)]
    df = build_frame(meals_with_h)
    ig2 = simulate(df)

    # Fill glucose from the final simulation and format for saving.
    df["glucose"] = np.round(ig2, 1)
    df["t"] = df["t"].dt.strftime("%d-%b-%Y %H:%M:%S")
    df["cho_label"] = df["cho_label"].replace("", np.nan)
    df["bolus_label"] = df["bolus_label"].replace("", np.nan)

    out_path = os.path.join(save_folder, OUT_NAME)
    df.to_parquet(out_path, index=False)

    print(f"[pass 2] glucose range: {df['glucose'].min():.1f}"
          f"–{df['glucose'].max():.1f} mg/dL; IG at H row = "
          f"{df['glucose'].iloc[h_row]:.1f} mg/dL")
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    freeze_support()
    main()
