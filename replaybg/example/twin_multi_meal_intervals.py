"""Interval (x0-chained) twinning example (multi-meal model).

Fits the multi-meal T1D model **one day at a time** and chains the days together:
the final physiological state of each day is carried into the next as the initial
condition ``x0`` (rescaled by that day's parameters ``theta_prev``). This is the
"interval" workflow — the way ReplayBG twins a multi-day recording without
lumping everything into a single segment.

The two-day recording ``data_day_1_2.parquet`` is split at 04:00, which is the
model's dinner/night -> breakfast insulin-sensitivity boundary, so every segment
starts fresh at the breakfast window and the overnight period stays intact. Each
day's twin is pickled to ``results/twin_multi_meal_intervals_day<n>.pkl`` so
``replay_multi_meal_intervals.py`` can load them and replay the same chain.
"""

from multiprocessing import freeze_support

import os
import pandas as pd
import matplotlib.pyplot as plt

from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.agata_analysis import analyze_twin
from utils.plot_twinning import plot_twinning_intervals, _simulate
from replaybg import ReplayBG


def split_days(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Splits a continuous recording into daily segments at every 04:00.

    04:00 is where the model switches its insulin sensitivity from the
    dinner/night value (``SI_D``, active for hours ``< 4``) to the breakfast
    value (``SI_B``), so cutting there keeps each day's B/L/D/S meal structure —
    and the overnight SI_D period — inside a single segment.

    Parameters
    ----------
    df : pandas.DataFrame
        The full recording, with a datetime ``t`` column.

    Returns
    -------
    list of pandas.DataFrame
        One sub-frame per day, index-reset, in chronological order.
    """
    marks = [i for i in range(len(df))
             if df['t'].dt.hour.iloc[i] == 4 and df['t'].dt.minute.iloc[i] == 0]
    bounds = [0] + marks + [len(df)]
    return [df.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:])]


if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1_2.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''), 'results')

    # Impute the first day's breakfast (clearly missing from the recording).
    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    day_dfs = split_days(df)

    # Create ReplayBg instance
    rbg = ReplayBG()

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.05), 'min': 0, 'max': .5},
        'f': {'prior': Normal(mu=0.8, sigma=0.05), 'min': 0, 'max': 1},
        'p2': {'prior': Normal(mu=0.11, sigma=0.05), 'min': 0, 'max': .5},
        'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
        'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
        'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
        'SI_B': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_L': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_D': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'kabs_B': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_L': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_D': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_S': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'beta_B': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_L': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_D': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_S': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }

    # Correlations are disabled for this example (parameters treated as independent).
    correlations = None

    # Twin each day in turn, carrying the previous day's final state forward.
    segments = []
    prev_x0 = None
    prev_theta = None
    for n, day_df in enumerate(day_dfs, start=1):
        save_name = f'multi_meal_intervals_day{n}'

        rbg_data = MultiMealT1DData(data=day_df, environment=rbg.environment)
        t_start = int((day_df["t"].iloc[0] - day_df["t"].iloc[0].normalize()).total_seconds() / 60)

        # Day 1 is a cold start; later days inherit x0 (carry-over state) and
        # theta_prev (previous-day parameters used to rescale that state).
        if prev_x0 is None:
            model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                                      t_start=t_start)
        else:
            model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                                      t_start=t_start, x0=prev_x0, theta_prev=prev_theta)

        result = rbg.twin(rbg_data=rbg_data,
                          model=model,
                          unknown_parameters_prior=unknown_parameters_prior,
                          correlations=correlations,
                          parallelize=True, n_jobs=-1, n_starts=8,
                          log_history=False,
                          path=save_folder, save_name=save_name)

        # Fold the AGATA fit metrics back into the saved twin pickle.
        analyze_twin(result, rbg_data, model, verbose=rbg.environment.verbose,
                     path=save_folder, save_name=save_name)

        # Simulate the fitted model forward so it holds this day's final state,
        # then read off the carry-over conditions for the next day.
        _simulate(model, rbg_data, result['theta'])
        prev_x0 = model.get_final_x0()
        prev_theta = model.get_theta()

        segments.append({'rbg_data': rbg_data, 'model': model, 'theta': result['theta']})

    # Continuity check: each day's fit should start where the previous ended.
    print("\nTwinning carry-over continuity (IG, mg/dL):")
    for n in range(1, len(segments)):
        prev_end = _simulate(segments[n - 1]['model'], segments[n - 1]['rbg_data'],
                             segments[n - 1]['theta'])[-1]
        this_start = _simulate(segments[n]['model'], segments[n]['rbg_data'],
                               segments[n]['theta'])[0]
        print(f"  day{n} end = {prev_end:8.3f}  |  day{n + 1} start = {this_start:8.3f}"
              f"  |  Δ = {abs(this_start - prev_end):.3e}")

    # Plot the whole multi-day fit as one continuous figure. Hide the t_hour channel.
    mask_inputs = [i for i, name in segments[0]['rbg_data'].data_to_input.items()
                   if name == 't_hour']
    fig = plot_twinning_intervals(segments, thresholds=[70, 180],
                                  input_groups=[[0, 1, 2, 3, 4], [5], [6]],
                                  mask_inputs=mask_inputs)
    fig.show()
    plt.show()
