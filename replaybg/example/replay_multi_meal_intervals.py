"""Interval (x0-chained) replay example (multi-meal model).

Loads the per-day twinning results produced by ``twin_multi_meal_intervals.py``
and replays them **one day at a time**, chaining the days together: the final
state of each replayed day is carried into the next as the initial condition
``x0`` (rescaled by that day's parameters ``theta_prev``). No twinning happens
here.

This mirrors the interval twinning workflow on the replay side, so the whole
multi-day simulation is reconstructed from independent per-day models joined by
their carry-over state, then shown as one continuous trace.
"""

from multiprocessing import freeze_support

import os
import matplotlib.pyplot as plt

from model.multi_meal_t1d import MultiMealT1DModel
from replaybg import ReplayBG
from utils.agata_analysis import analyze_replay
from utils.load_results import load_results
from utils.numba_dicts import to_typed_f64_dict
from utils.plot_replay import plot_replay_intervals

if __name__ == '__main__':
    freeze_support()
    save_folder = os.path.join(os.path.abspath(''), 'results')

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Discover how many days the twinning step produced.
    day_names = []
    n = 1
    while os.path.exists(os.path.join(save_folder, f'twin_multi_meal_intervals_day{n}.pkl')):
        day_names.append(f'multi_meal_intervals_day{n}')
        n += 1
    if not day_names:
        raise FileNotFoundError(
            "No twin_multi_meal_intervals_day<n>.pkl found — run "
            "twin_multi_meal_intervals.py first.")

    # Replay each day in turn, carrying the previous day's final state forward.
    replay_list = []
    prev_model = None
    for save_name in day_names:
        twin = load_results(save_folder, save_name, prefix='twin')
        theta_estimated = twin['theta']
        rbg_data = twin['rbg_data']

        # Day 1 is a cold start; later days inherit x0 (carry-over state) and
        # theta_prev (previous-day parameters) from the previous replayed model.
        if prev_model is None:
            model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                                      theta0=to_typed_f64_dict(theta_estimated),
                                      t_start=rbg_data.t_start)
        else:
            model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                                      theta0=to_typed_f64_dict(theta_estimated),
                                      t_start=rbg_data.t_start,
                                      x0=prev_model.get_final_x0(),
                                      theta_prev=prev_model.get_theta())

        out = rbg.replay(rbg_data=rbg_data,
                         model=model,
                         path=save_folder, save_name=save_name)

        # Fold the AGATA metrics back into the saved replay pickle.
        analyze_replay(out, ts=5, verbose=rbg.environment.verbose,
                       path=save_folder, save_name=save_name)

        # After replay the model holds this day's final state; keep it to seed
        # the next day.
        prev_model = model
        replay_list.append(out)

    # Continuity check: each replayed day should start where the previous ended.
    print("\nReplay carry-over continuity (IG, mg/dL):")
    for i in range(1, len(replay_list)):
        prev_end = replay_list[i - 1]['output'][-1]
        this_start = replay_list[i]['output'][0]
        print(f"  day{i} end = {prev_end:8.3f}  |  day{i + 1} start = {this_start:8.3f}"
              f"  |  Δ = {abs(this_start - prev_end):.3e}")

    # Plot the whole multi-day replay as one continuous figure. Hide the t_hour channel.
    mask_inputs = [i for i, name in replay_list[0]["data_to_input"].items()
                   if name == 't_hour']
    fig = plot_replay_intervals(replay_list, thresholds=[70, 180],
                                input_groups=[[0, 1, 2, 3, 4], [5], [6]],
                                mask_inputs=mask_inputs)
    fig.show()
    plt.show()
