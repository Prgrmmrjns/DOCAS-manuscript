"""Replay-only example (multi-meal model).

Loads the twinning results produced by ``twin_multi_meal.py`` (the fitted ``theta``
and the prepared ``rbg_data``) and runs a forward simulation. No twinning happens here.
"""

from multiprocessing import freeze_support

import os
import matplotlib.pyplot as plt

from model.multi_meal_t1d import MultiMealT1DModel
from replaybg import ReplayBG
from utils.agata_analysis import analyze_replay
from utils.load_results import load_results
from utils.numba_dicts import to_typed_f64_dict
from utils.plot_replay import plot_replay

if __name__ == '__main__':
    freeze_support()
    save_folder = os.path.join(os.path.abspath(''), 'results')
    save_name = 'multi_meal_day_1'

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Load the twinning results (theta + rbg_data) saved by twin_multi_meal.py
    twin = load_results(save_folder, save_name, prefix='twin')
    theta_estimated = twin['theta']
    rbg_data = twin['rbg_data']

    # Initialize the model with the estimated parameters. t_start is recovered from
    # the prepared data, so no original DataFrame is needed.
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta_estimated),
                              t_start=rbg_data.t_start)

    # Run replay and save the simulation
    out = rbg.replay(rbg_data=rbg_data,
                     model=model,
                     path=save_folder, save_name=save_name)

    # Analyze the replayed glucose with AGATA and fold the metrics back into the
    # saved replay_<save_name>.pkl (adds an 'analysis' key).
    analyze_replay(out, ts=5, verbose=rbg.environment.verbose,
                   path=save_folder, save_name=save_name)

    # Plot the replayed output (with thresholds) and the inputs that drove it.
    # Hide the t_hour channel.
    mask_inputs = [i for i, name in out["data_to_input"].items() if name == 't_hour']
    fig = plot_replay(out, thresholds=[70, 180], mask_inputs=mask_inputs)
    fig.show()
    plt.show()