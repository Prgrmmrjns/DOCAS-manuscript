"""Replay-only example with closed-loop callbacks and a noisy CGM sensor.

Loads the twinning results produced by ``twin_multi_meal.py`` and replays them twice:
once as a plain baseline, and once with a correction-bolus and a hypo-treatment policy
acting on a realistic, noisy CGM measurement. No twinning happens here.
"""

from multiprocessing import freeze_support

import os
import pandas as pd
import matplotlib.pyplot as plt

from model.multi_meal_t1d import MultiMealT1DModel
from replaybg import ReplayBG
from callbacks import CorrectionBolus, HypoTreatment
from sensors import Vettoretti19CGM
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

    # --- Baseline replay (no policy) ---
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta_estimated), t_start=rbg_data.t_start)
    baseline = rbg.replay(rbg_data=rbg_data, model=model, path=save_folder, save_name='baseline')

    # --- Replay with a correction-bolus callback, acting on a noisy CGM sensor ---
    # The sensor turns the model output (interstitial glucose) into a realistic CGM
    # measurement. Callbacks read ctx.measurement, so they close the loop on the noisy
    # signal instead of the true output.
    correction_bolus_callback = CorrectionBolus(threshold=180, target=120, cf=50, lockout_min=60)
    # HypoTreatment reads the bolus CorrectionBolus published on ctx.shared: if a
    # correction went in within the last 90 min that insulin is still working, so the
    # rescue is sized up. Neither policy holds a reference to the other.
    hypotreatment_callback = HypoTreatment(threshold=70, carbs=15, lockout_min=30,
                                           stacking_window_min=90, stacking_carbs=10)
    sensor = Vettoretti19CGM()
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta_estimated), t_start=rbg_data.t_start)
    controlled = rbg.replay(rbg_data=rbg_data, model=model, path=save_folder, save_name='controlled',
                            callbacks=[correction_bolus_callback, hypotreatment_callback],
                            sensor=sensor)

    # Analyze both replays with AGATA so the effect of the control policy is
    # visible in the glycemic metrics (the controlled run also analyzes the
    # noisy sensor measurement trace).
    analyze_replay(baseline, ts=5, verbose=rbg.environment.verbose,
                   path=save_folder, save_name='baseline')
    analyze_replay(controlled, ts=5, verbose=rbg.environment.verbose,
                   path=save_folder, save_name='controlled')

    # Exhaustive log of what the callbacks did, as a tidy table
    actions = pd.DataFrame(controlled['actions'])
    print(actions)

    # Plot the controlled replay: output + measurement, the inputs that drove it,
    # and the callback actions. Hide the t_hour channel.
    mask_inputs = [i for i, name in controlled["data_to_input"].items() if name == 't_hour']
    fig = plot_replay(controlled, thresholds=[70, 180], mask_inputs=mask_inputs)
    fig.show()
    plt.show()
