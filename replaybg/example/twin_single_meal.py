"""Twinning-only example (single-meal model).

Fits the single-meal T1D model to one day of CGM + insulin + meal data and saves
the twinning results to ``results/twin_<save_name>.pkl``. Run ``replay_single_meal.py``
afterwards to load those results and simulate.
"""

from multiprocessing import freeze_support

import os
import pandas as pd
import matplotlib.pyplot as plt

from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.agata_analysis import analyze_twin
from utils.plot_twinning import plot_twinning
from utils.plot_twinning_history import plot_twinning_history
from replaybg import ReplayBG


if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''), 'results')
    save_name = 'single_meal_day_1'

    # Impute breakfast (clearly missing)
    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Create data in required format
    rbg_data = SingleMealT1DData(data=df,
                                 body_weight=100,
                                 environment=rbg.environment)

    # Initialize model
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': .5},
        'p2': {'prior': Normal(mu=0.11, sigma=0.004), 'min': 0, 'max': .5},
        'f': {'prior': Normal(mu=0.8, sigma=0.05), 'min': 0, 'max': 1},
        'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
        'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
        'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
        'SI': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'kabs': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'beta': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }

    # Run twinning. The results (theta, history, rbg_data) are pickled to
    # results/twin_<save_name>.pkl so replay_single_meal.py can load them.
    result = rbg.twin(rbg_data=rbg_data,
                      model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      parallelize=True, n_jobs=-1, n_starts=16,
                      log_history=True,
                      path=save_folder, save_name=save_name)

    # Analyze the fit with AGATA (glycemic profile of the fitted trace + the
    # fit-error metrics vs. the reference CGM) and fold the metrics back into
    # the saved twin_<save_name>.pkl (adds an 'analysis' key).
    analyze_twin(result, rbg_data, model, verbose=rbg.environment.verbose,
                 path=save_folder, save_name=save_name)

    # Plot the twinning fit: simulated model output (with estimated theta) vs the
    # observed data, plus the inputs that drove the fit. Hide the t_hour channel.
    mask_inputs = [i for i, name in rbg_data.data_to_input.items() if name == 't_hour']
    fig = plot_twinning(rbg_data, model, theta=result['theta'],
                        thresholds=[70, 180], mask_inputs=mask_inputs)
    fig.show()

    if result['history'] is not None:
        plot_twinning_history(result['history'], param_names=list(unknown_parameters_prior.keys()))

    plt.show()
