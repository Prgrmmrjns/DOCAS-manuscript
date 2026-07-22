"""Twinning-only example (multi-meal extended model).

Fits the multi-meal extended T1D model (two days, with second-day meal labels) and
saves the twinning results to ``results/twin_<save_name>.pkl``. Run
``replay_multi_meal_extended.py`` afterwards to load those results and simulate.
"""

from multiprocessing import freeze_support

import os
import pandas as pd
import matplotlib.pyplot as plt

from data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData
from model.multi_meal_extended_t1d import MultiMealExtendedT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.agata_analysis import analyze_twin
from utils.plot_twinning import plot_twinning
from utils.plot_twinning_history import plot_twinning_history
from replaybg import ReplayBG


if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1_extended.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''), 'results')
    save_name = 'multi_meal_extended'

    # Impute breakfast (clearly missing)
    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    df = df.iloc[0:348, :]
    # Modify label of second day
    df.loc[312, "cho_label"] = 'B2'

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Create data in required format
    rbg_data = MultiMealExtendedT1DData(data=df,
                                        environment=rbg.environment)

    # Initialize model
    t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
    model = MultiMealExtendedT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.05), 'min': 0, 'max': .5},
        'p2': {'prior': Normal(mu=0.11, sigma=0.05), 'min': 0, 'max': .5},
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
        'kabs_B2': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_L2': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_S2': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'beta_B2': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_L2': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_S2': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'SI_B2': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
    }

    # Run twinning. The results (theta, history, rbg_data) are pickled to
    # results/twin_<save_name>.pkl so replay_multi_meal_extended.py can load them.
    result = rbg.twin(rbg_data=rbg_data,
                      model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      parallelize=True, n_jobs=-1, n_starts=64,
                      log_history=False,
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
