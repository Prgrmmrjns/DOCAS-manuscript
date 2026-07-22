"""Get started with ReplayBG (multi-meal model).

This is the one-file, end-to-end tour of ReplayBG. It:

  1. loads a day of CGM + insulin + meal data,
  2. **twins** the physiological model to that data (creates the digital twin), and
  3. **replays** the twin twice — once as-is (baseline), and once under a
     counterfactual "what-if" scenario (30% less bolus insulin) — to show how the
     same digital twin answers "what would this person's glucose have looked like?".

Everything is kept in memory so you can follow a single linear flow. The
task-specific example scripts (``twin_multi_meal.py``, ``replay_multi_meal.py``,
``replay_with_callbacks.py``, ...) split twinning and replay across files and
communicate through a pickle; start here first.

Run it from the repository root::

    uv run python example/get_started.py
"""

import os
import sys
from multiprocessing import freeze_support

import pandas as pd
import matplotlib.pyplot as plt

# Make the repo-root modules (replaybg, model, data, utils, ...) importable when
# this script is run directly, regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replaybg import ReplayBG
from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.numba_dicts import to_typed_f64_dict
from utils.plot_twinning import plot_twinning
from utils.plot_replay import plot_replay
from utils.agata_analysis import analyze_replay


# ``twin(parallelize=True)`` uses multiprocessing, so the whole script must sit
# behind the standard ``if __name__ == '__main__'`` / ``freeze_support()`` guard.
if __name__ == '__main__':
    freeze_support()

    # ------------------------------------------------------------------
    # 0. Load and prepare the data
    # ------------------------------------------------------------------
    # ReplayBG expects a pandas DataFrame with one row every 5 minutes and these
    # columns:
    #   t            timestamp (datetime)          — 5-minute grid
    #   glucose      CGM, mg/dL                    — may contain NaN
    #   cho          ingested carbs, g             — 0 when no meal
    #   bolus        bolus insulin, U              — 0 when no bolus
    #   basal        basal insulin, U/min          — per-row rate
    #   bolus_label  label of each bolus event
    #   cho_label    meal type: 'B','L','D','S','H'
    #                (breakfast/lunch/dinner/snack/hypo-treatment)
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_parquet(os.path.join(here, "data_day_1_2.parquet"))
    df['t'] = pd.to_datetime(df['t'])   # the 't' column MUST be datetime

    # This synthetic trace is missing its first breakfast; add it back so the
    # meal-absorption dynamics have something to fit against.
    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    # ------------------------------------------------------------------
    # 1. Create the ReplayBG object and the model-ready data
    # ------------------------------------------------------------------
    rbg = ReplayBG()

    # MultiMealT1DData converts the raw DataFrame into the input matrix and
    # observations the model/twinner consume (u2ss, tsteps, ...).
    rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)

    # The multi-meal model needs to know the time of day of the first sample
    # (minutes past midnight) so its time-of-day insulin sensitivity lines up.
    t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)

    # ------------------------------------------------------------------
    # 2. STEP 1 — Twin: fit the model to the data (create the digital twin)
    # ------------------------------------------------------------------
    # unknown_parameters_prior declares which physiological parameters to estimate
    # and the prior belief about each. Twinning finds the parameter set (theta)
    # that best explains this person's data (MAP estimate). The multi-meal model
    # estimates a per-time-of-day insulin sensitivity (SI_B/SI_L/SI_D) and
    # per-meal absorption (kabs_*) and delay (beta_*).
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

    # Correlations between parameters can be encoded as a Gaussian-copula prior.
    # Here insulin sensitivity is negatively correlated with the insulin-action
    # rate p2.
    correlations = {('SI_B', 'p2'): -.5,
                    ('SI_L', 'p2'): -.5,
                    ('SI_D', 'p2'): -.5}

    # Twinning runs a multi-start optimisation. n_starts is kept small here so the
    # example finishes in a couple of minutes — bump it up (e.g. 64) for real fits.
    result = rbg.twin(rbg_data=rbg_data,
                      model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      correlations=correlations,
                      parallelize=True, n_jobs=-1, n_starts=4)

    theta = result['theta']   # the estimated parameters — this IS the digital twin
    print("Estimated parameters (theta):")
    for name, value in theta.items():
        print(f"  {name:8s} = {value:.5f}")

    # Plot the fit: model output with the estimated theta vs the observed CGM.
    # mask_inputs hides the internal 't_hour' channel from the input panel.
    mask_inputs = [i for i, name in rbg_data.data_to_input.items() if name == 't_hour']
    fig_fit = plot_twinning(rbg_data, model, theta=theta,
                            input_groups=[[0, 1, 2, 3, 4], [5], [6]],
                            thresholds=[70, 180], mask_inputs=mask_inputs)
    fig_fit.show()

    # ------------------------------------------------------------------
    # 3. STEP 2a — Replay: forward-simulate the twin as-is (baseline)
    # ------------------------------------------------------------------
    # Rebuild the model with the estimated parameters (theta0 must go through
    # to_typed_f64_dict — the model consumes a Numba typed dict, not a plain one).
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
    baseline = rbg.replay(rbg_data=rbg_data, model=model)

    fig_baseline = plot_replay(baseline, thresholds=[70, 180], mask_inputs=mask_inputs)
    fig_baseline.show()

    # ------------------------------------------------------------------
    # 4. STEP 2b — Replay a counterfactual "what-if" scenario
    # ------------------------------------------------------------------
    # The point of the digital twin: change the inputs and ask "what would have
    # happened?". Here we cut every bolus by 30% and replay the SAME twin (same
    # theta) — only the insulin dosing changes, the physiology is unchanged.
    df_whatif = df.copy()
    df_whatif['bolus'] = df_whatif['bolus'] * 0.7
    rbg_data_whatif = MultiMealT1DData(data=df_whatif, environment=rbg.environment)

    model = MultiMealT1DModel(u2ss=rbg_data_whatif.u2ss, tsteps=rbg_data_whatif.tsteps,
                              theta0=to_typed_f64_dict(theta), t_start=rbg_data_whatif.t_start)
    whatif = rbg.replay(rbg_data=rbg_data_whatif, model=model)

    fig_whatif = plot_replay(whatif, thresholds=[70, 180], mask_inputs=mask_inputs)
    fig_whatif.show()

    # ------------------------------------------------------------------
    # 5. Compare the two scenarios
    # ------------------------------------------------------------------
    # AGATA computes standard glycemic metrics (time-in-range, mean glucose, ...).
    # With 30% less insulin we expect higher glucose / more time above range.
    print("\nBaseline glycemic metrics:")
    analyze_replay(baseline, ts=5, verbose=rbg.environment.verbose)
    print("\nWhat-if (-30% bolus) glycemic metrics:")
    analyze_replay(whatif, ts=5, verbose=rbg.environment.verbose)

    # Overlay the two glucose traces so the effect is visible at a glance.
    fig_cmp, ax = plt.subplots(figsize=(11, 4))
    ax.plot(baseline['output'], label='baseline')
    ax.plot(whatif['output'], label='what-if (-30% bolus)')
    ax.axhspan(70, 180, color='green', alpha=0.08, label='target range')
    ax.set_xlabel('time (min)')
    ax.set_ylabel('glucose (mg/dL)')
    ax.set_title('Replay: baseline vs. counterfactual')
    ax.legend()
    fig_cmp.tight_layout()
    fig_cmp.show()

    plt.show()
