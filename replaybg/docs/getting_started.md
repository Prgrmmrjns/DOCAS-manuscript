# Getting Started

This page walks you through your first end-to-end ReplayBG session: **install**,
**twin** a physiological model to a day of data, and **replay** the twin — first
as-is, then under a counterfactual "what-if" scenario. It mirrors the runnable
[`example/get_started.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/get_started.py)
script.

## Installation

ReplayBG is not published on PyPI — run it from a clone. It requires
**Python ≥ 3.12** and uses [`uv`](https://docs.astral.sh/uv/) for dependency
management.

```bash
git clone https://github.com/DIANA-UNIPD/replaybg
cd replaybg
uv sync
```

Modules live at the repository root and are imported directly (`from replaybg
import ReplayBG`, `from model.multi_meal_t1d import MultiMealT1DModel`, …), so run
your scripts and the tests **from the repo root**.

You can run the complete get-started tour with:

```bash
uv run python example/get_started.py
```

## Preparation: the data

ReplayBG consumes a pandas `DataFrame` with **one row every 5 minutes** and the
following columns:

| Column        | Type / units          | Notes                                                    |
|---------------|-----------------------|----------------------------------------------------------|
| `t`           | datetime              | 5-minute grid; **must** be converted with `pd.to_datetime` |
| `glucose`     | float, mg/dL          | CGM; `NaN` allowed                                        |
| `cho`         | float, g              | ingested carbs; `0` when no meal                         |
| `bolus`       | float, U              | bolus insulin; `0` when no bolus                         |
| `basal`       | float, U/min          | basal insulin rate (per row)                             |
| `bolus_label` | str                   | label of each bolus event                                |
| `cho_label`   | str                   | meal type: `B`/`L`/`D`/`S`/`H` (breakfast/lunch/dinner/snack/hypo-treatment) |

!!! warning
    Data must follow strict format requirements. See
    [Choosing Data for Twinning](data_requirements.md) for the full rules and
    best practices.

```python
import pandas as pd
from replaybg import ReplayBG
from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.numba_dicts import to_typed_f64_dict

df = pd.read_parquet("example/data_day_1_2.parquet")
df['t'] = pd.to_datetime(df['t'])          # the 't' column MUST be datetime
```

## Step 0: the `ReplayBG` object, data and model

Create the top-level [`ReplayBG`](replaybg_object.md) object, then build the
**data** object and the **model** — together, a data class + model class form a
*blueprint* (see [Choosing a Blueprint](blueprints/choosing.md)). Here we use the
multi-meal blueprint.

```python
rbg = ReplayBG()
rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)

# The multi-meal model needs the time of day of the first sample (minutes past
# midnight) so its time-of-day insulin sensitivity lines up.
t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
```

!!! note
    ReplayBG does **not** pick a blueprint for you from a string flag. You choose
    it by instantiating the matching data + model classes. See
    [Choosing a Blueprint](blueprints/choosing.md).

## Step 1: create the digital twin (`twin`)

`unknown_parameters_prior` declares **which** physiological parameters to estimate
and the **prior belief** about each. `twin()` finds the parameter set (`theta`)
that best explains the data — the MAP estimate — via a multi-start optimisation.

```python
unknown_parameters_prior = {
    'Gb':     {'prior': Normal(mu=119.13, sigma=7.11),      'min': 70, 'max': 150},
    'SG':     {'prior': LogNormal(mu=-3.8, sigma=0.05),     'min': 0,  'max': .5},
    'f':      {'prior': Normal(mu=0.8, sigma=0.05),         'min': 0,  'max': 1},
    'p2':     {'prior': Normal(mu=0.11, sigma=0.05),        'min': 0,  'max': .5},
    'ka2':    {'prior': LogNormal(mu=-4.2875, sigma=0.4274),'min': 0,  'max': .5},
    'kd':     {'prior': LogNormal(mu=-3.5090, sigma=0.6187),'min': 0,  'max': .5},
    'kempt':  {'prior': LogNormal(mu=-1.9646, sigma=0.7069),'min': 0,  'max': .75},
    'SI_B':   {'prior': Gamma(alpha=3.3, beta=1 / 5e-4),    'min': 0,  'max': .1},
    'SI_L':   {'prior': Gamma(alpha=3.3, beta=1 / 5e-4),    'min': 0,  'max': .1},
    'SI_D':   {'prior': Gamma(alpha=3.3, beta=1 / 5e-4),    'min': 0,  'max': .1},
    'kabs_B': {'prior': LogNormal(mu=-5.4591, sigma=1.4396),'min': 0,  'max': .5},
    'kabs_L': {'prior': LogNormal(mu=-5.4591, sigma=1.4396),'min': 0,  'max': .5},
    'kabs_D': {'prior': LogNormal(mu=-5.4591, sigma=1.4396),'min': 0,  'max': .5},
    'kabs_S': {'prior': LogNormal(mu=-5.4591, sigma=1.4396),'min': 0,  'max': .5},
    'beta_B': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    'beta_L': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    'beta_D': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    'beta_S': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
}

# Optional: encode prior correlations between parameters (Gaussian copula).
correlations = {('SI_B', 'p2'): -.5, ('SI_L', 'p2'): -.5, ('SI_D', 'p2'): -.5}

result = rbg.twin(rbg_data=rbg_data, model=model,
                  unknown_parameters_prior=unknown_parameters_prior,
                  correlations=correlations,
                  parallelize=True, n_jobs=-1, n_starts=4)   # bump n_starts for real fits

theta = result['theta']   # the estimated parameters — this IS the digital twin
```

!!! tip
    `twin(parallelize=True)` uses multiprocessing, so the whole script must sit
    behind the standard `if __name__ == '__main__':` / `freeze_support()` guard.
    `n_starts` is kept small here so the example finishes quickly — bump it up
    (e.g. 64) for real fits. See the [Twinning Procedure](twinning.md).

## Step 2: run replay simulations (`replay`)

Rebuild the model with the estimated parameters (`theta0` must be a Numba typed
dict, so pass it through `to_typed_f64_dict`) and forward-simulate. First the
**baseline** — the data exactly as recorded:

```python
model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
baseline = rbg.replay(rbg_data=rbg_data, model=model)
```

Then a **counterfactual "what-if"** — the reason the digital twin exists. Change
the inputs and ask *what would have happened?* Here we cut every bolus by 30% and
replay the **same** twin (same `theta`); only the insulin dosing changes, the
physiology does not:

```python
df_whatif = df.copy()
df_whatif['bolus'] = df_whatif['bolus'] * 0.7
rbg_data_whatif = MultiMealT1DData(data=df_whatif, environment=rbg.environment)

model = MultiMealT1DModel(u2ss=rbg_data_whatif.u2ss, tsteps=rbg_data_whatif.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data_whatif.t_start)
whatif = rbg.replay(rbg_data=rbg_data_whatif, model=model)
```

Each `replay()` returns a dict with the predicted `output` (interstitial
glucose), the applied `input` matrix, and an `actions` log. See the
[Replay guide](replay.md) for the full contract and closed-loop control.

## Step 3: analyze and visualize

Compare the two scenarios with the [AGATA](analyzing_results.md) glycemic metrics
(with 30% less insulin, expect higher glucose / more time above range):

```python
from utils.agata_analysis import analyze_replay
analyze_replay(baseline, ts=5, verbose=True)
analyze_replay(whatif, ts=5, verbose=True)
```

And plot the fit and the replays (see [Plotting Utilities](plotting.md)):

```python
from utils.plot_twinning import plot_twinning
from utils.plot_replay import plot_replay

# Hide the internal 't_hour' channel from the input panels.
mask_inputs = [i for i, name in rbg_data.data_to_input.items() if name == 't_hour']

plot_twinning(rbg_data, model, theta=theta, thresholds=[70, 180], mask_inputs=mask_inputs)
plot_replay(baseline, thresholds=[70, 180], mask_inputs=mask_inputs)
plot_replay(whatif, thresholds=[70, 180], mask_inputs=mask_inputs)
```

## Where to next?

- [The ReplayBG Object](replaybg_object.md) — the full `twin()` / `replay()` API.
- [Choosing a Blueprint](blueprints/choosing.md) — pick the right model + data pair.
- [Twinning Procedure](twinning.md) — single meal, multi meal, extended, and
  multi-day intervals.
- [Replay](replay.md) — closed-loop control with callbacks and a CGM sensor.
