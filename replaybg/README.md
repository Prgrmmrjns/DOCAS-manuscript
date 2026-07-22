# ReplayBG

**ReplayBG** is a digital-twin framework for Type 1 Diabetes (T1D) glucose dynamics. It fits a
physiological ODE model to a person's CGM + insulin + meal data (**twinning**), then uses the
fitted model to simulate counterfactual "what-if" scenarios (**replay**) — *what would this
person's glucose have looked like with a different insulin dose, a different meal, or a control
algorithm in the loop?*

```
 raw DataFrame            Data class              Model (Numba jitclass)
 t, glucose, cho,   ─▶   SingleMealT1DData  ─▶   SingleMealT1DModel   ─┐
 bolus, basal, ...       MultiMealT1DData        MultiMealT1DModel     │
                                                                       │
        ┌──────────────────────────────────────────────────────────  ┘
        │
        ├─▶  rbg.twin()    MAP estimation (multi-start Powell)  ─▶  theta (the digital twin)
        │
        └─▶  rbg.replay()  forward simulation with theta        ─▶  predicted glucose trace
```

## Installation

ReplayBG is not published on PyPI; run it from a clone. It requires **Python ≥ 3.12** and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <this-repo-url>
cd replaybg
uv sync
```

Modules live at the repository root and are imported directly (`from replaybg import ReplayBG`,
`from model.multi_meal_t1d import MultiMealT1DModel`, …), so run scripts and tests from the repo
root.

## Get started

The snippets below mirror [`example/get_started.py`](example/get_started.py) — a single,
runnable, end-to-end tour (twin → baseline replay → counterfactual replay). Run it with:

```bash
uv run python example/get_started.py
```

### Preparation

ReplayBG consumes a pandas `DataFrame` with **one row every 5 minutes** and the following
columns:

| Column        | Type / units          | Notes                                                    |
|---------------|-----------------------|----------------------------------------------------------|
| `t`           | datetime              | 5-minute grid; **must** be converted with `pd.to_datetime` |
| `glucose`     | float, mg/dL          | CGM; `NaN` allowed                                        |
| `cho`         | float, g              | ingested carbs; `0` when no meal                         |
| `bolus`       | float, U              | bolus insulin; `0` when no bolus                         |
| `basal`       | float, U/min          | basal insulin rate (per row)                             |
| `bolus_label` | str                   | label of each bolus event                                |
| `cho_label`   | str                   | meal type: `B`/`L`/`D`/`S`/`H` (breakfast/lunch/dinner/snack/hypo-treatment) |

```python
import pandas as pd
from replaybg import ReplayBG
from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.numba_dicts import to_typed_f64_dict

df = pd.read_parquet("example/data_day_1_2.parquet")
df['t'] = pd.to_datetime(df['t'])          # the 't' column MUST be datetime

rbg = ReplayBG()
rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)

t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
```

### Step 1 — Create the digital twin (`twin`)

`unknown_parameters_prior` declares which physiological parameters to estimate and the prior
belief about each. `twin()` finds the parameter set (`theta`) that best explains the data (MAP
estimate) via a multi-start optimisation.

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

### Step 2 — Run replay simulations (`replay`)

Rebuild the model with the estimated parameters (`theta0` must be a Numba typed dict, so pass it
through `to_typed_f64_dict`) and forward-simulate. First the **baseline** (the data as
recorded):

```python
model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
baseline = rbg.replay(rbg_data=rbg_data, model=model)
```

Then a **counterfactual "what-if"** — the reason the digital twin exists. Change the inputs and
ask *what would have happened?* Here we cut every bolus by 30% and replay the **same** twin
(same `theta`); only the insulin dosing changes, the physiology does not:

```python
df_whatif = df.copy()
df_whatif['bolus'] = df_whatif['bolus'] * 0.7
rbg_data_whatif = MultiMealT1DData(data=df_whatif, environment=rbg.environment)

model = MultiMealT1DModel(u2ss=rbg_data_whatif.u2ss, tsteps=rbg_data_whatif.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data_whatif.t_start)
whatif = rbg.replay(rbg_data=rbg_data_whatif, model=model)
```

Each `replay()` returns a dict with the predicted `output` (interstitial glucose), the applied
`input` matrix, and an `actions` log. Compare the two scenarios with the AGATA glycemic metrics
(with 30% less insulin, expect higher glucose / more time above range):

```python
from utils.agata_analysis import analyze_replay
analyze_replay(baseline, ts=5)
analyze_replay(whatif, ts=5)
```

See [`example/get_started.py`](example/get_started.py) for the full runnable code, including the
diagnostic plots (`plot_twinning`, `plot_replay`).

## More examples

All example scripts and their sample `.parquet` data live in [`example/`](example/):

| Script | What it shows |
|--------|---------------|
| [`get_started.py`](example/get_started.py) | **Start here** — one-file twin → replay → counterfactual (multi-meal) |
| [`twin_single_meal.py`](example/twin_single_meal.py) / [`replay_single_meal.py`](example/replay_single_meal.py) | Simplest variant: single-meal twinning, then replay from the saved fit |
| [`twin_multi_meal.py`](example/twin_multi_meal.py) / [`replay_multi_meal.py`](example/replay_multi_meal.py) | Multi-meal model with time-of-day insulin sensitivity |
| [`twin_multi_meal_intervals.py`](example/twin_multi_meal_intervals.py) / [`replay_multi_meal_intervals.py`](example/replay_multi_meal_intervals.py) | Multi-day recordings twinned day-by-day with carried-over state (x0 chaining) |
| [`twin_multi_meal_extended.py`](example/twin_multi_meal_extended.py) / [`replay_multi_meal_extended.py`](example/replay_multi_meal_extended.py) | Two-day extended model with second-day meal labels |
| [`replay_with_callbacks.py`](example/replay_with_callbacks.py) | Closed-loop replay: correction-bolus & hypo-treatment policies on a noisy CGM sensor |

The task-specific `twin_*.py` / `replay_*.py` pairs communicate through a pickle in
`example/results/`: the twinning script saves `theta` + prepared data, and the replay script
reloads it. `get_started.py` keeps everything in memory instead.

## How it works

- **Twinning** (`rbg.twin`) performs MAP estimation — it maximises the log-posterior
  (`log_prior + log_likelihood`) with a multi-start Powell optimiser (`scipy.optimize.minimize`),
  optionally parallelised across CPUs. It returns `{'theta', 'correlations', 'history'}`.
- **Replay** (`rbg.replay`) is a closed-loop forward simulation. It integrates the model step by
  step; before each step it invokes optional `callbacks` (control policies) through a
  `ReplayContext`, and can feed them a noisy, sub-sampled reading from an optional `sensor`
  instead of the true glucose. It returns `{'output', 'input', 'data_to_input', 'actions'}`
  (plus `measurement`/`measurement_time` when a sensor is used).

## Documentation

The full documentation — getting-started guide, blueprints, twinning, replay,
callbacks, sensors, plotting, analysis, and an auto-generated API reference — is
built with MkDocs (Material theme):

```bash
uv run mkdocs serve      # live-reload at http://127.0.0.1:8000
uv run mkdocs build      # static site into site/
```

Sources live in [`docs/`](docs/); the site is deployed to GitHub Pages by
[`.github/workflows/docs.yml`](.github/workflows/docs.yml) on every push to `main`.

## Testing

```bash
uv run pytest                 # full suite
uv run pytest -m "not slow"   # skip the heavy end-to-end twinning tests
```

The `slow`-marked tests pin golden parameter values and run a real twinning end to end, so they
fail if the model, optimiser, or dependencies drift. See [`tests/README.md`](tests/README.md)
for the per-module breakdown.

## Project layout

| Path | Role |
|------|------|
| `replaybg.py` | `ReplayBG` — top-level API: `.twin()` and `.replay()` |
| `twinner/` | `Twinner` — MAP optimiser (multi-start + optional parallelism) |
| `model/` | Numba `@jitclass` ODE models (single-meal, multi-meal, extended) |
| `data/` | DataFrame → model-ready `rbg_data` preparation classes |
| `distributions/` | Prior distributions (Normal, Gamma, LogNormal, Uniform) |
| `callbacks/`, `sensors/` | Closed-loop control policies and CGM sensor models for replay |
| `environment/` | `Environment` config + the `jitclass_` JIT on/off switch |
| `utils/` | Typed-dict helpers, AGATA analysis, plotting, save/load |

## Reference

If you use ReplayBG in your research, please cite:

> G. Cappon, M. Vettoretti, G. Sparacino, S. Del Favero, and A. Facchinetti, "ReplayBG: A
> Digital Twin-Based Methodology to Identify a Personalized Model From Type 1 Diabetes Data and
> Simulate Glucose Concentrations to Assess Alternative Therapies," *IEEE Transactions on
> Biomedical Engineering*, vol. 70, no. 11, pp. 3227–3238, Nov. 2023,
> doi: [10.1109/TBME.2023.3286856](https://doi.org/10.1109/TBME.2023.3286856).

```bibtex
@article{cappon2023replaybg,
  author  = {Cappon, Giacomo and Vettoretti, Martina and Sparacino, Giovanni and Del Favero, Simone and Facchinetti, Andrea},
  title   = {{ReplayBG}: A Digital Twin-Based Methodology to Identify a Personalized Model From Type 1 Diabetes Data and Simulate Glucose Concentrations to Assess Alternative Therapies},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {70},
  number  = {11},
  pages   = {3227--3238},
  year    = {2023},
  doi     = {10.1109/TBME.2023.3286856},
}
```
