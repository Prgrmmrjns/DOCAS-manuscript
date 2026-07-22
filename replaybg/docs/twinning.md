# Twinning Procedure

Twinning is the step that **creates a digital twin**: it personalizes a
[blueprint](blueprints/choosing.md) by estimating the set of unknown model
parameters $\boldsymbol{\theta}_{phy}$ that best explains a person's recorded data.

## How it works: MAP estimation

Given the physiological model

$$
\begin{cases}
    \dot{\boldsymbol{x}}_{phy}(t) = \boldsymbol{f}_{phy}(\boldsymbol{x}_{phy}, \boldsymbol{u}_{phy}, t, \boldsymbol{\theta}_{phy}) \\
    y(t) = CGM(t)
\end{cases}
$$

twinning estimates $\boldsymbol{\theta}_{phy}$ by **Maximum A Posteriori (MAP)**
estimation. Using the *a priori* information $p_{\boldsymbol{\theta}}(\boldsymbol{\theta})$
on the parameters, MAP finds the $\boldsymbol{\theta}_{phy}$ that maximizes the
posterior:

$$
p_{\boldsymbol{\theta}|Y,U}(\boldsymbol{\theta}|Y,U) \propto
p_{Y|\boldsymbol{\theta},U}(Y|\boldsymbol{\theta},U)\, p_{\boldsymbol{\theta}}(\boldsymbol{\theta})
$$

Concretely, ReplayBG **minimizes the negative log-posterior**
($-(\log\text{-prior} + \log\text{-likelihood})$) with a bounded **Powell**
optimisation (`scipy.optimize.minimize`), run from `n_starts` different initial
guesses (multi-start) sampled from the priors, optionally in parallel across CPUs.
The best start is kept. The likelihood uses a Gaussian error model with a constant
5% coefficient of variation.

!!! note "Only MAP — no MCMC"
    The current ReplayBG uses **MAP only**. It produces a single point estimate
    `theta` (the digital twin). The old py_replay_bg's MCMC method and the
    multiple-realizations machinery are not part of this codebase.

## How to twin

Twinning is performed by [`ReplayBG.twin()`](replaybg_object.md#twin):

```python
result = rbg.twin(
    rbg_data=rbg_data,
    model=model,
    unknown_parameters_prior=unknown_parameters_prior,
    correlations=None,
    n_starts=64,
    parallelize=True, n_jobs=-1,
    log_history=False,
    path=None, save_name=None,
)
theta = result['theta']   # {name: value} — the digital twin
```

!!! warning "Guard multiprocessing"
    `parallelize=True` uses multiprocessing, so the whole script must sit behind
    `if __name__ == '__main__':` with `from multiprocessing import freeze_support`
    /` freeze_support()` at the top of `main`.

### The `unknown_parameters_prior` dict

This dict declares **which** parameters to estimate and the **prior** belief about
each. Each entry maps a parameter name to:

```python
{
    'param_name': {
        'prior': <distribution>,   # Normal / LogNormal / Gamma / Uniform
        'min': float,              # lower bound
        'max': float,              # upper bound
        'integer': bool,           # optional, default False
    },
    ...
}
```

The `prior` is one of the [distributions](api/distributions.md) — each provides
`evaluate()`, `cdf()` and `sample()`. Only the parameters you list are estimated;
every other model parameter keeps its physiological default. So if your data has
no lunch, simply omit `kabs_L` / `beta_L`.

### Correlated priors (Gaussian copula)

You can encode prior **correlations** between parameters via the `correlations`
argument — pairwise `{(name_a, name_b): rho}` with `rho` in `[-1, 1]`:

```python
correlations = {('SI_B', 'p2'): -.5, ('SI_L', 'p2'): -.5, ('SI_D', 'p2'): -.5}
result = rbg.twin(..., correlations=correlations)
```

The joint prior becomes a **Gaussian copula** over the named parameters: their
marginals stay exactly as declared, and only the dependence structure is added.
Correlated parameters must use a distribution with a `cdf` method (all built-in
ones qualify). Passing `None` treats parameters as independent. See
`docs/gaussian_copula_priors.pdf` in the repo for the math.

## Twinning single portions of data

To twin a single meal or a single day, build the blueprint and call `twin()`. This
single-meal example (see
[`example/twin_single_meal.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/twin_single_meal.py))
also records the optimisation history and saves the results:

```python
from multiprocessing import freeze_support
import os, pandas as pd
from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from replaybg import ReplayBG

if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1.parquet"); df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''), 'results')

    rbg = ReplayBG()
    rbg_data = SingleMealT1DData(data=df, body_weight=100, environment=rbg.environment)
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

    unknown_parameters_prior = {
        'Gb':    {'prior': Normal(mu=119.13, sigma=7.11),       'min': 70, 'max': 150},
        'SG':    {'prior': LogNormal(mu=-3.8, sigma=0.5),       'min': 0,  'max': .5},
        'p2':    {'prior': Normal(mu=0.11, sigma=0.004),        'min': 0,  'max': .5},
        'f':     {'prior': Normal(mu=0.8, sigma=0.05),          'min': 0,  'max': 1},
        'ka2':   {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0,  'max': .5},
        'kd':    {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0,  'max': .5},
        'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0,  'max': .75},
        'SI':    {'prior': Gamma(alpha=3.3, beta=1 / 5e-4),     'min': 0,  'max': .1},
        'kabs':  {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0,  'max': .5},
        'beta':  {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }

    result = rbg.twin(rbg_data=rbg_data, model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      parallelize=True, n_jobs=-1, n_starts=16, log_history=True,
                      path=save_folder, save_name='single_meal_day_1')
```

The **multi-meal** and **multi-meal extended** variants are identical apart from
the blueprint classes and the parameters you estimate:

=== "Multi meal"

    ```python
    from data.multi_meal_t1d_data import MultiMealT1DData
    from model.multi_meal_t1d import MultiMealT1DModel

    rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)
    t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
    # estimate Gb, SG, f, p2, ka2, kd, kempt, SI_B/L/D, kabs_B/L/D/S, beta_B/L/D/S
    ```

=== "Multi meal extended"

    ```python
    from data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData
    from model.multi_meal_extended_t1d import MultiMealExtendedT1DModel

    rbg_data = MultiMealExtendedT1DData(data=df, environment=rbg.environment)
    t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
    model = MultiMealExtendedT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
    # additionally estimate kabs_B2/L2/S2, beta_B2/L2/S2, SI_B2
    ```

See [`example/twin_multi_meal.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/twin_multi_meal.py)
and [`example/twin_multi_meal_extended.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/twin_multi_meal_extended.py).

## Twinning multi-day intervals

To twin a recording spanning **more than one day**, twin it **one day at a time**
and chain the days: the final physiological state of each day becomes the initial
condition of the next. This removes the steady-state assumption for every day
after the first.

The key mechanism: after fitting a day, simulate the model forward to its final
state, then read off the carry-over conditions with `model.get_final_x0()` (the
end state) and `model.get_theta()` (the parameters used to rescale that state).
Pass them into the **next** day's model as `x0` and `theta_prev`.

```python
from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from utils.plot_twinning import _simulate

def split_days(df):
    """Split a recording into daily segments at every 04:00 (the SI_D→SI_B boundary)."""
    marks = [i for i in range(len(df))
             if df['t'].dt.hour.iloc[i] == 4 and df['t'].dt.minute.iloc[i] == 0]
    bounds = [0] + marks + [len(df)]
    return [df.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:])]

day_dfs = split_days(df)

prev_x0 = None
prev_theta = None
for n, day_df in enumerate(day_dfs, start=1):
    rbg_data = MultiMealT1DData(data=day_df, environment=rbg.environment)
    t_start = int((day_df["t"].iloc[0] - day_df["t"].iloc[0].normalize()).total_seconds() / 60)

    # Day 1 is a cold start; later days inherit the carry-over state + previous theta.
    if prev_x0 is None:
        model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)
    else:
        model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                                  t_start=t_start, x0=prev_x0, theta_prev=prev_theta)

    result = rbg.twin(rbg_data=rbg_data, model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      parallelize=True, n_jobs=-1, n_starts=8,
                      path='results', save_name=f'multi_meal_intervals_day{n}')

    # Advance the fitted model to its final state, then read the carry-over.
    _simulate(model, rbg_data, result['theta'])
    prev_x0 = model.get_final_x0()
    prev_theta = model.get_theta()
```

The full runnable version — including a carry-over continuity check and a stitched
multi-day plot — is
[`example/twin_multi_meal_intervals.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/twin_multi_meal_intervals.py).

!!! warning "Use the same `u2ss` across the interval"
    Cutting at 04:00 keeps each day's B/L/D/S meal structure and the overnight
    `SI_D` window inside a single segment. Keep the same steady-state basal `u2ss`
    across the interval so the equilibrium is consistent from day to day.

## What is returned and saved

`twin()` returns `{'theta', 'correlations', 'history'}`. When `path` is given, it
pickles `{theta, correlations, history, rbg_data}` to
`<path>/twin_<save_name>.pkl`. See [Saving Results](saving_results.md), and
[Analyzing Results](analyzing_results.md) for folding the AGATA fit metrics back
into the saved file.
