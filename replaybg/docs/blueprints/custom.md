# Custom Blueprint

A blueprint is a **model class** paired with a **data class**. To build your own
digital-twin structure, you implement both, respecting the small interface
contracts that `ReplayBG.twin()` and `ReplayBG.replay()` rely on. The built-in
[single-meal](single_meal.md) pair is the best minimal reference implementation —
read `model/single_meal_t1d.py` and `data/single_meal_t1d_data.py` alongside this
page.

## The model class contract

Model classes are compiled with Numba's `@jitclass` via the project's `jitclass_`
switch, so all state arrays and scalar fields must be declared up front in a
`JITCLASS_SPEC` with explicit Numba types.

```python
from environment import jitclass_

JITCLASS_SPEC = [
    ("u2ss", float64),
    ("G", float64[:]),
    # ... every scalar/array field with its Numba type ...
]

@jitclass_(JITCLASS_SPEC)
class MyT1DModel:
    ...
```

!!! note "The `jitclass_` switch"
    `jitclass_` (from `environment/__init__.py`) applies real JIT compilation, or
    falls back to a no-op identity decorator when `DEBUG=True` — flip that flag to
    debug your model in plain Python. Parameters are always passed between Python
    and the jitclass through **Numba typed dicts** (`numba.typed.Dict`), which you
    build with [`to_typed_f64_dict`](../api/utils.md).

Your model must implement these methods:

| Method | Role |
|--------|------|
| `__init__(u2ss, theta0, x0, tsteps, [t_start,] theta_prev)` | Allocate state arrays and initialize. `u2ss` is the steady-state basal insulin; `theta0`/`x0`/`theta_prev` are typed dicts (empty ⇒ defaults / cold start). |
| `reset(theta0)` | Set parameters from a typed dict and re-initialize all state. Called by the twinner on **every** objective evaluation. |
| `step(u, t)` | Advance the ODE by one integration step given the input row `u` at time index `t`. |
| `output(t)` | Return the observable model output at time `t` (the CGM-comparable signal). |
| `get_final_x0()` | Return a typed dict of end-of-segment state, to seed the next segment's `x0`. |
| `get_theta()` | Return a typed dict of the current parameters, to seed the next segment's `theta_prev`. |

!!! warning "Method names"
    The carry-over methods are **`get_final_x0()`** and **`get_theta()`**. (Some
    older internal notes referred to `apply_x0` / `extract_final_x0`; those do not
    exist.)

The constructor signature of the built-in models is:

```python
# single-meal
SingleMealT1DModel(u2ss, theta0=<empty typed dict>, x0=<empty typed dict>,
                   tsteps=1440, theta_prev=<empty typed dict>)
# multi-meal / extended add t_start (minute-of-day the segment starts)
MultiMealT1DModel(u2ss, theta0=..., x0=..., tsteps=1440, t_start=240, theta_prev=...)
```

Missing keys in `theta0` fall back to physiological defaults set inside `reset`;
missing keys in `x0` fall back to the steady state. This is what lets you estimate
only a subset of parameters (the rest keep population values).

## The data class contract

The data class turns a raw `DataFrame` (one row every **5 minutes**) into the
arrays the model and twinner consume. Its constructor signature matches the
built-ins:

```python
MyT1DData(data: pd.DataFrame = None, data_to_input=None,
          body_weight=100., environment: Environment = None)
```

It must expose (at least) these attributes:

| Attribute | Meaning |
|-----------|---------|
| `u` | Input matrix, shape `(tsteps, n_channels)`, in **model units**. |
| `data_to_input` | Mapping `{channel_index: channel_name}` describing `u`'s columns. |
| `tsteps` | Number of integration steps (from the `t` column and the sampling time). |
| `yts` | Data sampling stride in integration steps (5, since data is 5-minutely). |
| `y` | Observed output (glucose) at data resolution; may contain `NaN`. |
| `y_idxs` | Indices of the non-missing observations (`np.where(~isnan(y))`). |
| `u2ss` | Steady-state basal insulin — `mean(basal) * 1000 / body_weight`. |
| `body_weight` | Patient body weight (used to normalize inputs; read by callbacks). |

The `data_to_input` order is the contract between the data class and the model:
the model's `step(u, t)` reads channels by that index. By convention the last two
channels are `forcing_ip` and `forcing_ra` (zero unless a caller overrides them
for a counterfactual forcing scenario).

## Wiring it up

Once both classes exist, they are used exactly like a built-in blueprint:

```python
rbg = ReplayBG()
rbg_data = MyT1DData(data=df, environment=rbg.environment)
model = MyT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

result = rbg.twin(rbg_data=rbg_data, model=model,
                  unknown_parameters_prior=unknown_parameters_prior)
```

The twinner is model-agnostic: it only calls `reset`, `step` and `output`, and
reads the priors you declare. As long as your two classes honor the contracts
above, everything downstream — twinning, replay, [callbacks](../replay.md),
[sensors](../cgm_model.md), [plotting](../plotting.md) and
[analysis](../analyzing_results.md) — works unchanged.

!!! tip "The output is model-agnostic too"
    Sensors and plotting only ever see `model.output(t)` and the channel names in
    `data_to_input`. Nothing downstream assumes what your output *means*, so a
    custom model with a differently-named output still plots and analyzes fine.
