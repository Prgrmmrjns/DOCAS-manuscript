# Replay

Once a digital twin exists, you **replay** it: forward-simulate the fitted model
to answer *what-if* questions. The possibilities are broad — reduce a bolus by
30%, test a bolus calculator, evaluate a hypo-treatment policy, run an artificial
pancreas algorithm, or just generate data from arbitrary inputs.

## The `replay` method

```python
results = rbg.replay(
    rbg_data,
    model=None,
    callbacks=None,
    sensor=None,
    path=None, save_name=None,
)
```

Replay is a **closed-loop forward simulation**. It integrates the model step by
step; before each step it invokes any `callbacks` (control policies) through a
`ReplayContext`, and — when a `sensor` is provided — feeds them a noisy,
sub-sampled measurement instead of the true glucose. See
[The ReplayBG Object](replaybg_object.md#replay) for the full parameter list.

### What it returns

```python
{
    'output':         np.ndarray,   # predicted interstitial glucose, at integration resolution
    'input':          np.ndarray,   # applied inputs, shape (tsteps, n_channels)
    'data_to_input':  dict,         # channel index -> name
    'actions':        list,         # flat list of action records logged by callbacks
    # present only when a sensor is supplied:
    'measurement':      np.ndarray, # sensor samples, at sensor cadence
    'measurement_time': np.ndarray, # integration-step index of each sample
}
```

## Counterfactual replay (no control)

The simplest replay changes the recorded inputs offline and simulates the **same**
twin. Rebuild the model with the estimated parameters (`theta0` must be a Numba
typed dict — pass it through `to_typed_f64_dict`):

```python
from utils.numba_dicts import to_typed_f64_dict

# Baseline: the data as recorded.
model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
baseline = rbg.replay(rbg_data=rbg_data, model=model)

# What-if: cut every bolus by 30% and replay the same twin.
df_whatif = df.copy()
df_whatif['bolus'] = df_whatif['bolus'] * 0.7
rbg_data_whatif = MultiMealT1DData(data=df_whatif, environment=rbg.environment)
model = MultiMealT1DModel(u2ss=rbg_data_whatif.u2ss, tsteps=rbg_data_whatif.tsteps,
                          theta0=to_typed_f64_dict(theta), t_start=rbg_data_whatif.t_start)
whatif = rbg.replay(rbg_data=rbg_data_whatif, model=model)
```

But editing inputs offline is not enough when the meal/insulin inputs depend on the
*current* glucose (a bolus calculator, a rescue-carb policy, a controller). For
that, you use **callbacks**.

## The `ReplayContext`

A single `ReplayContext` is created per `replay()` call and refreshed **in place**
each minute. It is the per-step view your callbacks read and write. It is
model-agnostic: inputs are addressed by **name** and expressed in the model's own
units — any human-unit conversion (insulin U, carbs g) is the callback's job.

Key attributes:

| Attribute | Meaning |
|-----------|---------|
| `k` | Current integration minute (the step being decided). |
| `t_hour` | Hour-of-day at step `k`. |
| `yts` | Data/sensor cadence in minutes (e.g. 5) — use it to self-gate. |
| `u` | Live, mutable input vector for this step (model units). |
| `output_history` / `input_history` | Trajectories so far, valid up to index `k-1`. |
| `measurement` | Latest sensor reading held between samples. **Equals the true output when no sensor.** Prefer this in control policies. |
| `measurement_history` | Sensor signal at integration resolution, valid up to `k-1`. |
| `data_to_input` | Channel index → name mapping. |
| `model` | The live model (read-only by convention) — e.g. `model.Ip` for insulin-on-board. |
| `shared` | Free-form dict for callbacks to exchange data by name. |

Key methods:

| Method | Role |
|--------|------|
| `get_input(name)` | Current value of an input channel (model units). |
| `set_input(name, value)` | Set an input channel. |
| `add_input(name, value)` | Add to an input channel. |
| `log(**fields)` | Append an action record to the log (`k` and `callback` auto-filled). |

## Callbacks

A **callback** is a control policy invoked once per integration minute, *before*
the model steps, so it can inspect the simulation and modify the inputs the model
is about to consume. Subclass `ReplayCallback`:

- store **hyperparameters** in `__init__`;
- keep **memory** in instance attributes (they persist across the whole run and
  remain inspectable afterward);
- share state with other callbacks through `ctx.shared`;
- override `action(ctx)` to read the state and modify the current-step inputs via
  `ctx.set_input` / `ctx.add_input`. The return value is ignored.

Callbacks run in the order you pass them to `replay()`; each mutates the same input
vector in place, so later callbacks observe earlier edits. The class attribute
`name` labels the callback in the action log, and `rbg_data` is injected before the
run so callbacks can read data-specific quantities such as `rbg_data.body_weight`.

### A minimal custom callback

```python
from callbacks import ReplayCallback

class CapBolus(ReplayCallback):
    """Never let the per-step bolus exceed `max_u` units."""
    def __init__(self, max_u=5.0):
        self.max_u = max_u

    def action(self, ctx):
        # inputs are in model units (mU/kg/min); convert the cap using body weight
        cap = self.max_u * (1000.0 / self.rbg_data.body_weight)
        if ctx.get_input("bolus") > cap:
            prev = ctx.get_input("bolus")
            ctx.set_input("bolus", cap)
            ctx.log(prev_u=prev, new_u=cap)
```

### Built-in callbacks

Two ready-made policies live in `callbacks`:

- **`CorrectionBolus(threshold=180, target=120, cf=40, lockout_min=60)`** — if
  `ctx.measurement > threshold` and the lockout has elapsed, deliver
  `(glucose - target) / cf` units of correction insulin. It publishes each bolus on
  `ctx.shared["last_bolus"] = {"u", "k"}`.
- **`HypoTreatment(threshold=70, carbs=15, lockout_min=15, stacking_window_min=0,
  stacking_carbs=0)`** — if `ctx.measurement < threshold` and the lockout has
  elapsed, add `carbs` grams of rescue carbohydrates (the "rule of 15"). If a bolus
  was published within `stacking_window_min` minutes, it adds `stacking_carbs`
  extra grams (insulin-stacking awareness) — without holding any reference to the
  bolus callback.

### Full closed-loop example

This replays a previously-saved twin twice — a plain baseline, and a controlled run
with a correction-bolus + hypo-treatment policy acting on a **noisy CGM sensor**
(see [`example/replay_with_callbacks.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/replay_with_callbacks.py)):

```python
from multiprocessing import freeze_support
import os, pandas as pd
from model.multi_meal_t1d import MultiMealT1DModel
from replaybg import ReplayBG
from callbacks import CorrectionBolus, HypoTreatment
from sensors import Vettoretti19CGM
from utils.load_results import load_results
from utils.numba_dicts import to_typed_f64_dict
from utils.agata_analysis import analyze_replay

if __name__ == '__main__':
    freeze_support()
    save_folder = os.path.join(os.path.abspath(''), 'results')

    rbg = ReplayBG()
    twin = load_results(save_folder, 'multi_meal_day_1', prefix='twin')
    theta, rbg_data = twin['theta'], twin['rbg_data']

    # Baseline (no policy)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
    baseline = rbg.replay(rbg_data=rbg_data, model=model)

    # Controlled: policies act on a realistic, noisy CGM measurement.
    correction = CorrectionBolus(threshold=180, target=120, cf=50, lockout_min=60)
    hypo = HypoTreatment(threshold=70, carbs=15, lockout_min=30,
                         stacking_window_min=90, stacking_carbs=10)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f64_dict(theta), t_start=rbg_data.t_start)
    controlled = rbg.replay(rbg_data=rbg_data, model=model,
                            callbacks=[correction, hypo],
                            sensor=Vettoretti19CGM())

    # Inspect exactly what the callbacks did:
    actions = pd.DataFrame(controlled['actions'])
    print(actions)

    # Compare with AGATA (the controlled run also analyzes the sensor trace).
    analyze_replay(baseline, ts=5, verbose=True)
    analyze_replay(controlled, ts=5, verbose=True)
```

The `actions` list is exactly what each callback logged via `ctx.log(...)`, one
record per action, with `k` and `callback` auto-filled — turn it into a tidy table
with `pd.DataFrame(results['actions'])`.

## Replaying multi-day intervals

Just like [interval twinning](twinning.md#twinning-multi-day-intervals), replay a
multi-day recording one day at a time, carrying each day's final state into the
next via `x0` / `theta_prev` when constructing the model. See
[`example/replay_multi_meal_intervals.py`](https://github.com/DIANA-UNIPD/replaybg/blob/main/example/replay_multi_meal_intervals.py).

## Saving and visualizing

Pass `path` / `save_name` to pickle the results (see
[Saving Results](saving_results.md)), analyze them with
[AGATA](analyzing_results.md), and visualize them with
[`plot_replay`](plotting.md).
