# The ReplayBG Object

The `ReplayBG` object is the core entry point of the framework. You create it
once, then use its two methods — `twin()` and `replay()` — for the whole workflow.

```python
from replaybg import ReplayBG

ReplayBG(ts: int = 1, seed: int = 1, plot_mode: bool = True, verbose: bool = True)
```

## Constructor parameters

- **`ts`** *(int, default `1`)* — the integration time step in minutes. Models
  integrate at this resolution.
- **`seed`** *(int, default `1`)* — the random seed, for reproducibility (used,
  e.g., when sampling the CGM sensor error during replay).
- **`plot_mode`** *(bool, default `True`)* — whether ReplayBG should show plots of
  the results.
- **`verbose`** *(bool, default `True`)* — verbosity of ReplayBG (twinning prints
  a run header, a convergence summary and the best-fit parameters).

These are stored on an [`Environment`](api/environment.md) object exposed as
`rbg.environment`, which is passed to the data classes:

```python
rbg = ReplayBG()
rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)
```

!!! note "This is different from the old py_replay_bg API"
    There is **no** `blueprint`, `save_folder`, `yts` or `exercise` argument on
    the constructor anymore. You choose a blueprint by instantiating the matching
    [data + model classes](blueprints/choosing.md), the data sampling time is
    fixed at 5 minutes by the data classes, and results are saved per-call via the
    `path` / `save_name` arguments of `twin()` / `replay()`.

## `twin()`

Runs the twinning procedure (MAP estimation) to identify the model parameters.

```python
rbg.twin(
    rbg_data,
    model: object = None,
    unknown_parameters_prior: dict = None,
    correlations: dict = None,
    n_starts: int = 64,
    parallelize: bool = False,
    n_jobs: int | None = None,
    log_history: bool = False,
    path: str | None = None,
    save_name: str | None = None,
) -> dict
```

| Parameter | Description |
|-----------|-------------|
| `rbg_data` | A data object (e.g. `MultiMealT1DData`) holding the inputs, timestamps and observed glucose. |
| `model` | A model instance implementing the [model interface contract](blueprints/custom.md) (`reset`, `step`, `output`). |
| `unknown_parameters_prior` | Dict defining which parameters to estimate and their priors/bounds. See [Twinning](twinning.md#the-unknown_parameters_prior-dict). |
| `correlations` | Optional pairwise prior correlations `{(name_a, name_b): rho}` (Gaussian copula). `None` ⇒ independent. |
| `n_starts` | Number of multi-start optimisations (default `64`). |
| `parallelize` | Whether to run the multi-start optimisation in parallel (multiprocessing). |
| `n_jobs` | Number of parallel jobs; `None` ⇒ all CPUs. |
| `log_history` | Whether to record the optimisation history (for [`plot_twinning_history`](plotting.md)). |
| `path` | Directory to pickle the results into; `None` ⇒ not saved. See [Saving Results](saving_results.md). |
| `save_name` | File name for the saved pickle; defaults to `twin_YYYY_mm_dd.pkl`. |

**Returns** a dict with keys:

- `theta` — the estimated parameters as a `{name: value}` mapping (this is the
  digital twin),
- `correlations` — the prior correlations used (or `None`),
- `history` — the optimisation history when `log_history=True`, else `None`.

## `replay()`

Forward-simulates the fitted model, optionally with closed-loop control policies
and a CGM sensor.

```python
rbg.replay(
    rbg_data,
    model: object = None,
    callbacks: list | None = None,
    sensor: object = None,
    path: str | None = None,
    save_name: str | None = None,
) -> dict
```

| Parameter | Description |
|-----------|-------------|
| `rbg_data` | The data object holding the inputs to replay. |
| `model` | The model instance, rebuilt with the estimated `theta0`. |
| `callbacks` | List of [`ReplayCallback`](replay.md#callbacks) control policies invoked before each step. |
| `sensor` | Optional [`Sensor`](cgm_model.md) producing a noisy, sub-sampled measurement exposed to callbacks as `ctx.measurement`. |
| `path` | Directory to pickle the results into; `None` ⇒ not saved. |
| `save_name` | File name for the saved pickle; defaults to `replay_YYYY_mm_dd.pkl`. |

**Returns** a dict with keys:

- `output` — predicted interstitial glucose at integration resolution
  (`np.ndarray`, length `tsteps`),
- `input` — the applied inputs, shape `(tsteps, n_channels)`,
- `data_to_input` — the channel-index → name mapping,
- `actions` — a flat list of action records logged by the callbacks,
- `measurement` and `measurement_time` — **only when a sensor is supplied** (the
  sensor samples and their integration-step indices).

See the [Replay guide](replay.md) for the full closed-loop story.
