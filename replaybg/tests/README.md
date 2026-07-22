# Tests

Unit test suite for ReplayBG, run with [pytest](https://docs.pytest.org/).

## Running

```bash
# All tests
uv run pytest

# A single file / test
uv run pytest tests/test_distributions.py
uv run pytest tests/test_numba_dicts.py::test_basic_conversion

# Verbose
uv run pytest -v

# Skip the slow end-to-end twinning tests
uv run pytest -m "not slow"
```

Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.
Because the package modules live at the repo root (e.g. `from distributions
import Normal`), `pythonpath = ["."]` is set there so imports resolve without an
install step.

## Layout

- `conftest.py` — shared fixtures: `env` (a default `Environment`) and the
  in-memory input dataframes `single_meal_df`, `multi_meal_df`,
  `multi_meal_extended_df` reused across the data tests.
- `dataclass_contract.py` — not a test module; shared assertion helpers
  (`assert_shared_contract`, `assert_meal_routing`) covering the identical
  time / insulin / meal-core / `u`-matrix pipeline of the multi-meal classes.
- `test_distributions.py` — prior distributions (`Normal`, `LogNormal`, `Gamma`,
  `Uniform`), full method coverage: constructor, `evaluate` (pdf), `cdf`
  (vs `scipy` + support/edge cases), and `sample` (bounds, seed-reproducibility,
  clipping).
- `test_numba_dicts.py` — `to_typed_f64_dict` conversion helper.
- `test_environment.py` — `Environment` defaults and the `identity` decorator.
- `test_single_meal_data.py` — `SingleMealT1DData`: full conversion contract
  (time/hour/minute expansion, glucose/`y_idxs`, insulin & meal scaling, labels,
  raw `*_data` arrays, `u` matrix, custom `data_to_input`).
- `test_multi_meal_data.py` — `MultiMealT1DData`: shared contract + B/L/D/S/H
  meal-label routing + `data_to_input` map.
- `test_multi_meal_extended_data.py` — `MultiMealExtendedT1DData`: shared
  contract + routing including the second-occurrence channels (B2/L2/S2).
- `test_twinner.py` — `Twinner` (MAP estimator): deterministic unit tests for
  `_build_correlation_structure` and the objective methods (`_log_prior`,
  `_log_likelihood`, `_log_posterior`, incl. the Gaussian-copula correction and
  integer rounding), plus end-to-end `twin()` runs on the real single-meal
  model — over one real day (`example/data_day_1.parquet` +
  `example/patient_info.parquet`) — checking the return contract,
  rounding/clipping, determinism and recovery of a known `SI` from noise-free
  synthetic data.
- `test_single_meal_model.py` — end-to-end twinning **replicability** for the
  single-meal model: fits the full 10-parameter prior over
  `example/data_two_day_extended.parquet` sliced to 04:00 of the second day
  (~22 h, 265 samples) and asserts that two `twin()` runs return byte-identical
  parameters, checked per `n_starts` setting (1 and 4). Also pins the twinned
  parameters as golden values (`GOLDEN_X`) so the fit can't drift silently
  across suite runs; regenerate them if the model/optimiser/deps change.
- `test_multi_meal_model.py` — the multi-meal counterpart of the above: same
  data slice and replicability + golden-value checks, but fitting the full
  18-parameter multi-meal prior (per-meal `SI_*` / `kabs_*` / `beta_*`
  channels) with the `MultiMealT1DModel` (needs `t_start`). Also `slow`.
- `test_multi_meal_extended_model.py` — the extended variant: replicability +
  golden checks with the `MultiMealExtendedT1DModel`, fitting 25 parameters
  (multi-meal prior + the second-occurrence channels `kabs_B2`/`kabs_L2`/
  `kabs_S2`/`beta_B2`/`beta_L2`/`beta_S2`/`SI_B2`). Uses the *entire*
  `data_two_day_extended` trace (~29 h, 349 samples) so the second-day B2/S2
  meals exercise those channels (there is no L2 meal, so `kabs_L2`/`beta_L2`
  stay unconstrained). Heaviest test; `slow`.
- `test_plot_utils.py` — the plotting utilities (`plot_replay`,
  `plot_twinning_history` and the shared `utils/plot_common.py`). Asserts what
  the figures *contain* rather than how they look: that every logged callback
  action reaches the canvas and is labelled with the field actually plotted, that
  channels are stemmed or stepped according to their sparsity, that
  `mask_inputs` / `input_groups` resolve as documented, that each history panel is
  filtered by its own non-finite values, and that the hover crosshair reads dense
  and sparse series correctly without disturbing the data limits. Renders on the
  `Agg` backend, so no display is needed.

The Numba `@jitclass` model classes are not yet covered — they are the natural
next tier to add.

## Adding tests

Add `test_*.py` files under `tests/`. Keep them fast and free of file/network
I/O. Note the model classes are Numba `@jitclass` compiled, so the first test
that touches them pays a one-time JIT compilation cost.
