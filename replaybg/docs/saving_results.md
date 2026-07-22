# Saving Results

Both `twin()` and `replay()` can persist their results to disk as a pickle. You
choose **where** and **under what name** with the `path` and `save_name` arguments;
if you omit `path`, nothing is written and the results are only returned in memory.

## Saving a twin

```python
result = rbg.twin(rbg_data=rbg_data, model=model,
                  unknown_parameters_prior=unknown_parameters_prior,
                  path='results', save_name='my_twin')
```

This writes `results/twin_my_twin.pkl`, containing a dict with:

- `theta` — the estimated parameters (the digital twin),
- `correlations` — the prior correlations used (or `None`),
- `history` — the optimisation history (when `log_history=True`, else `None`),
- `rbg_data` — the prepared data object (so replay can reload it).

## Saving a replay

```python
results = rbg.replay(rbg_data=rbg_data, model=model,
                     path='results', save_name='baseline')
```

This writes `results/replay_baseline.pkl`, containing the returned replay dict
(`output`, `input`, `data_to_input`, `actions`, plus `measurement` /
`measurement_time` when a sensor was used).

## Naming and location

- `path` is a directory; it is created if missing (`os.makedirs(..., exist_ok=True)`).
- The file name is `f"{prefix}_{save_name}.pkl"`, where `prefix` is `twin` or
  `replay`. If `save_name` is `None`, it defaults to today's date, e.g.
  `twin_2026_07_18.pkl`.

So a `path='results'` folder ends up looking like:

```
results/
├── twin_my_twin.pkl
├── replay_baseline.pkl
└── replay_controlled.pkl
```

## Loading results back

Use the paired helpers in `utils` (the `prefix` must match how the file was
saved):

```python
from utils.load_results import load_results

twin = load_results('results', 'my_twin', prefix='twin')
theta, rbg_data = twin['theta'], twin['rbg_data']

replay = load_results('results', 'baseline', prefix='replay')
```

This is exactly how the split `twin_*.py` / `replay_*.py` example scripts
communicate: the twinning script saves `theta` + prepared data, and the replay
script reloads it.

## Analysis re-saves the file

The [AGATA analysis](analyzing_results.md) helpers `analyze_twin` /
`analyze_replay` accept the same `path` / `save_name`. When given, they add an
`analysis` key to the results and **re-save** the pickle in place, so the fit
metrics and glycemic profile live alongside the twin/replay they describe:

```python
from utils.agata_analysis import analyze_twin
analyze_twin(result, rbg_data, model, path='results', save_name='my_twin')
```
