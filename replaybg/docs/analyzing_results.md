# Analyzing Results

ReplayBG analyzes twinning and replay outputs by integrating
[AGATA](https://github.com/gcappon/py_agata) (`py_agata`), a dedicated toolbox for
glucose data analysis. The two helpers live in `utils.agata_analysis` and run
AGATA on the resulting glucose traces, optionally printing a compact summary and
folding the metrics back into the saved pickle.

## `analyze_replay`

Runs AGATA's full glucose-profile analysis on a replay's `output` — and, when the
replay used a [sensor](cgm_model.md), also on the noisy `measurement` trace.

```python
from utils.agata_analysis import analyze_replay

analysis = analyze_replay(replay_results, ts=5, verbose=True)
```

| Parameter | Meaning |
|-----------|---------|
| `replay_results` | The dict returned by `replay()` (must contain `output`). |
| `ts` | AGATA glucose sampling cadence in minutes (default `5`). The `output` is downsampled to this cadence. |
| `integration_ts` | Minutes per integration step (default `1`, = `environment.ts`). |
| `glycemic_target` | Glycemic target set forwarded to AGATA (default `"diabetes"`). |
| `verbose` | Print a compact metric summary. |
| `path` / `save_name` | When given, add the analysis under an `analysis` key and re-save the pickle. |

**Returns** a dict with key `output` (the AGATA profile of the model output) and,
when a sensor was used, `measurement` (the profile of the sensor trace).

## `analyze_twin`

Simulates the fitted model forward to get the predicted trace, runs the profile
analysis on it, **and** computes the comprehensive fit-error metric set against the
reference CGM (`rbg_data.y`).

```python
from utils.agata_analysis import analyze_twin

analysis = analyze_twin(twin_results, rbg_data, model, verbose=True)
```

**Returns** a dict with:

- `profile` — the AGATA glucose profile of the fitted trace,
- `error` — the fit-error metrics: `rmse`, `mard`, `cod`, Clarke EGA zones
  (`clarke`), and `g_rmse`.

## What AGATA computes

`analyze_glucose_profile` returns a rich set of standardized glucose metrics,
grouped as:

- **variability** — mean/median/std glucose, CV, range, IQR, AUC, GMI, CONGA,
  J-index, MAGE, MODD, and more;
- **time_in_ranges** — time in target, and in (level 1/2) hypo- and
  hyperglycemia;
- **risk** — ADRR, LBGI, HBGI, BGRI, GRI;
- **glycemic_transformation** — GRADE scores, IGC, hypo/hyper index, MR index;
- **events** — hypo/hyperglycemic events (start/end/duration/frequency);
- **data_quality** — days of observation and missing-glucose percentage.

The `verbose=True` printout gives a curated subset (mean/std/CV glucose, GMI,
time-in-target, time hypo/hyper, LBGI/HBGI) plus, for `analyze_twin`, the fit-error
block (RMSE, MARD, COD, gRMSE, Clarke A/B/C/D/E).

## Typical workflow

```python
# 1. Twin, then analyze the fit and fold the metrics into the saved pickle.
result = rbg.twin(rbg_data=rbg_data, model=model,
                  unknown_parameters_prior=unknown_parameters_prior,
                  path='results', save_name='my_twin')
analyze_twin(result, rbg_data, model, verbose=True,
             path='results', save_name='my_twin')

# 2. Replay two scenarios and compare their glycemic metrics.
baseline = rbg.replay(rbg_data=rbg_data, model=model)
whatif   = rbg.replay(rbg_data=rbg_data_whatif, model=model_whatif)
analyze_replay(baseline, ts=5, verbose=True)
analyze_replay(whatif,   ts=5, verbose=True)
```

For the metric definitions and the complete list of indices, see the
[AGATA documentation](https://github.com/gcappon/py_agata).
