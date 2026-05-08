# iSHAP

Rule-guided synthetic data augmentation for tabular models.
The pipeline optimizes synthetic points to improve a combined objective of:

- model performance (RMSE, lower is better), and
- SHAP-based feasibility score from directional SCM rules (higher is better).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Expected at repo root:

- `d1namo_combined.csv`
- `cancer patient data sets.csv`

## Run

```bash
python scripts/main.py
```

## Outputs

- `results/<dataset>/feasibility.json`
- `results/<dataset>/before_feature_contrib.csv`
- `results/<dataset>/after_feature_contrib.csv`
- `images/<dataset>/...png` (SCM, beeswarm, interaction network)

## Main Files

- `scripts/main.py` — run configuration, dataset list, plotting.
- `scripts/lib.py` — training loop, synthetic optimization, feasibility scoring.
- `scripts/model.py` — model definition and shared parameters.
- `scripts/d1namo.py` — d1namo dataset + SCM rules.
- `scripts/cancer_air_pollution.py` — cancer dataset + SCM rules.
- `scripts/visuals.py` — beeswarm, SCM graph, and interaction network plots.
- `manuscript/main.tex` — paper draft.

## What To Change

- Change which datasets run: edit `DATASETS` in `scripts/main.py`.
- Change optimization knobs (`n_trials`, `synth_points_per_round`, `objective_metric_weight`): edit `RUN_PIPELINE_KWARGS` in `scripts/main.py`.
- Change model hyperparameters (LightGBM settings): edit `MODEL_KW` in `scripts/model.py`.
- Change rule set for a dataset: edit `SCM_RULES` in the dataset file (for example `scripts/d1namo.py` or `scripts/cancer_air_pollution.py`).
- Change task type (regression vs classification): edit `TASK` in the dataset file.
- Change interaction graph edge threshold: edit `MIN_ABS_PEARSON_FOR_INTERACTION_GRAPH` in `scripts/main.py`.
- Change synthetic-point generator backend: edit `scripts/synthesis.py`.
- Change combined scoring logic (metric-feasibility trade-off): edit `scripts/scoring.py`.
- Change manuscript text and figures: edit `manuscript/main.tex` (figures are read from `manuscript/images/...`).

## TODO

1. try out with other datasets / remove cancer dataset as it has perfect accuracy
2. define better SCM  
3. make smarter search space  
4. write paper  
5. try out implementing shapiq  
6. do-shapley value calculation instead of optuna
