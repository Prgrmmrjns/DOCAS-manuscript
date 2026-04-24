# iSHAP / SHAPPROP

NSGA-II over **custom LightGBM objectives** (covariance bias toward a sign prior) vs. held-out **performance** (RMSE / ROC-AUC) and **feasibility** from `pred_contrib` means.

## Setup

```bash
pip install -r requirements.txt
```

## Data (repo root)

| File | Module |
|------|--------|
| `d1namo_combined.csv` | `d1namo` |
| `heart_disease.csv` | `heart_disease` |
| `cancer_risk_factors.csv` | `cancer_risk` |

## Run

```bash
python scripts/main.py
```

Writes `results/shapprop_run.json`. Edit `Settings` in `scripts/main.py` for `datasets`, `n_trials`, etc.

## Layout

- `scripts/main.py` — Optuna NSGA-II, refit custom vs default objective, reports.
- `scripts/lib.py` — train/test split, feasibility score.
- `scripts/objkit.py` — shared MSE / logistic + sign-penalty objective factory.
- `scripts/backend_lightgbm.py` — LightGBM fit + `pred_contrib` summary.
- `scripts/<dataset>.py` — `load(project_root) -> (X, y)`; `TASK`, `NAME`.
- `scripts/objectives/<dataset>.py` — `SIGNS` / `WORLD_MODEL`, `suggest_params`, `make_objective`.
