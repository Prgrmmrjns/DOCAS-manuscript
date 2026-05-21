# iSHAP

Interventional audits and causal fixes for tabular models (Ohio T1DM case study).

Two main approaches:

- **Latent confounder imputation** — impute hidden columns from SHAP interactions (NSGA-II over imputer weights).
- **Synthetic augmentation** — add SCM-guided counterfactual training rows (NSGA-II).

Head-to-head compares baseline, monotonic LightGBM, preprocessing, feature engineering, synthetic augmentation, and latent confounders across all Ohio patients.

## Setup

```bash
pip install -r requirements.txt
```

## Data

- `datasets/ohio_t1dm.csv` — built automatically from `OhioT1DM/**/*.xml` when missing.
- Raw XML lives under `OhioT1DM/` (2018 and 2020 cohorts).

## Run

Full head-to-head (all patients, writes presentation table/figures):

```bash
python scripts/main.py
```

Static slides only (IRC/SHAP/preprocess figures, no Optuna):

```bash
python scripts/presentation_build.py
```

Rebuild Ohio CSV from XML:

```bash
python scripts/ohio_t1dm.py
```

## Outputs

- `results/ohio_t1dm/<patient_id>/` — feasibility, latent imputer params, head-to-head metrics.
- `presentation/images/` — figures for `presentation/causalml_project_presentation.tex`.
- `manuscript/images/ohio_t1dm/<patient_id>/` — manuscript figures for the reference patient (540).

## Main files

| File | Role |
|------|------|
| `scripts/main.py` | Head-to-head entry point |
| `scripts/head_to_head.py` | Per-patient approach comparison |
| `scripts/lib.py` | Latent imputer, feasibility, pipeline |
| `scripts/synthetic_augment_experiment.py` | Synthetic augmentation |
| `scripts/ohio_t1dm.py` | Data build/load, SCM rules |
| `scripts/model.py` | LightGBM wrapper |
| `scripts/visuals.py` | Plots |
| `scripts/presentation_build.py` | Static presentation figures |
| `manuscript/main.tex` | Paper draft |
| `presentation/causalml_project_presentation.tex` | Beamer deck |
