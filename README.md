# Insulin Dose–Response Alignment for Glucose Forecasting in Type 1 Diabetes

Code for the revision of:

> J. C. Wolber, M. E. Samadi, J. Sellin, M. Mücke, A. Schuppert.
> *Insulin Dose–Response Alignment for Glucose Forecasting in Type 1 Diabetes.*
> Intelligence-Based Medicine, IBMED-D-26-00789.

The paper makes two contributions:

- The **alignment error (AE)** measures how far a forecaster’s insulin response is from a stated population dose–response curve (DRC). The response is the individual conditional expectation (ICE) on the test set. The target is the UVa/Padova curve. AE is a plausibility check, not patient-specific ground truth.
- **DOCAS** (dose–response curve alignment via synthetic augmentation) adds counterfactual training rows at insulin doses that were never given. Each row copies a real context and is labelled from that DRC. The forecaster is trained once, unchanged, on the real and synthetic rows together.

A standalone package is at [github.com/Prgrmmrjns/docas](https://github.com/Prgrmmrjns/docas). This repository is the study.

## 1. Install

Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Prgrmmrjns/DOCAS-manuscript.git
cd DOCAS-manuscript
uv sync
```

`uv sync` installs the lockfile, including [ReplayBG](https://github.com/DIANA-UNIPD/replaybg) at the commit used for the paper. The study runs on CPU.

The flowchart is exported from SVG with `rsvg-convert` (`brew install librsvg` or `apt install librsvg2-bin`). The other figures do not need it.

## 2. Add the datasets

None of the datasets can be redistributed. They are gitignored. Run the commands from the repository root.

### MetaboNet v1.0

The revision reads OhioT1DM, AZT1D, HUPA-UCM, and T1D-UOM from MetaboNet v1.0 (Wolff et al., 2026), one model per person on the full record. Download the public train and test parquet files from [metabo-net.org](https://metabo-net.org) and place them here:

```text
datasets/
├── train.parquet
└── test.parquet
```

The first run builds `datasets/metabonet_full/` (one parquet per person) and `datasets/metabonet_known/`.

### OhioT1DM and AZT1D raw files

`scripts/main.py` and the patient-588 figures still read the original releases.

OhioT1DM is released after a data-use agreement. Request steps are on [webpages.charlotte.edu/rbunescu/ohiot1dm.html](https://webpages.charlotte.edu/rbunescu/ohiot1dm.html) (Marling & Bunescu, 2020):

1. Complete the form at [ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl](https://ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl).
2. Email the signed form to [liguo21@ohio.edu](mailto:liguo21@ohio.edu).

Place the archive, with `2018/` and `2020/` each containing `train/` and `test/`, at `OhioT1DM/`. Leave the XML files unchanged.

```bash
find OhioT1DM \( -name '*-ws-training.xml' -o -name '*-ws-testing.xml' \) | wc -l
```

AZT1D is `AZT1D 2025.zip` (776 MB) on Mendeley Data, [doi.org/10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1) (Khamesian et al., 2025):

```bash
curl -L -o "AZT1D 2025.zip" \
  "https://data.mendeley.com/public-files/datasets/gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
unzip "AZT1D 2025.zip"
mv "AZT1D 2025" azt1d
```

Subject 14 is excluded in code: its CGM and fingerstick readings share one column.

No separate preprocessing step. `scripts/preprocessing.py` builds current CGM, bolus insulin on board, carbohydrates on board, and the 30- or 60-minute CGM change on a 5-minute grid. Scalers are fit on the training split only.

## 3. Reproduce the revision

Each command is resumable. Finished patients are skipped. Results land in `results/revision/`. Expect many hours.

```bash
uv run python scripts/ablation.py cohort
uv run python scripts/ablation.py
uv run python scripts/metabonet_comparators.py
uv run python scripts/replay_eval.py revision
uv run python scripts/visuals.py
```

| Command | Writes | Where in the paper |
|---|---|---|
| `ablation.py cohort` | `results/revision/full_recent/` | Table 3 (Baseline and DOCAS) and Table 5 context-adjusted DOCAS forecasting. One JSON per person. Latest 8,000 training rows, seed 42, chronological 80/20 split. OhioT1DM, AZT1D, HUPA-UCM, T1D-UOM. |
| `ablation.py` | `results/revision/full_ablation.json` | Table 5 forecasting columns. Budgets, uniform probes, post-hoc DRC, 0.5×/2× insulin sensitivity, pump-settings DRC. `weight` and `weight-replay` are the sample-weight arms. |
| `metabonet_comparators.py` | `results/revision/full_comparators.json` | Appendix B. Monotone CatBoost, MLP, penalty MLP, MLP DOCAS, 12-lag CatBoost. Also the 90/120-minute endpoints. `lags` reruns the OhioT1DM and AZT1D lag fits. |
| `replay_eval.py revision` | `results/revision/full_replay_open.json`, `results/revision/ablation_replay.json`, `results/revision/alt_replay.json` | Table 4, Table 5 ReplayBG, Appendix C, and Table 6 ReplayBG. |
| `visuals.py` | `results/revision/pdp_mean_ices.json`, `manuscript/figures/` | Cohort-mean insulin curves, then the manuscript figures. |

The LaTeX tables in `manuscript/tables/` are the manuscript sources. They are filled from those JSON files. `scripts/main.py` is the first-submission OhioT1DM and AZT1D pipeline (five seeds, then ablations and ReplayBG). It does not rebuild the revision tables.

### Reproducibility

- Revision fits use seed 42.
- `scripts/main.py` averages CatBoost and MLP seeds 42–46. Its ablations and ReplayBG use seed 42.
- The same seed reproduces a CatBoost RMSE and AE. MLP digits can move across machines because of PyTorch.
- Study settings are the `STUDY KNOBS` block at the top of `scripts/main.py`.

## 4. Code layout

```text
scripts/
├── main.py                 # study settings; first-submission pipeline
├── core.py                 # target DRC, synthetic rows, alignment error
├── preprocessing.py        # OhioT1DM, AZT1D, and MetaboNet loaders
├── forecasters.py          # CatBoost, monotone CatBoost, MLP
├── metabonet_comparators.py  # Appendix B, strata, long horizons, 12-lag fits
├── ablation.py             # cohort fit, context-adjusted DOCAS, budgets, wrapper, sample weight
├── replay_eval.py          # ReplayBG twins, corrective-bolus search, revision windows
├── visuals.py              # figures
└── tables.py               # first-submission LaTeX tables
```

## 5. Citation

Cite the paper above, the datasets (Marling & Bunescu, 2020; Khamesian et al., 2025; Wolff et al., 2026), and ReplayBG (Cappon et al., 2023).
