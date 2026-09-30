# Insulin Dose–Response Alignment for Glucose Forecasting in Type 1 Diabetes

Code to reproduce every table, figure, and number in the paper:

> J. C. Wolber, M. E. Samadi, J. Sellin, M. Mücke, A. Schuppert.
> *Insulin Dose–Response Alignment for Glucose Forecasting in Type 1 Diabetes.*
> Submitted to Artificial Intelligence in Medicine.

The paper makes two contributions:

- The **alignment error (AE)** measures how far the insulin response of a glucose forecaster is from a target dose–response curve (DRC). The response is read from individual conditional expectation (ICE) curves on the test set. The target is a population DRC taken from the UVa/Padova model.
- **DOCAS** (dose–response curve alignment via synthetic augmentation) adds synthetic training rows at insulin doses that were never given. Each row is labelled from the target DRC, and the unchanged forecaster is trained once on real and synthetic rows together.

A standalone, pip-installable implementation of DOCAS is available at [github.com/Prgrmmrjns/docas](https://github.com/Prgrmmrjns/docas). This repository contains only the study.

## 1. Install

You need Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Prgrmmrjns/DOCAS-manuscript.git
cd DOCAS-manuscript
uv sync
```

`uv sync` installs the exact package versions from `uv.lock`. This includes [ReplayBG](https://github.com/DIANA-UNIPD/replaybg) 2.x, pinned to the commit used for the paper. The code runs on the CPU and needs no GPU.

The flowchart and the graphical abstract are exported from SVG with `rsvg-convert` (librsvg). Install it with `brew install librsvg` on macOS or `apt install librsvg2-bin` on Debian/Ubuntu. Without it, all other results are still produced.

## 2. Add the datasets

Neither dataset can be redistributed, so both are excluded from the repository. Run the commands below from the repository root. The loaders read only these paths.

### OhioT1DM (12 participants)

OhioT1DM is released only after a data-use agreement. The current request steps are on [webpages.charlotte.edu/rbunescu/ohiot1dm.html](https://webpages.charlotte.edu/rbunescu/ohiot1dm.html) (Marling & Bunescu, 2020):

1. Complete the data-use agreement at [ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl](https://ohio.qualtrics.com/jfe/form/SV_02QtWEVm7ARIKIl).
2. Email the signed form to [liguo21@ohio.edu](mailto:liguo21@ohio.edu). Ohio University sends the dataset.

The archive contains `2018/` and `2020/`, each with `train/` and `test/`. Place that directory at the repository root under the name `OhioT1DM`. Leave the XML files unchanged. The study uses the official train/test split.

```text
OhioT1DM/
├── 2018/
│   ├── train/
│   │   ├── 559-ws-training.xml
│   │   ├── 563-ws-training.xml
│   │   ├── 570-ws-training.xml
│   │   ├── 575-ws-training.xml
│   │   ├── 588-ws-training.xml
│   │   └── 591-ws-training.xml
│   └── test/
│       ├── 559-ws-testing.xml
│       ├── 563-ws-testing.xml
│       ├── 570-ws-testing.xml
│       ├── 575-ws-testing.xml
│       ├── 588-ws-testing.xml
│       └── 591-ws-testing.xml
└── 2020/
    ├── train/
    │   ├── 540-ws-training.xml
    │   ├── 544-ws-training.xml
    │   ├── 552-ws-training.xml
    │   ├── 567-ws-training.xml
    │   ├── 584-ws-training.xml
    │   └── 596-ws-training.xml
    └── test/
        ├── 540-ws-testing.xml
        ├── 544-ws-testing.xml
        ├── 552-ws-testing.xml
        ├── 567-ws-testing.xml
        ├── 584-ws-testing.xml
        └── 596-ws-testing.xml
```

```bash
find OhioT1DM \( -name '*-ws-training.xml' -o -name '*-ws-testing.xml' \) | wc -l
# 24
```

### AZT1D (24 of 25 participants)

Download `AZT1D 2025.zip` (776 MB) from Mendeley Data, [doi.org/10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1) (Khamesian et al., 2025). The same file is [on the dataset page](https://data.mendeley.com/datasets/gk9m674wcx/1). From the repository root:

```bash
curl -L -o "AZT1D 2025.zip" \
  "https://data.mendeley.com/public-files/datasets/gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded"
unzip "AZT1D 2025.zip"
mv "AZT1D 2025" azt1d
```

The zip unpacks as `AZT1D 2025/`. Renaming that folder to `azt1d` is the layout the loader expects. It also contains `Visual Statistics/`, `README.txt`, and `Manuscript.pdf`. The study reads only the subject CSVs:

```text
azt1d/
└── CGM Records/
    ├── Subject 1/Subject 1.csv
    ├── Subject 2/Subject 2.csv
    └── Subject 3/ … Subject 25/Subject 25.csv
```

```bash
find azt1d/"CGM Records" -name 'Subject *.csv' | wc -l
# 25
```

Subject 14 is in the zip and is excluded in the code because it has no isolated CGM channel: its CGM and fingerstick readings share one column. Each remaining subject is split chronologically, 80% train and 20% test.

### Preprocessing

No separate preprocessing step is needed. `scripts/preprocessing.py` reads the raw XML/CSV files on first use and builds the model inputs on a 5-minute grid:

- current CGM;
- bolus insulin on board, using the ReplayBG subcutaneous insulin model;
- carbohydrates on board;
- the target, which is the 30- or 60-minute change in CGM.

Scalers are fitted on the training split only.

## 3. Reproduce the paper

```bash
uv run python scripts/main.py
```

This one command runs the full study in dependency order and writes every result to `results/` as JSON:

| Step | Where in the paper |
|---|---|
| CatBoost baseline vs DOCAS, both cohorts, 30/60 min, seeds 42–46 | Held-out accuracy and alignment (table and figure), trade-off figure |
| MLP (five seeds) and CatBoost with 12 lag features (seed 42) | Supplementary: MLP and lag features |
| Ablations: synthetic budget 0.1×–100×, uniform-only probes, DRC after prediction, 0.5×/2× insulin sensitivity | Ablation studies |
| Pump-settings target DRC | Ablation studies, Supplementary: pump-settings target DRC |
| ReplayBG corrective-bolus decision support on OhioT1DM: guarded, unguarded, pump-settings DRC, dose-penalty sweep | In-silico decision support, Supplementary: hypoglycaemia guard |
| AZT1D automated-delivery profile | In-silico decision support |

The full run fits several thousand models and ReplayBG digital twins, so expect it to take many hours on a laptop. Tables and figures are then written to `manuscript/tables/` and `manuscript/figures/`. The manuscript itself is not part of this repository.

To rebuild only the tables and figures from existing `results/`:

```bash
uv run python scripts/tables.py
```

### Reproducibility

- All random choices are seeded:
  - The main CatBoost and MLP results average seeds 42–46.
  - Ablations, lag features, and ReplayBG use seed 42.
  - ReplayBG twinning uses the library default seed.
- Re-running a CatBoost fit with the same seed reproduces the stored RMSE and AE exactly.
- MLP results can differ in the last digits across machines because of floating-point differences in PyTorch.
- Study settings (seeds, horizons, synthetic budget, noise, target DRC parameters, sensor profiles) are defined in the `STUDY KNOBS` block at the top of `scripts/main.py`.

## 4. Code layout

```text
scripts/
├── main.py           # study settings and the full pipeline (entry point)
├── core.py           # DOCAS: target DRC, synthetic rows, alignment error, ICE curves
├── preprocessing.py  # OhioT1DM and AZT1D loaders, insulin/carbohydrates on board
├── forecasters.py    # CatBoost and MLP forecasters
├── lgbm_lags.py      # lag-feature variant (12 × 5-minute lags per channel)
├── ablation.py       # budget, sampling, wrapper, sensitivity, and pump-settings ablations
├── replay_eval.py    # ReplayBG digital twins and the corrective-bolus decision support system
├── tables.py         # LaTeX tables and statistics (Wilcoxon, Holm–Bonferroni)
└── visuals.py        # figures
```

## 5. Citation

If you use this code, please cite the paper above. Please also cite the datasets (Marling & Bunescu, 2020; Khamesian et al., 2025) and ReplayBG (Cappon et al., 2023).
