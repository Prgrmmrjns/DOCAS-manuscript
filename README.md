# DOCAS-manuscript (private)

Manuscript + OhioT1DM / ReplayBG experiments.

**Library (PyPI / public):** [`docas`](https://pypi.org/project/docas/) · [github.com/Prgrmmrjns/docas](https://github.com/Prgrmmrjns/docas)

## Setup

```bash
cd ~/Documents/DOCAS-manuscript
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# place OhioT1DM XML at OhioT1DM/
```

If `pip show docas` lists torch/optuna/shap, you have an old editable install — fix with:

```bash
pip uninstall -y docas
pip install --no-cache-dir 'docas>=0.1.0'
pip show docas   # should require only numpy
```

## Run

```bash
python scripts/main.py            # uncomment steps in main()
python scripts/manuscript.py      # tables + figures
./manuscript/build.sh             # PDF
```

LSTM cross-check: `pip install '.[lstm]'` (or TensorFlow 2.16 in a separate venv).

## Layout

```text
scripts/         # Ohio eval, ReplayBG, ablation, manuscript assets
manuscript/      # LaTeX paper
replaybg/        # vendored ReplayBG
results/         # local caches (gitignored)
```
