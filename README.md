# DOCAS-manuscript (private)

Manuscript, OhioT1DM / ReplayBG experiments, and the DOCAS library used in the paper.

Local clone: `~/Documents/DOCAS-manuscript`  
**Public package:** [Prgrmmrjns/DOCAS](https://github.com/Prgrmmrjns/DOCAS) (`~/Documents/DOCAS`)  
(`pip install` / examples live there; this repo vendors the same `src/docas` for the study.)

> On macOS (case-insensitive disk), do **not** also use a folder named `docas` — it collides with `DOCAS`.

## Layout

```text
src/docas/       # library (synced with public DOCAS)
examples/        # package demos
scripts/         # Ohio eval, ReplayBG, ablation, manuscript assets
manuscript/      # LaTeX paper
replaybg/        # vendored ReplayBG dependency
results/         # local caches (gitignored)
```

## Research setup

```bash
pip install -e ".[research]"
# place OhioT1DM XML at OhioT1DM/
python scripts/main.py           # uncomment steps in main()
python scripts/manuscript.py     # tables + figures
./manuscript/build.sh            # PDF
```

## Package-only install

```bash
pip install -e ".[examples]"
python examples/01_quickstart.py
```
