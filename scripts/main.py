from __future__ import annotations

import time
import warnings
from pathlib import Path

import ohio_t1dm
from head_to_head import (
    HEAD_TO_HEAD_KEYS,
    run_all_patients,
    write_presentation_artifacts,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
PATIENT_ID = "540"

N_TRIALS = 5000
SELECTION_RMSE_WEIGHT = 0.5  # 0= only feasibility, 1= only RMSE
N_HIDDEN_CONFOUNDERS = 1
FEASIBILITY_GRID_SIZE = 10
DO_CURVE_MAX_SAMPLES = 100

# Synthetic augmentation: SCM counterfactual rows per Optuna trial (single NSGA-II pass).
N_SYNTH_ROWS_PER_ROUND = 100
SHOW_PROGRESSBAR = False

PRES_IMG_DIR = ROOT / "presentation" / "images"


def _run_kw() -> dict:
    return dict(
        root=ROOT,
        seed=SEED,
        n_trials=N_TRIALS,
        selection_rmse_weight=SELECTION_RMSE_WEIGHT,
        n_hidden_confounders=N_HIDDEN_CONFOUNDERS,
        feasibility_grid_size=FEASIBILITY_GRID_SIZE,
        do_curve_max_samples=DO_CURVE_MAX_SAMPLES,
        n_synth_rows=N_SYNTH_ROWS_PER_ROUND,
        show_progress_bar=SHOW_PROGRESSBAR,
    )


def run_head_to_head() -> None:
    n = len(ohio_t1dm.patient_ids(str(ROOT)))
    print(f"Head-to-head ({n} patients, n_trials={N_TRIALS}, NSGA-II w={SELECTION_RMSE_WEIGHT}) …", flush=True)
    t0 = time.perf_counter()
    rows, summary = run_all_patients(manuscript_patient=PATIENT_ID, **_run_kw())
    wall_s = time.perf_counter() - t0
    write_presentation_artifacts(
        rows, summary, PRES_IMG_DIR,
        n_trials=N_TRIALS, selection_rmse_weight=SELECTION_RMSE_WEIGHT,
    )
    print("\nSummary (all patients):", flush=True)
    for key in HEAD_TO_HEAD_KEYS:
        s = summary.get(key, {})
        if s:
            print(
                f"  {s['label']:22s}  insulin_feas {s['insulin_feas_mean']:.3f}±{s['insulin_feas_std']:.3f}  "
                f"val_RMSE {s['val_rmse_mean']:.2f}  test_RMSE {s['test_rmse_mean']:.2f}  "
                f"runtime {s['runtime_s_mean']:.1f}±{s['runtime_s_std']:.1f}s",
                flush=True,
            )
    print(f"\nTotal wall time: {wall_s:.1f}s ({wall_s / 60:.1f} min)", flush=True)
    print(f"\nPresentation → {PRES_IMG_DIR / 'head_to_head_insulin_ircs.png'}", flush=True)


if __name__ == "__main__":
    run_head_to_head()
