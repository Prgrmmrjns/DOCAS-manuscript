"""Build static figures for causalml_project_presentation.tex (no full pipeline runs)."""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import ohio_t1dm
from lib import rule_do_curve
from model import make_model
from visuals import (
    save_irc_alignment_examples_figure,
    save_irc_six_relationship_types_figure,
    save_shap_beeswarm_single,
    save_single_rule_do_curve_figure,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
INSULIN_IRC_PATIENT = "588"
SEED = 42
FEASIBILITY_GRID_SIZE = 10
TARGET = ohio_t1dm.TARGET

_IRC_SPECS = (
    (INSULIN_IRC_PATIENT, "insulin", "01_insulin_interventional_response_curve.png"),
    ("591", "carbs", "01_carbs_interventional_response_curve.png"),
    ("567", "pa_accel", "01_pa_accel_interventional_response_curve.png"),
)


def _baseline_model(patient_id: str) -> tuple[Any, pd.DataFrame, list[str]]:
    X_df, y_s, _, _ = ohio_t1dm.load_train_test(str(ROOT), patient_id)
    latent_candidates = [c for c in X_df.columns if X_df[c].isna().all()]
    obs_cols = [c for c in X_df.columns if c not in latent_candidates]
    X_np = X_df[obs_cols].to_numpy(np.float64)
    y_np = y_s.to_numpy(np.float64)
    X_train, X_val, y_train, _ = train_test_split(
        X_np, y_np, test_size=ohio_t1dm.TEST_SIZE, random_state=SEED,
    )
    model = make_model(random_state=SEED, n_jobs=1)
    model.fit(X_train, y_train)
    return model, pd.DataFrame(X_val, columns=obs_cols), obs_cols


def build_figures(img_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)

    for patient_id, feat, fname in _IRC_SPECS:
        sub = ohio_t1dm._patient_sub(str(ROOT), patient_id)
        rules = list(ohio_t1dm.scm_rules_for(sub))
        rule = next((r for r in rules if str(r["start"]) == feat), None)
        if rule is None:
            continue
        model, X_val_df, obs_cols = _baseline_model(patient_id)
        save_single_rule_do_curve_figure(
            rule=rule,
            model=model,
            X_eval_df=X_val_df,
            cols=obs_cols,
            target_name=TARGET,
            out_path=img_dir / fname,
            label="Before",
            line_color="#c0392b",
            grid_size=FEASIBILITY_GRID_SIZE,
        )

    save_irc_six_relationship_types_figure(img_dir / "irc_six_relationship_types.png")

    rules = list(ohio_t1dm.scm_rules_for(ohio_t1dm._patient_sub(str(ROOT), INSULIN_IRC_PATIENT)))
    insulin_rule = next(r for r in rules if str(r["start"]) == "insulin")
    model, X_val_df, obs_cols = _baseline_model(INSULIN_IRC_PATIENT)
    grid, avg = rule_do_curve(
        model, X_val_df[obs_cols].to_numpy(np.float64), obs_cols, insulin_rule, TARGET,
        grid_size=FEASIBILITY_GRID_SIZE,
    )
    save_irc_alignment_examples_figure(
        img_dir / "irc_alignment_examples.png",
        grid=grid, avg=avg, patient_id=INSULIN_IRC_PATIENT,
        increasing=False, feature="insulin",
    )

    ins_model, ins_val, _ = _baseline_model(INSULIN_IRC_PATIENT)
    save_shap_beeswarm_single(
        ins_model, ins_val, img_dir / "01_shap_baseline_problem.png",
        title=f"Baseline LightGBM SHAP — patient {INSULIN_IRC_PATIENT}",
    )

    ohio_t1dm.ensure_ohio_csv(str(ROOT))
    ohio_root = ROOT / "OhioT1DM"
    overview = ohio_t1dm.find_overview_patient_xml(ohio_root)
    if overview:
        pid, xmlp = overview
        plot = ohio_t1dm.save_example_insulin_glucose_plot(ROOT, ohio_root, xml_path=xmlp, patient_subdir=pid)
        if plot and plot.is_file():
            shutil.copy(plot, img_dir / "preprocess_example_insulin_glucose.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation figures for causalml_project_presentation.tex")
    _ = parser.parse_args()
    img_dir = ROOT / "presentation" / "images"
    t0 = time.perf_counter()
    build_figures(img_dir)
    print(f"Figures → {img_dir} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
