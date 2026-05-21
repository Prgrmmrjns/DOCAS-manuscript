"""Head-to-head: baseline vs synthetic vs latent (all Ohio patients)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import ohio_t1dm
from lib import compute_feasibility, holdout_rmse, iSHAP, rule_do_curve
from model import make_model
from synthetic_augment_experiment import _drop_latent_columns, run_synthetic_augmentation

HEAD_TO_HEAD_KEYS = ("baseline", "synthetic_augment", "latent_confounder")
HEAD_TO_HEAD_LABELS = {
    "baseline": "Baseline",
    "synthetic_augment": "Synthetic augmentation",
    "latent_confounder": "Latent confounder",
}
HEAD_TO_HEAD_COLORS = {
    "baseline": "#c0392b",
    "synthetic_augment": "#2980b9",
    "latent_confounder": "#27ae60",
}
NORM_IRC_GRID = 25


@dataclass
class PatientApproachRow:
    patient_id: str
    key: str
    val_rmse: float
    test_rmse: float
    insulin_feasibility: float
    runtime_s: float
    norm_insulin_curve: np.ndarray | None = None


def _insulin_rule(rules: list[dict]) -> dict | None:
    target = ohio_t1dm.TARGET
    for r in rules:
        if str(r["start"]) == "insulin" and str(r["end"]) == target:
            return r
    return None


def _insulin_from_report(report: dict[str, Any], target: str) -> float:
    for item in report.get("rules", ()):
        if str(item.get("feature")) == "insulin" and str(item.get("rule", "")).endswith(f"->{target}"):
            v = item.get("aligned_step_fraction")
            return float(v) if v is not None else float("nan")
    return float("nan")


def normalized_insulin_curve(
    model: Any,
    X_val: np.ndarray,
    cols: list[str],
    rules: list[dict],
    *,
    latent_cols: tuple[str, ...] = (),
    grid_size: int,
) -> np.ndarray | None:
    r = _insulin_rule(rules)
    if r is None:
        return None
    grid, avg = rule_do_curve(
        model, X_val, cols, r, ohio_t1dm.TARGET,
        latent_columns=latent_cols, grid_size=grid_size,
    )
    if grid.size < 2:
        return None
    g0, g1 = float(np.min(grid)), float(np.max(grid))
    if g1 <= g0 + 1e-12:
        return None
    t = (grid - g0) / (g1 - g0)
    tgt = np.linspace(0.0, 1.0, NORM_IRC_GRID)
    return np.interp(tgt, t, avg).astype(np.float64)


def _load_patient(patient_id: str, root: Path) -> tuple[
    list[dict], pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None,
]:
    sub = ohio_t1dm._patient_sub(str(root), patient_id)
    rules = list(ohio_t1dm.scm_rules_for(sub))
    X, y, X_test, y_test = ohio_t1dm.load_train_test(str(root), patient_id)
    return rules, X, y, X_test, y_test


def _attr(dm: object, name: str, default: object = ()) -> object:
    v = getattr(dm, name, default)
    return default if v is None else v


def run_patient_approaches(
    patient_id: str,
    *,
    root: Path,
    seed: int,
    n_trials: int,
    selection_rmse_weight: float,
    n_hidden_confounders: int,
    feasibility_grid_size: int,
    do_curve_max_samples: int,
    n_synth_rows: int,
    show_progress_bar: bool,
    save_latent_figures: bool = False,
    img_dir: Path | None = None,
    run_name: str | None = None,
) -> list[PatientApproachRow]:
    from visuals import (
        centered_radial_positions,
        save_shap_beeswarm_before_after_figure,
        save_shap_interaction_network_before_after_graph,
    )

    rules, X_df, y_s, X_test_df, y_test_s = _load_patient(patient_id, root)
    target = str(ohio_t1dm.TARGET)

    cols = list(X_df.columns)
    latent_candidates = [c for c in cols if X_df[c].isna().all()]
    n_z = max(1, min(n_hidden_confounders, len(latent_candidates)))
    active_hidden = latent_candidates[:n_z]
    inactive = set(latent_candidates[n_z:])
    if inactive:
        cols = [c for c in cols if c not in inactive]
        X_df = X_df[cols]
        if X_test_df is not None:
            X_test_df = X_test_df[cols]

    obs_cols = [c for c in cols if c not in active_hidden]
    obs_idx = np.array([cols.index(c) for c in obs_cols], dtype=np.intp)

    X_np = X_df[cols].to_numpy(np.float64)
    y_np = y_s.to_numpy(np.float64)
    X_test_np = X_test_df[cols].to_numpy(np.float64) if X_test_df is not None else None
    y_test_np = y_test_s.to_numpy(np.float64) if y_test_s is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_np, test_size=ohio_t1dm.TEST_SIZE, random_state=seed,
    )
    if X_test_np is None:
        X_test_np, y_test_np = X_val, y_val

    X_train_obs = X_train[:, obs_idx]
    X_val_obs = X_val[:, obs_idx]
    X_test_obs = X_test_np[:, obs_idx]
    X_train_syn = _drop_latent_columns(pd.DataFrame(X_train_obs, columns=obs_cols))
    syn_cols = list(X_train_syn.columns)
    X_train_syn = X_train_syn.to_numpy(np.float64)
    X_val_syn = _drop_latent_columns(pd.DataFrame(X_val_obs, columns=obs_cols))[syn_cols].to_numpy(np.float64)
    X_test_syn = _drop_latent_columns(pd.DataFrame(X_test_obs, columns=obs_cols))[syn_cols].to_numpy(np.float64)

    rows: list[PatientApproachRow] = []

    t0 = time.perf_counter()
    m0 = make_model(random_state=seed, n_jobs=1)
    m0.fit(X_train_obs, y_train)
    rep0 = compute_feasibility(
        m0, X_val_obs, obs_cols, rules, target,
        random_state=seed, max_samples=None, grid_size=feasibility_grid_size,
    )
    runtime_b = time.perf_counter() - t0
    rows.append(PatientApproachRow(
        patient_id, "baseline",
        holdout_rmse(m0, X_val_obs, y_val),
        holdout_rmse(m0, X_test_obs, y_test_np),
        _insulin_from_report(rep0, target),
        runtime_b,
        normalized_insulin_curve(m0, X_val_obs, obs_cols, rules, grid_size=feasibility_grid_size),
    ))

    t0 = time.perf_counter()
    m_syn = run_synthetic_augmentation(
        X_train_syn, y_train, X_val_syn, y_val, syn_cols, rules, target,
        n_trials=n_trials,
        selection_rmse_weight=selection_rmse_weight,
        random_state=seed,
        feas_max_samples=do_curve_max_samples,
        feas_grid_size=feasibility_grid_size,
        synth_per_round=n_synth_rows,
        show_progress_bar=show_progress_bar,
    )
    runtime_syn = time.perf_counter() - t0
    rep_syn = compute_feasibility(
        m_syn, X_val_syn, syn_cols, rules, target,
        random_state=seed, max_samples=None, grid_size=feasibility_grid_size,
    )
    rows.append(PatientApproachRow(
        patient_id, "synthetic_augment",
        holdout_rmse(m_syn, X_val_syn, y_val),
        holdout_rmse(m_syn, X_test_syn, y_test_np),
        _insulin_from_report(rep_syn, target),
        runtime_syn,
        normalized_insulin_curve(m_syn, X_val_syn, syn_cols, rules, grid_size=feasibility_grid_size),
    ))

    latent_name = run_name or f"{ohio_t1dm.NAME}/{patient_id}"
    do_path = str(img_dir / "do_curve_before_after.png") if save_latent_figures and img_dir else None
    t0 = time.perf_counter()
    result = iSHAP.run_pipeline(
        X_df[cols], y_s,
        rules=rules,
        target=target,
        root=str(root),
        name=latent_name,
        test_size=float(ohio_t1dm.TEST_SIZE),
        X_test=X_test_df,
        y_test=y_test_s,
        do_curve_figure_path=do_path,
        random_state=seed,
        n_trials=n_trials,
        selection_rmse_weight=selection_rmse_weight,
        feasibility_grid_size=feasibility_grid_size,
        do_curve_max_samples=do_curve_max_samples,
        n_hidden_confounders=n_hidden_confounders,
        show_progress_bar=show_progress_bar,
    )
    runtime_lat = time.perf_counter() - t0
    m_lat = result["model_after"]
    X_val_lat = result["X_val"].to_numpy(np.float64)
    cols_lat = list(result["cols"])
    lat_cols = tuple(result["active_latent_confounders"])
    rows.append(PatientApproachRow(
        patient_id, "latent_confounder",
        float(result["rmse_metrics"]["val"]["after"]),
        float(result["rmse_metrics"]["test"]["after"]),
        _insulin_from_report(result["feasibility_report_after"], target),
        runtime_lat,
        normalized_insulin_curve(m_lat, X_val_lat, cols_lat, rules, latent_cols=lat_cols, grid_size=feasibility_grid_size),
    ))

    if save_latent_figures and img_dir is not None:
        latents = tuple(_attr(ohio_t1dm, "LATENT_COLUMNS")) + tuple(result["active_latent_confounders"])
        cols_after = result["cols"]
        cols_before = list(result.get("cols_before", cols_after))
        feat_after = tuple(c for c in cols_after if c != target)
        pos = centered_radial_positions(feat_after, target)
        pos_before = centered_radial_positions(tuple(c for c in cols_before if c != target), target)
        save_shap_interaction_network_before_after_graph(
            cols=cols_after, cols_before=cols_before, target_name=target,
            X_df_before=result["X_eval_before"], model_before=result["model_before"],
            X_df_after=result["X_eval_after"], model_after=result["model_after"],
            out_path=img_dir / "shap_interaction_network_before_after_synthetic.png",
            seed=seed, pos=pos, pos_before=pos_before,
            bolt_latents_after=latents,
            latent_fallback_corr_after=dict(_attr(ohio_t1dm, "LATENT_FALLBACK_EDGE_TO_TARGET", {})),
        )
        save_shap_beeswarm_before_after_figure(
            model_before=result["model_before"], model_after=result["model_after"],
            X_val_df=result["X_val"], X_val_before_df=result.get("X_val_before"),
            out_path=img_dir / "shap_beeswarm_before_after_synthetic.png",
            random_state=seed,
        )

    return rows


def summarize_rows(all_rows: list[PatientApproachRow]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in HEAD_TO_HEAD_KEYS:
        sub = [r for r in all_rows if r.key == key]
        if not sub:
            continue
        summary[key] = {
            "label": HEAD_TO_HEAD_LABELS[key],
            "val_rmse_mean": float(np.mean([r.val_rmse for r in sub])),
            "val_rmse_std": float(np.std([r.val_rmse for r in sub], ddof=0)),
            "test_rmse_mean": float(np.mean([r.test_rmse for r in sub])),
            "test_rmse_std": float(np.std([r.test_rmse for r in sub], ddof=0)),
            "insulin_feas_mean": float(np.mean([r.insulin_feasibility for r in sub])),
            "insulin_feas_std": float(np.std([r.insulin_feasibility for r in sub], ddof=0)),
            "runtime_s_mean": float(np.mean([r.runtime_s for r in sub])),
            "runtime_s_std": float(np.std([r.runtime_s for r in sub], ddof=0)),
            "n_patients": len(sub),
        }
    return summary


def write_head_to_head_table_tex(summary: dict[str, dict[str, float]], out_path: Path) -> None:
    def _cell(mean: float, std: float) -> str:
        return f"${mean:.2f} \\pm {std:.2f}$"

    def _time_cell(mean: float, std: float) -> str:
        return f"${mean:.1f} \\pm {std:.1f}$\\,s"

    lines = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        " & Val RMSE & Test RMSE & Insulin feasibility & Runtime \\\\",
        "\\midrule",
    ]
    for key in HEAD_TO_HEAD_KEYS:
        s = summary.get(key)
        if not s:
            continue
        lines.append(
            f"{s['label']} & {_cell(s['val_rmse_mean'], s['val_rmse_std'])} "
            f"& {_cell(s['test_rmse_mean'], s['test_rmse_std'])} "
            f"& {_cell(s['insulin_feas_mean'], s['insulin_feas_std'])} "
            f"& {_time_cell(s['runtime_s_mean'], s['runtime_s_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def write_head_to_head_json(
    all_rows: list[PatientApproachRow],
    summary: dict[str, dict[str, float]],
    out_path: Path,
    *,
    n_trials: int,
    selection_rmse_weight: float,
) -> None:
    payload = {
        "preliminary": True,
        "setup": f"scripts/main.py (NSGA-II, selection_rmse_weight={selection_rmse_weight}, n_trials={n_trials})",
        "evaluation": "insulin rule from compute_feasibility on full validation set",
        "n_patients": int(summary.get("baseline", {}).get("n_patients", 0)),
        "feasibility_rule": "insulin monotonicity (decreasing)",
        "wall_clock_s": round(float(summary.get("_wall_clock_s", 0.0)), 1),
        "summary": {k: v for k, v in summary.items() if not k.startswith("_")},
        "patients": [
            {
                "patient_id": r.patient_id,
                "approach": r.key,
                "val_rmse": round(r.val_rmse, 4),
                "test_rmse": round(r.test_rmse, 4),
                "insulin_feasibility": round(r.insulin_feasibility, 4),
                "runtime_s": round(r.runtime_s, 2),
            }
            for r in all_rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))


def write_presentation_artifacts(
    all_rows: list[PatientApproachRow],
    summary: dict[str, dict[str, float]],
    pres_dir: Path,
    *,
    n_trials: int,
    selection_rmse_weight: float,
) -> None:
    from visuals import save_head_to_head_insulin_irc_figure

    pres_dir.mkdir(parents=True, exist_ok=True)
    curves: dict[str, list[np.ndarray]] = {k: [] for k in HEAD_TO_HEAD_KEYS}
    for row in all_rows:
        if row.norm_insulin_curve is not None and row.key in curves:
            curves[row.key].append(row.norm_insulin_curve)

    save_head_to_head_insulin_irc_figure(
        curves_by_approach=curves,
        approach_labels=HEAD_TO_HEAD_LABELS,
        approach_colors=HEAD_TO_HEAD_COLORS,
        out_path=pres_dir / "head_to_head_insulin_ircs.png",
    )
    write_head_to_head_table_tex(summary, pres_dir / "head_to_head_table.tex")
    write_head_to_head_json(
        all_rows, summary, pres_dir / "head_to_head.json",
        n_trials=n_trials, selection_rmse_weight=selection_rmse_weight,
    )


def run_all_patients(
    *,
    root: Path,
    seed: int,
    n_trials: int,
    selection_rmse_weight: float,
    n_hidden_confounders: int,
    feasibility_grid_size: int,
    do_curve_max_samples: int,
    n_synth_rows: int,
    show_progress_bar: bool,
    manuscript_patient: str | None = None,
) -> tuple[list[PatientApproachRow], dict[str, dict[str, float]]]:
    all_rows: list[PatientApproachRow] = []
    pids = ohio_t1dm.patient_ids(str(root))
    t_all = time.perf_counter()
    for i, pid in enumerate(pids, 1):
        print(f"  head-to-head patient {pid} ({i}/{len(pids)}) …", flush=True)
        save_figs = pid == manuscript_patient
        img_dir = root / "manuscript" / "images" / f"{ohio_t1dm.NAME}/{pid}" if save_figs else None
        if save_figs and img_dir is not None:
            img_dir.mkdir(parents=True, exist_ok=True)
        all_rows.extend(run_patient_approaches(
            pid,
            root=root,
            seed=seed,
            n_trials=n_trials,
            selection_rmse_weight=selection_rmse_weight,
            n_hidden_confounders=n_hidden_confounders,
            feasibility_grid_size=feasibility_grid_size,
            do_curve_max_samples=do_curve_max_samples,
            n_synth_rows=n_synth_rows,
            show_progress_bar=show_progress_bar,
            save_latent_figures=save_figs,
            img_dir=img_dir,
        ))
    summary = summarize_rows(all_rows)
    summary["_wall_clock_s"] = float(time.perf_counter() - t_all)
    return all_rows, summary
