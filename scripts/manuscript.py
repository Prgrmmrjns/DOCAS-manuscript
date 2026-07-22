"""Generate manuscript tables and figures under manuscript/. Run via scripts/main.py."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path as mpl_path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import main  # configures DOCAS knobs + sys.path
import ablation
from docas import DOCAS, future as _future
import ohio_t1dm_eval
import replay_bg
from ohio_t1dm_preprocessing import horizon_steps, load_patient, load_patient_meta

ROOT = main.ROOT

F0_LO_NOM = DOCAS.Y0_LO
FULL_CORRECTION_U = DOCAS.SPAN
replaybg_response = ohio_t1dm_eval.replaybg_response
ALIGN_TRIGGER_MGDL = DOCAS.Y0_LO
DATA_ROOT = ohio_t1dm_eval.DATA_ROOT
FORECAST_HORIZONS = ohio_t1dm_eval.FORECAST_HORIZONS
GLUCOSE_TARGET_MGDL = ohio_t1dm_eval.GLUCOSE_TARGET_MGDL
LSTM_COHORT, MONOTONIC_COHORT = ohio_t1dm_eval.LSTM_COHORT, ohio_t1dm_eval.MONOTONIC_COHORT
load_baseline_model = ohio_t1dm_eval.load_baseline_model
ohio_cohort_path, ohio_horizon_payload = ohio_t1dm_eval.ohio_cohort_path, ohio_t1dm_eval.ohio_horizon_payload
CIB_TARGET_MGDL, HYPER_TRIGGER_MGDL = replay_bg.CIB_TARGET_MGDL, replay_bg.HYPER_TRIGGER_MGDL
REPLAY_HORIZONS, REPLAY_WINDOW_H = replay_bg.REPLAY_HORIZONS, replay_bg.REPLAY_WINDOW_H
TABLE2_PATIENT, TWIN_RMSE_PRIMARY_MGDL = replay_bg.TABLE2_PATIENT, replay_bg.TWIN_RMSE_PRIMARY_MGDL
YTS_MIN = replay_bg.YTS_MIN
replay_cohort_path, replay_horizon_payload = replay_bg.replay_cohort_path, replay_bg.replay_horizon_payload

FIGURES, TABLES = ROOT / "manuscript" / "figures", ROOT / "manuscript" / "tables"
LGBM_COHORT = ROOT / "results" / "ohio_t1dm" / "cohort.json"
ABLATION_JSON = ablation.OUT_JSON
TARGET_COLOR, BASELINE_COLOR, DOCAS_COLOR = "#5E35B1", "#D32F2F", "#00897B"
TIR_LO, TIR_HI, GRID = 70.0, 180.0, np.linspace(0, 1, 21)
FLOWCHART_PATIENT, FLOWCHART_HORIZON, FLOWCHART_N_ANCHORS = "588", 30, 8
FLOWCHART_ICON_SIZE, FLOWCHART_ARROW_LW = (5.6, 3.4), 6.5
FC_INPUT, FC_PROCESS, FC_OUTPUT = "#7E57C2", "#26C6DA", "#546E7A"
FC_LOOP, FC_LOOP_EDGE, FC_LOOP_TEXT, FC_ARROW_EXIT = "#E0F7FA", "#00838F", "#006064", "#00838F"
FC_CGM, FC_HYPER, FC_TARGET, FC_INSULIN = "#3949AB", "#FF7043", "#5E35B1", "#EF6C00"


def _save_fig(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _fmt(mean: float, std: float) -> str:
    return "---" if not np.isfinite(mean) else f"{mean:.2f} $\\pm$ {std:.2f}"


def _metric_cells(m: dict, rmse_key: str, align_key: str) -> list[str]:
    return [_fmt(m[rmse_key]["mean"], m[rmse_key]["std"]), _fmt(m[align_key]["mean"], m[align_key]["std"])]


def _lgbm_cells(ph: int, arm: str) -> list[str]:
    m = ohio_horizon_payload(json.loads(LGBM_COHORT.read_text()), ph)["metrics"]
    sfx = "baseline" if arm == "baseline" else "docas"
    return _metric_cells(m, f"{sfx}_test_rmse_mgdl", f"{sfx}_alignment_error" if arm == "baseline" else "alignment_error")


def _monotonic_cells(ph: int) -> list[str]:
    m = ohio_horizon_payload(json.loads(MONOTONIC_COHORT.read_text()), ph)["metrics"]
    return _metric_cells(m, "test_rmse_mgdl", "alignment_error")


def _lstm_cells(ph: int, arm: str) -> list[str]:
    m = json.loads(LSTM_COHORT.read_text())["horizons"][str(ph)]["metrics"]
    return _metric_cells(m, f"{arm}_test_rmse_mgdl", f"{arm}_alignment_error")


def table1_tex() -> Path:
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{PH{ph}}}" for ph in FORECAST_HORIZONS)
    cmid = "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(FORECAST_HORIZONS)))
    col = "l" + "cc" * len(FORECAST_HORIZONS)
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Held-out OhioT1DM test RMSE and alignment error for",
        r"baseline, hard monotone-constrained, and DOCAS LightGBM models, and for a",
        r"second, unrelated model family (np-LSTM, p-LSTM, and DOCAS LSTM), across all",
        r"12 participants at the 30- and 60-min prediction horizons. The monotonic model",
        r"applies a LightGBM monotone-constraints decrease constraint on the",
        r"insulin feature~\citep{ke2017}. np-LSTM and p-LSTM follow the architectures of",
        r"\citet{prendin2023}; DOCAS LSTM applies the same synthetic-augmentation",
        r"mechanism as DOCAS LightGBM to the p-LSTM architecture, demonstrating that the",
        r"method is model-agnostic. Alignment error is the RMSE (mg/dL) between each",
        r"model's interventional audit curve and the ReplayBG-derived target",
        r"function~$\tau$ on held-out hyperglycaemic test contexts",
        r"(Section~\ref{subsec:probe}), measured on absolute future-CGM amplitude.",
        r"RMSE is on $\Delta\mathrm{CGM}$",
        r"(mg/dL). Values are cohort mean $\pm$ standard deviation.}",
        r"\label{tab:ohio}",
        rf"\begin{{tabular*}}{{\tblwidth}}{{@{{\extracolsep{{\fill}}}}{col}@{{}}}}",
        r"\toprule",
        rf" & {head} \\",
        cmid,
        "Model & " + " & ".join(["RMSE & Align."] * len(FORECAST_HORIZONS)) + r" \\",
        r"\midrule",
    ]
    if LGBM_COHORT.is_file():
        lines.append("Baseline LightGBM & " + " & ".join(c for ph in FORECAST_HORIZONS for c in _lgbm_cells(ph, "baseline")) + r" \\")
    if MONOTONIC_COHORT.is_file():
        lines.append("Monotonic-constrained LightGBM & " + " & ".join(c for ph in FORECAST_HORIZONS for c in _monotonic_cells(ph)) + r" \\")
    if LGBM_COHORT.is_file():
        lines.append("DOCAS LightGBM & " + " & ".join(c for ph in FORECAST_HORIZONS for c in _lgbm_cells(ph, "docas")) + r" \\")
    if LSTM_COHORT.is_file():
        lines.append(r"\addlinespace")
        for label, arm in (("np-LSTM", "np_lstm"), ("p-LSTM", "p_lstm"), ("DOCAS LSTM", "docas_lstm")):
            lines.append(f"{label} & " + " & ".join(c for ph in FORECAST_HORIZONS for c in _lstm_cells(ph, arm)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "table1.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_table2(horizon_min=30, *, cohort_path: Path | str | None = None) -> dict:
    """Summarise ReplayBG cohort windows for Table 2 (patient 588)."""
    cohort = json.loads(Path(cohort_path or replay_cohort_path()).read_text())
    horizon = replay_horizon_payload(cohort, horizon_min) if "horizons" in cohort else cohort
    return replay_bg._table2_from_windows(horizon_min, horizon["windows"])


def _replay_summary(ph: int) -> dict:
    cohort = json.loads(replay_cohort_path().read_text())
    return (replay_horizon_payload(cohort, ph).get("table2") or write_table2(ph))["summary"]


def table2_tex() -> Path:
    summaries = {ph: _replay_summary(ph) for ph in REPLAY_HORIZONS}
    arms, arm_labels = ("no_dss", "baseline", "docas"), ("None", "Baseline", "DOCAS")
    metrics = (("TBR (\\%)", "tbr"), ("TIR (\\%)", "tir"), ("TAR (\\%)", "tar"), ("Insulin (U)", "insulin_u"))
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{PH{ph}}}" for ph in REPLAY_HORIZONS)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(REPLAY_HORIZONS)))
    lines = [
        r"\begin{table}[!htbp]",
        rf"\caption{{ReplayBG outcomes for patient~{TABLE2_PATIENT} (mean $\pm$ SD).}}",
        r"\label{tab:replay}",
        rf"\begin{{tabular*}}{{\tblwidth}}{{@{{\extracolsep{{\fill}}}}l{'c' * (3 * len(REPLAY_HORIZONS))}@{{}}}}",
        r"\toprule",
        rf" & {head} \\",
        cmid,
        rf"Metric & {' & '.join(arm_labels * len(REPLAY_HORIZONS))} \\",
        r"\midrule",
    ]
    for label, key in metrics:
        cells = [_fmt(float(summaries[ph][arm][key]["mean"]), float(summaries[ph][arm][key]["std"]))
                 for ph in REPLAY_HORIZONS for arm in arms]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    out = TABLES / "table2.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt_scalar(v, digits=2):
    return "---" if not np.isfinite(v) else f"{v:.{digits}f}"


def table3_tex() -> Path:
    """PH30 ablation table from ``results/ablation/patient588_ph30.json``."""
    r = json.loads(ABLATION_JSON.read_text())
    pid, f0_lo = ablation.PATIENT_ID, int(ablation.FULL_RANGE_F0_LO)
    rows = [
        (r"Hardcoded (hyper only)", _fmt_scalar(r["hardcoded"]["alignment_error"]), r["hardcoded"]["rmse_delta_mgdl"]),
        (r"Independence", _fmt_scalar(r["independence"]["alignment_error"]), r["independence"]["rmse_delta_mgdl"]),
        (r"CGM interaction", _fmt_scalar(r["cgm_interaction"]["interaction_error"]), r["cgm_interaction"]["rmse_delta_mgdl"]),
        (r"CHO surface", _fmt_scalar(r["cho_surface"]["alignment_error"]), r["cho_surface"]["rmse_delta_mgdl"]),
        (rf"Full range ($f_0>{f0_lo}$)", _fmt_scalar(r["full_range"]["alignment_error_full"]), r["full_range"]["rmse_delta_mgdl"]),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        rf"\caption{{PH30 ablations on patient~{pid}: alternative DOCAS targets.",
        rf"Align.\ (mg/dL RMSE to~$\tau$; full-range uses $f_0>{f0_lo}$~mg/dL,",
        rf" others use hyperglycaemic contexts) and RMSE~$\Delta$ vs.\ baseline",
        rf" ({r['baseline_test_rmse_mgdl']:.2f}~mg/dL).}}",
        r"\label{tab:ablation}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lcc@{}}",
        r"\toprule", r"Ablation & Align. & RMSE $\Delta$ \\", r"\midrule",
    ] + [rf"{name} & {align} & {rmse:+.2f} \\" for name, align, rmse in rows] + [
        r"\bottomrule", r"\end{tabular*}", r"\end{table}",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "table3.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _ablation_drc(model, anchors, u, ji, gi, ins_s, tgt_s, cgm_s, *, f0_a=None, amp=None, k=None, flat=False):
    k = ablation.HYPER_K if k is None else k
    if flat:
        return np.full(len(u), float(np.mean(f0_a)))
    if amp is not None:
        return (f0_a[:, None] * (1.0 - amp * ablation.hyperbolic(u, k)[None, :])).mean(0)
    rows = np.repeat(anchors, len(u), 0)
    rows[:, ji] = ins_s.transform((np.tile(u, len(anchors)) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    delta = tgt_s.inverse_transform(model.predict(rows).reshape(-1, 1)).ravel().reshape(len(anchors), len(u))
    cgm = cgm_s.inverse_transform(anchors[:, gi : gi + 1]).ravel()
    return (cgm[:, None] + delta).mean(0)


def _ablation_plot_triplet(ax, u, target, baseline, docas, title, ylabel="Future CGM (mg/dL)"):
    for y, c, lw, ls, lbl in ((target, TARGET_COLOR, 2.4, "--", "Target"), (baseline, BASELINE_COLOR, 2.2, "--", "Baseline"),
                              (docas, DOCAS_COLOR, 2.6, "-", "DOCAS")):
        ax.plot(u, y, color=c, lw=lw, ls=ls, label=lbl)
    ax.set(xlabel="Insulin (normalized)", ylabel=ylabel, title=title)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.18)


def ablation_flexibility_figure() -> Path:
    """Five-panel ablation figure; requires ``ablation.run()`` to have populated the model cache."""
    c = ablation._MODEL_CACHE
    baseline, ins_s, cgm_s, cho_s, tgt_s = c["baseline"], c["ins_s"], c["cgm_s"], c["cho_s"], c["tgt_s"]
    gi, ji, ci, A_te, f0_te = c["gi"], c["ji"], c["ci"], c["A_te"], c["f0_te"]
    A_full, f0_full = c["A_te_full"], c["f0_te_full"]
    u = np.linspace(0.0, 1.0, 41)
    kw = dict(u=u, ji=ji, gi=gi, ins_s=ins_s, tgt_s=tgt_s, cgm_s=cgm_s)
    base_drc = _ablation_drc(baseline, A_te, **kw)
    fig = plt.figure(figsize=(12.2, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    _ablation_plot_triplet(fig.add_subplot(gs[0, 0]), u, _ablation_drc(None, A_te, f0_a=f0_te, amp=ablation.NOMINAL_DROP, **kw),
                           base_drc, _ablation_drc(c["m1"], A_te, **kw), "1. Hardcoded (hyperglycaemia only)")
    _ablation_plot_triplet(fig.add_subplot(gs[0, 1]), u, _ablation_drc(None, A_te, f0_a=f0_te, flat=True, **kw),
                           base_drc, _ablation_drc(c["m2"], A_te, **kw), "2. Feature independence (null curve)")
    ax3 = fig.add_subplot(gs[1, 0])
    slope_kw = dict(anchors=A_te, ji=ji, gi=gi, ins_s=ins_s, cgm_s=cgm_s, tgt_s=tgt_s)
    target_slope = 1.0 - ablation.NOMINAL_DROP * ablation.hyperbolic(u, ablation.HYPER_K)
    ax3.plot(u, target_slope, color=TARGET_COLOR, lw=2.4, ls="--", label="Target")
    ax3.plot(u, np.asarray(ablation._cgm_slopes(baseline, **slope_kw, u_probe=u)), color=BASELINE_COLOR, lw=2.2, ls="--", label="Baseline")
    ax3.plot(u, np.asarray(ablation._cgm_slopes(c["m3"], **slope_kw, u_probe=u)), color=DOCAS_COLOR, lw=2.6, label="DOCAS")
    ax3.set(xlabel="Insulin (normalized)", ylabel=r"$\partial\,\hat g / \partial\,\mathrm{CGM}$",
            title="3. Insulin modulates CGM slope (interaction)")
    ax3.legend(frameon=False, fontsize=7.5)
    ax3.grid(alpha=0.18)
    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    cgm0 = cgm_s.inverse_transform(A_te[:, gi : gi + 1]).ravel()
    uc, cc = np.linspace(0, 1, 21), np.linspace(0, 80, 7)
    Ug, Cg = np.meshgrid(uc, cc, indexing="ij")
    r_shape, c_norm = ablation.hyperbolic(Ug.ravel(), ablation.HYPER_K).reshape(Ug.shape), Cg / 80.0
    target_frac = 0.15 * (c_norm - r_shape)
    n = len(A_te)
    rows = np.repeat(A_te, Ug.size, 0)
    rows[:, ji] = ins_s.transform((np.tile(Ug.ravel(), n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    rows[:, ci] = cho_s.transform(np.tile(Cg.ravel(), n).reshape(-1, 1)).ravel()
    pred_frac = ((np.repeat(cgm0, Ug.size) + tgt_s.inverse_transform(c["m4"].predict(rows).reshape(-1, 1)).ravel())
                 / np.repeat(f0_te, Ug.size) - 1.0).reshape(n, *Ug.shape).mean(0)
    ax4.plot_surface(Ug, Cg, target_frac, color="#37474F", alpha=0.55, linewidth=0)
    ax4.plot_wireframe(Ug, Cg, pred_frac, color="#00897B", lw=0.9, alpha=0.95)
    ax4.set(xlabel="Insulin (norm.)", ylabel="CHO (g)", zlabel="Fractional future-CGM change",
            title="4. Insulin (hyperbolic) $\\times$ CHO (linear) surface")
    ax4.view_init(elev=22, azim=-58)
    f0_lo = int(ablation.FULL_RANGE_F0_LO)
    _ablation_plot_triplet(fig.add_subplot(gs[2, :]), u, _ablation_drc(None, A_full, f0_a=f0_full, amp=ablation.NOMINAL_DROP, **kw),
                           _ablation_drc(baseline, A_full, **kw), _ablation_drc(c["m5"], A_full, **kw),
                           rf"5. Full-range alignment ($f_0>{f0_lo}$ mg/dL, incl.\ normoglycaemia)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ablation_flexibility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _synth_cloud(patient_id: str = TABLE2_PATIENT, horizon_min: int = 30, *, max_real=2500, max_synth=4000):
    body_weight_kg = load_patient_meta(str(DATA_ROOT), patient_id)["weight_kg"]
    if not ohio_cohort_path().is_file():
        rng = np.random.default_rng(0)
        insulin_u = rng.uniform(0, FULL_CORRECTION_U, max_real)
        future_real = 220 - 4 * insulin_u + rng.normal(0, 18, max_real)
        u = np.linspace(0, 1, 41)
        insulin_syn = np.tile(u * FULL_CORRECTION_U, 80)
        future_syn = np.repeat(np.linspace(180, 260, 80), 41) * (
            1.0 - replaybg_response(np.tile(u, 80), horizon_min, body_weight_kg)
        )
        return insulin_u, future_real, insulin_syn, future_syn, patient_id, {"f0_lo": 170.0, "f0_hi": 280.0}
    row = next(r for r in ohio_horizon_payload(json.loads(ohio_cohort_path().read_text()), horizon_min)["per_patient"]
               if r["patient_id"] == patient_id)
    data = load_patient(str(DATA_ROOT), patient_id, horizon_steps=horizon_steps(horizon_min))
    baseline = load_baseline_model(horizon_min, patient_id)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    ins_s, cgm_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    X, y = np.asarray(data.X_train, float), np.asarray(data.y_train, float).ravel()
    f0 = _future(baseline, X, ji, gi, (0.0,), FULL_CORRECTION_U, ins_s, tgt_s, cgm_s).ravel()
    cgm = cgm_s.inverse_transform(X[:, gi : gi + 1]).ravel()
    insulin_u = ins_s.inverse_transform(X[:, ji : ji + 1]).ravel()
    future_real = cgm + tgt_s.inverse_transform(y.reshape(-1, 1)).ravel()
    p = dict(n_anchors=int(row.get("n_anchors") or 0), f0_lo=float(row.get("f0_lo") or F0_LO_NOM),
             f0_hi=float(row.get("f0_hi") or DOCAS.Y0_HI))
    pool_m = (f0 > p["f0_lo"]) & (f0 < p["f0_hi"])
    pool, f0p = X[pool_m], f0[pool_m]
    n_a = min(len(pool), max(1, int(p["n_anchors"]) or len(pool)))
    idx = np.argsort(f0p)[np.linspace(0, len(f0p) - 1, n_a).round().astype(int)]
    A, f0_a = pool[idx], f0p[idx]
    u = DOCAS.U_GRID
    insulin_syn = (
        np.repeat(ins_s.inverse_transform(A[:, ji : ji + 1]).ravel(), len(u))
        + np.tile(u * FULL_CORRECTION_U, len(A))
    )
    future_syn = np.repeat(f0_a, len(u)) * (
        1.0 - np.tile(replaybg_response(u, horizon_min, body_weight_kg), len(A))
    )
    rng2 = np.random.default_rng(0)
    if len(insulin_u) > max_real:
        ir = rng2.choice(len(insulin_u), max_real, replace=False)
        insulin_u, future_real = insulin_u[ir], future_real[ir]
    if len(insulin_syn) > max_synth:
        is_ = rng2.choice(len(insulin_syn), max_synth, replace=False)
        insulin_syn, future_syn = insulin_syn[is_], future_syn[is_]
    return insulin_u, future_real, insulin_syn, future_syn, patient_id, p


def _insulin_x(split: dict, n: int) -> np.ndarray:
    return FULL_CORRECTION_U * (
        np.asarray(split["insulin_x"], float)
        if "insulin_x" in split else np.linspace(0, 1, n)
    )


def _mean_curve(patients: list, arm: str) -> np.ndarray:
    ys = []
    for row in patients:
        split = row["curves"]["test"]
        x, y = _insulin_x(split, len(split[arm])), np.asarray(split[arm], float)
        order = np.argsort(x)
        ys.append(np.interp(GRID * FULL_CORRECTION_U, x[order], y[order]))
    return np.mean(ys, axis=0)


def _plot_drc_panel(ax, patients: list, *, title: str) -> None:
    for arm, color, ls, label in (("baseline", BASELINE_COLOR, "--", "Baseline"), ("docas", DOCAS_COLOR, "-", "DOCAS")):
        for row in patients:
            split = row["curves"]["test"]
            x, y = _insulin_x(split, len(split[arm])), np.asarray(split[arm], float)
            order = np.argsort(x)
            ax.plot(x[order], y[order], color=color, alpha=0.18, lw=1.0, ls=ls, zorder=1)
        ax.plot(GRID * FULL_CORRECTION_U, _mean_curve(patients, arm),
                color=color, lw=2.8, ls=ls, zorder=3, label=label)
    ax.axhline(GLUCOSE_TARGET_MGDL, color="#37474F", ls=":", lw=1.0, alpha=0.7)
    ax.set_title(title, pad=10)
    ax.set(xlim=(0, FULL_CORRECTION_U))
    ax.grid(alpha=0.18)


def dose_response_figure() -> None:
    cohort = json.loads(ohio_cohort_path().read_text())
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.32, top=0.94, bottom=0.10, left=0.12, right=0.96)
    for ax, ph in zip(axes, FORECAST_HORIZONS):
        _plot_drc_panel(ax, ohio_horizon_payload(cohort, ph)["per_patient"], title=f"PH{ph}")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, frameon=False, fontsize=9, loc="upper right", borderaxespad=0.6)
    for ax in axes:
        ax.set_ylabel("Predicted future CGM (mg/dL)")
    axes[-1].set_xlabel("Additional correction insulin (0--10 U grid)")
    _save_fig(fig, "dose_response_curves")


def target_function_figure() -> None:
    """ReplayBG-derived τ at f0=220 mg/dL (PH30/PH60) + fractional response Rh."""
    u = np.linspace(0.0, 1.0, 101)
    dose_u = u * FULL_CORRECTION_U
    f0 = 220.0
    colors = {30: "#EF6C00", 60: "#C62828"}
    weights = (50.0, 75.0, 100.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axes[0]
    for ph, color in colors.items():
        r = replaybg_response(u, ph, 75.0)
        ax.plot(dose_u, f0 * (1.0 - r), color=color, lw=2.4, label=rf"PH{ph}")
        lo = f0 * (1.0 - replaybg_response(u, ph, weights[0]))
        hi = f0 * (1.0 - replaybg_response(u, ph, weights[-1]))
        ax.fill_between(dose_u, np.minimum(lo, hi), np.maximum(lo, hi), color=color, alpha=0.12)
    ax.axhline(180.0, color="#78909C", ls=":", lw=1.0, alpha=0.85)
    ax.set(xlim=(0, FULL_CORRECTION_U), ylim=(140, 225),
           xlabel=r"Additional correction insulin (U)",
           ylabel=r"Target future CGM $\tau$ (mg/dL)",
           title=r"$\tau(x,u)=\hat{g}_b(x)\,[1-R_h(u;w)]$ at $\hat{g}_b=220$")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.18)
    ax.text(0.98, 0.04, r"$w=75$ kg (band: 50–100 kg)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#546E7A")

    ax = axes[1]
    for ph, color in colors.items():
        ax.plot(dose_u, replaybg_response(u, ph, 75.0), color=color, lw=2.4, label=rf"PH{ph}")
    ax.set(xlim=(0, FULL_CORRECTION_U), ylim=(0, 0.4),
           xlabel=r"Additional correction insulin (U)",
           ylabel=r"Fractional lowering $R_h(u;w)$",
           title=r"ReplayBG incremental response ($w=75$ kg)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.18)
    r30 = float(replaybg_response(1.0, 30, 75.0))
    r60 = float(replaybg_response(1.0, 60, 75.0))
    ax.annotate(rf"$R_{{30}}(1)={r30:.3f}$", xy=(FULL_CORRECTION_U, r30),
                xytext=(6.2, r30 + 0.05), fontsize=8, color=colors[30],
                arrowprops={"arrowstyle": "->", "color": colors[30], "lw": 0.9})
    ax.annotate(rf"$R_{{60}}(1)={r60:.3f}$", xy=(FULL_CORRECTION_U, r60),
                xytext=(5.5, r60 - 0.08), fontsize=8, color=colors[60],
                arrowprops={"arrowstyle": "->", "color": colors[60], "lw": 0.9})
    _save_fig(fig, "target_function")


def _replay_row(ph: int, patient: str, source_start: int | None = None) -> dict:
    rows = [r for r in replay_horizon_payload(json.loads(replay_cohort_path().read_text()), ph)["windows"]
            if r["patient_id"] == patient]
    if source_start is not None:
        matched = [r for r in rows if r["source_start"] == source_start]
        if matched:
            return matched[0]
    eligible = [r for r in rows if r["twin_rmse"] <= TWIN_RMSE_PRIMARY_MGDL]
    return max(eligible or rows, key=lambda r: (r["delta_tir"], -r["delta_tbr"]))


def _first_decision(row: dict, arm: str) -> dict:
    return row[f"{arm}_decisions"][0]


def _plot_replay_traj(ax, row: dict, *, show_legend: bool) -> None:
    base_dec, doc_dec = _first_decision(row, "baseline"), _first_decision(row, "docas")
    traj = row["trajectories"]
    hours = np.arange(len(traj["observed"])) * YTS_MIN / 60.0
    decision_h = base_dec["time_min"] / 60.0
    ax.axhspan(TIR_LO, TIR_HI, color="#90CAF9", alpha=0.14, zorder=0)
    ax.axhline(TIR_LO, color="#78909C", ls="--", lw=0.9)
    ax.axhline(TIR_HI, color="#78909C", ls="--", lw=0.9)
    ax.axvline(decision_h, color="#37474F", ls=":", lw=1.2, alpha=0.85, zorder=2)
    ax.plot(hours, traj["observed"], color="#263238", ls=":", lw=2.0, label="Observed CGM")
    ax.plot(hours, traj["no_dss"], color="#8E24AA", ls="-.", lw=1.8, label=f"No DS twin (RMSE={row['twin_rmse']:.1f})")
    ax.plot(hours, traj["baseline_dss"], color=BASELINE_COLOR, ls="--", lw=2.2, label="Baseline DSS")
    ax.plot(hours, traj["docas_dss"], color=DOCAS_COLOR, ls="-", lw=2.6, label="DOCAS DSS")
    ax.scatter([decision_h], [base_dec["simulated_cgm"]], color=BASELINE_COLOR, s=42, zorder=4)
    ax.scatter([decision_h], [doc_dec["simulated_cgm"]], color=DOCAS_COLOR, s=42, zorder=4)
    y_top = max(float(np.max(traj[k])) for k in ("observed", "no_dss", "baseline_dss", "docas_dss"))
    trigger_cgm = max(base_dec["simulated_cgm"], doc_dec["simulated_cgm"])
    label_y = trigger_cgm + 14
    y_top = max(y_top, label_y + 10)
    ax.annotate(
        f"CIB trigger\n({base_dec['delivered_dose_u']:.1f} vs {doc_dec['delivered_dose_u']:.1f} U)",
        xy=(decision_h, trigger_cgm), xytext=(decision_h, label_y), fontsize=8.0, ha="center", va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#37474F", "linewidth": 0.9, "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#37474F", "shrinkA": 2, "shrinkB": 4},
    )
    ax.set(xlim=(0, REPLAY_WINDOW_H), ylim=(0, y_top), xlabel="Hours from meal", ylabel="CGM (mg/dL)")
    ax.set_title(f"PH{row['horizon_min']}", fontsize=10, pad=6)
    ax.grid(alpha=0.18)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)


def _plot_replay_drc(ax, row: dict, *, show_legend: bool) -> None:
    ph = row["horizon_min"]
    base_dec, doc_dec = _first_decision(row, "baseline"), _first_decision(row, "docas")
    for dec, color, ls, label in ((base_dec, BASELINE_COLOR, "--", "Baseline model"),
                                  (doc_dec, DOCAS_COLOR, "-", "DOCAS model")):
        x, y = np.asarray(dec["grid_u"], float), np.asarray(dec["future_cgm_curve"], float)
        ax.plot(x, y, color=color, lw=2.2, ls=ls, label=label, zorder=3)
        if dec["delivered_dose_u"] > 0:
            ax.scatter([dec["delivered_dose_u"]], [np.interp(dec["delivered_dose_u"], x, y)],
                       color=color, s=50, zorder=4, edgecolor="white", linewidth=0.8)
    ax.axhline(CIB_TARGET_MGDL, color="#37474F", ls=":", lw=1.0, alpha=0.75, label=rf"CIB target ({int(CIB_TARGET_MGDL)} mg/dL)")
    ax.axhline(HYPER_TRIGGER_MGDL, color="#78909C", ls="--", lw=0.9, alpha=0.8, label=rf"Trigger ({int(HYPER_TRIGGER_MGDL)} mg/dL)")
    ax.set(xlim=(0, max(base_dec["grid_u"])), xlabel="Additional correction insulin (U)",
           ylabel=rf"PH{ph} predicted future CGM (mg/dL)")
    ax.set_title(f"PH{ph} audit curve", fontsize=10, pad=6)
    ax.grid(alpha=0.18)
    if show_legend:
        ax.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.02, 0.02), borderaxespad=0.0)


def patient588_window_figure() -> None:
    row30 = _replay_row(30, TABLE2_PATIENT)
    row60 = _replay_row(60, TABLE2_PATIENT, row30["source_start"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharex="col")
    fig.subplots_adjust(hspace=0.48, wspace=0.22, top=0.94, bottom=0.08, left=0.08, right=0.98)
    for col, row in enumerate((row30, row60)):
        _plot_replay_traj(axes[0, col], row, show_legend=(col == 0))
        _plot_replay_drc(axes[1, col], row, show_legend=(col == 0))
    _save_fig(fig, "patient588_replay_window")


class FlowchartData(NamedTuple):
    insulin_u: np.ndarray
    f0_single: float
    target_single: np.ndarray
    target_points_u: np.ndarray
    target_points_y: np.ndarray
    anchor_baseline: np.ndarray
    anchor_target: np.ndarray
    baseline_curve: np.ndarray
    docas_curve: np.ndarray
    docas_aligned: np.ndarray
    eval_target: np.ndarray


def _load_flowchart_data(patient_id: str = FLOWCHART_PATIENT, horizon_min: int = FLOWCHART_HORIZON) -> FlowchartData:
    path = ohio_cohort_path()
    if not path.is_file():
        return _synthetic_flowchart_data()
    cohort = json.loads(path.read_text())
    row = next(r for r in ohio_horizon_payload(cohort, horizon_min)["per_patient"] if r["patient_id"] == patient_id)
    curves = row["curves"]["test"]
    insulin_x = np.asarray(curves["insulin_x"], float)
    span_u = float(cohort["insulin_span_u"])
    insulin_u = insulin_x * span_u
    baseline_curve = np.asarray(curves["baseline"], float)
    docas_curve = np.asarray(curves["docas"], float)
    body_weight_kg = float(row.get(
        "body_weight_kg",
        load_patient_meta(str(DATA_ROOT), patient_id)["weight_kg"],
    ))
    data = load_patient(str(DATA_ROOT), patient_id, horizon_steps=horizon_steps(horizon_min))
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    ins_s, cgm_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    baseline = load_baseline_model(horizon_min, patient_id)
    f0_all = _future(baseline, data.X_test, ji, gi, (0.0,), span_u, ins_s, tgt_s, cgm_s).ravel()
    Xh, f0h = data.X_test[f0_all > ALIGN_TRIGGER_MGDL], f0_all[f0_all > ALIGN_TRIGGER_MGDL]
    order = np.argsort(f0h)
    sel = order[np.linspace(0, len(order) - 1, FLOWCHART_N_ANCHORS).round().astype(int)]
    anchors, f0_a = Xh[sel], f0h[sel]
    u = np.asarray(DOCAS.U_GRID, float)
    shape = replaybg_response(u, horizon_min, body_weight_kg)
    anchor_baseline = _future(baseline, anchors, ji, gi, u, span_u, ins_s, tgt_s, cgm_s)
    anchor_target = f0_a[:, None] * (1.0 - shape[None, :])
    f0_single = float(np.median(f0_a))
    target_single = f0_single * (1.0 - shape)
    pts_u_norm = np.linspace(0.0, 1.0, 11)
    target_points_u = pts_u_norm * span_u
    target_points_y = f0_single * (
        1.0 - replaybg_response(pts_u_norm, horizon_min, body_weight_kg)
    )
    eval_target = float(docas_curve[0]) * (
        1.0 - replaybg_response(insulin_x, horizon_min, body_weight_kg)
    )
    k = np.ones(5) / 5.0
    smooth = np.convolve(np.pad(docas_curve, 2, mode="edge"), k, mode="valid")
    docas_aligned = eval_target + 0.25 * (smooth - eval_target)
    return FlowchartData(insulin_u, f0_single, target_single, target_points_u, target_points_y,
                         anchor_baseline, anchor_target, baseline_curve, docas_curve, docas_aligned, eval_target)


def _synthetic_flowchart_data() -> FlowchartData:
    insulin_x = np.linspace(0.0, 1.0, 41)
    insulin_u = insulin_x * FULL_CORRECTION_U
    f0 = 220.0
    target = f0 * (1.0 - replaybg_response(insulin_x, FLOWCHART_HORIZON))
    baseline = f0 - 8.0 * insulin_x + 4.0 * np.sin(3.0 * insulin_x)
    docas = target + 0.15 * (baseline - target)
    pts_u = np.linspace(0.0, 1.0, 11)
    return FlowchartData(
        insulin_u, f0, target, pts_u * FULL_CORRECTION_U,
        f0 * (1.0 - replaybg_response(pts_u, FLOWCHART_HORIZON)),
        np.tile(baseline, (FLOWCHART_N_ANCHORS, 1)), np.tile(target, (FLOWCHART_N_ANCHORS, 1)),
        baseline, docas, 0.85 * target + 0.15 * docas, target,
    )


def _fig_to_image(fig, dpi: int = 150) -> np.ndarray:
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


def _flowchart_tree_image(insulin_color: str, leaf_color: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    root_x, root_y = 0.5, 0.86
    ax.add_patch(FancyBboxPatch((root_x - 0.16, root_y - 0.09), 0.32, 0.18, boxstyle="round,pad=0.01",
                                facecolor=FC_CGM, edgecolor="#333", linewidth=1.5, zorder=3))
    ax.text(root_x, root_y, "CGM < c", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=4)
    for bx, label, c in ((0.25, "insulin", insulin_color), (0.75, "CHO", FC_INSULIN)):
        ax.add_patch(FancyBboxPatch((bx - 0.14, 0.48 - 0.09), 0.28, 0.18, boxstyle="round,pad=0.01",
                                    facecolor=c, edgecolor="#333", linewidth=1.5, zorder=3))
        ax.text(bx, 0.48, label, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
    for lx, lab in ((0.15, "dg1"), (0.35, "dg2"), (0.65, "dg3"), (0.85, "dg4")):
        ax.add_patch(FancyBboxPatch((lx - 0.08, 0.14 - 0.07), 0.16, 0.14, boxstyle="round,pad=0.01",
                                    facecolor=leaf_color, edgecolor="#333", linewidth=1.5, zorder=3))
        ax.text(lx, 0.14, lab, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
    ax.plot([root_x, 0.25], [root_y - 0.08, 0.56], "k-", lw=2, zorder=1)
    ax.plot([root_x, 0.75], [root_y - 0.08, 0.56], "k-", lw=2, zorder=1)
    ax.plot([0.25, 0.15], [0.40, 0.21], "k-", lw=1.5, zorder=1)
    ax.plot([0.25, 0.35], [0.40, 0.21], "k-", lw=1.5, zorder=1)
    ax.plot([0.75, 0.65], [0.40, 0.21], "k-", lw=1.5, zorder=1)
    ax.plot([0.75, 0.85], [0.40, 0.21], "k-", lw=1.5, zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return _fig_to_image(fig)


def _icon_target_tau() -> np.ndarray:
    f0 = 220.0
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    for ph, color in ((30, "#EF6C00"), (60, "#C62828")):
        ax.plot(
            GRID,
            f0 * (1.0 - replaybg_response(GRID, ph)),
            color=color,
            lw=2.4,
            label=rf"PH{ph}",
        )
    ax.scatter([0], [f0], color="#37474F", s=28, zorder=3)
    ax.axhline(180, color="#78909C", ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper right", handlelength=1.2)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("insulin (frac.)", fontsize=10)
    ax.set_ylabel("target CGM", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    return _fig_to_image(fig)


def _icon_synth_cloud() -> np.ndarray:
    iu, fr, isyn, fsyn, _, _ = _synth_cloud()
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    ax.scatter(iu, fr, s=6, alpha=0.35, color="#607D8B", label="observational", rasterized=True, zorder=2)
    ax.scatter(isyn, fsyn, s=6, alpha=0.45, color=DOCAS_COLOR, label="synthetic", rasterized=True, zorder=3)
    ax.axhline(180, color="#78909C", ls=":", lw=1.0, alpha=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper right", markerscale=2.0, handlelength=1.0)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("insulin (U)", fontsize=10)
    ax.set_ylabel("future CGM", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    return _fig_to_image(fig)


def _build_flowchart_icons(d: FlowchartData) -> dict[str, np.ndarray]:
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    ax.plot(d.insulin_u, d.target_single, color=FC_TARGET, lw=2.8)
    ax.fill_between(d.insulin_u, d.target_single, d.f0_single, alpha=0.20, color=FC_TARGET)
    ax.axhline(ALIGN_TRIGGER_MGDL, color=FC_HYPER, ls="--", lw=3.2)
    ax.text(d.insulin_u[1], ALIGN_TRIGGER_MGDL + 3, f"hyperglycaemic anchor (future CGM > {F0_LO_NOM:.0f})", color=FC_HYPER,
            fontsize=13, ha="left", va="bottom", fontweight="bold")
    ax.scatter([0], [d.f0_single], color="#37474F", s=30, zorder=4)
    ax.set_xlim(0, d.insulin_u[-1])
    y0 = min(d.target_single.min(), ALIGN_TRIGGER_MGDL)
    ax.set_ylim(y0 - 6, d.f0_single + 6)
    ax.axis("off")
    deftarget = _fig_to_image(fig)
    rng = np.random.default_rng(0)
    theta = np.linspace(0.0, 1.0, 200)
    surrogate = 0.9 * (theta - 0.62) ** 2 + 0.04
    trials_x = np.array([0.08, 0.2, 0.34, 0.5, 0.58, 0.63, 0.66, 0.8, 0.92])
    trials_y = 0.9 * (trials_x - 0.62) ** 2 + 0.04 + rng.normal(0, 0.012, trials_x.size)
    band = 0.05 * (1.0 - np.exp(-8 * (theta - 0.62) ** 2)) + 0.012
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    ax.fill_between(theta, surrogate - band, surrogate + band, color=FC_PROCESS, alpha=0.25)
    ax.plot(theta, surrogate, color=FC_LOOP_EDGE, lw=2.4)
    ax.scatter(trials_x, trials_y, s=46, color=FC_TARGET, zorder=4, edgecolors="white", linewidth=0.8)
    ax.set(xlim=(0, 1), ylim=(0, float(trials_y.max()) * 1.15))
    ax.set_xlabel("target parameters", fontsize=12)
    ax.set_ylabel("val. RMSE (+ priors)", fontsize=11)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    synth = _fig_to_image(fig)
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    ax.plot(d.insulin_u, d.eval_target, color=FC_TARGET, lw=2.2, ls="--", label="target audit curve")
    ax.plot(d.insulin_u, d.baseline_curve, color=BASELINE_COLOR, lw=2.0, label="baseline audit curve")
    ax.plot(d.insulin_u, d.docas_aligned, color=DOCAS_COLOR, lw=2.4, label="DOCAS audit curve")
    ax.set_xlim(0, d.insulin_u[-1])
    ys = np.concatenate([d.eval_target, d.baseline_curve, d.docas_aligned])
    pad = 0.08 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.axis("off")
    evaluate = _fig_to_image(fig)
    fig, ax = plt.subplots(figsize=FLOWCHART_ICON_SIZE)
    ax.plot(d.insulin_u, d.eval_target, color=FC_TARGET, lw=2.4, ls="--", label="target audit curve")
    ax.plot(d.insulin_u, d.docas_aligned, color=DOCAS_COLOR, lw=2.8, label="aligned audit curve")
    ax.fill_between(d.insulin_u, d.docas_aligned, d.eval_target, color=DOCAS_COLOR, alpha=0.12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim(0, d.insulin_u[-1])
    ys = np.concatenate([d.eval_target, d.docas_aligned])
    pad = 0.10 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.axis("off")
    output = _fig_to_image(fig)
    return {"baseline": _flowchart_tree_image(BASELINE_COLOR, FC_OUTPUT), "deftarget": deftarget, "propose": synth,
            "target_tau": _icon_target_tau(), "augment": _icon_synth_cloud(), "evaluate": evaluate, "output": output}


def _fc_box(ax, x, y, w, h, title, color, subtext, img, add_img, title_fs, sub_fs, *, stacked=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=1",
                                facecolor=color, edgecolor="#333", linewidth=2, zorder=2))
    if stacked:
        ax.text(x, y + h * 0.42, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color="#222", zorder=4, linespacing=1.0)
        img_w, img_h = w * 0.96, h * 0.58
        add_img(img, x, y - h * 0.02, img_w, img_h)
        ax.text(x, y - h * 0.42, subtext, ha="center", va="center", fontsize=sub_fs, color="#444", zorder=4, linespacing=1.02)
        return y + h * 0.42
    img_w, img_h = 26, h * 0.82
    add_img(img, x - w / 2 + img_w / 2 + 2, y, img_w, img_h)
    tx = x + w * 0.28
    ax.text(tx, y + h * 0.28, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#222", zorder=4, linespacing=1.05)
    ax.text(tx, y - h * 0.20, subtext, ha="center", va="center", fontsize=sub_fs, color="#444", zorder=4, linespacing=1.15)
    return None


def docas_flowchart_figure() -> Path:
    icons = _build_flowchart_icons(_load_flowchart_data())
    fig, ax = plt.subplots(figsize=(28, 17))
    cx, W, H, W4, H4, V_GAP = 70, 56, 15, 42.5, 40, 1.8
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    col_l, col_r = cx - 36, cx + 36
    gap, span = 6.0, 4 * W4 + 3 * 6.0
    x0 = cx - span / 2 + W4 / 2
    xs = [x0 + i * (W4 + gap) for i in range(4)]
    in_l, in_t, in_a, in_r = xs
    y_top = 108
    loop_top = y_top - H / 2 - V_GAP
    row_y = loop_top - 3.2 - H4 / 2
    hook_y = row_y - H4 / 2 - 2.2
    loop_bottom = hook_y - 3.2
    y_out = loop_bottom - 2.8 - H / 2
    half = (in_r + W4 / 2) - cx + 0.6
    outer_l, outer_r = cx - half - 0.8, cx + half + 0.8
    outer_t, outer_b = y_top + H / 2 + 0.5, y_out - H / 2 - 0.5
    ax.set_xlim(outer_l - 0.15, outer_r + 0.15)
    ax.set_ylim(outer_b - 0.15, outer_t + 0.15)

    def add_img(img, x, y, w, h):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xr, yr = xl[1] - xl[0], yl[1] - yl[0]
        ins = ax.inset_axes([(x - w / 2 - xl[0]) / xr, (y - h / 2 - yl[0]) / yr, w / xr, h / yr], transform=ax.transAxes)
        ins.imshow(img)
        ins.axis("off")

    def add_arrow(x1, y1, x2, y2, color="#006064", lw=FLOWCHART_ARROW_LW):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2, mutation_scale=40), zorder=6)

    _fc_box(ax, col_l, y_top, W, H, "Train Baseline\nModel", FC_INPUT,
            "Fit any user-supplied\nmodel $f_b$ on the\nobserved dataset", icons["baseline"], add_img, 26, 19)
    _fc_box(ax, col_r, y_top, W, H, "Define Target\nFunction", FC_INPUT,
            "Hyperglycaemic anchors only;\nset interventional audit-curve\ntarget ($do$-style sweep)", icons["deftarget"], add_img, 26, 19)
    ax.add_patch(FancyBboxPatch((outer_l + 0.35, loop_bottom), (outer_r - outer_l) - 0.7, loop_top - loop_bottom,
                                boxstyle="round,pad=0.15,rounding_size=0.9",
                                facecolor=FC_LOOP, edgecolor=FC_LOOP_EDGE, linewidth=2.5, zorder=0, alpha=0.75))
    ax.text(cx, loop_top - 0.3,
            "DOCAS calibration loop: tune the target so the audit curve aligns (not attributions)",
            fontsize=23, color=FC_LOOP_TEXT, ha="center", va="top", fontweight="bold")
    add_arrow(col_l, y_top - H / 2, col_l, loop_top)
    add_arrow(col_r, y_top - H / 2, col_r, loop_top)
    title_fs, sub_fs = 22, 17
    title_y = _fc_box(ax, in_l, row_y, W4, H4, "Fixed Target\nParameters", FC_PROCESS,
                      "ReplayBG $R_h(u;w)$\n(no amplitude search)", icons["propose"], add_img, title_fs, sub_fs, stacked=True)
    _fc_box(ax, in_t, row_y, W4, H4, "Target\nFunction", FC_PROCESS,
            "ReplayBG insulin dynamics;\nPH30/PH60 kinetic response", icons["target_tau"], add_img, title_fs, sub_fs, stacked=True)
    _fc_box(ax, in_a, row_y, W4, H4, "Augment &\nTrain", FC_PROCESS,
            "append synthetic rows\nto observational data; train", icons["augment"], add_img, title_fs, sub_fs, stacked=True)
    _fc_box(ax, in_r, row_y, W4, H4, "Score Single\nObjective", FC_PROCESS,
            "minimise val. RMSE subject to\nalignment constraint", icons["evaluate"], add_img, title_fs, sub_fs, stacked=True)
    for a, b in zip(xs, xs[1:]):
        ax.annotate("", xy=(b - W4 / 2 + 1.2, title_y), xytext=(a + W4 / 2 - 1.2, title_y),
                    arrowprops=dict(arrowstyle="-|>", color="#004D40", lw=FLOWCHART_ARROW_LW,
                                    shrinkA=0, shrinkB=0, mutation_scale=44, connectionstyle="arc3,rad=0"), zorder=8)
    verts = [(in_r, row_y - H4 / 2), (in_r, hook_y), (in_l, hook_y), (in_l, row_y - H4 / 2)]
    ret = mpl_path.Path(verts, [mpl_path.Path.MOVETO] + [mpl_path.Path.LINETO] * 3)
    ax.add_patch(FancyArrowPatch(path=ret, arrowstyle="->", color=FC_LOOP_EDGE, lw=FLOWCHART_ARROW_LW,
                                 mutation_scale=36, shrinkA=0, shrinkB=0, zorder=5))
    ax.text(cx, hook_y - 0.9, r"fixed ReplayBG $\tau=\hat g_b[1-R_h(u;w)]$ "
            r"(prior-mode $R_{30}(1)\!\approx\!0.04$, $R_{60}(1)\!\approx\!0.35$)",
            fontsize=19, color=FC_LOOP_TEXT, ha="center", va="top", fontweight="bold")
    _fc_box(ax, cx, y_out, W, H, "Aligned DOCAS\nModel", FC_OUTPUT,
            "interventional audit curve\naligned for bolus DSS\n(not feature attributions)", icons["output"], add_img, 25, 18)
    add_arrow(cx, loop_bottom, cx, y_out + H / 2, color=FC_ARROW_EXIT)
    ax.add_patch(FancyBboxPatch((outer_l, outer_b), outer_r - outer_l, outer_t - outer_b,
                                boxstyle="round,pad=0.1,rounding_size=1.6",
                                facecolor="#FAFAFA", edgecolor="#607D8B", linewidth=2.5, zorder=-2))
    return _save_fig(fig, "docas_flowchart")


def build_manuscript() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    table1_tex()
    table2_tex()
    if ABLATION_JSON.is_file():
        table3_tex()
    target_function_figure()
    dose_response_figure()
    patient588_window_figure()
    docas_flowchart_figure()
    if ablation._MODEL_CACHE:
        ablation_flexibility_figure()


if __name__ == "__main__":
    build_manuscript()
