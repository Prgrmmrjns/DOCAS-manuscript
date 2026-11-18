"""Manuscript figures (PNG). Import study symbols from main."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from preprocessing import horizon_steps

import main as m
from core import (
    DOCAS, abs_tau_drop, frac_tau_drop, calibrated_target, docas_fit_kw,
    future as _future, replaybg_drop_trace, replaybg_response, sample_synthetic,
    sync_cfg,
)
import replay_eval as replay_study

ROOT = m.ROOT
FIGURES = ROOT / "manuscript" / "figures"
TARGET_COLOR, BASELINE_COLOR, DOCAS_COLOR = "#5E35B1", "#D32F2F", "#00897B"
TIR_LO, TIR_HI, GRID = 70.0, 180.0, np.linspace(0, 1, 21)
FLOWCHART_PATIENT, FLOWCHART_HORIZON = "588", 60
REPLAY_SHOWCASE_PH = replay_study.REPLAY_HORIZONS[0]
FC_HYPER = "#FF7043"
TABLE2_PATIENT = replay_study.TABLE2_PATIENT
YTS_MIN = replay_study.YTS_MIN
REPLAY_WINDOW_H = replay_study.REPLAY_WINDOW_H
TWIN_RMSE_PRIMARY_MGDL = replay_study.TWIN_RMSE_PRIMARY_MGDL
TWIN_RMSE_PRIMARY_MGDL = TWIN_RMSE_PRIMARY_MGDL
ALIGN_TRIGGER_MGDL = 170.0  # flowchart annotation only
FORECAST_HORIZONS = m.FORECAST_HORIZONS
DATA_ROOT = m.DATA_ROOT
load_patient = m.load_patient
load_baseline_model = m.load_baseline_model
ohio_cohort_path = m.ohio_cohort_path
ohio_horizon_payload = m.ohio_horizon_payload
replay_cohort_path = replay_study.replay_cohort_path
replay_horizon_payload = replay_study.replay_horizon_payload
insulin_span_u = m.insulin_span_u
TIR_LO, TIR_HI = TIR_LO, TIR_HI
REPLAY_WINDOW_H = REPLAY_WINDOW_H
HYPO_FLOOR_MGDL = getattr(replay_study, "HYPO_FLOOR_MGDL", getattr(replay_study, "HYPO_FLOOR_MGDL"))
HYPER_TRIGGER_MGDL = getattr(replay_study, "HYPER_TRIGGER_MGDL", getattr(replay_study, "HYPER_TRIGGER_MGDL"))


def _legend(ax, *args, **kwargs):
    """Upper-right legend with white framed box (manuscript-wide)."""
    kw = dict(
        loc="upper right", frameon=True, fancybox=False,
        facecolor="white", edgecolor="0.35", framealpha=1.0,
        fontsize=7.5, borderaxespad=0.4)
    kw.update(kwargs)
    leg = ax.legend(*args, **kw)
    fr = leg.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("0.35")
    fr.set_linewidth(0.8)
    fr.set_alpha(1.0)
    return leg


_legend = _legend


def _save_fig(fig, name: str, *, dest: Path | None = None) -> Path:
    folder = dest if dest is not None else FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


_save_fig = _save_fig


def _study_drop_curve(u_norm: np.ndarray, ph: int, *, span: float | None = None,
                      cgm0: float | None = None) -> np.ndarray:
    """Study DRC as ΔCGM (mg/dL) on u∈[0,1]; fractional mode scales by ambient cgm0."""
    cfg = m.STUDY_CFG
    sp = float(span if span is not None else (
        m.DOCAS.SPAN if m.DOCAS.SPAN is not None else 1.0))
    mode = str(cfg.get("tau_mode", "fractional")).lower()
    drop_mode = str(cfg.get("drop_mode", "endpoint"))
    kw = dict(body_weight_kg=float(cfg["body_weight_kg"]), SI=float(cfg["SI"]),
              span=sp, drop_mode=drop_mode)
    if mode.startswith("frac"):
        drop, u_ref = frac_tau_drop(
            int(ph), sp, si=float(cfg["SI"]), body_weight_kg=float(cfg["body_weight_kg"]),
            drop_mode=drop_mode)
        g0 = float(cgm0 if cgm0 is not None else m.G0_TAU)
        # Convert fractional edge drop to absolute Δ at ambient g0 for plotting.
        abs_drop = drop * g0
    else:
        drop, u_ref = abs_tau_drop(
            int(ph), sp, si=float(cfg["SI"]), body_weight_kg=float(cfg["body_weight_kg"]),
            drop_mode=drop_mode)
        abs_drop = drop
    u = np.asarray(u_norm, float).ravel()
    R = np.asarray(replaybg_response(u, int(ph), **kw), float).ravel()
    r_ref = float(replaybg_response(np.array([u_ref]), int(ph), **kw)[0]) or 1.0
    return abs_drop * (R / r_ref)


def _target_ice_at(ph: int, u_abs, cgm, span: float) -> np.ndarray:
    """Study target ICE (ΔCGM mg/dL) at each (IOB, ambient CGM)."""
    u = np.asarray(u_abs, float).ravel() / max(float(span), 1e-6)
    g = np.asarray(cgm, float).ravel()
    umax = float(max(1.0, np.nanmax(u))) if u.size else 1.0
    grid = np.linspace(0.0, umax, 129)
    unit = _study_drop_curve(grid, int(ph), span=float(span), cgm0=1.0)
    return np.interp(u, grid, unit) * g


_abs_drop_curve = _study_drop_curve


_MEAN_CGM: dict[tuple[str, int], float] = {}


def _mean_test_cgm(pid: str, ph: int) -> float:
    """Ambient CGM the audit target is scaled by, averaged over the test contexts.

    Curves plotted as a mean over contexts must be compared against τ at that
    same mean ambient level, not at the nominal G0.
    """
    key = (str(pid), int(ph))
    if key not in _MEAN_CGM:
        data = load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
        gi = data.feature_names.index("CGM")
        _MEAN_CGM[key] = float(np.mean(data.train_scalers[gi].inverse_transform(
            np.asarray(data.X_test, float)[:, gi: gi + 1]).ravel()))
    return _MEAN_CGM[key]


def _synth_future_cgm(Xs, ys, yi, y_s, dy_s, meta) -> np.ndarray:
    """Future CGM (mg/dL) of synthetic rows, honouring core's fit space.

    ``sample_synthetic`` emits level-scaled labels only when ``fit_space`` is
    level; the default delta space needs the row's own ambient CGM added back.
    """
    ys = np.asarray(ys, float).reshape(-1, 1)
    if bool(meta.get("predicts_level")):
        return y_s.inverse_transform(ys).ravel()
    cgm = y_s.inverse_transform(np.asarray(Xs, float)[:, yi : yi + 1]).ravel()
    return cgm + dy_s.inverse_transform(ys).ravel()


def _core_context_probes(data, *, target, drop, u_ref, tau_mode, feat_span, anchor,
                         n=15, ctx_jitter=0.0, emp_frac=1.0, seed=0):
    """Probes for one training context, generated by ``core.sample_synthetic``.

    IOB is resampled from the full training pool (``emp_frac=1``) so D/E show
    the observational dose density, not a uniform necklace along τ.
    """
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    ci = data.feature_names.index("CHO")
    ins_s, cgm_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    X = np.asarray(data.X_train, float)
    y = np.asarray(data.y_train, float).ravel()
    a = int(anchor)
    Xs, ys, meta = sample_synthetic(
        X, y, target=target, ji=ji, yi=gi, span=float(feat_span),
        a_s=ins_s, dy_s=tgt_s, y_s=cgm_s, rng=np.random.default_rng(seed),
        tau_drop=float(drop), u_ref=float(u_ref), tau_mode=str(tau_mode),
        synth_ratio=1.0, emp_frac=float(emp_frac), ctx_jitter=float(ctx_jitter),
        cho_idx=ci, cho_s=data.train_scalers[ci],
        anchors=np.full(int(n), a, dtype=int))
    probe = ins_s.inverse_transform(Xs[:, ji : ji + 1]).ravel()
    fut = _synth_future_cgm(Xs, ys, gi, cgm_s, tgt_s, meta)
    o = np.argsort(probe)
    return probe[o], fut[o]


def _synth_cloud(patient_id: str = TABLE2_PATIENT, horizon_min: int = 30, *,
                 max_real=2500, max_synth=4000, emp_frac=None):
    """Observational + study-τ synthetic cloud (replace probes to train-max IOB)."""
    data = load_patient(str(DATA_ROOT), patient_id, horizon_steps=horizon_steps(horizon_min))
    feat_span = float(insulin_span_u(data, horizon_min=horizon_min))
    DOCAS.SPAN = feat_span
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    ins_s, cgm_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    X, y = np.asarray(data.X_train, float), np.asarray(data.y_train, float).ravel()
    cgm = cgm_s.inverse_transform(X[:, gi : gi + 1]).ravel()
    insulin_u = ins_s.inverse_transform(X[:, ji : ji + 1]).ravel()
    future_real = cgm + tgt_s.inverse_transform(y.reshape(-1, 1)).ravel()
    cfg = m.STUDY_CFG
    fit_kw, drop, u_ref, _, _ = docas_fit_kw(data, feat_span, horizon_min, cfg=cfg)
    mode = str(fit_kw.get("tau_mode", "fractional"))
    Xs, ys, meta = sample_synthetic(
        X, y, target=fit_kw["target"], ji=ji, yi=gi, span=feat_span,
        a_s=ins_s, dy_s=tgt_s, y_s=cgm_s, rng=np.random.default_rng(0),
        tau_drop=drop, u_ref=u_ref, tau_mode=mode,
        synth_ratio=float(cfg["synth_ratio"]),
        emp_frac=float(cfg.get("emp_frac", 0.5) if emp_frac is None else emp_frac),
        ctx_jitter=float(cfg["ctx_jitter"]),
        cho_idx=data.feature_names.index("CHO"), cho_s=data.train_scalers[data.feature_names.index("CHO")],
    )
    insulin_syn = ins_s.inverse_transform(Xs[:, ji : ji + 1]).ravel()
    future_syn = _synth_future_cgm(Xs, ys, gi, cgm_s, tgt_s, meta)
    p = dict(
        n_anchors=int(meta["n_anchors"]),
        cgm=cgm, insulin=insulin_u, future=future_real,
        span=feat_span, drop=float(drop), u_ref=float(u_ref),
        target=fit_kw["target"], tau_mode=mode,
        u=np.asarray(meta["u"], float),
        anchor_idx=np.asarray(meta["anchor_idx"], int),
    )
    rng2 = np.random.default_rng(0)
    if len(insulin_u) > max_real:
        ir = rng2.choice(len(insulin_u), max_real, replace=False)
        insulin_u, future_real = insulin_u[ir], future_real[ir]
        p["plot_idx"] = ir
    else:
        p["plot_idx"] = np.arange(len(insulin_u))
    if len(insulin_syn) > max_synth:
        is_ = rng2.choice(len(insulin_syn), max_synth, replace=False)
        insulin_syn, future_syn = insulin_syn[is_], future_syn[is_]
        p["u"], p["anchor_idx"] = p["u"][is_], p["anchor_idx"][is_]
    return insulin_u, future_real, insulin_syn, future_syn, patient_id, p


def _insulin_x(split: dict, n: int) -> np.ndarray:
    return (np.asarray(split["insulin_x"], float)
            if "insulin_x" in split else np.linspace(0, 1, n))


def _delta_curve(split: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Audit curve as Δ vs u=0 ambient, on the stored insulin grid."""
    x = _insulin_x(split, len(split[arm]))
    y = np.asarray(split[arm], float)
    order = np.argsort(x)
    return x[order], y[order] - y[order][0]


def _abs_iob_curve(row: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    u, d = _delta_curve(row["curves"]["test"], arm)
    return u * float(row.get("insulin_span_u") or 1.0), d


def _ohio_context_curves() -> dict:
    """Per-patient context-DOCAS mean ICE on the same grid as the Ohio cohort figure."""
    path = ROOT / "results" / "revision" / "ohio_ctx_pdp.json"
    if path.is_file():
        return json.loads(path.read_text())
    import os
    from joblib import Parallel, delayed
    import ablation as cd
    from core import DOCAS, docas_fit_kw, sync_cfg, future
    from forecasters import train_baseline, CAT_KWARGS
    ohio = json.loads(ohio_cohort_path().read_text())

    def _one(ph: int, row: dict) -> tuple[str, int, list[float]]:
        CAT_KWARGS["thread_count"] = 1
        m.set_dataset("ohio_t1dm")
        pid = str(row["patient_id"])
        data = m.load_patient(str(m.DATA_ROOT), pid, horizon_steps=horizon_steps(ph))
        span = float(row["insulin_span_u"])
        u = np.asarray(row["curves"]["test"]["insulin_x"], float)
        DOCAS.SPAN, DOCAS.SEED = span, 42
        sync_cfg(SEED=42)
        fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=m.STUDY_CFG)
        base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
        _, R_tab = cd._pop_R(ph, span, m.STUDY_CFG)
        theta = cd._search_theta(data, span, ph, R_tab, seed=42)
        ctx, _ = cd._fit_ctx(data, span, ph, theta, R_tab, seed=42, base_mdl=base)
        ji = int(data.insulin_idx)
        gi = data.feature_names.index("CGM")
        X = np.asarray(data.X_test, float)
        curve = future(
            ctx, X, ji, gi, u, span,
            data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi],
            level=False,
        ).mean(0)
        return pid, ph, [round(float(v), 2) for v in curve]

    tasks = [
        (ph, row)
        for ph in FORECAST_HORIZONS
        for row in ohio_horizon_payload(ohio, ph)["per_patient"]
    ]
    got = Parallel(n_jobs=max(1, (os.cpu_count() or 4) - 1), backend="loky")(
        delayed(_one)(ph, row) for ph, row in tasks)
    out: dict = {}
    for pid, ph, curve in got:
        out.setdefault(str(ph), {})[pid] = curve
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


PDP_SOURCES = ("OhioT1DM", "AZT1D", "HUPA-UCM", "T1D-UOM")
COHORT_PDP_PATH = ROOT / "results" / "revision" / "cohort_pdps.json"


def _cohort_pdp_one(pid: str, ph: int) -> dict:
    """Mean held-out ICE for baseline and population DOCAS, same grid as Figure 3."""
    import forecasters
    from forecasters import train_baseline

    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    span = float(m.insulin_span_u(data, horizon_min=ph))
    u = np.linspace(0.0, 1.0, int(m.N_GRID_POINTS))
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=m.STUDY_CFG)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    docas = DOCAS(**{**fit_kw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    ji, gi = int(data.insulin_idx), data.feature_names.index("CGM")
    X = np.asarray(data.X_test, float)
    a_s, dy_s, y_s = data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]

    def _mean(model) -> list[float]:
        acc = np.zeros(len(u), float)
        n = 0
        for i in range(0, len(X), 400):
            sl = X[i:i + 400]
            acc += _future(
                model, sl, ji, gi, u, span, a_s, dy_s, y_s, level=False).sum(0)
            n += len(sl)
        return [round(float(v), 2) for v in acc / max(n, 1)]

    g = y_s.inverse_transform(X[:, gi:gi + 1]).ravel()
    return {
        "patient_id": str(pid),
        "source": str(pid).split("|", 1)[0],
        "insulin_span_u": float(span),
        "cgm": float(np.mean(g)),
        "curves": {"test": {
            "insulin_x": [round(float(v), 4) for v in u],
            "baseline": _mean(base),
            "docas": _mean(docas),
        }},
    }


def cohort_pdps(*, recompute: bool = False) -> dict:
    """Baseline and population-DOCAS PDPs for the 68 evaluated subjects."""
    if COHORT_PDP_PATH.is_file() and not recompute:
        return json.loads(COHORT_PDP_PATH.read_text())
    import os
    from joblib import Parallel, delayed

    pids = []
    for path in sorted((ROOT / "results" / "revision" / "full_recent").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("source") in PDP_SOURCES and "horizons" in row:
            pids.append(row["patient_id"])
    tasks = [(pid, ph) for ph in FORECAST_HORIZONS for pid in pids]
    got = Parallel(n_jobs=max(1, (os.cpu_count() or 4) - 1), backend="loky")(
        delayed(_cohort_pdp_one)(pid, ph) for pid, ph in tasks)
    out: dict = {}
    for (pid, ph), row in zip(tasks, got):
        out.setdefault(str(ph), []).append(row)
    COHORT_PDP_PATH.parent.mkdir(parents=True, exist_ok=True)
    COHORT_PDP_PATH.write_text(json.dumps(out))
    return out


def _plot_drc_panel(ax, patients: list, *, title: str, ph: int = 30,
                    relative: bool = False, scales: dict | None = None,
                    autoscale_y: bool = True) -> None:
    """Audit curves per patient; relative=True rescales ΔCGM to % of ambient CGM."""
    spans = [float(r.get("insulin_span_u") or 1.0) for r in patients]
    x_hi = float(max(spans)) if spans else 1.0
    y_vals = []
    if scales is None:
        scale = {
            str(r["patient_id"]): (100.0 / _mean_test_cgm(str(r["patient_id"]), ph)
                                   if relative else 1.0)
            for r in patients
        }
    else:
        scale = scales
    for arm, color, ls, label in (
        ("baseline", BASELINE_COLOR, "--", "Baseline"),
        ("docas", DOCAS_COLOR, "-", "DOCAS"),
    ):
        first = True
        for row in patients:
            x, d = _abs_iob_curve(row, arm)
            d = d * scale[str(row["patient_id"])]
            y_vals.append(d)
            ax.plot(x, d, color=color, alpha=0.5, lw=1.1, ls=ls, zorder=2,
                    label=label if first else None)
            first = False
    ax.axhline(0.0, color="#78909C", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(title, pad=6)
    ax.set(xlim=(0.0, x_hi * 1.02))
    if autoscale_y:
        floor = 3.0 if relative else 5.0
        stack = np.concatenate([np.asarray(v, float).ravel() for v in y_vals]) if y_vals else np.array([-floor, floor])
        y_lo = float(min(np.nanmin(stack), -floor))
        y_hi = float(max(np.nanmax(stack), floor))
        pad = 0.12 * max(y_hi - y_lo, 2.0 * floor)
        ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(alpha=0.18)


def _plot_mean_pdp_panel(ax, ice: dict, *, title: str, relative: bool = True) -> None:
    """Cohort-mean centred ICE for baseline, population DOCAS, and context DOCAS."""
    I = np.asarray(ice.get("I") or [], float)
    mean = ice.get("mean_rel" if relative else "mean") or ice.get("mean") or {}
    styles = (
        ("baseline", BASELINE_COLOR, "--", "Baseline"),
        ("pop", DOCAS_COLOR, "-", "Population DOCAS"),
        ("ctx", "#0277BD", "-", "Context DOCAS"),
    )
    y_vals = []
    for key, col, ls, lab in styles:
        y = mean.get(key)
        if not y:
            continue
        yy = np.asarray([np.nan if v is None else v for v in y], float)
        y_vals.append(yy[np.isfinite(yy)])
        ax.plot(I, yy, color=col, lw=2.0, ls=ls, zorder=3, label=lab)
    ax.axhline(0.0, color="#78909C", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(title, pad=8)
    ax.set(xlim=(0.0, 5.0 * 1.02))
    floor = 3.0 if relative else 5.0
    stack = np.concatenate([np.asarray(v, float).ravel() for v in y_vals]) if y_vals else np.array([-floor, floor])
    y_lo = float(min(np.nanmin(stack), -floor))
    y_hi = float(max(np.nanmax(stack), floor))
    pad = 0.12 * max(y_hi - y_lo, 2.0 * floor)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(alpha=0.18)


def dose_response_figure(*, relative: bool = True) -> Path:
    """Two panels: every evaluated patient's baseline and DOCAS PDP at 30 and 60 minutes."""
    payload = cohort_pdps()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.55), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.16, left=0.08, right=0.98)
    for ax, ph, tag in zip(axes, FORECAST_HORIZONS, ("A", "B")):
        patients = payload[str(ph)]
        scales = {
            str(r["patient_id"]): (100.0 / float(r["cgm"]) if relative else 1.0)
            for r in patients
        }
        _plot_drc_panel(
            ax, patients,
            title=f"{tag} · {ph} min (n={len(patients)})",
            ph=ph, relative=relative, scales=scales)
        for line in ax.get_lines():
            if line.get_label() and not line.get_label().startswith("_"):
                line.set_linewidth(1.4)
            else:
                line.set_linewidth(0.8)
                line.set_alpha(0.28)
    h, l = axes[1].get_legend_handles_labels()
    if l:
        _legend(axes[1], h, l, fontsize=7.5)
    ylab = (r"$\Delta$ predicted CGM (\% of ambient)" if relative
            else r"$\Delta$ predicted CGM (mg/dL)")
    for ax in axes:
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel(r"Bolus IOB (U)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
    name = "dose_response_curves" if relative else "dose_response_curves_absolute"
    return _save_fig(fig, name)


def _replay_row(ph: int, patient: str | None = None, source_start: int | None = None,
                *, scope: str = "", json_path: Path | None = None) -> dict:
    """One replayed window to plot: a named patient, or the cohort's best case.

    Without a patient the choice is made over the windows that enter the
    reported summary, by the same rule for either cohort: the largest time in
    range gain, then the largest difference between the doses the two arms pick.
    """
    path = json_path if json_path is not None else replay_cohort_path(scope)
    payload = replay_horizon_payload(json.loads(path.read_text()), ph)
    rows = ([r for r in payload["windows"]
             if r["patient_id"] == patient and r["twin_rmse"] <= TWIN_RMSE_PRIMARY_MGDL]
            if patient is not None else replay_study._primary_windows(payload["windows"]))
    if source_start is not None:
        return next(r for r in rows if r["source_start"] == source_start)
    # Prefer a window where DOCAS and baseline select different doses and TIR improves.
    def _score(r):
        bd = (r.get("baseline_decisions") or [{}])[0]
        dd = (r.get("docas_decisions") or [{}])[0]
        du = abs(float(dd.get("delivered_dose_u", 0)) - float(bd.get("delivered_dose_u", 0)))
        return (float(r.get("delta_tir") or 0.0), du)
    return max(rows, key=_score)


def _first_decision(row: dict, arm: str) -> dict:
    return row[f"{arm}_decisions"][0]


def _load_replay_showcase() -> dict:
    return _replay_row(REPLAY_SHOWCASE_PH, TABLE2_PATIENT)


def patient588_replay_window_figure() -> Path:
    """CGM + dose-selection audit: chosen bolus marked on each arm's curve."""
    return _replay_window_figure(
        _load_replay_showcase(), f"Patient {TABLE2_PATIENT} · ReplayBG window",
        "patient588_replay_window")


def _dec_key(dec: dict, *names):
    for n in names:
        if n in dec and dec[n] is not None:
            return dec[n]
    raise KeyError(names)


def safeguard_patient_figure() -> Path:
    """Patient-level hypoglycaemia-guard loop (Supplementary)."""
    row = _load_replay_showcase()
    doc = _first_decision(row, "docas")
    start = int(_dec_key(row, "source_start", "source_start"))
    pid = str(_dec_key(row, "patient_id", "patient_id"))
    try:
        raw = json.loads(replay_study.unguarded_cohort_path("").read_text())
        wins = (raw["horizons"][str(REPLAY_SHOWCASE_PH)]["windows"]
                if "horizons" in raw else raw["windows"])
        urow = next(r for r in wins
                    if str(r["patient_id"]) == str(pid)
                    and int(r["source_start"]) == int(start))
        u_doc = _first_decision(urow, "docas")
    except (FileNotFoundError, StopIteration, KeyError, TypeError, ValueError):
        u_doc = None

    traj = row.get("trajectories") or {}
    cgm = np.asarray(traj.get("no_dss") or row.get("cgm_no_dss"), float)
    hours = np.arange(len(cgm)) * (YTS_MIN / 60.0)
    t0 = float(_dec_key(doc, "time_min", "time_min")) / 60.0
    grid = np.asarray(_dec_key(doc, "grid_u", "grid_u"), float)
    fut = np.asarray(_dec_key(doc, "future_cgm_curve", "future_cgm_curve"), float)
    dy = fut - fut[0]
    try:
        nadir_k = np.asarray(
            _dec_key(doc, "nadir_worst_case_curve", "nadir_worst_case_curve"), float)
    except KeyError:
        pending = replay_study.pending_action_ratio(replay_study.PH)
        k_si = replay_study.SAFETY_SI_FACTOR
        nadir_k = fut[0] - k_si * (1.0 + pending) * (fut[0] - fut)
    floor = float(replay_study.HYPO_FLOOR_MGDL)
    trigger = float(replay_study.HYPER_TRIGGER_MGDL)
    d_sel = float(_dec_key(doc, "delivered_dose_u", "delivered_dose_u"))
    y_sel = float(np.interp(d_sel, grid, dy))
    g_sel = float(np.interp(d_sel, grid, nadir_k))

    fig, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.15),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
        constrained_layout=True)
    ax_t, ax_ice, ax_g = axes

    ax_t.axhspan(TIR_LO, TIR_HI, color="#90CAF9", alpha=0.14, zorder=0)
    ax_t.axhline(trigger, color="#C62828", ls="--", lw=1.1, zorder=2)
    ax_t.plot(hours, cgm, color="#455A64", lw=1.9, zorder=3)
    ax_t.axvline(t0, color="#455A64", ls=":", lw=1.0, alpha=0.85)
    g0 = float(np.interp(t0, hours, cgm))
    ax_t.scatter([t0], [g0], s=42, color="#C62828", zorder=4,
                 edgecolors="white", linewidths=0.5)
    ax_t.text(0.03, 0.96, rf"Trigger: CGM $> {trigger:.0f}$",
              transform=ax_t.transAxes, ha="left", va="top", fontsize=8.5)
    ax_t.set(xlim=(0, REPLAY_WINDOW_H),
             ylim=(0, max(280.0, float(np.nanmax(cgm)) + 20)),
             xlabel="Hours from meal", ylabel="CGM (mg/dL)")
    ax_t.set_title(f"A · Patient {pid}", fontsize=10, pad=5)
    ax_t.grid(alpha=0.18)

    span = float(max(grid.max(), 1e-6))
    u_tau = np.linspace(0.0, 1.0, 61)
    tau_dy = _abs_drop_curve(
        u_tau, REPLAY_SHOWCASE_PH, span=span,
        cgm0=float(_dec_key(doc, "observed_cgm", "observed_cgm")))
    ax_ice.plot(u_tau * span, tau_dy, color=TARGET_COLOR, ls=":", lw=1.7,
                label=r"Target ICE", zorder=1)
    ax_ice.axhline(0.0, color="#78909C", ls="--", lw=0.8, alpha=0.7)
    ax_ice.plot(grid, dy, color=DOCAS_COLOR, lw=2.1, label="DOCAS ICE", zorder=2)
    ax_ice.scatter(grid, dy, s=18, color=DOCAS_COLOR, zorder=3,
                   edgecolors="white", linewidths=0.35)
    ax_ice.scatter([d_sel], [y_sel], s=88, color=DOCAS_COLOR, zorder=5,
                   edgecolors="#212121", linewidths=0.8)
    ax_ice.set(xlim=(0, span), xlabel="Candidate bolus (U)",
               ylabel=r"$\Delta$ predicted CGM (mg/dL)")
    ax_ice.set_title("B · Score the ICE", fontsize=10, pad=5)
    _legend(ax_ice, fontsize=7)
    ax_ice.grid(alpha=0.18)

    ok = nadir_k >= floor
    ax_g.axhline(floor, color="#C62828", ls="--", lw=1.2, zorder=2)
    ax_g.fill_between(grid, 0.0, floor, where=~ok, color="#FFCDD2",
                      alpha=0.55, zorder=0)
    ax_g.plot(grid, nadir_k, color=DOCAS_COLOR, lw=2.1, zorder=3)
    ax_g.scatter(grid[ok], nadir_k[ok], s=18, color=DOCAS_COLOR, zorder=4,
                 edgecolors="white", linewidths=0.35)
    ax_g.scatter(grid[~ok], nadir_k[~ok], s=22, color="#C62828", zorder=4,
                 marker="x", linewidths=1.1)
    ax_g.scatter([d_sel], [g_sel], s=88, color=DOCAS_COLOR, zorder=5,
                 edgecolors="#212121", linewidths=0.8)
    ax_g.text(d_sel, g_sel + 8.0, f"{d_sel:g} U", ha="center", va="bottom",
              fontsize=7.5, color=DOCAS_COLOR, fontweight="bold")
    if u_doc is not None:
        d_u = float(_dec_key(u_doc, "delivered_dose_u", "delivered_dose_u"))
        if abs(d_u - d_sel) > 0.2:
            g_u = float(np.interp(d_u, grid, nadir_k))
            ax_g.scatter([d_u], [g_u], s=70, facecolors="none",
                         edgecolors="#6A1B9A", linewidths=1.3, zorder=5)
            ax_g.annotate("unguarded", xy=(d_u, g_u),
                          xytext=(d_u - 0.15, g_u - 22),
                          fontsize=7, color="#6A1B9A", ha="right",
                          arrowprops=dict(arrowstyle="-", color="#6A1B9A", lw=0.7))
    ax_g.set(xlim=(0, span), ylim=(max(0.0, float(nadir_k.min()) - 25),
                                   max(float(nadir_k.max()) + 20, floor + 40)),
             xlabel="Candidate bolus (U)",
             ylabel="Projected nadir (mg/dL)")
    ax_g.set_title(rf"C · Keep nadir $\geq {floor:.0f}$", fontsize=10, pad=5)
    ax_g.grid(alpha=0.18)
    return _save_fig(fig, "safeguard_patient588")


def _replay_window_figure(row: dict, title: str, name: str) -> Path:
    base_dec, doc_dec = _first_decision(row, "baseline"), _first_decision(row, "docas")
    traj = row.get("trajectories") or {}
    cgm_no = np.asarray(traj.get("no_dss") or row.get("cgm_no_dss"), float)
    cgm_base = np.asarray(traj.get("baseline_dss") or row.get("cgm_baseline"), float)
    cgm_doc = np.asarray(traj.get("docas_dss") or row.get("cgm_docas"), float)
    hours = np.arange(len(cgm_no)) * (YTS_MIN / 60.0)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True)
    ax.axhspan(TIR_LO, TIR_HI, color="#90CAF9", alpha=0.14, zorder=0)
    ax.plot(hours, cgm_no, color="#78909C", lw=1.8, label="No DSS")
    ax.plot(hours, cgm_base, color=BASELINE_COLOR, ls="--", lw=2.0, label="Baseline")
    ax.plot(hours, cgm_doc, color=DOCAS_COLOR, lw=2.2, label="DOCAS")
    t0 = float(base_dec["time_min"]) / 60.0
    ax.axvline(t0, color="#455A64", ls=":", lw=1.0, alpha=0.8)
    ax.set(xlim=(0, REPLAY_WINDOW_H), ylim=(0, max(280, float(np.nanmax(cgm_no)) + 20)),
           xlabel="Hours from meal", ylabel="CGM (mg/dL)")
    ax.set_title(title, fontsize=10, pad=6)
    _legend(ax, fontsize=8)
    ax.grid(alpha=0.18)

    # Right: ICE at the decision; selected dose labelled under the marker.
    span = float(max(float(max(base_dec["grid_u"])), float(max(doc_dec["grid_u"])), 1e-6))
    u_tau = np.linspace(0.0, 1.0, 61)
    tau_dy = _abs_drop_curve(u_tau, REPLAY_SHOWCASE_PH, span=span,
                             cgm0=float(doc_dec["observed_cgm"]))
    ax2.plot(u_tau * span, tau_dy, color=TARGET_COLOR, ls=":", lw=1.8,
             label="Target DRC", zorder=1)
    ax2.axhline(0.0, color="#78909C", ls="--", lw=0.9, alpha=0.7)
    y_all = [tau_dy]
    picks = []
    for dec, color, ls, label in (
        (base_dec, BASELINE_COLOR, "--", "Baseline"),
        (doc_dec, DOCAS_COLOR, "-", "DOCAS"),
    ):
        dose_grid = np.asarray(dec["grid_u"], float)
        fut = np.asarray(dec["future_cgm_curve"], float)
        dy = fut - fut[0]
        y_all.append(dy)
        d_sel = float(dec["delivered_dose_u"])
        y_sel = float(np.interp(d_sel, dose_grid, dy))
        picks.append((d_sel, y_sel, color))
        ax2.plot(dose_grid, dy, color=color, ls=ls, lw=2.2, label=label, zorder=2)
        ax2.scatter(
            dose_grid, dy, s=22, color=color, zorder=4,
            edgecolors="white", linewidths=0.4, alpha=0.85)
        ax2.axvline(d_sel, color=color, ls=":", lw=1.0, alpha=0.55, zorder=1)
        ax2.scatter(
            [d_sel], [y_sel], s=95, color=color, zorder=5,
            edgecolors="#212121", linewidths=0.9)
    y_stack = np.concatenate(y_all)
    y_lo, y_hi = float(np.min(y_stack)), float(np.max(y_stack))
    ax2.set_ylim(y_lo - 16.0, max(y_hi + 20.0, 34.0))
    for d_sel, y_sel, color in picks:
        ax2.text(d_sel, y_sel - 8.0, f"{d_sel:g} U", ha="center", va="top",
                 fontsize=12, color=color, fontweight="bold", zorder=6)
    ax2.set(xlim=(0, span), xlabel="Candidate bolus dose (U)",
            ylabel=r"$\Delta$ predicted CGM (mg/dL)")
    ax2.set_title("Dose selection on the audit curve", fontsize=10, pad=6)
    _legend(ax2, fontsize=7)
    ax2.grid(alpha=0.18)
    return _save_fig(fig, name)


def _plot_target_ice(ax, *, talk: bool = False, titled: bool = True,
                    legend_fs: float = 8) -> None:
    """Figure 2B / flowchart B: 30- and 60-min target ICE vs correction bolus."""
    from preprocessing import CIB_MAX_U

    t_min, drop = replaybg_drop_trace(1.0, horizon_min=270)
    d30_1u = float(drop[30])
    d60_1u = float(drop[60])
    span = float(CIB_MAX_U)
    doses = np.linspace(0.0, span, 81)
    u = doses / span
    tau30 = _abs_drop_curve(u, 30, span=span)
    tau60 = _abs_drop_curve(u, 60, span=span)
    ax.plot(doses, tau30, color="#EF6C00", lw=2.4, label="30 min")
    ax.plot(doses, tau60, color=TARGET_COLOR, lw=2.4, label="60 min")
    ax.axvline(1.0, color="#90A4AE", ls=":", lw=1.0)
    ax.scatter([1.0], [d30_1u], color="#EF6C00", s=36, zorder=4,
               edgecolors="#212121", linewidths=0.5)
    ax.scatter([1.0], [d60_1u], color=TARGET_COLOR, s=36, zorder=4,
               edgecolors="#212121", linewidths=0.5)
    y_lo_b = min(float(tau60.min()) * 1.08, -1)
    kw = dict(
        xlim=(0, span * 1.02), ylim=(y_lo_b, 2),
        xlabel="Correction bolus (U)",
        ylabel=(r"Observed glucose drop (mg/dL)" if not talk
                else r"Target DRC (mg/dL)"))
    if titled:
        kw["title"] = ("B · Target ICE vs bolus" if talk
                       else "B · Derived target functions")
    ax.set(**kw)
    ax.text(1.08, 0.96 * y_lo_b, "1 U", fontsize=7.5, color="#607D8B",
            ha="left", va="bottom")
    _legend(ax, loc="center right", bbox_to_anchor=(0.98, 0.62), fontsize=legend_fs)
    ax.grid(alpha=0.18)


def iob_feature_figure(*, name: str = "iob_feature", dest: Path | None = None) -> Path:
    """Logged bolus → population action-remaining IOB vs a new correction DRC."""
    from preprocessing import bolus_iob_fraction, CIB_MAX_U

    ages = np.arange(0, 8 * 60 + 1, 1, dtype=float)
    frac = bolus_iob_fraction(ages)
    doses = np.linspace(0.0, float(CIB_MAX_U), 81)
    span = float(CIB_MAX_U)
    u = doses / span
    tau60 = _abs_drop_curve(u, 60, span=span)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55), constrained_layout=True)
    ax = axes[0]
    ax.plot(ages / 60.0, frac, color="#00897B", lw=2.4)
    ax.fill_between(ages / 60.0, 0.0, frac, color="#00897B", alpha=0.12)
    ax.set(xlim=(0, 8), ylim=(0, 1.05),
           xlabel="Hours since bolus",
           ylabel="Action remaining",
           title="A · Population IOB curve")
    ax.text(
        0.97, 0.08,
        "Ohio: total dose at delivery\n"
        "AZT1D: food + correction + automatic",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#455A64")
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.plot(doses, tau60, color=TARGET_COLOR, lw=2.4)
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.axvline(1.0, color="#90A4AE", ls=":", lw=1.0)
    ax.set(xlim=(0, span * 1.02),
           ylim=(min(float(tau60.min()) * 1.08, -1), 2),
           xlabel="New correction bolus (U)",
           ylabel=r"Target $\Delta$CGM (mg/dL)",
           title="B · Target DRC from basal")
    ax.text(
        0.97, 0.08,
        "Simulated from basal steady state,\n"
        "not from residual IOB",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#455A64")
    ax.grid(alpha=0.18)
    return _save_fig(fig, name, dest=dest)


def context_adjust_figure(*, name: str = "context_adjust", dest: Path | None = None) -> Path:
    """How carbohydrate mixes the slopes, and how glucose scales the drop.

    The personal step is the shallow limit of the published clamp (one quarter
    of the population step). Dose support is taken as saturated, so the figure
    varies only glucose and carbohydrate.
    """
    u = np.linspace(0.0, 1.0, 121)
    dose = u * 5.0

    def pop(g: float) -> np.ndarray:
        return _study_drop_curve(u, 60, span=5.0, cgm0=float(g))

    def mix(g: float, cho: float) -> np.ndarray:
        gate = 0.95 * (cho ** 2) / (cho ** 2 + 30.0 ** 2)
        p = pop(g)
        return (1.0 - gate) * p + gate * (0.25 * p)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55), sharey=True)
    fig.subplots_adjust(wspace=0.12, top=0.86, bottom=0.16, left=0.08, right=0.98)

    ax = axes[0]
    carb_styles = (
        (0.0, TARGET_COLOR, "0 g, population step"),
        (20.0, "#7E57C2", "20 g"),
        (40.0, "#00897B", "40 g"),
        (80.0, "#00695C", "80 g"),
    )
    for cho, col, lab in carb_styles:
        ax.plot(dose, mix(140.0, cho), color=col, lw=2.2, label=lab)
    ax.plot(dose, 0.25 * pop(140.0), color="#90A4AE", lw=1.6, ls="--",
            label="Personal step, shallow limit")
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.set(xlim=(0, 5), xlabel="Insulin (U)", ylabel="Change from zero dose (mg/dL)",
           title="A · Carbohydrate at 140 mg/dL")
    _legend(ax, fontsize=7)
    ax.grid(alpha=0.18)

    ax = axes[1]
    glucose_styles = (
        (90.0, "#F9A825", "90 mg/dL"),
        (140.0, "#FB8C00", "140 mg/dL"),
        (200.0, "#EF6C00", "200 mg/dL"),
        (280.0, "#E65100", "280 mg/dL"),
    )
    for g, col, lab in glucose_styles:
        ax.plot(dose, mix(g, 40.0), color=col, lw=2.2, label=lab)
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.set(xlim=(0, 5), xlabel="Insulin (U)",
           title="B · Glucose at 40 g carbohydrate")
    _legend(ax, fontsize=7)
    ax.grid(alpha=0.18)
    return _save_fig(fig, name, dest=dest)


def context_drc_figure(*, name: str = "context_drc", dest: Path | None = None) -> Path:
    """Inter-patient fasting curves (Ohio) and three contexts for the extremes."""
    path = ROOT / "results" / "revision" / "context_docas.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    rows = payload["ohio_t1dm"]["patients"]
    I = np.asarray(rows[0]["I"], float)
    # Rank by fasting drop at 5 U.
    ends = [(abs(r["curves"]["fast"][-1]), r) for r in rows]
    ends.sort(key=lambda t: t[0], reverse=True)
    mild, steep = ends[-1][1], ends[0][1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    ax = axes[0]
    for r in rows:
        ax.plot(I, r["curves"]["fast"], color="#90A4AE", lw=1.0, alpha=0.85)
    ax.plot(I, rows[0]["pop_curves"]["fast"], color=TARGET_COLOR, lw=2.4, label="Population")
    ax.plot(I, steep["curves"]["fast"], color="#D32F2F", lw=2.0, label=f"Steep ({steep['patient_id']})")
    ax.plot(I, mild["curves"]["fast"], color="#00897B", lw=2.0, label=f"Mild ({mild['patient_id']})")
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.set(xlim=(0, 5), xlabel="Insulin (U)", ylabel=r"Change from zero dose (mg/dL)",
           title=r"A · Fasting ($G{=}110$, CHO${=}0$)")
    _legend(ax, fontsize=7.5)
    ax.grid(alpha=0.18)

    ax = axes[1]
    tones = {"fast": ("#EF6C00", "Fasting 110 / 0 g"),
             "meal": ("#1565C0", "Meal 140 / 40 g"),
             "high": ("#6A1B9A", "Correction 200 / 5 g")}
    for key, (col, lab) in tones.items():
        pop = np.mean([r["pop_curves"][key] for r in rows], axis=0)
        ctx = np.mean([r["curves"][key] for r in rows], axis=0)
        ax.plot(I, pop, color=col, lw=1.5, ls="--", label=f"Population, {lab}")
        ax.plot(I, ctx, color=col, lw=2.2, label=f"Context, {lab}")
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.set(xlim=(0, 5), xlabel="Insulin (U)", ylabel=r"Change from zero dose (mg/dL)",
           title="B · Population (dashed) and context scale (solid)")
    _legend(ax, fontsize=6.5)
    ax.grid(alpha=0.18)
    return _save_fig(fig, name, dest=dest)


def iob_pharmacology_figure(patient_id: str = "588", *, name: str = "iob_pharmacology",
                           dest: Path | None = None, talk: bool = False) -> Path:
    """1 U CGM drop and the derived 30- and 60-min target DRCs."""
    t_min, drop = replaybg_drop_trace(1.0, horizon_min=270)
    d30_1u = float(drop[30])
    d60_1u = float(drop[60])
    peak_i = int(np.argmin(drop))
    d_peak = float(drop[peak_i])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45), constrained_layout=True)
    ax = axes[0]
    ax.plot(t_min / 60.0, drop, color="#EF6C00", lw=2.4)
    ax.axhline(0.0, color="#B0BEC5", lw=0.8)
    ax.scatter([peak_i / 60.0], [d_peak], color="#37474F", s=36, zorder=4)
    ax.annotate(
        f"peak drop @ {peak_i} min\n{d_peak:.1f}",
        xy=(peak_i / 60.0, d_peak),
        xytext=(2.4, d_peak + 0.15 * abs(d_peak) if d_peak < 0 else d_peak + 2),
        fontsize=8, color="#37474F", ha="left",
        arrowprops=dict(arrowstyle="-", color="#90A4AE", lw=0.8))
    ax.axvline(0.5, color="#EF6C00", ls=":", lw=1.1, alpha=0.85)
    ax.axvline(1.0, color=TARGET_COLOR, ls=":", lw=1.1, alpha=0.85)
    ax.scatter([0.5], [d30_1u], color="#EF6C00", s=42, zorder=5,
               edgecolors="#212121", linewidths=0.6)
    ax.scatter([1.0], [d60_1u], color=TARGET_COLOR, s=42, zorder=5,
               edgecolors="#212121", linewidths=0.6)
    y_lo = min(float(drop.min()) * 1.12, -1)
    ax.text(0.06, d30_1u - 0.06 * (0 - y_lo), f"30 min\n{d30_1u:.1f}",
            ha="left", va="top", fontsize=7.5, color="#EF6C00")
    ax.text(1.06, d60_1u, f"60 min\n{d60_1u:.1f}",
            ha="left", va="center", fontsize=7.5, color=TARGET_COLOR)
    ax.set(xlim=(0, 4.5), ylim=(y_lo, 2),
           xlabel="Hours after 1 U bolus",
           ylabel=r"$\Delta$ CGM vs control (mg/dL)",
           title=("A · 1 U bolus vs no-bolus" if talk
                  else r"A · Absolute drop (1 U, $G_0{=}220$)"))
    ax.text(
        0.97, 0.04,
        rf"SI ${m.SI_MODE * 1e3:.1f}\times 10^{{-3}}$ · {m.BODY_WEIGHT_KG:.0f} kg",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#455A64")
    ax.grid(alpha=0.18)
    _plot_target_ice(axes[1], talk=talk)
    return _save_fig(fig, name, dest=dest)


def _context_scale(g: float, cho: float, theta: dict) -> float:
    """Patient scale on the population slope, clipped to [0.45, 1.55]."""
    zg = (float(g) - 140.0) / 40.0
    zc = float(cho) / (float(cho) + 30.0)
    eta = (
        float(theta.get("t0", 0.0))
        + float(theta.get("tg", 0.0)) * zg
        + float(theta.get("tc", 0.0)) * zc
        + float(theta.get("tgc", 0.0)) * zg * zc
    )
    return float(np.exp(np.clip(eta, np.log(0.45), np.log(1.55))))


def _plot_context_drc(ax) -> None:
    """Carbohydrate-gated synthetic sample weight w_ctx(c) in Context DOCAS."""
    cho = np.linspace(0.0, 120.0, 200)
    zc = cho / (cho + 25.0)
    w_ctx = 0.52 * (1.0 - zc) + 0.02 * zc
    ax.plot(cho, np.ones_like(cho), color=TARGET_COLOR, lw=1.6, ls=":",
            label=r"Population DOCAS ($w{=}1.0$)")
    ax.plot(cho, w_ctx, color="#1565C0", lw=2.4,
            label=r"Context DOCAS ($w_{\mathrm{ctx}}(c)$)")
    for c_mark, lab, col in ((0.0, "Fasting", "#00897B"),
                             (25.0, "25 g", "#6A1B9A"),
                             (60.0, "60 g", "#EF6C00")):
        wm = 0.52 * (1.0 - c_mark / (c_mark + 25.0)) + 0.02 * (c_mark / (c_mark + 25.0))
        ax.scatter([c_mark], [wm], color=col, s=38, zorder=5,
                   edgecolors="#212121", linewidths=0.6)
    ax.set(
        xlim=(-2, 120),
        ylim=(-0.03, 1.08),
        xlabel="Carbohydrates on board $c$ (g)",
        ylabel="Synthetic sample weight",
        title="C · Carbohydrate-gated synthetic weight",
    )
    _legend(ax, fontsize=7)
    ax.grid(alpha=0.18)


def _flowchart_panel_crops(*, panel_bg: str = "#ffffff",
                           show_trigger_band: bool = False,
                           label_fs: float | None = None) -> dict[str, str]:
    """Shared matplotlib crops for flowchart + graphical abstract (base64 PNGs)."""
    import base64
    import io

    ph, pid = FLOWCHART_HORIZON, FLOWCHART_PATIENT
    data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(ph))
    feat_span = float(insulin_span_u(data, horizon_min=ph))
    DOCAS.SPAN = feat_span
    u = np.linspace(0.0, 1.0, 81)
    dose = u * feat_span
    f0_ref = float(m.G0_TAU)
    iu, fr, isyn, fsyn, _, meta = _synth_cloud(
        pid, ph, max_real=2500, max_synth=4000, emp_frac=1.0)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    a_s, y_s, dy_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    Xte = np.asarray(data.X_test, float)
    baseline = load_baseline_model(ph, pid)
    bundle = m._load_bundle(ph, pid)
    docas = bundle["docas"]
    base_d = _future(baseline, Xte, ji, gi, u, feat_span, a_s, dy_s, y_s).mean(0)
    docas_d = _future(docas, Xte, ji, gi, u, feat_span, a_s, dy_s, y_s).mean(0)
    base_d = base_d - base_d[0]
    docas_d = docas_d - docas_d[0]
    tau_d = _abs_drop_curve(u, ph, span=feat_span, cgm0=_mean_test_cgm(pid, ph))

    rng = np.random.default_rng(1)
    fut_all, ins_all, cgm_all = meta["future"], meta["insulin"], meta["cgm"]
    top = np.flatnonzero(
        (fut_all > 200.0) & (fut_all < 260.0)
        & (ins_all < 0.40 * feat_span)
    )
    if not top.size:
        top = np.flatnonzero((fut_all > 190.0) & (ins_all < 0.55 * feat_span))
    if not top.size:
        top = np.arange(len(fut_all))
    a = int(top[np.argmax(fut_all[top])])
    anchors = np.asarray(meta.get("anchor_idx"), int)
    counts = np.bincount(anchors, minlength=len(fut_all))
    scored = [(int(counts[i]), float(fut_all[i]), int(i)) for i in top]
    scored.sort(reverse=True)
    a = scored[0][2]
    x0 = float(ins_all[a])
    y_fut = float(fut_all[a])
    y_ctx = float(cgm_all[a])
    target = meta["target"]
    drop = float(meta["drop"])
    tau_mode = str(meta.get("tau_mode") or m.STUDY_CFG.get("tau_mode", "fractional"))
    core_kw = dict(target=target, drop=drop, u_ref=float(meta["u_ref"]),
                   tau_mode=tau_mode, feat_span=feat_span)
    probe, fut_lab = _core_context_probes(data, anchor=a, n=80, seed=2, **core_kw)
    drc_x, drc_y = _core_context_probes(data, anchor=a, n=61, emp_frac=0.0, **core_kw)
    y_band = float(ALIGN_TRIGGER_MGDL)
    iob_lab = "Horizon bolus IOB (U)"
    x_hi = float(max(
        feat_span,
        float(np.percentile(iu, 99)) if len(iu) else 0.0,
        float(np.percentile(isyn, 99)) if len(isyn) else 0.0,
    ))
    ok = np.isfinite(iu) & np.isfinite(fr) & (iu >= 0) & (fr > 0)
    if ok.sum() >= 20:
        slope_obs, intercept_obs = np.polyfit(iu[ok], fr[ok], 1)
    else:
        slope_obs, intercept_obs = 0.0, float(np.nanmean(fr)) if len(fr) else 180.0

    def _b64(fig) -> str:
        fig.patch.set_facecolor(panel_bg)
        for ax in fig.axes:
            ax.set_facecolor(panel_bg)
            if label_fs:
                ax.tick_params(labelsize=label_fs - 1)
                ax.xaxis.label.set_size(label_fs)
                ax.yaxis.label.set_size(label_fs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                    facecolor=panel_bg, edgecolor="none", pad_inches=0.04)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    fig, ax = plt.subplots(figsize=(5.2, 2.9), facecolor=panel_bg)
    ax.scatter(iu, fr, s=6, alpha=0.22, color="#90A4AE", rasterized=True, zorder=1)
    if show_trigger_band:
        ax.axhline(y_band, color=FC_HYPER, ls="--", lw=1.0, alpha=0.85, zorder=2)
    x_line = np.linspace(0.0, x_hi, 60)
    ax.plot(x_line, intercept_obs + slope_obs * x_line, color=BASELINE_COLOR, ls="--",
            lw=2.0, label="Observational fit", zorder=3)
    ax.plot(dose, f0_ref + tau_d, color=TARGET_COLOR, ls=":", lw=2.0,
            label="Target DRC", zorder=4)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    _legend(ax, fontsize=7)
    ax.grid(alpha=0.15)
    img_a = _b64(fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.9), facecolor=panel_bg)
    _plot_target_ice(ax, titled=False, legend_fs=7)
    img_b = _b64(fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.9), facecolor=panel_bg)
    ax.scatter(iu, fr, s=6, alpha=0.25, color="#78909C", rasterized=True, zorder=2)
    ax.scatter(isyn, fsyn, s=5, alpha=0.18, color=DOCAS_COLOR, rasterized=True, zorder=3)
    ax.plot(drc_x, drc_y, color=TARGET_COLOR, ls=":", lw=2.0, zorder=4)
    ax.scatter(probe, fut_lab, s=26, alpha=0.8, color=DOCAS_COLOR, edgecolors="#004D40",
               linewidths=0.5, zorder=5)
    ax.scatter([x0], [y_fut], s=120, marker="*", color=FC_HYPER,
               edgecolors="#37474F", linewidths=0.7, zorder=6)
    if show_trigger_band:
        ax.axhline(y_band, color=FC_HYPER, ls="--", lw=1.0, alpha=0.85)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    ax.grid(alpha=0.15)
    img_e = _b64(fig)

    fig, ax = plt.subplots(figsize=(5.4, 2.6), facecolor=panel_bg)
    ax.plot(dose, base_d, color=BASELINE_COLOR, ls="--", lw=2.0, label="Baseline")
    ax.plot(dose, docas_d, color=DOCAS_COLOR, lw=2.2, label="DOCAS")
    ax.axhline(0.0, color="#78909C", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(feat_span, color="#78909C", ls=":", lw=1.0, alpha=0.85)
    ax.set(xlim=(0, feat_span), xlabel=iob_lab,
           ylabel="Change in model prediction (mg/dL)")
    ax2 = ax.twinx()
    ax2.plot(dose, tau_d, color=TARGET_COLOR, ls=":", lw=1.9,
             label="Target DRC (observed drop)")
    y_stack = np.concatenate([base_d, docas_d, tau_d, [0.0]])
    pad = 0.08 * max(float(np.ptp(y_stack)), 1.0)
    y_lo, y_hi = float(np.min(y_stack)) - pad, float(np.max(y_stack)) + pad
    ax.set_ylim(y_lo, y_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_ylabel("Observed glucose drop (mg/dL)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, h1 + h2, l1 + l2, loc="center right",
            bbox_to_anchor=(0.98, 0.62), fontsize=7.5)
    ax.grid(alpha=0.18)
    img_f = _b64(fig)

    return {"a": img_a, "b": img_b, "e": img_e, "f": img_f}


def docas_flowchart_figure() -> Path:
    """GA-style vertical flowchart: Phase 1 → 2 → 3 (SVG chrome + panel crops)."""
    import base64
    import io
    import subprocess

    ph, pid = FLOWCHART_HORIZON, FLOWCHART_PATIENT
    data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(ph))
    feat_span = float(insulin_span_u(data, horizon_min=ph))
    DOCAS.SPAN = feat_span
    u = np.linspace(0.0, 1.0, 81)
    dose = u * feat_span
    f0_ref = float(m.G0_TAU)
    iu, fr, isyn, fsyn, _, meta = _synth_cloud(
        pid, ph, max_real=2500, max_synth=4000, emp_frac=1.0)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    a_s, y_s, dy_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    Xte = np.asarray(data.X_test, float)
    baseline = load_baseline_model(ph, pid)
    bundle = m._load_bundle(ph, pid)
    docas = bundle["docas"]
    base_d = _future(baseline, Xte, ji, gi, u, feat_span, a_s, dy_s, y_s).mean(0)
    docas_d = _future(docas, Xte, ji, gi, u, feat_span, a_s, dy_s, y_s).mean(0)
    base_d = base_d - base_d[0]
    docas_d = docas_d - docas_d[0]
    tau_d = _abs_drop_curve(u, ph, span=feat_span, cgm0=_mean_test_cgm(pid, ph))

    rng = np.random.default_rng(1)
    fut_all, ins_all, cgm_all = meta["future"], meta["insulin"], meta["cgm"]
    anchors = np.asarray(meta.get("anchor_idx"), int)
    top = np.flatnonzero(
        (fut_all > 200.0) & (fut_all < 260.0)
        & (ins_all < 0.40 * feat_span)
    )
    if not top.size:
        top = np.flatnonzero((fut_all > 190.0) & (ins_all < 0.55 * feat_span))
    if not top.size:
        top = np.arange(len(fut_all))
    # Prefer a context that was actually drawn as a synth anchor (panel E highlight).
    counts = np.bincount(anchors, minlength=len(fut_all))
    scored = [(int(counts[i]), float(fut_all[i]), int(i)) for i in top]
    scored.sort(reverse=True)
    a = scored[0][2]
    x0 = float(ins_all[a])
    y_fut = float(fut_all[a])
    y_ctx = float(cgm_all[a])
    target = meta["target"]
    drop = float(meta["drop"])
    tau_mode = str(meta.get("tau_mode") or m.STUDY_CFG.get("tau_mode", "fractional"))
    core_kw = dict(target=target, drop=drop, u_ref=float(meta["u_ref"]),
                   tau_mode=tau_mode, feat_span=feat_span)
    # D/E points: empiric IOB. τ line: uniform probes for a smooth ICE.
    probe, fut_lab = _core_context_probes(data, anchor=a, n=80, seed=2, **core_kw)
    probe_e, fut_e = probe, fut_lab
    iob_lab = "Horizon bolus IOB (U)"
    x_hi = float(max(
        feat_span,
        float(np.percentile(iu, 99)) if len(iu) else 0.0,
        float(np.percentile(isyn, 99)) if len(isyn) else 0.0,
    ))
    tau_x, fut_tau = _core_context_probes(
        data, anchor=a, n=61, emp_frac=0.0, **core_kw)

    # Observational association on training cloud (often wrong-signed vs physiology).
    ok = np.isfinite(iu) & np.isfinite(fr) & (iu >= 0) & (fr > 0)
    if ok.sum() >= 20:
        slope_obs, intercept_obs = np.polyfit(iu[ok], fr[ok], 1)
    else:
        slope_obs, intercept_obs = 0.0, float(np.nanmean(fr)) if len(fr) else 180.0

    canvas = "#e8e8f2"
    card = "#ffffff"  # contrast vs canvas (GA-style cards)
    panel_bg = card

    def _b64(fig) -> str:
        fig.patch.set_facecolor(panel_bg)
        for ax in fig.axes:
            ax.set_facecolor(panel_bg)
            ax.tick_params(labelsize=12, pad=2)
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.xaxis.labelpad = 3
            ax.yaxis.labelpad = 3
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                    facecolor=panel_bg, edgecolor="none", pad_inches=0.03)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # --- A: observational train cloud + cohort bias ---
    fig, ax = plt.subplots(figsize=(5.8, 3.0), facecolor=panel_bg)
    ax.scatter(iu, fr, s=7, alpha=0.22, color="#90A4AE", rasterized=True, zorder=1)
    x_line = np.linspace(0.0, x_hi, 60)
    ax.plot(x_line, intercept_obs + slope_obs * x_line, color=BASELINE_COLOR, ls="--",
            lw=2.2, label="Observational fit", zorder=3)
    ax.plot(dose, f0_ref + tau_d, color=TARGET_COLOR, ls=":", lw=2.2,
            label="Target DRC", zorder=4)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    _legend(ax, fontsize=10)
    ax.grid(alpha=0.15)
    img_a = _b64(fig)

    # --- B: same subplot as Figure 2B ---
    fig, ax = plt.subplots(figsize=(5.8, 3.0), facecolor=panel_bg)
    _plot_target_ice(ax, titled=False, legend_fs=10)
    img_b = _b64(fig)

    # --- C: one context ---
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=panel_bg)
    ax.scatter(iu, fr, s=7, alpha=0.22, color="#90A4AE", rasterized=True, zorder=1)
    ax.scatter([x0], [y_fut], s=130, marker="*", color=FC_HYPER,
               edgecolors="#37474F", linewidths=0.7, zorder=4)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    ax.grid(alpha=0.15)
    img_c = _b64(fig)

    # --- D: one context's synthetic probes (core.sample_synthetic) ---
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=panel_bg)
    ax.scatter(iu, fr, s=6, alpha=0.18, color="#90A4AE", rasterized=True, zorder=1)
    ax.plot(tau_x, fut_tau, color=TARGET_COLOR, ls=":", lw=2.0, zorder=3)
    ax.scatter(probe, fut_lab, s=22, alpha=0.65, color=DOCAS_COLOR, zorder=4,
               edgecolors="#004D40", linewidths=0.4)
    ax.scatter([x0], [y_fut], s=130, marker="*", color=FC_HYPER,
               edgecolors="#37474F", linewidths=0.7, zorder=5)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    ax.grid(alpha=0.15)
    img_d = _b64(fig)

    # --- E: full cloud (sample_synthetic) + per-context curves ---
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=panel_bg)
    ax.scatter(iu, fr, s=6, alpha=0.22, color="#78909C", rasterized=True, zorder=2)
    ax.scatter(isyn, fsyn, s=4, alpha=0.14, color=DOCAS_COLOR, rasterized=True, zorder=3)
    ax.plot(tau_x, fut_tau, color=TARGET_COLOR, ls=":", lw=2.0, zorder=5)
    ax.scatter(probe_e, fut_e, s=22, alpha=0.7, color=DOCAS_COLOR, edgecolors="#004D40",
               linewidths=0.5, zorder=6)
    ax.scatter([x0], [y_fut], s=130, marker="*", color=FC_HYPER,
               edgecolors="#37474F", linewidths=0.7, zorder=7)
    ax.set(xlim=(0, x_hi), ylim=(60, 320), xlabel=iob_lab, ylabel="Future CGM (mg/dL)")
    ax.grid(alpha=0.15)
    img_e = _b64(fig)

    # --- F: audit ---
    fig, ax = plt.subplots(figsize=(5.8, 3.0), facecolor=panel_bg)
    ax.plot(dose, base_d, color=BASELINE_COLOR, ls="--", lw=2.2, label="Baseline")
    ax.plot(dose, docas_d, color=DOCAS_COLOR, lw=2.4, label="DOCAS")
    ax.axhline(0.0, color="#78909C", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(feat_span, color="#78909C", ls=":", lw=1.0, alpha=0.85)
    ax.set(xlim=(0, feat_span), xlabel=iob_lab,
           ylabel="Change in model prediction (mg/dL)")
    ax2 = ax.twinx()
    ax2.plot(dose, tau_d, color=TARGET_COLOR, ls=":", lw=2.0,
             label="Target DRC (observed drop)")
    y_stack = np.concatenate([base_d, docas_d, tau_d, [0.0]])
    pad = 0.08 * max(float(np.ptp(y_stack)), 1.0)
    y_lo, y_hi = float(np.min(y_stack)) - pad, float(np.max(y_stack)) + pad
    ax.set_ylim(y_lo, y_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_ylabel("Observed glucose drop (mg/dL)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, h1 + h2, l1 + l2, loc="center right",
            bbox_to_anchor=(0.98, 0.62), fontsize=11)
    ax.grid(alpha=0.18)
    img_f = _b64(fig)

    ink, mute = "#101075", "#3f2670"
    W = 1600
    margin = 10
    gutter = 14
    content_x = margin
    content_w = W - 2 * margin
    phase_gap = 72  # room for vertical arrows between phases

    # Phase 1 — flush to side edges
    p1_ty = 24
    r1_y = p1_ty + 14
    r1_h = 290
    r1_w = (content_w - gutter) / 2
    a_box = (content_x, r1_y, r1_w, r1_h)
    b_box = (content_x + r1_w + gutter, r1_y, r1_w, r1_h)

    # Phase 2
    p2_ty = r1_y + r1_h + phase_gap
    r2_y = p2_ty + 14
    r2_h = 300
    arr_w = 78
    r2_w = (content_w - 2 * arr_w) / 3
    c_box = (content_x, r2_y, r2_w, r2_h)
    d_box = (c_box[0] + r2_w + arr_w, r2_y, r2_w, r2_h)
    e_box = (d_box[0] + r2_w + arr_w, r2_y, r2_w, r2_h)

    # Phase 3 — same card size as A/B, centered
    p3_ty = r2_y + r2_h + phase_gap
    r3_y = p3_ty + 14
    r3_h = r1_h
    r3_w = r1_w
    f_box = (content_x + (content_w - r3_w) / 2, r3_y, r3_w, r3_h)
    H = int(r3_y + r3_h + 10)

    def _panel(x, y, w, h, href, title):
        ix, it, ib = 6, 34, 6
        return (
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{card}" stroke="#101010" stroke-width="1.5"/>\n'
            f'  <text x="{x + w / 2:.1f}" y="{y + 23:.1f}" text-anchor="middle" '
            f'font-size="19" font-weight="600" fill="{ink}">{title}</text>\n'
            f'  <image x="{x + ix:.1f}" y="{y + it:.1f}" '
            f'width="{w - 2 * ix:.1f}" height="{h - it - ib:.1f}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'href="data:image/png;base64,{href}"/>\n'
        )

    def _harr(x0, x1, y, lab):
        xm = 0.5 * (x0 + x1)
        lines = lab.split("\n")
        n = len(lines)
        line_h = 18
        y_last = y - 30
        y0 = y_last - line_h * (n - 1)
        tspans = "\n".join(
            f'    <tspan x="{xm:.1f}" y="{y0 + line_h * i:.1f}">{ln}</tspan>'
            for i, ln in enumerate(lines)
        )
        return (
            f'  <line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
            f'stroke="#101010" stroke-width="2.2" marker-end="url(#arr)"/>\n'
            f'  <text text-anchor="middle" font-size="15" font-style="italic" '
            f'fill="{mute}">\n{tspans}\n  </text>\n'
        )

    def _varr(x, y0, y1):
        return (
            f'  <line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
            f'stroke="#101010" stroke-width="2.2" marker-end="url(#arr)"/>\n'
        )

    cx = W / 2
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" '
        f'role="img" aria-labelledby="title desc">\n'
        '  <title id="title">DOCAS flowchart</title>\n'
        '  <desc id="desc">Phase 1: observational bias and target DRC. '
        "Phase 2: synthetic probes. Phase 3: audit after retrain.</desc>\n"
        "  <defs>\n"
        '    <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3.5" '
        'orient="auto" markerUnits="strokeWidth">\n'
        '      <path d="M0,0 L7,3.5 L0,7 z" fill="#101010"/>\n'
        "    </marker>\n"
        "  </defs>\n"
        f'  <rect width="{W}" height="{H}" fill="{canvas}"/>\n'
        f'  <text x="{cx}" y="{p1_ty}" text-anchor="middle" font-size="20" '
        f'font-weight="700" fill="{ink}">Phase 1 · stated target vs observational bias</text>\n'
        + _panel(*a_box, img_a, "A · Observational cloud (confounding)")
        + _panel(*b_box, img_b, "B · Stated target DRC")
        + _varr(cx, r1_y + r1_h + 10, p2_ty - 18)
        + f'  <text x="{cx}" y="{p2_ty}" text-anchor="middle" font-size="20" '
        f'font-weight="700" fill="{ink}">Phase 2 · synthetic do(u) rows</text>\n'
        + _panel(*c_box, img_c, "C · One real context")
        + _panel(*d_box, img_d, "D · Probes labelled by DRC")
        + _panel(*e_box, img_e, "E · Real + synthetic cloud")
        + _harr(c_box[0] + r2_w + 6, d_box[0] - 6, r2_y + r2_h / 2, "rewrite\ntreatment")
        + _harr(d_box[0] + r2_w + 6, e_box[0] - 6, r2_y + r2_h / 2, "repeat")
        + _varr(cx, r2_y + r2_h + 10, p3_ty - 18)
        + f'  <text x="{cx}" y="{p3_ty}" text-anchor="middle" font-size="20" '
        f'font-weight="700" fill="{ink}">Phase 3 · train once &amp; audit</text>\n'
        + _panel(*f_box, img_f, "F · Partial dependence plot DOCAS vs Baseline")
        + "</svg>\n"
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES / "docas_flowchart.svg"
    svg_path.write_text(svg)
    out = FIGURES / "docas_flowchart.png"
    try:
        import shutil
        rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
        subprocess.run(
            [rsvg, "-d", "300", "-p", "300", str(svg_path), "-o", str(out)],
            check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        fig = plt.figure(figsize=(12.8, 10.0), facecolor="white")
        fig.text(0.5, 0.97, "DOCAS flowchart (install librsvg for GA-style export)",
                 ha="center", fontsize=10, color=mute)
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
    return out


def _ga_numbers() -> dict:
    """Headline numbers for the graphical abstract, read from the study JSONs."""
    red, cost = [], []
    for path in (m.CAT_COHORT, m.RESULTS / "azt1d" / "cohort.json"):
        cohort = json.loads(path.read_text())
        for ph in FORECAST_HORIZONS:
            mt = ohio_horizon_payload(cohort, ph)["metrics"]
            red.append(1.0 - mt["alignment_error"]["mean"]
                       / mt["baseline_alignment_error"]["mean"])
            cost.append(mt["docas_test_rmse_mgdl"]["mean"]
                        / mt["baseline_test_rmse_mgdl"]["mean"] - 1.0)
    rp = replay_cohort_path(replay_study._SCOPE["ohio_t1dm"])
    co = replay_horizon_payload(json.loads(rp.read_text()),
                                replay_study.REPLAY_HORIZONS[0])["cohort"]
    post = co["summary_post"]
    return dict(
        red_lo=100 * min(red), red_hi=100 * max(red),
        cost_lo=100 * min(cost), cost_hi=100 * max(cost),
        tir_base=float(post["baseline"]["tir"]["mean"]),
        tir_docas=float(post["docas"]["tir"]["mean"]),
        tbr_docas=float(post["docas"]["tbr"]["mean"]),
    )


def graphical_abstract_figure() -> Path:
    """Three-step graphical abstract: confounded logs, DOCAS, aligned response."""
    import shutil
    import subprocess

    crops = _flowchart_panel_crops(panel_bg="#ffffff", label_fs=11)
    k = _ga_numbers()

    ink, mute, canvas, line = "#1a1a4e", "#4a4a6a", "#eef0f6", "#c5c9d8"
    W, H = 1600, 620
    mx, gap = 28, 58
    cw = (W - 2 * mx - 2 * gap) / 3
    cy, ch = 108, 486
    xs = [mx + i * (cw + gap) for i in range(3)]
    accents = [BASELINE_COLOR, DOCAS_COLOR, TARGET_COLOR]
    titles = ["Confounded clinical logs", "DOCAS: realistic synthetic doses",
              "Aligned insulin response"]

    def t(x, y, s, size=14, weight="400", fill=ink, anchor="middle", style="normal"):
        return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
                f'font-weight="{weight}" font-style="{style}" fill="{fill}">{s}</text>\n')

    def img(x, y, w, h, key):
        return (f'  <image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{crops[key]}"/>\n')

    def chip(x, y, w, label, fill, stroke):
        return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="38" rx="19" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>\n'
                + t(x + w / 2, y + 25, label, 14, "600"))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" '
        'role="img" aria-labelledby="title desc">\n',
        '  <title id="title">DOCAS graphical abstract</title>\n',
        '  <desc id="desc">Confounded logs teach forecasters that insulin raises '
        'glucose; DOCAS adds synthetic dose rows labelled by a target dose-response '
        'curve; the aligned forecaster follows the curve at a small accuracy cost.</desc>\n',
        '  <defs>\n'
        '    <marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="4" '
        'orient="auto" markerUnits="strokeWidth">\n'
        f'      <path d="M0,0 L8,4 L0,8 z" fill="{ink}"/>\n'
        '    </marker>\n'
        '    <filter id="sh" x="-5%" y="-5%" width="110%" height="115%">\n'
        '      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1a1a4e" '
        'flood-opacity="0.12"/>\n'
        '    </filter>\n'
        '  </defs>\n',
        f'  <rect width="{W}" height="{H}" fill="{canvas}"/>\n',
        t(W / 2, 36, "Insulin dose–response alignment for glucose forecasting", 28, "700"),
        t(W / 2, 64, "Held-out accuracy does not test the insulin response.", 16, "600", mute),
        t(W / 2, 86, "(1) Alignment error quantifies it.  "
          "(2) DOCAS corrects it with synthetic doses.", 16, "400", mute),
    ]

    for i, (x, col, title) in enumerate(zip(xs, accents, titles)):
        parts += [
            f'  <rect x="{x:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" rx="16" '
            f'fill="#ffffff" stroke="{line}" stroke-width="1.2" filter="url(#sh)"/>\n',
            f'  <rect x="{x:.1f}" y="{cy}" width="{cw:.1f}" height="8" rx="4" fill="{col}"/>\n',
            f'  <circle cx="{x + 34:.1f}" cy="{cy + 42}" r="17" fill="{col}"/>\n',
            t(x + 34, cy + 48, str(i + 1), 18, "700", "#ffffff"),
            t(x + 62, cy + 49, title, 19, "700", ink, "start"),
        ]
        if i < 2:
            ya = cy + ch / 2
            parts.append(f'  <line x1="{x + cw + 10:.1f}" y1="{ya}" x2="{x + cw + gap - 12:.1f}" '
                         f'y2="{ya}" stroke="{ink}" stroke-width="2.6" marker-end="url(#arr)"/>\n')

    x = xs[0]
    parts += [
        img(x + 14, cy + 72, cw - 28, 300, "a"),
        t(x + cw / 2, cy + 408, "Insulin is given when glucose is already high.", 15, "600"),
        t(x + cw / 2, cy + 432, "The dose–response curve falls with more insulin,", 15),
        t(x + cw / 2, cy + 454, "but the model learns the rising line in the logs.", 15),
        t(x + cw / 2, cy + 470 - 4 + 0, "", 1),
    ]

    x = xs[1]
    w1, w2, w3 = 128, 150, 118
    row = w1 + w2 + w3 + 2 * 36
    x0 = x + (cw - row) / 2
    ycr = cy + 392
    parts += [
        img(x + 14, cy + 72, cw - 28, 300, "e"),
        chip(x0, ycr, w1, "Observed rows", "#f3f4f8", "#90A4AE"),
        t(x0 + w1 + 18, ycr + 26, "+", 22, "700"),
        chip(x0 + w1 + 36, ycr, w2, "Synthetic doses", "#e0f2f1", DOCAS_COLOR),
        t(x0 + w1 + w2 + 54, ycr + 26, "→", 22, "700"),
        chip(x0 + w1 + w2 + 72, ycr, w3, "Train once", "#ede7f6", TARGET_COLOR),
        t(x + cw / 2, cy + 454, "Each row copies a real situation at an insulin dose", 15),
        t(x + cw / 2, cy + 476, "never given, labelled from the dose–response curve.", 15),
    ]

    x = xs[2]
    tw, th, tg = (cw - 28 - 2 * 12) / 3, 104, 12
    ty = cy + 358
    tiles = [
        (f"{k['red_lo']:.0f}–{k['red_hi']:.0f}%", "lower alignment", "error vs baseline", DOCAS_COLOR),
        (f"+{k['cost_lo']:.1f}–{k['cost_hi']:.1f}%", "higher RMSE", "small cost", BASELINE_COLOR),
        (f"{k['tir_base']:.1f} → {k['tir_docas']:.1f}%", "time in range",
         f"{k['tbr_docas']:.0f}% below range", TARGET_COLOR),
    ]
    parts.append(img(x + 14, cy + 72, cw - 28, 252, "f"))
    for j, (big, lab, sub, col) in enumerate(tiles):
        tx = x + 14 + j * (tw + tg)
        parts += [
            f'  <rect x="{tx:.1f}" y="{ty}" width="{tw:.1f}" height="{th}" rx="12" '
            f'fill="#f7f8fb" stroke="{line}" stroke-width="1.1"/>\n',
            f'  <rect x="{tx:.1f}" y="{ty + 14}" width="4" height="{th - 28}" rx="2" fill="{col}"/>\n',
            t(tx + tw / 2, ty + 40, big, 18, "700", col),
            t(tx + tw / 2, ty + 66, lab, 14, "600"),
            t(tx + tw / 2, ty + 87, sub, 13, "400", mute, style="italic"),
        ]
    parts.append("</svg>\n")
    svg = "".join(pt for pt in parts if 'font-size="1"' not in pt)

    FIGURES.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES / "graphical_abstract.svg"
    svg_path.write_text(svg)
    out = FIGURES / "graphical_abstract.png"
    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-z", "2", str(svg_path), "-o", str(out)],
                   check=True, capture_output=True)
    return out


def alignment_pareto_figure() -> Path:
    """RMSE-vs-alignment frontier over the synthetic budget, per horizon."""
    budget_path = ROOT / "results" / "ablation" / "budget_ohio.json"
    cohort = json.loads(ohio_cohort_path().read_text())
    bud = json.loads(budget_path.read_text())
    fig, axes = plt.subplots(1, len(FORECAST_HORIZONS), figsize=(9.0, 3.9))
    for ax, ph in zip(np.atleast_1d(axes), FORECAST_HORIZONS):
        hz = bud["horizons"][str(ph)]
        met = hz["metrics"]
        ae = [met[f"{b:g}x"]["alignment_error"]["mean"] for b in m_budgets(bud)]
        rm = [met[f"{b:g}x"]["test_rmse_mgdl"]["mean"] for b in m_budgets(bud)]
        rows = cohort["horizons"][str(ph)]["per_patient"]
        base = hz.get("baseline") or {}
        b_ae = float((base.get("alignment_error") or {}).get(
            "mean", np.mean([r["baseline_alignment_error"] for r in rows])))
        b_rm = float((base.get("test_rmse_mgdl") or {}).get(
            "mean", np.mean([r["baseline_test_rmse_mgdl"] for r in rows])))
        # budget → 0 recovers the observational baseline, so close the frontier there
        order = np.argsort(m_budgets(bud))
        ax.plot([b_ae] + [ae[i] for i in order], [b_rm] + [rm[i] for i in order],
                "-", color=DOCAS_COLOR, lw=1.4, zorder=2)
        ax.plot(ae, rm, "o", color=DOCAS_COLOR, ms=5, zorder=3,
                label="DOCAS (synthetic budget)")
        b_min = min(m_budgets(bud))
        # high budgets crowd the left end, so alternate the label side there
        for rank, i in enumerate(np.argsort(m_budgets(bud))[::-1]):
            b, x, y = m_budgets(bud)[i], ae[i], rm[i]
            below = b != b_min and rank % 2
            ax.annotate(f"{b:g}×", (x, y), textcoords="offset points",
                        xytext=((-6, -13) if b == b_min or below else (7, 4)),
                        ha=("right" if b == b_min or below else "left"),
                        fontsize=7.5, color=DOCAS_COLOR)
        ax.plot([b_ae], [b_rm], "s", color=BASELINE_COLOR, ms=7, zorder=4,
                label="observational baseline (0×)")
        lo, hi = min(rm + [b_rm]), max(rm + [b_rm])
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.6 * (hi - lo))
        x_hi = max(ae + [b_ae])
        ax.set_xlim(0.0, x_hi * 1.08)
        ax.set_xlabel("alignment error (mg/dL)")
        ax.set_ylabel("held-out RMSE (mg/dL)")
        ax.set_title(f"{ph} min", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
    _legend(np.atleast_1d(axes)[-1], loc="upper right")
    fig.tight_layout()
    return _save_fig(fig, "alignment_pareto")


ALIGN_EXAMPLE_PH = 60
ALIGN_EXAMPLE_PATIENT = "544"
ALIGN_IOB_CUT, ALIGN_CGM_CUT = 3.0, 180.0
# Short post-meal windows: (pid, day, t0_h, t1_h, forecast_origin_h).
ALIGN_SCATTER_PIDS = None
ALIGN_BAD_MEAL = ("544", "2027-06-25", 13.45, 15.70, 14.583)
ALIGN_GOOD_MEAL = ("563", "2021-11-04", 9.50, 11.55, 10.35)


def _inv_1d(scaler, a) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(a, float).reshape(-1, 1)).ravel()


def _delta_mgdl(model, X, dy_s) -> np.ndarray:
    return _inv_1d(dy_s, model.predict(np.asarray(X, float)))


def _rewrite_feature(data, X, name: str, v) -> np.ndarray:
    X = np.asarray(X, float).copy()
    j = int(data.feature_names.index(name))
    u = np.broadcast_to(np.asarray(v, float).ravel(), (len(X),))
    X[:, j] = data.train_scalers[j].transform(u.reshape(-1, 1)).ravel()
    return X


def _centred_feat(model, data, X, name: str) -> np.ndarray:
    """Ceteris paribus ΔCGM of the factual feature versus zero."""
    y_u = _delta_mgdl(model, X, data.cgm_scaler)
    y_0 = _delta_mgdl(model, _rewrite_feature(data, X, name, 0.0), data.cgm_scaler)
    return y_u - y_0


def _ice_surface(data, model, g, u, cho: float) -> np.ndarray:
    """Centred ICE on a (CGM × IOB) grid; CHO held at the training median."""
    ji, ci = int(data.insulin_idx), int(data.feature_names.index("CHO"))
    G, U = np.meshgrid(g, u, indexing="ij")
    X = np.empty((G.size, len(data.feature_names)))
    X[:, 0] = data.train_scalers[0].transform(G.ravel().reshape(-1, 1)).ravel()
    X[:, ji] = data.train_scalers[ji].transform(U.ravel().reshape(-1, 1)).ravel()
    X[:, ci] = data.train_scalers[ci].transform(np.full((G.size, 1), cho)).ravel()
    X0 = X.copy()
    X0[:, ji] = data.train_scalers[ji].transform(np.zeros((G.size, 1))).ravel()
    return (_delta_mgdl(model, X, data.cgm_scaler)
            - _delta_mgdl(model, X0, data.cgm_scaler)).reshape(G.shape)


def _bin_mean(x, y, val, x_edges, y_edges) -> np.ndarray:
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)
    acc = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    cnt = np.zeros_like(acc)
    inside = ((x >= x_edges[0]) & (x <= x_edges[-1])
              & (y >= y_edges[0]) & (y <= y_edges[-1]))
    np.add.at(acc, (yi[inside], xi[inside]), val[inside])
    np.add.at(cnt, (yi[inside], xi[inside]), 1)
    out = np.full_like(acc, np.nan)
    ok = cnt > 0
    out[ok] = acc[ok] / cnt[ok]
    return out


def _heldout_day(pid: str, day: str, ph: int = ALIGN_EXAMPLE_PH) -> dict:
    """One Ohio test calendar day: CGM, events, and 60-min predicted glucose."""
    import pandas as pd
    import preprocessing as pp
    m.set_dataset("ohio_t1dm")
    phs = horizon_steps(ph)
    data = load_patient(str(DATA_ROOT), pid, horizon_steps=phs)
    path = pp._patient_xml(str(DATA_ROOT), pid, "test")
    pat = pp._parse(path)
    grid, cgm_g = pp._uniform_grid(pat["gluc"])
    df = pp._frame_from_xml(path, interpolate_cgm=False, ph_steps=phs)
    bundle = m._load_bundle(ph, pid)
    t = pd.to_datetime(grid)
    day_ts = pd.Timestamp(day)
    on_day = (t.normalize() == day_ts.normalize())
    hours = (t - day_ts.normalize()).total_seconds().to_numpy() / 3600.0
    mt, mv = pp._ns([(x, v) for x, v in pat["meal"] if v > 0])
    bt, bv = pp._ns(pat["bolus"])
    meal_t, bol_t = pd.to_datetime(mt), pd.to_datetime(bt)
    meals = [(float((pd.Timestamp(x) - day_ts.normalize()).total_seconds() / 3600.0),
              float(v))
             for x, v in zip(meal_t, mv)
             if pd.Timestamp(x).normalize() == day_ts.normalize()]
    bols = [(float((pd.Timestamp(x) - day_ts.normalize()).total_seconds() / 3600.0),
             float(v))
            for x, v in zip(bol_t, bv)
            if pd.Timestamp(x).normalize() == day_ts.normalize()]
    iob = pp._causal_horizon_iob(grid, bt, bv, phs)
    cho = pp._causal_horizon_cho(grid, mt, mv, phs)
    y_raw = pp._target_delta(df["CGM"].values, 1, phs)
    idx = np.arange(len(y_raw))
    keep = np.isfinite(y_raw) & np.isfinite(df["CGM"].values[idx])
    t_fut = t[idx[keep] + phs]
    fut_on = t_fut.normalize() == day_ts.normalize()
    d_b = _delta_mgdl(bundle["baseline"], data.X_test, data.cgm_scaler)
    d_d = _delta_mgdl(bundle["docas"], data.X_test, data.cgm_scaler)
    now_g = df["CGM"].values[idx[keep]]
    h_fut = ((t_fut - day_ts.normalize()).total_seconds().to_numpy() / 3600.0)[fut_on]
    return dict(
        pid=str(pid), day=day, ph=int(ph),
        hours=hours[on_day], cgm=cgm_g[on_day],
        iob=iob[on_day], cho=cho[on_day],
        h_pred=h_fut, pred_b=now_g[fut_on] + d_b[fut_on],
        pred_d=now_g[fut_on] + d_d[fut_on],
        meals=meals, boluses=bols,
    )


def _origin_forecast(day: dict, origin: float) -> tuple[float, ...]:
    """Snap origin to a 60-min forecast; return origin, arrival, CGM, preds."""
    h, g = day["hours"], day["cgm"]
    hp = day["h_pred"]
    ok = np.isfinite(h) & np.isfinite(g)
    g_ok, h_ok = g[ok], h[ok]
    arrival = origin + day["ph"] / 60.0
    j = int(np.argmin(np.abs(hp - arrival)))
    arrival = float(hp[j])
    origin = arrival - day["ph"] / 60.0
    current = float(np.interp(origin, h_ok, g_ok))
    actual = float(np.interp(arrival, h_ok, g_ok))
    return (origin, arrival, current, actual,
            float(day["pred_b"][j]), float(day["pred_d"][j]))


def _hhmm(h: float) -> str:
    tot = int(round(float(h) * 60.0))
    return f"{tot // 60:02d}:{tot % 60:02d}"


def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.12, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="right", va="bottom", clip_on=False)


def _plot_meal(ax, day: dict, t0: float, t1: float, origin: float,
               *, title: str, ylabel: bool, xlabel: bool = False,
               letter: str | None = None, mark: dict | None = None) -> float:
    h, g = day["hours"], day["cgm"]
    in_win = (h >= t0) & (h <= t1)
    ok = np.isfinite(h) & np.isfinite(g)
    ax.plot(h[in_win & ok], g[in_win & ok], color="#37474F", lw=2.0, zorder=5,
            label="CGM")
    origin, arrival, current, actual, baseline, docas = _origin_forecast(
        day, origin)
    ax.plot([origin, arrival], [current, baseline], color=BASELINE_COLOR,
            ls="--", lw=1.6, zorder=6, label="Baseline forecast")
    ax.plot([origin, arrival], [current, docas], color=DOCAS_COLOR,
            lw=1.6, zorder=6, label="DOCAS forecast")
    ax.scatter([arrival], [actual], s=36, color="#37474F", zorder=7)
    ax.scatter([arrival], [baseline], s=36, marker="s",
               color=BASELINE_COLOR, zorder=7)
    ax.scatter([arrival], [docas], s=36, color=DOCAS_COLOR, zorder=7)
    if mark:
        ax.scatter([origin], [current], s=55, c=mark["color"],
                   marker=mark["marker"], edgecolors="white", linewidths=0.8,
                   zorder=8)
    shown = np.concatenate([g[in_win & ok], [baseline, docas]])
    shown = shown[np.isfinite(shown)]
    lo, hi = float(np.min(shown)), float(np.max(shown))
    ax.set_ylim(lo - 12, hi + 18)
    ax.set_title(title, fontsize=10, pad=3)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=8)
    if ylabel:
        ax.set_ylabel("CGM (mg/dL)", fontsize=9)
    ax.set_xlim(t0, t1)
    ticks = np.arange(np.ceil(t0 * 2) / 2, t1 + 1e-6, 0.5)
    ax.set_xticks(ticks, [_hhmm(x) for x in ticks])
    if xlabel:
        ax.set_xlabel("Time", fontsize=9)
    else:
        ax.tick_params(labelbottom=False)
    if letter:
        _panel_letter(ax, letter)
    return origin


def _row_times(df, grid, lb: int, ph: int):
    import pandas as pd
    import preprocessing as pp
    y = pp._target_delta(df["CGM"].values, lb, ph)
    X = np.concatenate(
        [pp._roll_features(np.asarray(df[c], float), lb, ph) for c in pp.FEATURE_COLUMNS],
        axis=1)
    nan = np.union1d(np.where(np.isnan(X))[0], np.where(np.isnan(y))[0])
    keep = np.ones(len(y), dtype=bool)
    keep[nan] = False
    t = pd.to_datetime(np.asarray(grid)[lb - 1: lb - 1 + len(y)])
    return t[keep]


def _test_times(dataset: str, pid: str, ph: int):
    import pandas as pd
    import preprocessing as pp
    phs = horizon_steps(ph)
    lb = int(pp.LOOKBACK)
    if dataset == "ohio_t1dm":
        path = pp._patient_xml(str(DATA_ROOT), pid, "test")
        grid, _ = pp._uniform_grid(pp._parse(path)["gluc"])
        df = pp._frame_from_xml(path, interpolate_cgm=False, ph_steps=phs)
        return _row_times(df, grid, lb, phs)
    raw = pp._normalize_columns(pd.read_csv(pp._subject_csv(str(DATA_ROOT), pid)))
    raw["EventDateTime"] = pd.to_datetime(raw["EventDateTime"])
    t0, t1 = raw["EventDateTime"].iloc[0], raw["EventDateTime"].iloc[-1]
    grid_ts = pd.date_range(t0.floor("5min"), t1.ceil("5min"), freq="5min")
    df = pp._frame_from_csv(
        pp._subject_csv(str(DATA_ROOT), pid), interpolate_cgm=True, ph_steps=phs)
    ok = ~np.isnan(df["CGM"].values)
    i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
    df = df.iloc[i0:i1]
    n = len(df)
    cut = max(int(n * pp.TRAIN_FRAC), lb + phs + 50)
    cut = min(cut, n - (lb + phs + 20))
    return _row_times(df.iloc[cut:], grid_ts[i0:i1][cut:], lb, phs)


def _alignment_pred_rows(ph: int = ALIGN_EXAMPLE_PH,
                         pids: tuple[str, ...] | None = ALIGN_SCATTER_PIDS) -> list[dict]:
    """Every Ohio held-out 60-min prediction with features, ICE, and d_err."""
    prev = m.DATASET
    m.set_dataset("ohio_t1dm")
    out = []
    if pids is None:
        pids = tuple(str(p) for p in m.list_patients(str(DATA_ROOT)))
    for pid in pids:
        data = m.load_patient(str(DATA_ROOT), str(pid),
                              horizon_steps=horizon_steps(ph))
        bundle = m._load_bundle(ph, str(pid))
        X = np.asarray(data.X_test, float)
        ji = int(data.insulin_idx)
        ci = int(data.feature_names.index("CHO"))
        iob = _inv_1d(data.train_scalers[ji], X[:, ji])
        cgm = _inv_1d(data.train_scalers[0], X[:, 0])
        cho = _inv_1d(data.train_scalers[ci], X[:, ci])
        y_mg = _inv_1d(data.cgm_scaler, data.y_test)
        span = float(insulin_span_u(data, horizon_min=ph))
        tau = _target_ice_at(ph, iob, cgm, span)
        ice_b = _centred_ice(bundle["baseline"], data, X, iob)
        ice_d = _centred_ice(bundle["docas"], data, X, iob)
        leftover = y_mg - tau
        pb = _delta_mgdl(bundle["baseline"], X, data.cgm_scaler)
        pd_ = _delta_mgdl(bundle["docas"], X, data.cgm_scaler)
        shift = pd_ - pb
        d_err = np.abs(pb - y_mg) - np.abs(pd_ - y_mg)
        hi = iob > 1.0
        for i in range(len(y_mg)):
            out.append(dict(
                patient_id=str(pid),
                cgm=float(cgm[i]), iob=float(iob[i]), cho=float(cho[i]),
                y=float(y_mg[i]), tau=float(tau[i]),
                pred_b=float(pb[i]), pred_d=float(pd_[i]),
                shift=float(shift[i]),
                ice_baseline=float(ice_b[i]), ice_docas=float(ice_d[i]),
                leftover=float(leftover[i]),
                ice_misalign=float(ice_b[i] - tau[i]),
                obs_vs_baseline=float(y_mg[i] - ice_b[i]),
                cho_hi_iob=float(cho[i] * hi[i]),
                leftover_hi_iob=float(leftover[i] * hi[i]),
                d_err=float(d_err[i])))
    m.set_dataset(prev)
    return out


ALIGN_EXPLAINER_WINNER = "y"
ALIGN_EXPLAINER_LABEL = "Observed 60-minute $\\Delta$CGM (mg/dL)"
ALIGN_EXPLAINER_TITLE = (
    "When DOCAS lowers the forecast: helps if glucose fell, hurts if it rose")
ALIGN_SHIFT_DOWN = 0.0


def _alignment_down_rows(ph: int = ALIGN_EXAMPLE_PH) -> list[dict]:
    """Hours where DOCAS pulled the 60-min forecast down."""
    return [r for r in _alignment_pred_rows(ph) if r["shift"] < ALIGN_SHIFT_DOWN]


def _plot_alignment_cohort(ax, examples: list[dict], *,
                           xkey: str = ALIGN_EXPLAINER_WINNER,
                           xlabel: str = ALIGN_EXPLAINER_LABEL,
                           title: str = ALIGN_EXPLAINER_TITLE) -> float:
    """PH60: downward-shift hours, observed Δ vs |error| gain, all Ohio."""
    rows = _alignment_down_rows(ALIGN_EXAMPLE_PH)
    x = np.array([r[xkey] for r in rows])
    y = np.array([r["d_err"] for r in rows])
    r_all = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    ax.scatter(x, y, c="#90A4AE", s=6, marker="o", alpha=0.22,
               edgecolors="none", zorder=3, rasterized=True)
    if len(x) > 2:
        mfit, b = np.polyfit(x, y, 1)
        pad = 0.05 * (float(x.max()) - float(x.min()) or 1.0)
        xs = np.linspace(x.min() - pad, x.max() + pad, 2)
        ax.plot(xs, mfit * xs + b, color="0.35", lw=1.2, zorder=2)
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axvline(0.0, color="0.55", lw=0.8)
    for ex in examples:
        ax.scatter([ex["x"]], [ex["y"]], s=70, c=ex["color"], marker=ex["marker"],
                   edgecolors="white", linewidths=0.9, zorder=6)
        ax.annotate(
            ex["label"], (ex["x"], ex["y"]), xytext=ex["offset"],
            textcoords="offset points", fontsize=10, fontweight="bold",
            color=ex["color"], ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=ex["color"], lw=0.9,
                            shrinkA=0, shrinkB=4))
    ax.set_xlabel(xlabel, fontsize=9, labelpad=5)
    ax.set_ylabel("Pointwise absolute error (baseline $-$ DOCAS)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(0.98, 0.04, f"Pearson $r={r_all:.2f}$  $n={len(rows)}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.25")
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=8)
    _panel_letter(ax, "A")
    return r_all


def alignment_individual_figure() -> Path:
    """Downward-shift scatter plus two CGM windows (no TreeSHAP)."""
    bpid, bday, bt0, bt1, borigin = ALIGN_BAD_MEAL
    gpid, gday, gt0, gt1, gorigin = ALIGN_GOOD_MEAL
    bad, good = _heldout_day(bpid, bday), _heldout_day(gpid, gday)

    def _ex(day, origin, letter, color, marker, offset):
        o, _, cur, act, base, doc = _origin_forecast(day, origin)
        return dict(
            origin=o, x=act - cur, y=abs(base - act) - abs(doc - act),
            label=letter, color=color, marker=marker, offset=offset)

    examples = [
        _ex(bad, borigin, "B", "#5D4037", "o", (10, -16)),
        _ex(good, gorigin, "C", "#EF6C00", "s", (-22, 12)),
    ]
    fig = plt.figure(figsize=(10.8, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38,
                          wspace=0.28, top=0.92, bottom=0.10, left=0.10,
                          right=0.98)
    ax_cohort = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    _plot_alignment_cohort(ax_cohort, examples)
    _plot_meal(ax_a, bad, bt0, bt1, examples[0]["origin"], ylabel=True,
               xlabel=True, letter="B", mark=examples[0],
               title=f"Patient {bad['pid']} · meal offsets insulin")
    _plot_meal(ax_b, good, gt0, gt1, examples[1]["origin"], ylabel=False,
               xlabel=True, letter="C", mark=examples[1],
               title=f"Patient {good['pid']} · insulin dominates")
    h, l = ax_a.get_legend_handles_labels()
    _legend(ax_b, h, l, loc="upper right")
    return _save_fig(fig, "alignment_individual")


def m_budgets(payload: dict) -> list[float]:
    return [float(b) for b in payload["budgets"]]


def _rewrite_insulin(data, X, u_abs) -> np.ndarray:
    X = np.asarray(X, float).copy()
    ji = int(data.insulin_idx)
    u = np.broadcast_to(np.asarray(u_abs, float).ravel(), (len(X),))
    X[:, ji] = data.train_scalers[ji].transform(u.reshape(-1, 1)).ravel()
    return X


def _centred_ice(model, data, X, u_abs) -> np.ndarray:
    y_u = _delta_mgdl(model, _rewrite_insulin(data, X, u_abs), data.cgm_scaler)
    y_0 = _delta_mgdl(model, _rewrite_insulin(data, X, 0.0), data.cgm_scaler)
    return y_u - y_0


def write_all_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (
        dose_response_figure,
        patient588_replay_window_figure,
        safeguard_patient_figure,
        iob_pharmacology_figure,
        context_adjust_figure,
        docas_flowchart_figure,
        graphical_abstract_figure,
        alignment_individual_figure,
    ):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skip figure {fn.__name__}: {e}", flush=True)


def run_pdp_mean_ices() -> None:
    """Ohio mean insulin curves used by the dose–response figure."""
    from ablation import _fit_ctx, _pop_R
    from core import docas_fit_kw
    from forecasters import PenaltyMLPForecaster, train_baseline, train_mlp, train_monotone
    out_path = ROOT / "results" / "revision" / "pdp_mean_ices.json"
    u_grid = np.linspace(0.0, 5.0, 21)
    arms = ("baseline", "mlp", "monotone", "penalty", "pop", "ctx")

    def thetas(ph: int) -> dict:
        found = {}
        for p in sorted((ROOT / "results" / "revision" / "full_recent").glob("OhioT1DM__*.json")):
            d = json.loads(p.read_text())
            if "horizons" in d:
                found[d["patient_id"].split("|", 1)[1]] = d["horizons"][str(ph)]["theta"]
        return found

    def models(pid, ph, theta):
        m.set_dataset("ohio_t1dm")
        data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
        span = float(m.insulin_span_u(data, horizon_min=ph))
        _, R_tab = _pop_R(ph, span, m.STUDY_CFG)
        fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=m.STUDY_CFG)
        DOCAS.SPAN, DOCAS.SEED = span, 42
        sync_cfg(SEED=42)
        base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
        mlp = train_mlp(data.X_train, data.y_train, model_kwargs={"seed": 42})
        mono = train_monotone(
            data.X_train, data.y_train, insulin_idx=data.insulin_idx,
            model_kwargs={"random_seed": 42})
        pop = DOCAS(**{**fit_kw, "seed": 42}).fit(
            data.X_train, data.y_train, baseline_model=base)["model"]
        ctx, _ = _fit_ctx(data, span, ph, theta, R_tab, seed=42, base_mdl=base)
        aligner = DOCAS(**{**fit_kw, "seed": 42})
        pen = PenaltyMLPForecaster(aligner=aligner, seed=42).fit(data.X_train, data.y_train)
        return dict(data=data, span=span, baseline=base, mlp=mlp, monotone=mono, penalty=pen, pop=pop, ctx=ctx)

    def one(ph: int) -> dict:
        per = []
        for pid, th in thetas(ph).items():
            pack = models(pid, ph, th)
            data, X, span = pack["data"], np.asarray(pack["data"].X_test, float), pack["span"]
            u_abs = np.clip(u_grid, 0.0, span)
            curves = {
                arm: [round(float(np.mean(_centred_ice(pack[arm], data, X, u))), 3) for u in u_abs]
                for arm in arms
            }
            yi = data.feature_names.index("CGM")
            g = data.train_scalers[yi].inverse_transform(X[:, yi: yi + 1]).ravel()
            per.append(dict(
                patient_id=str(pid), span=span, cgm=round(float(np.mean(g)), 2),
                I=[round(float(x), 3) for x in u_abs], curves=curves))
        mean_abs = {arm: [] for arm in arms}
        mean_rel = {arm: [] for arm in arms}
        for i, u in enumerate(u_grid):
            for arm in arms:
                vals = [r["curves"][arm][i] for r in per if u <= r["span"] + 1e-9]
                rel = [r["curves"][arm][i] * 100.0 / r["cgm"] for r in per if u <= r["span"] + 1e-9]
                mean_abs[arm].append(round(float(np.mean(vals)), 3) if vals else None)
                mean_rel[arm].append(round(float(np.mean(rel)), 3) if rel else None)
        return dict(I=[round(float(x), 3) for x in u_grid], mean=mean_abs, mean_rel=mean_rel, per_patient=per)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(ph): one(ph) for ph in (30, 60)}))


if __name__ == "__main__":
    run_pdp_mean_ices()
    write_all_figures()
