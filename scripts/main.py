"""First-submission study on OhioT1DM and AZT1D.

uv run python scripts/main.py

The revision (MetaboNet full records, seed 42, latest 8,000 training rows)
is the scripts listed in the README, not this entry point.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for k in ("OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"):
    os.environ.setdefault(f"{k}_NUM_THREADS", "1")

import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT
sys.path[:0] = [str(ROOT / "scripts")]

from forecasters import train_baseline, train_mlp
import preprocessing as pp
from preprocessing import horizon_steps, rmse_mgdl
from core import (
    Cfg, DOCAS, calibrated_target, docas_fit_kw, future, sync_cfg,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# STUDY KNOBS
# =============================================================================
LOAD = False
SEED = 42
BODY_WEIGHT_KG = 75.0
SI_MODE = 2.3 * 5e-4  # mode of ReplayBG's SI prior, Gamma(alpha=3.3, beta=1/5e-4)
G0_TAU = 220.0
N_GRID_POINTS = 80
FORECAST_HORIZONS = (30, 60)
ABS_DOSE_GRID_U = 5.0  # common absolute audit window (U), span-free AE
# Weight on alignment error when picking a validation budget (RMSE + λ·AE).
AE_WEIGHT = 1.0
BUDGET_CAP = 20.0  # largest selectable budget; bigger ones are swept for the frontier only

ABS = dict(
    synth_ratio=20,   # synth rows ≈ this × train (selected at AE_WEIGHT; ablation.run_budget_sweep)
    emp_frac=0.5,     # IOB mix: empiric vs Uniform[0,1]
    ctx_jitter=10,    # context noise σ (CGM mg/dL, CHO g)
)
# Primary protocol: fractional τ scaled by the ambient CGM at dosing time.
STUDY_CFG = {
    **ABS, "SI": SI_MODE, "body_weight_kg": BODY_WEIGHT_KG,
    "tau_mode": "fractional", "drop_mode": "endpoint",
    "level_ref": "current", "fit_space": "delta",
}
# Supplementary absolute-τ sensitivity (fixed G0 drop).
ABS_CFG = {
    **ABS, "SI": SI_MODE, "body_weight_kg": BODY_WEIGHT_KG,
    "tau_mode": "absolute", "drop_mode": "endpoint",
    "level_ref": "current", "fit_space": "delta",
}
SEEDS = (42, 43, 44, 45, 46)
MLP_JOBS = max(1, (os.cpu_count() or 4) - 2)  # parallel (patient, seed) MLP fits

SENSOR = {
    "ohio_t1dm": dict(profile="rapid_sq_enlite", tau=8.0, alpha=10.0),
    "azt1d": dict(profile="rapid_sq_g6", tau=8.0, alpha=5.0),
    "hupa": dict(profile="rapid_sq_g6", tau=8.0, alpha=5.0),
    "metabonet": dict(profile="rapid_sq_g6", tau=8.0, alpha=5.0),
    "metabonet_full": dict(profile="rapid_sq_g6", tau=8.0, alpha=5.0),
}
# ReplayBG single-meal glucose model → absolute-τ target R(u) and drop scale (core.replaybg_response)
RBG = dict(
    Gb=119,           # euglycemic setpoint for hypo-risk term (mg/dL)
    SG=0.017,         # glucose effectiveness
    p2=0.11,          # insulin-action coupling
    SI=SI_MODE,       # insulin sensitivity (study lock; overrides ReplayBG default)
    ka2=0.011,        # subcutaneous insulin PK rate 2
    kd=0.021,         # subcutaneous insulin PK rate 1
    ke=0.127,         # plasma insulin elimination
    VG=1.45,          # glucose distribution volume (dL/kg)
    VI=0.135,         # interstitial insulin volume (L/kg)
    alpha=7,          # CGM sensor smoothing (dataset SENSOR overrides at runtime)
    tau=8,            # bolus delivery lag (min; SENSOR overrides at runtime)
    r1=1.44,          # hypo-risk curve shape
    r2=0.81,          # hypo-risk curve exponent
)

RESULTS, DATASET = ROOT / "results", "ohio_t1dm"
# =============================================================================

sync_cfg(
    LOAD=LOAD, SEED=SEED, N_GRID_POINTS=N_GRID_POINTS,
    SI_MODE=SI_MODE, BODY_WEIGHT_KG=BODY_WEIGHT_KG, G0_TAU=G0_TAU,
    DATASET=DATASET, SENSOR=SENSOR, RBG=RBG, AE_WEIGHT=AE_WEIGHT,
)
DOCAS.train_fn = lambda X, y, model_kwargs=None, n_factual=None, sample_weight=None, **_: train_baseline(
    X, y, model_kwargs=model_kwargs, n_factual=n_factual, sample_weight=sample_weight)

list_patients, load_patient, load_patient_meta = (
    pp.list_patients_ohio, pp.load_patient_ohio, pp.load_patient_meta_ohio)
_DATASETS = {
    "ohio_t1dm": (pp.list_patients_ohio, pp.load_patient_ohio, pp.load_patient_meta_ohio),
    "azt1d": (pp.list_patients_azt1d, pp.load_patient_azt1d, pp.load_patient_meta_azt1d),
    "hupa": (pp.list_patients_hupa, pp.load_patient_hupa, pp.load_patient_meta_hupa),
    "metabonet": (pp.list_patients_metabonet, pp.load_patient_metabonet, pp.load_patient_meta_metabonet),
    "metabonet_full": (
        pp.list_patients_metabonet_full,
        pp.load_patient_metabonet_full,
        pp.load_patient_meta_metabonet_full,
    ),
}
_METRIC_KEYS = (
    "baseline_test_rmse_mgdl docas_test_rmse_mgdl "
    "alignment_error baseline_alignment_error "
    "alignment_error_abs baseline_alignment_error_abs "
    "ape_mgdl baseline_ape_mgdl "
    "train_alignment baseline_train_alignment"
).split()
CAT_COHORT = RESULTS / "ohio_t1dm" / "cohort.json"
MLP_COHORT = RESULTS / "ohio_t1dm" / "mlp_cohort.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def _protocol_meta() -> dict:
    c = STUDY_CFG
    return dict(
        protocol="fractional_tau",
        tau_mode=str(c.get("tau_mode", "fractional")),
        drop_mode=str(c.get("drop_mode", "endpoint")),
        level_ref=str(c.get("level_ref", "current")),
        fit_space=str(c.get("fit_space", "delta")),
        SI=float(c["SI"]), body_weight_kg=float(c["body_weight_kg"]),
        synth_ratio=float(c["synth_ratio"]),
        ctx_jitter=float(c["ctx_jitter"]),
        emp_frac=float(c["emp_frac"]),
        n_grid_points=int(DOCAS.N_GRID_POINTS),
        seeds=list(SEEDS),
        pharm_profile=str(dict(SENSOR.get(str(DATASET).lower().replace("-", "_"), SENSOR["ohio_t1dm"])).get("profile", "")),
    )


def results_dir() -> Path:
    p = RESULTS / DATASET
    p.mkdir(parents=True, exist_ok=True)
    return p


DSS_SPAN_FLOOR_U = 5.0


def insulin_span_u(data, horizon_min: int | None = None) -> float:
    """Probe span for synthetic doses: training-max bolus IOB, floored at 5 U.

    The floor matches the ReplayBG search. It uses training rows only, so a
    higher test dose cannot widen the span. Patients already above 5 U keep
    their training maximum.
    """
    del horizon_min
    ji = int(data.insulin_idx)
    a_s = data.train_scalers[ji]
    v = a_s.inverse_transform(np.asarray(data.X_train, float)[:, ji: ji + 1]).ravel()
    return float(max(float(np.nanmax(v)), DSS_SPAN_FLOOR_U))


def set_dataset(name: str = "ohio_t1dm") -> None:
    global DATASET, list_patients, load_patient, load_patient_meta
    name = str(name).lower().replace("-", "_")
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    DATASET = name
    Cfg.DATASET = name
    list_patients, load_patient, load_patient_meta = _DATASETS[name]


def mean_std(values) -> dict:
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1))}


def log_means(label: str, ph: int, rows: list[dict], rmse_key: str, align_key: str) -> None:
    print(f"  {label} PH{ph}:  test RMSE {mean_std([r[rmse_key] for r in rows])['mean']:.2f}  "
          f"align {mean_std([r[align_key] for r in rows])['mean']:.2f}", flush=True)


def ohio_cohort_path(scope: str = "") -> Path: return results_dir() / f"cohort{scope}.json"
def ohio_horizon_payload(cohort: dict, horizon_min: int) -> dict:
    h = dict((cohort.get("horizons") or {}).get(str(horizon_min)) or {})
    if "metrics" not in h and h.get("per_patient"):
        h["metrics"] = {k: mean_std([r[k] for r in h["per_patient"]]) for k in _METRIC_KEYS}
    return h


def model_path(horizon_min: int, patient_id: str, *, subdir: str = "models") -> Path:
    path = results_dir() / f"ph{horizon_min}" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{patient_id}.joblib"


def _load_bundle(horizon_min: int, patient_id: str, *, subdir: str = "models") -> dict:
    import forecasters
    sys.modules.setdefault("model", forecasters)
    return joblib.load(model_path(horizon_min, patient_id, subdir=subdir))


def save_models(horizon_min: int, patient_id: str, data, *, baseline=None, docas=None,
                docas_meta=None, subdir: str = "models") -> Path:
    path = model_path(horizon_min, patient_id, subdir=subdir)
    bundle = {}
    if path.is_file():
        try:
            bundle = dict(joblib.load(path))
        except Exception:
            bundle = {}
    for k, v in (("baseline", baseline), ("docas", docas), ("docas_meta", docas_meta)):
        if v is not None:
            bundle[k] = v
    bundle.update(feature_names=list(data.feature_names), patient_id=str(patient_id),
                  horizon_min=int(horizon_min))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_baseline_model(horizon_min: int, patient_id: str, *, subdir: str = "models"):
    return _load_bundle(horizon_min, patient_id, subdir=subdir)["baseline"]


def _patient_row(horizon_min, pid, data, baseline, docas, span, u_norm, *, meta=None) -> dict:
    span = float(span)
    DOCAS.SPAN = span
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    meta = meta or {}
    a_s, dy_s, y_s = data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]
    ctx_te, ctx_tr = np.asarray(data.X_test, float), np.asarray(data.X_train, float)
    bp = meta.get("best_params") or {}
    drop = float(meta["tau_drop"])
    u_ref = float(bp.get("u_ref", meta.get("u_ref", 1.0)))
    tau_mode = str(bp.get("tau_mode", meta.get("tau_mode", STUDY_CFG.get("tau_mode", "fractional"))))
    level_ref = str(bp.get("level_ref", meta.get("level_ref", STUDY_CFG.get("level_ref", "current"))))
    predicts_level = bool(meta.get("predicts_level", False))
    aligner = DOCAS(
        calibrated_target(int(horizon_min),
                          body_weight_kg=float(bp.get("body_weight_kg", BODY_WEIGHT_KG)),
                          SI=float(bp.get("SI", SI_MODE)), span=span,
                          drop_mode=str(bp.get("drop_mode", STUDY_CFG.get("drop_mode", "endpoint")))),
        intervention_idx=ji, outcome_idx=gi, action_scaler=a_s, level_scaler=y_s,
        delta_scaler=dy_s, span=span, tau_drop=drop, u_ref=u_ref, tau_mode=tau_mode,
        level_ref=level_ref, predicts_level=predicts_level,
        synth_ratio=float(bp.get("synth_ratio", ABS["synth_ratio"])),
    )
    train_a = aligner.alignment(docas, ctx_tr)
    train_b = aligner.alignment(baseline, ctx_tr)
    test_a = aligner.alignment(docas, ctx_te)
    test_b = aligner.alignment(baseline, ctx_te)
    abs_a = aligner.alignment_absolute(docas, ctx_te, max_dose_u=ABS_DOSE_GRID_U)
    abs_b = aligner.alignment_absolute(baseline, ctx_te, max_dose_u=ABS_DOSE_GRID_U)
    cd = future(docas, ctx_te, ji, gi, u_norm, span, a_s, dy_s, y_s, level=predicts_level)
    cb = future(baseline, ctx_te, ji, gi, u_norm, span, a_s, dy_s, y_s, level=False)
    rnd = lambda a, dig=2: [round(float(v), dig) for v in np.asarray(a, float).ravel()]
    base_rmse = float(rmse_mgdl(data.y_test, baseline.predict(data.X_test), dy_s))
    if predicts_level:
        # Convert level-space predictions → scaled ΔCGM for fair RMSE vs baseline.
        pred_level = y_s.inverse_transform(
            np.asarray(docas.predict(data.X_test), float).reshape(-1, 1)).ravel()
        cgm0 = y_s.inverse_transform(ctx_te[:, gi: gi + 1]).ravel()
        pred_delta_s = dy_s.transform((pred_level - cgm0).reshape(-1, 1)).ravel()
        docas_rmse = float(rmse_mgdl(data.y_test, pred_delta_s, dy_s))
    else:
        docas_rmse = float(rmse_mgdl(data.y_test, docas.predict(data.X_test), dy_s))
    return {
        "patient_id": str(pid), "horizon_min": int(horizon_min),
        "baseline_test_rmse_mgdl": base_rmse,
        "docas_test_rmse_mgdl": docas_rmse,
        "alignment_error": float(test_a), "baseline_alignment_error": float(test_b),
        "alignment_error_abs": float(abs_a), "baseline_alignment_error_abs": float(abs_b),
        "ape_mgdl": float(docas_rmse + test_a),
        "baseline_ape_mgdl": float(base_rmse + test_b),
        "train_alignment": float(train_a), "baseline_train_alignment": float(train_b),
        "insulin_span_u": float(span),
        "tau_mode": tau_mode, "predicts_level": predicts_level,
        "n_synth": int(meta.get("n_synth") or 0), "n_anchors": int(meta.get("n_anchors") or 0),
        "curves": {"test": {
            "insulin_x": rnd(u_norm, 4),
            "docas": rnd(cd.mean(0), 2),
            "baseline": rnd(cb.mean(0), 2),
        }},
    }


def train_baseline_horizon(horizon_min: int, *, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    for pid in pids:
        data = load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon_min))
        save_models(horizon_min, pid, data,
                    baseline=DOCAS.train_fn(data.X_train, data.y_train, model_kwargs={}))
    return {"horizon_min": horizon_min, "patient_ids": list(pids)}


def _eval_patient(pid, horizon_min, *, train_fn=None, model_kwargs=None, baseline=None,
                  subdir: str = "models", cfg=None, seed: int | None = None):
    data = load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon_min))
    span = float(insulin_span_u(data, horizon_min=horizon_min))
    DOCAS.SPAN = span
    study_cfg = dict(cfg or STUDY_CFG)
    if seed is not None:
        sync_cfg(SEED=int(seed))
        DOCAS.SEED = int(seed)
    kw, fn, old = dict(model_kwargs or {}), (train_fn or DOCAS.train_fn), DOCAS.train_fn
    DOCAS.train_fn = fn
    try:
        if baseline is None:
            baseline = fn(data.X_train, data.y_train, model_kwargs=kw)
        fit_kw, drop, u_ref, _, _ = docas_fit_kw(
            data, span, horizon_min, cfg=study_cfg, model_kwargs=kw)
        if seed is not None:
            fit_kw["seed"] = int(seed)
        result = DOCAS(**fit_kw).fit(data.X_train, data.y_train, baseline_model=baseline)
        docas = result["model"]
        tau_mode = str(fit_kw.get("tau_mode", study_cfg.get("tau_mode", "fractional")))
        meta_extra = dict(
            protocol=str(study_cfg.get("tau_mode", "fractional")) + "_tau",
            tau_drop=drop, u_ref=u_ref, tau_mode=tau_mode,
            level_ref=str(fit_kw.get("level_ref", "current")),
            predicts_level=bool(result.get("predicts_level", False)),
            best_params={**study_cfg, "tau_drop": drop, "u_ref": u_ref,
                         "synth_ratio": float(fit_kw["synth_ratio"]),
                         "tau_mode": tau_mode},
        )
    finally:
        DOCAS.train_fn = old
    meta = dict(
        n_synth=int(result.get("n_synth") or 0), n_anchors=int(result.get("n_anchors") or 0),
        insulin_grid_n=int(result["grid_n"]), insulin_span_u=float(span), **meta_extra)
    save_models(horizon_min, pid, data, baseline=baseline, docas=docas, docas_meta=meta, subdir=subdir)
    u = np.linspace(0.0, 1.0, int(result["grid_n"]))
    return _patient_row(horizon_min, pid, data, baseline, docas, span, u, meta=meta)


def run_horizon(horizon_min: int, *, patient_ids=None, seeds=None) -> dict:
    """Evaluate CatBoost baseline vs DOCAS; average over seeds when given."""
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    seeds = tuple(seeds if seeds is not None else SEEDS)
    if not DOCAS.LOAD:
        train_baseline_horizon(horizon_min, patient_ids=pids)

    # Accumulate per-patient metrics across seeds.
    acc: dict[str, dict[str, list]] = {str(pid): {k: [] for k in _METRIC_KEYS} for pid in pids}
    last_row: dict[str, dict] = {}
    for seed in seeds:
        sync_cfg(SEED=int(seed))
        DOCAS.SEED = int(seed)
        for pid in pids:
            if DOCAS.LOAD and len(seeds) == 1:
                data = load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon_min))
                bundle = _load_bundle(horizon_min, pid)
                meta = bundle["docas_meta"]
                span = float(meta["insulin_span_u"])
                u = np.linspace(0.0, 1.0, int(meta["insulin_grid_n"]))
                row = _patient_row(
                    horizon_min, pid, data, bundle["baseline"], bundle["docas"], span, u, meta=meta)
            else:
                row = _eval_patient(
                    pid, horizon_min, baseline=load_baseline_model(horizon_min, pid),
                    seed=int(seed))
            last_row[str(pid)] = row
            for k in _METRIC_KEYS:
                acc[str(pid)][k].append(float(row[k]))

    rows = []
    for pid in pids:
        base = dict(last_row[str(pid)])
        for k in _METRIC_KEYS:
            vals = acc[str(pid)][k]
            base[k] = float(np.mean(vals))
            base[f"{k}_seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        base["n_seeds"] = len(seeds)
        rows.append(base)

    metrics = {k: mean_std([r[k] for r in rows]) for k in _METRIC_KEYS}
    from tables import paired_diff_test, holm_correct
    pairs = (
        ("rmse", "baseline_test_rmse_mgdl", "docas_test_rmse_mgdl"),
        ("align", "baseline_alignment_error", "alignment_error"),
    )
    raw = []
    for name, bk, dk in pairs:
        t = paired_diff_test([r[bk] for r in rows], [r[dk] for r in rows])
        raw.append((name, t))
    adj = holm_correct([t["p"] for _, t in raw])
    sig = {}
    for (name, t), p_adj in zip(raw, adj):
        sig[f"{name}_p"] = round(t["p"], 4)
        sig[f"{name}_p_holm"] = round(p_adj, 4)
        sig[f"{name}_test"] = t["test"]
        sig[f"{name}_effect"] = t.get("effect")
        sig[f"{name}_ci"] = t.get("ci")
        sig[f"{name}_sig"] = bool(p_adj < 0.05)
        sig[f"{name}_dir"] = t["direction"] if p_adj < 0.05 else None
    metrics["significance"] = sig
    log_means("baseline", horizon_min, rows, "baseline_test_rmse_mgdl", "baseline_alignment_error")
    log_means("DOCAS", horizon_min, rows, "docas_test_rmse_mgdl", "alignment_error")
    return {"horizon_min": int(horizon_min), "metrics": metrics, "per_patient": rows,
            "seeds": list(seeds)}


def _eval_comparator_patient(pid, horizon_min, *, arm: str) -> dict:
    """Evaluate persistence / monotone / insulin-removed observational baselines."""
    from forecasters import train_persistence, train_monotone, train_insulin_removed
    from core import future as fut
    data = load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon_min))
    span = float(insulin_span_u(data, horizon_min=horizon_min))
    DOCAS.SPAN = span
    ji = int(data.insulin_idx)
    gi = data.feature_names.index("CGM")
    a_s, dy_s, y_s = data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]
    if arm == "persistence":
        model = train_persistence(data.X_train, data.y_train, delta_scaler=dy_s)
    elif arm == "monotone":
        model = train_monotone(data.X_train, data.y_train, insulin_idx=ji)
    elif arm == "insulin_removed":
        model = train_insulin_removed(data.X_train, data.y_train, insulin_idx=ji)
    else:
        raise ValueError(arm)
    # Align against study τ.
    fit_kw, drop, u_ref, _, _ = docas_fit_kw(data, span, horizon_min, cfg=STUDY_CFG)
    aligner = DOCAS(
        fit_kw["target"], intervention_idx=ji, outcome_idx=gi,
        action_scaler=a_s, level_scaler=y_s, delta_scaler=dy_s, span=span,
        tau_drop=drop, u_ref=u_ref, tau_mode=str(fit_kw["tau_mode"]),
        level_ref=str(fit_kw.get("level_ref", "current")), predicts_level=False,
    )
    ae = float(aligner.alignment(model, np.asarray(data.X_test, float)))
    rmse = float(rmse_mgdl(data.y_test, model.predict(data.X_test), dy_s))
    return dict(
        patient_id=str(pid), horizon_min=int(horizon_min), arm=arm,
        test_rmse_mgdl=rmse, alignment_error=ae, ape_mgdl=rmse + ae,
        insulin_span_u=span,
    )


def run_comparators(*, patient_ids=None) -> dict:
    """Persistence, monotone CatBoost, insulin-removed trunk on OhioT1DM."""
    set_dataset("ohio_t1dm")
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    arms = ("persistence", "monotone", "insulin_removed")
    horizons = {}
    for ph in FORECAST_HORIZONS:
        per = {arm: [] for arm in arms}
        for pid in pids:
            for arm in arms:
                per[arm].append(_eval_comparator_patient(pid, ph, arm=arm))
        metrics = {
            arm: {k: mean_std([r[k] for r in rows])
                  for k in ("test_rmse_mgdl", "alignment_error", "ape_mgdl")}
            for arm, rows in per.items()
        }
        horizons[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=per)
        print(f"  comparators PH{ph}: " + " | ".join(
            f"{arm} RMSE={metrics[arm]['test_rmse_mgdl']['mean']:.2f} "
            f"AE={metrics[arm]['alignment_error']['mean']:.2f}"
            for arm in arms), flush=True)
    out = RESULTS / "ohio_t1dm" / "comparators.json"
    payload = dict(patient_ids=[str(p) for p in pids], horizons=horizons, **_protocol_meta())
    _write_json(out, payload)
    return payload


def run_personal_tau(*, patient_ids=None) -> dict:
    """Sensitivity: DOCAS with per-patient SI from 1800-rule / TDD vs modal SI."""
    set_dataset("ohio_t1dm")
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    horizons = {}
    for ph in FORECAST_HORIZONS:
        rows = []
        for pid in pids:
            meta = load_patient_meta(str(DATA_ROOT), str(pid))
            si_p = float((meta.get("target") or {}).get("SI") or meta.get("SI_personal") or SI_MODE)
            cfg = {**STUDY_CFG, "SI": si_p}
            row = _eval_patient(pid, ph, cfg=cfg, seed=SEED)
            row["SI_personal"] = si_p
            row["tdd_u"] = meta.get("tdd_u")
            rows.append(row)
        metrics = {k: mean_std([r[k] for r in rows]) for k in _METRIC_KEYS}
        horizons[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=rows)
        print(f"  personal-τ PH{ph}: DOCAS RMSE {metrics['docas_test_rmse_mgdl']['mean']:.2f} "
              f"AE {metrics['alignment_error']['mean']:.2f}", flush=True)
    out = RESULTS / "ohio_t1dm" / "personal_tau.json"
    payload = dict(patient_ids=[str(p) for p in pids], horizons=horizons, **_protocol_meta())
    _write_json(out, payload)
    return payload


def run(*, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    all_p = tuple(list_patients(str(DATA_ROOT)))
    scope = "" if pids == all_p else "_patients-" + "-".join(pids)
    horizons = {str(ph): run_horizon(ph, patient_ids=pids) for ph in FORECAST_HORIZONS}
    path = ohio_cohort_path(scope)
    payload = {"patient_ids": [str(p) for p in pids], "horizons": horizons, **_protocol_meta()}
    _write_json(path, payload)
    return {"path": str(path), **payload}


def _run_struct_cohort(*, out: Path, keys: list[str], eval_fn, label: str,
                       patient_ids=None, require_key: str | None = None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    need = require_key or keys[0]
    horizons = {} if not DOCAS.LOAD else dict(
        (json.loads(out.read_text()) if out.is_file() else {}).get("horizons") or {})
    for ph in FORECAST_HORIZONS:
        h = horizons.setdefault(str(ph), {"horizon_min": ph, "per_patient": []})
        rows = [r for r in (h.get("per_patient") or []) if need in r]
        done = {str(r["patient_id"]) for r in rows}
        for pid in pids:
            if str(pid) in done:
                continue
            rows.append(eval_fn(pid, ph))
            done.add(str(pid))
            h["per_patient"] = rows
        if rows:
            h["metrics"] = {k: mean_std([r[k] for r in rows]) for k in keys}
        horizons[str(ph)] = h
        _write_json(out, {"patient_ids": list(pids), "horizons": horizons, **_protocol_meta()})
        if rows:
            log_means("baseline", ph, rows, "baseline_test_rmse_mgdl", "baseline_alignment_error")
            log_means("DOCAS", ph, rows, "docas_test_rmse_mgdl", "alignment_error")
    payload = {"patient_ids": list(pids), "horizons": horizons, **_protocol_meta()}
    _write_json(out, payload)
    return {"path": str(out), **payload}


def _mlp_train(X, y, model_kwargs=None, **_):
    return train_mlp(X, y, model_kwargs=model_kwargs)


def _mlp_job(dataset: str, ph: int, pid: str, seed: int) -> dict:
    """One (patient, seed) MLP fit; run in its own process by run_mlp.

    Imported by name rather than reused from the parent frame: when this file
    runs as a script the workers would otherwise get a fresh, unconfigured
    ``core.Cfg``, which the module-level ``sync_cfg`` call populates on import.
    """
    import importlib
    import torch
    torch.set_num_threads(1)  # workers already saturate the cores
    m = importlib.import_module("main")
    m.set_dataset(dataset)
    return m._eval_patient(pid, ph, train_fn=m._mlp_train, subdir="models_mlp",
                           seed=int(seed))


def run_mlp(*, patient_ids=None, seeds=None, n_jobs: int | None = None) -> dict:
    """MLP baseline vs DOCAS; average over seeds.

    The (patient, seed) fits are independent, so they run in parallel processes;
    each carries its own seed, so the result does not depend on the worker count.
    """
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    seeds = tuple(seeds if seeds is not None else SEEDS)
    n_jobs = int(n_jobs if n_jobs is not None else MLP_JOBS)
    out = results_dir() / "mlp_cohort.json"
    horizons = {}
    for ph in FORECAST_HORIZONS:
        acc: dict[str, dict[str, list]] = {str(pid): {k: [] for k in _METRIC_KEYS} for pid in pids}
        last_row: dict[str, dict] = {}
        jobs = [(str(pid), int(seed)) for seed in seeds for pid in pids]
        print(f"  MLP PH{ph}: {len(jobs)} fits on {n_jobs} workers", flush=True)
        done = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_mlp_job)(DATASET, ph, pid, seed) for pid, seed in jobs)
        for (pid, _seed), row in zip(jobs, done):
            last_row[pid] = row
            for k in _METRIC_KEYS:
                acc[pid][k].append(float(row[k]))
        rows = []
        for pid in pids:
            base = dict(last_row[str(pid)])
            for k in _METRIC_KEYS:
                vals = acc[str(pid)][k]
                base[k] = float(np.mean(vals))
                base[f"{k}_seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            base["n_seeds"] = len(seeds)
            rows.append(base)
        metrics = {k: mean_std([r[k] for r in rows]) for k in _METRIC_KEYS}
        from tables import paired_diff_test, holm_correct
        pairs = (
            ("rmse", "baseline_test_rmse_mgdl", "docas_test_rmse_mgdl"),
            ("align", "baseline_alignment_error", "alignment_error"),
        )
        raw = [(n, paired_diff_test([r[bk] for r in rows], [r[dk] for r in rows]))
               for n, bk, dk in pairs]
        adj = holm_correct([t["p"] for _, t in raw])
        sig = {}
        for (name, t), p_adj in zip(raw, adj):
            sig[f"{name}_p"] = round(t["p"], 4)
            sig[f"{name}_p_holm"] = round(p_adj, 4)
            sig[f"{name}_test"] = t["test"]
            sig[f"{name}_effect"] = t.get("effect")
            sig[f"{name}_ci"] = t.get("ci")
            sig[f"{name}_sig"] = bool(p_adj < 0.05)
            sig[f"{name}_dir"] = t["direction"] if p_adj < 0.05 else None
        metrics["significance"] = sig
        horizons[str(ph)] = {"horizon_min": ph, "metrics": metrics, "per_patient": rows,
                             "seeds": list(seeds)}
        log_means("baseline", ph, rows, "baseline_test_rmse_mgdl", "baseline_alignment_error")
        log_means("DOCAS", ph, rows, "docas_test_rmse_mgdl", "alignment_error")
        _write_json(out, {"patient_ids": list(pids), "horizons": horizons, **_protocol_meta()})
    payload = {"patient_ids": list(pids), "horizons": horizons, **_protocol_meta()}
    _write_json(out, payload)
    return {"path": str(out), **payload}

def _log_cohort(path: Path) -> None:
    cohort = json.loads(path.read_text())
    for ph in FORECAST_HORIZONS:
        rows = cohort["horizons"][str(ph)]["per_patient"]
        log_means("baseline", ph, rows, "baseline_test_rmse_mgdl", "baseline_alignment_error")
        log_means("DOCAS", ph, rows, "docas_test_rmse_mgdl", "alignment_error")


def _reset_seed() -> None:
    sync_cfg(SEED=SEED)
    DOCAS.SEED = SEED


def main() -> None:
    """Every result in the paper, in dependency order; then tables and figures."""
    import ablation, metabonet_comparators, replay_eval
    try:
        print(f"LOAD={DOCAS.LOAD} → "
              f"{'reuse caches' if DOCAS.LOAD else 'retrain (ignore caches)'}", flush=True)
        for ds in ("ohio_t1dm", "azt1d"):
            set_dataset(ds)
            base = RESULTS / ds
            print(f"\nCatBoost, five seeds ({ds})", flush=True)
            if DOCAS.LOAD and (cat_path := base / "cohort.json").is_file():
                _log_cohort(cat_path)
            else:
                run()
            print(f"\nMLP, five seeds ({ds})", flush=True)
            if DOCAS.LOAD and (mlp_path := base / "mlp_cohort.json").is_file():
                _log_cohort(mlp_path)
            else:
                run_mlp()
            _reset_seed()
        print("\nCatBoost with lag features", flush=True)
        metabonet_comparators.run_lags(backbones=metabonet_comparators.LAG_BACKBONES[:1])
        _reset_seed()
        print("\nAblation (seed 42)", flush=True)
        ablation.run_ablations()
        for ds in ablation.ABLATION_DATASETS:
            ablation.run_uniform_only(dataset=ds)
            ablation.run_budget_sweep(dataset=ds)
        print("\nTarget DRC at 0.5x / 2x insulin sensitivity", flush=True)
        ablation.run_misspec_all()
        print("\nPump-settings target DRC", flush=True)
        ablation.run_clinical_tau_all()
        print("\nReplayBG decision support (OhioT1DM)", flush=True)
        replay_eval.maybe_run()
        replay_eval.run_unguarded()
        replay_eval.run_clinical_tau_replay()
        pp.azt1d_aid_profile(str(ROOT))
    finally:
        set_dataset("ohio_t1dm")
        import tables
        tables.build_manuscript()

if __name__ == "__main__":
    main()
