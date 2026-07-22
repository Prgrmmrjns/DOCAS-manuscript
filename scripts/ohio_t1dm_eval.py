"""Train DOCAS models on OhioT1DM, save to disk, and evaluate held-out accuracy."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib, numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_ROOT / "scripts"))

from docas import DOCAS, F0_REF, future, interp_target, target_delta_ref
from forecasters import LSTMForecaster, NPLSTMForecaster, PLSTMForecaster, train_baseline, train_monotonic
from ohio_t1dm_preprocessing import PaperGlucoseModel, horizon_steps, list_patients, load_patient, load_patient_meta, rmse_mgdl

DOCAS.train_fn = lambda X, y, model_kwargs=None, sample_weight=None: train_baseline(
    X, y, model_kwargs=model_kwargs, sample_weight=sample_weight)
RESULTS = DATA_ROOT / "results"
OHIO_BACKEND = "lgbm"
MONOTONIC_COHORT = RESULTS / "ohio_t1dm" / "monotonic_cohort.json"
LSTM_COHORT = RESULTS / "ohio_t1dm" / "lstm_cohort.json"
LSTM_ANCHORS, LSTM_GRID = 384, 41
FORECAST_HORIZONS = (30, 60)
GLUCOSE_TARGET_MGDL = 145.0
AUDIT_CONTEXTS = 128
ALIGN_DECIMALS = 2
# ReplayBG prior modes (Ohio T1D target τ); manuscript §target / Cappon 2023.
# At w=75 kg, U_max=10 U: R_30(1)≈0.040, R_60(1)≈0.347 (G_ref=220).
RBG_PARAMS = dict(
    Gb=119.13, SG=float(np.exp(-3.8 - 0.25)), p2=0.11, SI=2.3 * 5e-4,
    ka2=float(np.exp(-4.2875 - 0.4274 ** 2)), kd=float(np.exp(-3.5090 - 0.6187 ** 2)),
    ke=0.127, VG=1.45, VI=0.135, alpha=7, tau=8, r1=1.4407, r2=0.8124)
_METRIC_KEYS = "baseline_test_rmse_mgdl docas_test_rmse_mgdl rmse_delta_mgdl alignment_error baseline_alignment_error alignment_improvement".split()
_LSTM_KEYS = "np_lstm_test_rmse_mgdl np_lstm_alignment_error p_lstm_test_rmse_mgdl p_lstm_alignment_error docas_lstm_test_rmse_mgdl docas_lstm_alignment_error".split()
_META_FROM_RESULT = "drop_fraction body_weight_kg target intervention n_synth n_anchors train_alignment".split()
_META_FROM_TUNE = "r1 target_amplitude".split()


def replaybg_response(u, horizon_min, body_weight_kg=75.0, *, tau=None, SI=None, **_):
    """Fractional CGM lowering R_h(u;w) from ReplayBG (start G=220); manuscript Eq. target."""
    p, bw = {**RBG_PARAMS}, float(body_weight_kg)
    if tau is not None:
        p["tau"] = float(tau)
    if SI is not None:
        p["SI"] = float(SI)
    doses = np.r_[np.asarray(u, float).ravel() * DOCAS.SPAN, 0.0]
    n, basal = len(doses), 1000.0 / (60.0 * bw)
    G, IG = np.full(n, 220.0), np.full(n, 220.0)
    X, Isc1, Isc2, Ip = np.zeros(n), np.full(n, basal / p["kd"]), np.full(n, basal / p["ka2"]), np.full(n, basal / p["ke"])
    for t in range(1, int(horizon_min) + 1):
        insulin = basal + (doses * 1000.0 / bw if t == int(round(p["tau"])) + 1 else 0.0)
        Isc1 = (Isc1 + insulin) / (1.0 + p["kd"])
        Isc2 = (Isc2 + p["kd"] * Isc1) / (1.0 + p["ka2"])
        Ip = (Ip + p["ka2"] * Isc2) / (1.0 + p["ke"])
        X = (X + p["p2"] * (p["SI"] / p["VI"]) * (Ip - basal / p["ke"])) / (1.0 + p["p2"])
        risk = np.where(G < p["Gb"], 1.0 + 10.0 * p["r1"] * (np.log(np.maximum(G, 60.0)) ** p["r2"] - np.log(p["Gb"]) ** p["r2"]) ** 2, 1.0)
        G = (G + p["SG"] * p["Gb"]) / (1.0 + p["SG"] + risk * X)
        IG = (p["alpha"] * IG + G) / (1.0 + p["alpha"])
    return ((IG[-1] - IG[:-1]) / 220.0).reshape(np.asarray(u).shape)


def ohio_target(horizon_min, body_weight_kg=75.0, *, tau=None, SI=None, **kw):
    """ReplayBG fractional lowering R_h(u;w) for τ=f0·[1−R]."""
    kw = {k: v for k, v in kw.items()
          if k not in {"u", "tau_curve", "delta", "emp", "mode", "blend", "drop_fraction",
                       "gain", "g0", "target_y0", "target_y_start", "target_y_end"}}
    knobs = {k: v for k, v in dict(tau=tau, SI=SI, **kw).items() if v is not None}
    return lambda u: replaybg_response(u, horizon_min, body_weight_kg, **knobs)


def resolve_delta_target(horizon_min, body_weight_kg, knobs=None, *, curve=None):
    """ΔBG at F0_REF from R_h (scaled by f0/F0_REF in alignment/synth)."""
    knobs = dict(knobs or {})
    if curve and curve.get("u") is not None and curve.get("delta") is not None:
        return interp_target(curve["u"], curve["delta"])
    if curve and curve.get("u") is not None and curve.get("tau") is not None:
        return lambda u: -F0_REF * np.asarray(interp_target(curve["u"], curve["tau"])(u), float)
    shape = ohio_target(horizon_min, body_weight_kg, **{k: knobs[k] for k in ("tau", "SI") if k in knobs})
    return lambda u: target_delta_ref(shape, u)


def _rmse_mgdl(model, X, y, tgt_s):
    return float(rmse_mgdl(y, model.predict(np.asarray(X, float)), tgt_s))


def ohio_results_dir() -> Path:
    path = RESULTS / "ohio_t1dm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_backend(backend: str = "lgbm") -> None:
    """Set DOCAS.train_fn (LightGBM)."""
    global OHIO_BACKEND
    backend = str(backend).lower()
    if backend != "lgbm":
        raise ValueError(f"unknown backend {backend!r}; only 'lgbm' is supported")
    OHIO_BACKEND = backend
    DOCAS.train_fn = lambda X, y, model_kwargs=None, sample_weight=None: train_baseline(
        X, y, model_kwargs=model_kwargs, sample_weight=sample_weight)
    print(f"[Ohio] backend={OHIO_BACKEND} → {ohio_results_dir()}", flush=True)


def round_metric(value, digits: int = ALIGN_DECIMALS) -> float:
    v = float(value)
    return float("nan") if not np.isfinite(v) else round(v, digits)


def mean_std(values, *, digits: int | None = None) -> dict:
    a = np.asarray(values, float)
    out = {k: float("nan") for k in ("mean", "std", "min", "max", "median")} if not len(a) else {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0., "min": float(a.min()), "max": float(a.max()), "median": float(np.median(a))}
    return out if digits is None else {k: round_metric(v, digits) for k, v in out.items()}


def format_cohort_summary(horizon_min: int, metrics: dict) -> str:
    b, d = metrics["baseline_test_rmse_mgdl"], metrics["docas_test_rmse_mgdl"]
    ba, da = metrics["baseline_alignment_error"], metrics["alignment_error"]
    ad = ALIGN_DECIMALS
    return (f"  Table1 PH{horizon_min}: baseline RMSE {b['mean']:.2f}±{b['std']:.2f} "
            f"align {ba['mean']:.{ad}f}±{ba['std']:.{ad}f} | "
            f"DOCAS RMSE {d['mean']:.2f}±{d['std']:.2f} align {da['mean']:.{ad}f}±{da['std']:.{ad}f}")


def ohio_cohort_path(scope: str = "") -> Path:
    return ohio_results_dir() / f"cohort{scope}.json"


def ohio_horizon_payload(cohort: dict, horizon_min: int) -> dict:
    return cohort["horizons"][str(horizon_min)]


def models_dir(horizon_min: int) -> Path:
    path = ohio_results_dir() / f"ph{horizon_min}" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(horizon_min: int, patient_id: str) -> Path:
    return models_dir(horizon_min) / f"{patient_id}.joblib"


def _load_bundle(horizon_min: int, patient_id: str) -> dict | None:
    path = model_path(horizon_min, patient_id)
    if not path.is_file():
        return None
    import forecasters
    sys.modules.setdefault("model", forecasters)
    try:
        return joblib.load(path)
    except Exception:
        return None


def save_models(horizon_min: int, patient_id: str, data, *, baseline=None, docas=None, docas_meta=None) -> Path:
    path = model_path(horizon_min, patient_id)
    bundle = _load_bundle(horizon_min, patient_id) or {}
    bundle.update({k: v for k, v in {"baseline": baseline, "docas": docas, "docas_meta": docas_meta}.items() if v is not None})
    bundle.update(train_scalers=data.train_scalers, cgm_scaler=data.cgm_scaler, feature_names=list(data.feature_names))
    joblib.dump(bundle, path)
    return path


def load_baseline_model(horizon_min: int, patient_id: str):
    return _load_bundle(horizon_min, patient_id)["baseline"]


def baseline_ready(horizon_min: int, patient_id: str) -> bool:
    return bool((b := _load_bundle(horizon_min, patient_id)) and "baseline" in b)


def docas_ready(horizon_min: int, patient_id: str) -> bool:
    return bool((b := _load_bundle(horizon_min, patient_id)) and "docas" in b)


def load_models(horizon_min: int, patient_id: str):
    bundle = _load_bundle(horizon_min, patient_id)
    scalers, cgm_scaler, names = bundle["train_scalers"], bundle["cgm_scaler"], bundle["feature_names"]
    for m in (bundle["baseline"], bundle["docas"]):
        raw = getattr(m, "_model", m)
        if hasattr(raw, "set_params"):
            try:
                raw.set_params(n_jobs=1)
            except (ValueError, TypeError):
                pass
    wrap = lambda m: PaperGlucoseModel(m, scalers, cgm_scaler)
    return wrap(bundle["baseline"]), wrap(bundle["docas"]), list(names)


def _sample_contexts(X, patient_id, n=AUDIT_CONTEXTS):
    X = np.asarray(X, float)
    if len(X) <= n:
        return X
    return X[np.random.default_rng(int(patient_id)).choice(len(X), n, replace=False)]


def _align_kw(data, gi, ji, u_norm, insulin_span_u, horizon_min, *,
              body_weight_kg=75.0, f0_lo=None, f0_hi=None, target_knobs=None, target_curve=None):
    u = np.asarray(u_norm, float)
    knobs = dict(target_knobs or {})
    delta_fn = resolve_delta_target(horizon_min, body_weight_kg, knobs, curve=target_curve)
    return dict(ji=ji, yi=gi, u=u, span=insulin_span_u, a_s=data.train_scalers[ji],
                dy_s=data.cgm_scaler, y_s=data.train_scalers[gi],
                response=np.asarray(delta_fn(u), float),  # ΔBG vs f0
                y0_lo=float(f0_lo if f0_lo is not None else DOCAS.Y0_LO),
                y0_hi=float(f0_hi if f0_hi is not None else DOCAS.Y0_HI))


def _alignment(model, baseline, X, *, ji, yi, u, span, a_s, dy_s, y_s, y0_lo, y0_hi, response, additive=True, **_):
    d = object.__new__(DOCAS)
    d.ji, d.yi, d.span, d.a_s, d.dy_s, d.y_s = ji, yi, span, a_s, dy_s, y_s
    d.y0_lo, d.y0_hi, d.additive, d.u_grid, d.target = y0_lo, y0_hi, additive, np.asarray(u, float), None
    return d.alignment(model, baseline, X, u=u, y0_lo=y0_lo, y0_hi=y0_hi, response=response)


def _patient_row(horizon_min, pid, data, baseline, docas, insulin_span_u, u_norm, *, meta=None) -> dict:
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    meta = meta or {}
    pmeta = load_patient_meta(str(DATA_ROOT), pid)
    knobs = dict(meta.get("target_knobs") or pmeta.get("target") or {})
    curve = meta.get("target_curve")
    body_weight_kg = float(meta.get("body_weight_kg", pmeta["weight_kg"]))
    drop_fraction = float(meta.get("drop_fraction", float(
        knobs.get("target_y_start", 0.0) - knobs.get("target_y_end", -F0_REF * 0.04))))
    # Always score on the clinical f0 band.
    f0_lo, f0_hi = float(DOCAS.Y0_LO), float(DOCAS.Y0_HI)
    align_kw = _align_kw(data, gi, ji, u_norm, insulin_span_u, horizon_min,
                         body_weight_kg=body_weight_kg, f0_lo=f0_lo, f0_hi=f0_hi,
                         target_knobs=knobs, target_curve=curve)
    base_rmse = rmse_mgdl(data.y_test, baseline.predict(data.X_test), data.cgm_scaler)
    docas_rmse = rmse_mgdl(data.y_test, docas.predict(data.X_test), data.cgm_scaler)
    test_contexts = _sample_contexts(data.X_test, pid)
    test_align = _alignment(docas, baseline, test_contexts, **align_kw)
    test_align_baseline = _alignment(baseline, baseline, test_contexts, **align_kw)
    insulin_x = np.asarray(u_norm, float).tolist()
    nan_curve = [float("nan")] * len(u_norm)
    contexts = _sample_contexts(data.X_test, pid)
    args = contexts, ji, gi, u_norm, insulin_span_u, data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]
    cd, cb = future(docas, *args), future(baseline, *args)
    hyper_mask = cb[:, 0] > DOCAS.Y0_LO
    cd = cd[hyper_mask]
    curve_data = {"test": {
        "insulin_x": insulin_x,
        "docas": cd.mean(0).tolist() if len(cd) else nan_curve,
        "baseline": cb[hyper_mask].mean(0).tolist() if hyper_mask.any() else nan_curve,
    }}
    return {"patient_id": pid, "horizon_min": horizon_min, "model_path": str(model_path(horizon_min, pid).relative_to(RESULTS)), "baseline_test_rmse_mgdl": base_rmse, "docas_test_rmse_mgdl": docas_rmse, "rmse_delta_mgdl": docas_rmse - base_rmse, "alignment_error": round_metric(test_align), "baseline_alignment_error": round_metric(test_align_baseline), "alignment_improvement": round_metric(test_align_baseline - test_align), "drop_fraction": drop_fraction, "body_weight_kg": body_weight_kg, "target": meta.get("target", "replaybg_prior_mode"), "intervention": meta.get("intervention", "additive_correction"), "insulin_grid_n": int(meta.get("insulin_grid_n", len(u_norm))), "insulin_span_u": insulin_span_u, "curves": curve_data, "n_synth": meta.get("n_synth", float("nan")), "n_anchors": meta.get("n_anchors", float("nan")), "train_alignment": round_metric(meta.get("train_alignment", float("nan"))), "train_rmse_delta_mgdl": meta.get("train_rmse_delta_mgdl", float("nan")), "f0_lo": meta.get("f0_lo"), "f0_hi": meta.get("f0_hi"), "grid_points": meta.get("grid_points"), "sampling": meta.get("sampling")}


def train_baseline_horizon(horizon_min: int, *, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    rows, model_kw = [], {"n_jobs": 1}
    for pid in pids:
        if baseline_ready(horizon_min, pid):
            continue
        data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(horizon_min))
        baseline = DOCAS.train_fn(data.X_train, data.y_train, model_kwargs=model_kw)
        save_models(horizon_min, pid, data, baseline=baseline)
        rmse = rmse_mgdl(data.y_test, baseline.predict(data.X_test), data.cgm_scaler)
        rows.append({"patient_id": pid, "test_rmse_mgdl": rmse})
        print(f"  baseline {pid} rmse {rmse:.1f}", flush=True)
    metrics = {"test_rmse_mgdl": mean_std([r["test_rmse_mgdl"] for r in rows])}
    return {"horizon_min": horizon_min, "patient_ids": list(pids), "metrics": metrics, "per_patient": rows}


def _calibrate_patient(horizon_min, pid, insulin_span_u, fixed_params=None):
    data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(horizon_min))
    pmeta = load_patient_meta(str(DATA_ROOT), pid)
    body_weight_kg, knobs = pmeta["weight_kg"], dict(pmeta.get("target") or {})
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    baseline = load_baseline_model(horizon_min, pid)
    fp = dict(fixed_params or {})
    # Fixed ReplayBG prior-mode τ (manuscript); body weight varies by patient.
    prior_kw = {k: v for k, v in knobs.items() if k in ("tau", "SI")}
    result = DOCAS(
        ohio_target(horizon_min, body_weight_kg, **prior_kw),
        intervention_idx=ji, outcome_idx=gi,
        action_scaler=data.train_scalers[ji], level_scaler=data.train_scalers[gi], delta_scaler=data.cgm_scaler,
        span=insulin_span_u, model_kwargs={"n_jobs": 1},
        target_name="replaybg_prior_mode",
    ).fit(data.X_train, data.y_train, baseline_model=baseline, fixed_params=fp)
    docas, u_norm = result["model"], np.linspace(0.0, 1.0, int(result["grid_n"]))
    tp = result.get("tune_params") or {}
    y_start = float(result.get("target_y_start", 0.0))
    y_end = float(result.get("target_y_end", tp.get("target_y_end", -F0_REF * float(result.get("r1", 0.0)))))
    knobs = {**knobs, "target_y_start": y_start, "target_y_end": y_end}
    ug = np.asarray(DOCAS.U_GRID, float)
    shape = ohio_target(horizon_min, body_weight_kg, **prior_kw)
    delta = target_delta_ref(shape, ug)
    target_curve = {"u": ug.tolist(), "delta": np.asarray(delta, float).tolist(), "f0_ref": F0_REF,
                    "r1": float(result.get("r1", tp.get("r1", float("nan"))))}
    docas_meta = {k: result[k] for k in _META_FROM_RESULT if k in result}
    docas_meta.update(
        body_weight_kg=body_weight_kg, target_knobs=knobs, target_curve=target_curve,
        target_y_start=y_start, target_y_end=y_end,
        insulin_grid_n=result["grid_n"], insulin_span_u=result["span"],
        train_rmse_delta_mgdl=result.get("train_rmse_delta"),
        f0_lo=DOCAS.Y0_LO, f0_hi=DOCAS.Y0_HI,
        drop_fraction=float(y_start - y_end),
    )
    docas_meta.update({k: tp.get(k) for k in _META_FROM_TUNE})
    docas_meta["sampling"] = tp.get("sampling", "f0_insulin_net")
    save_models(horizon_min, pid, data, docas=docas, docas_meta=docas_meta)
    row = _patient_row(horizon_min, pid, data, baseline, docas, insulin_span_u, u_norm, meta=docas_meta)
    bundle = _load_bundle(horizon_min, pid) or {}
    bundle["cohort_row"] = row
    joblib.dump(bundle, model_path(horizon_min, pid))
    return row


def run_horizon(horizon_min: int, *, patient_ids=None, fixed_params=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    missing = tuple(pid for pid in pids if not baseline_ready(horizon_min, pid))
    if missing:
        train_baseline_horizon(horizon_min, patient_ids=missing)
    insulin_span_u, rows_by_pid = DOCAS.SPAN, {}
    todo = [] if DOCAS.LOAD else list(pids)
    print(f"[DOCAS PH{horizon_min}]", flush=True)
    if DOCAS.LOAD:
        for pid in pids:
            data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(horizon_min))
            baseline = load_baseline_model(horizon_min, pid)
            bundle = _load_bundle(horizon_min, pid)
            if bundle.get("cohort_row"):
                row = bundle["cohort_row"]
            else:
                meta = bundle.get("docas_meta") or {}
                row = _patient_row(horizon_min, pid, data, baseline, bundle["docas"], insulin_span_u,
                                   np.linspace(0.0, 1.0, int(n)) if (n := meta.get("insulin_grid_n")) is not None else np.asarray(DOCAS.U_GRID, float), meta=meta)
            rows_by_pid[pid] = row
            print(f"  {pid} load rmse {row['docas_test_rmse_mgdl']:.1f} align {row['alignment_error']:.2f}", flush=True)
    for pid in todo:
        row = _calibrate_patient(horizon_min, pid, insulin_span_u, fixed_params)
        rows_by_pid[pid] = row
        print(f"  {pid} rmse {row['docas_test_rmse_mgdl']:.1f} align {row['alignment_error']:.2f}", flush=True)
    rows = [rows_by_pid[pid] for pid in pids]
    metrics = {k: mean_std([r[k] for r in rows], digits=ALIGN_DECIMALS if "alignment" in k else None) for k in _METRIC_KEYS}
    print(format_cohort_summary(horizon_min, metrics), flush=True)
    return {"horizon_min": horizon_min, "insulin_span_u": insulin_span_u, "patient_ids": list(pids), "metrics": metrics, "per_patient": rows}


def run(*, patient_ids=None, fixed_params_by_horizon=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    all_patients = tuple(list_patients(str(DATA_ROOT)))
    scope = "" if pids == all_patients else "_patients-" + "-".join(pids)
    params = dict(fixed_params_by_horizon or {})
    print(f"[Ohio eval · {OHIO_BACKEND}]", flush=True)
    horizons = {str(ph): run_horizon(ph, patient_ids=pids, fixed_params=params.get(ph))
                for ph in FORECAST_HORIZONS}
    path = ohio_cohort_path(scope)
    payload = {"patient_ids": list(pids), "backend": OHIO_BACKEND, "insulin_span_u": DOCAS.SPAN, "horizons": horizons}
    path.write_text(json.dumps(payload, indent=2))
    return {"path": str(path), **payload}


def run_monotonic(*, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    horizons, u_norm = {}, np.asarray(DOCAS.U_GRID, float)
    print("[Ohio · monotonic LightGBM]", flush=True)
    for ph in FORECAST_HORIZONS:
        rows = []
        for pid in pids:
            data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(ph))
            gi, ji = data.feature_names.index("CGM"), data.insulin_idx
            model = train_monotonic(data.X_train, data.y_train, insulin_idx=ji, n_features=len(data.feature_names), model_kwargs={"n_jobs": 1})
            rmse = rmse_mgdl(data.y_test, model.predict(data.X_test), data.cgm_scaler)
            contexts = _sample_contexts(data.X_test, pid)
            f0 = future(model, contexts, ji, gi, (0.0,), DOCAS.SPAN, data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]).ravel()
            pmeta = load_patient_meta(str(DATA_ROOT), pid)
            band = contexts[f0 > DOCAS.Y0_LO]
            align_err = _alignment(model, model, band if len(band) else contexts, **_align_kw(
                data, gi, ji, u_norm, DOCAS.SPAN, ph,
                body_weight_kg=pmeta["weight_kg"], target_knobs=pmeta.get("target")))
            rows.append({"patient_id": pid, "test_rmse_mgdl": rmse, "alignment_error": round_metric(align_err)})
            print(f"  PH{ph} {pid} rmse {rmse:.1f} align {align_err:.2f}", flush=True)
        horizons[str(ph)] = {"horizon_min": ph, "metrics": {
            "test_rmse_mgdl": mean_std([r["test_rmse_mgdl"] for r in rows]),
            "alignment_error": mean_std([r["alignment_error"] for r in rows], digits=ALIGN_DECIMALS),
        }, "per_patient": rows}
        print(f"  PH{ph} mean RMSE {horizons[str(ph)]['metrics']['test_rmse_mgdl']['mean']:.2f} "
              f"align {horizons[str(ph)]['metrics']['alignment_error']['mean']:.2f}", flush=True)
    MONOTONIC_COHORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"patient_ids": list(pids), "horizons": horizons}
    MONOTONIC_COHORT.write_text(json.dumps(payload, indent=2))
    return {"path": str(MONOTONIC_COHORT), **payload}


def _lstm_augment_rows(A, f0_a, ji, gi, ins_s, cgm_s, tgt_s, *, horizon_min, body_weight_kg, target_knobs=None, grid_points=LSTM_GRID):
    from docas import scale_delta_to_f0
    u = np.linspace(0.0, 1.0, grid_points)
    knobs = dict(target_knobs or {})
    delta_ref = np.asarray(resolve_delta_target(horizon_min, body_weight_kg, knobs)(u), float)
    n, g = len(A), len(u)
    cgm = np.repeat(cgm_s.inverse_transform(A[:, gi : gi + 1]).ravel(), g)
    f0_rep = np.repeat(f0_a, g)
    future = f0_rep + scale_delta_to_f0(np.tile(delta_ref, n), f0_rep)
    y = tgt_s.transform((future - cgm).reshape(-1, 1)).ravel()
    xd = np.repeat(A, g, 0)
    xd[:, ji] = ins_s.transform((np.repeat(ins_s.inverse_transform(A[:, ji:ji + 1]).ravel(), g) + np.tile(u, n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    return xd, y


def _eval_lstm_patient(pid: str, horizon_min: int) -> dict:
    data = load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(horizon_min))
    pmeta = load_patient_meta(str(DATA_ROOT), pid)
    body_weight_kg, knobs = pmeta["weight_kg"], dict(pmeta.get("target") or {})
    gi, ji, ci = 0, data.insulin_idx, data.feature_names.index("CHO")
    ins_s, cgm_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.cgm_scaler
    u = np.linspace(0.0, 1.0, LSTM_GRID)
    align_kw = _align_kw(data, gi, ji, u, DOCAS.SPAN, horizon_min, body_weight_kg=body_weight_kg, target_knobs=knobs)
    pad = 18 if int(horizon_min) <= 30 else 20
    p_model = PLSTMForecaster(ji=ji, ci=ci, pad=pad)
    np_lstm = NPLSTMForecaster().fit(data.X_train, data.y_train)
    p_lstm = p_model.fit(data.X_train, data.y_train)
    Xp_train = p_model.smooth(data.X_train)
    f0 = future(p_lstm, Xp_train, ji, gi, (0.0,), DOCAS.SPAN, ins_s, tgt_s, cgm_s).ravel()
    hyper = f0 > DOCAS.Y0_LO
    if hyper.sum():
        idx = np.random.default_rng(int(pid)).integers(0, int(hyper.sum()), LSTM_ANCHORS)
        A, f0_a = Xp_train[hyper][idx], f0[hyper][idx]
        xd, yd = _lstm_augment_rows(A, f0_a, ji, gi, ins_s, cgm_s, tgt_s, horizon_min=horizon_min, body_weight_kg=body_weight_kg, target_knobs=knobs)
        docas_lstm = LSTMForecaster(dense_activation="tanh").fit(np.vstack([Xp_train, xd]), np.concatenate([data.y_train, yd]))
    else:
        docas_lstm = p_lstm
    Xp_test = p_model.smooth(data.X_test)
    ctx_np = _sample_contexts(data.X_test, pid)
    ctx_p = _sample_contexts(Xp_test, pid)
    f0_np = future(np_lstm, ctx_np, ji, gi, (0.0,), DOCAS.SPAN, ins_s, tgt_s, cgm_s).ravel()
    f0_p = future(p_lstm, ctx_p, ji, gi, (0.0,), DOCAS.SPAN, ins_s, tgt_s, cgm_s).ravel()
    band_np = ctx_np[f0_np > DOCAS.Y0_LO]
    band_p = ctx_p[f0_p > DOCAS.Y0_LO]
    return dict(patient_id=pid, horizon_min=horizon_min,
                np_lstm_test_rmse_mgdl=_rmse_mgdl(np_lstm, data.X_test, data.y_test, tgt_s),
                np_lstm_alignment_error=round_metric(_alignment(np_lstm, np_lstm, band_np if len(band_np) else ctx_np, **align_kw)),
                p_lstm_test_rmse_mgdl=_rmse_mgdl(p_lstm, Xp_test, data.y_test, tgt_s),
                p_lstm_alignment_error=round_metric(_alignment(p_lstm, p_lstm, band_p if len(band_p) else ctx_p, **align_kw)),
                docas_lstm_test_rmse_mgdl=_rmse_mgdl(docas_lstm, Xp_test, data.y_test, tgt_s),
                docas_lstm_alignment_error=round_metric(_alignment(docas_lstm, p_lstm, band_p if len(band_p) else ctx_p, **align_kw)))


def run_lstm(*, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    horizons = {}
    print("[Ohio · LSTM np / p / DOCAS]", flush=True)
    for ph in FORECAST_HORIZONS:
        rows = []
        for pid in pids:
            row = _eval_lstm_patient(pid, ph)
            rows.append(row)
            print(f"  {pid} PH{ph} np-LSTM {row['np_lstm_test_rmse_mgdl']:.1f}/{row['np_lstm_alignment_error']:.2f}"
                  f"  p-LSTM {row['p_lstm_test_rmse_mgdl']:.1f}/{row['p_lstm_alignment_error']:.2f}"
                  f"  DOCAS-LSTM {row['docas_lstm_test_rmse_mgdl']:.1f}/{row['docas_lstm_alignment_error']:.2f}", flush=True)
        horizons[str(ph)] = {"horizon_min": ph, "metrics": {
            k: mean_std([r[k] for r in rows], digits=ALIGN_DECIMALS if "alignment" in k else None) for k in _LSTM_KEYS
        }, "per_patient": rows}
    LSTM_COHORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"patient_ids": list(pids), "horizons": horizons}
    LSTM_COHORT.write_text(json.dumps(payload, indent=2))
    return {"path": str(LSTM_COHORT), **payload}
