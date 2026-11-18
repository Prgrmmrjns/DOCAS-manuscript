"""MetaboNet comparators, strata, long horizons, and trajectory.

Arms for Table 2 / Appendix B:
  monotone CatBoost (seed 42)
  MLP baseline (seeds 42-46, last 8000 train rows)
  Penalty MLP (seed 42, last 8000)
  MLP DOCAS (seeds 42-46, last 8000)
  12-lag CatBoost baseline + DOCAS (seed 42)

Also writes 60-min strata (seed 42) and 90/120 endpoints + 5-min trajectory.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
for _k in ("OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

import catboost as cb  # noqa: F401
import numpy as np
from joblib import Parallel, delayed

import ablation as ab
import main as m
from core import DOCAS, docas_fit_kw, sync_cfg
from forecasters import PenaltyMLPForecaster, train_baseline, train_mlp, train_monotone
from preprocessing import PatientData, horizon_steps, _impulses, _valid_feature_rows
from preprocessing import LOOKBACK, _BOLUS_LOOKBACK_MIN, _read_metabonet_split, _metabonet_key
from preprocessing import _frame_from_metabonet_split, bolus_iob_fraction, _carb_irf

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"
OUT = REV / "full_comparators.json"
OUT_STRATA = REV / "full_strata.json"
OUT_LONG = REV / "full_long.json"
CAP = 8000
JOBS = max(1, (os.cpu_count() or 4) - 1)
SEEDS5 = (42,)
DS = "metabonet_full"
SOURCES = {"OhioT1DM", "AZT1D", "HUPA-UCM", "T1D-UOM"}
# Interim Replace-BG block. Full cohort replaces this list later.
REPLACE_KEEP = {
    "ReplaceBG|2", "ReplaceBG|10", "ReplaceBG|11", "ReplaceBG|14", "ReplaceBG|15",
    "ReplaceBG|16", "ReplaceBG|17", "ReplaceBG|18", "ReplaceBG|19", "ReplaceBG|20",
}

LAG_LOOKBACK = 12  # 12 × 5 min = 60 min history per feature
LAG_DATASETS = ("ohio_t1dm", "azt1d")
LAG_BACKBONES = (
    ("catboost", "models_lags12", "catboost_lags12_cohort.json", train_baseline),
    ("mlp", "models_lags12_mlp", "mlp_lags12_cohort.json",
     lambda X, y, model_kwargs=None, **_: train_mlp(X, y, model_kwargs=model_kwargs)),
)


def _col_idx(lb: int) -> tuple[int, int, int]:
    """Current (most recent) column for CGM / insulin / CHO blocks."""
    return lb - 1, 2 * lb - 1, 3 * lb - 1


def _patch_lag_layout(data) -> tuple[int, int, int]:
    """Expand names/scalers so study helpers index current columns correctly.

    Idempotent: ``load_patient`` is LRU-cached and returns a shared object, so a
    second backbone must not re-read indices 0/1/2 after expansion (those are all
    CGM scalers) or every channel collapses onto the CGM MinMaxScaler.
    """
    lb = int(data.lookback)
    gi, ji, cho_i = _col_idx(lb)
    n = len(data.train_scalers)
    if n == 3:
        y_s, a_s, cho_s = data.train_scalers
    elif n == 3 * lb:
        y_s = data.train_scalers[0]
        a_s = data.train_scalers[lb]
        cho_s = data.train_scalers[2 * lb]
    else:
        raise ValueError(f"unexpected train_scalers length {n} for lookback={lb}")
    names = []
    for feat in ("CGM", "insulin", "CHO"):
        for k in range(lb):
            names.append(feat if k == lb - 1 else f"{feat}_L{lb - 1 - k}")
    data.feature_names = tuple(names)
    data.train_scalers = [y_s] * lb + [a_s] * lb + [cho_s] * lb
    data.cgm_idx, data.insulin_idx = gi, ji
    return gi, ji, cho_i


def _eval(pid: str, horizon_min: int, *, train_fn, subdir: str) -> dict:
    # Copy: cached PatientData is shared across backbones; patch mutates scalers/names.
    data = copy.deepcopy(m.load_patient(
        str(ROOT), str(pid),
        horizon_steps=pp.horizon_steps(horizon_min), lookback=LAG_LOOKBACK))
    _patch_lag_layout(data)

    span = float(m.insulin_span_u(data, horizon_min=horizon_min))
    DOCAS.SPAN = span
    old = DOCAS.train_fn
    DOCAS.train_fn = train_fn
    try:
        baseline = train_fn(data.X_train, data.y_train, model_kwargs={})
        fit_kw, drop, u_ref, _, _ = docas_fit_kw(data, span, horizon_min, cfg=m.STUDY_CFG)
        result = DOCAS(**fit_kw).fit(data.X_train, data.y_train, baseline_model=baseline)
        docas = result["model"]
    finally:
        DOCAS.train_fn = old

    tau_mode = str(fit_kw.get("tau_mode", m.STUDY_CFG.get("tau_mode", "fractional")))
    meta = dict(
        protocol=tau_mode + "_tau", tau_drop=drop, u_ref=u_ref, tau_mode=tau_mode,
        predicts_level=bool(result.get("predicts_level", False)),
        n_synth=int(result.get("n_synth") or 0),
        n_anchors=int(result.get("n_anchors") or 0),
        insulin_grid_n=int(result["grid_n"]), insulin_span_u=float(span),
        best_params={**m.STUDY_CFG, "tau_drop": drop, "u_ref": u_ref,
                     "synth_ratio": float(fit_kw["synth_ratio"]), "tau_mode": tau_mode},
        lookback=LAG_LOOKBACK, n_features=int(data.X_train.shape[1]),
    )
    m.save_models(horizon_min, pid, data, baseline=baseline, docas=docas,
                      docas_meta=meta, subdir=subdir)
    u = np.linspace(0.0, 1.0, int(result["grid_n"]))
    return m._patient_row(horizon_min, pid, data, baseline, docas, span, u, meta=meta)


def run_lag_dataset(dataset: str, *, patient_ids=None, backbones=None) -> dict:
    m.set_dataset(dataset)
    DOCAS.LOAD = False
    pids = tuple(patient_ids or m.list_patients(str(ROOT)))
    bbs = tuple(backbones or LAG_BACKBONES)
    results = {}
    for name, subdir, cohort_name, train_fn in bbs:
        out: dict = {
            "dataset": dataset, "backbone": name, "lookback": LAG_LOOKBACK,
            "n_features": 3 * LAG_LOOKBACK,
            "patient_ids": [str(p) for p in pids], "horizons": {},
        }
        for ph in m.FORECAST_HORIZONS:
            print(f"\n=== {dataset}  {name.upper()} lags={LAG_LOOKBACK}  PH{ph}  "
                  f"({3 * LAG_LOOKBACK} feats) ===", flush=True)
            rows = [_eval(pid, ph, train_fn=train_fn, subdir=subdir) for pid in pids]
            metrics = {k: m.mean_std([r[k] for r in rows]) for k in m._METRIC_KEYS}
            m.log_means("baseline", ph, rows, "baseline_test_rmse_mgdl",
                            "baseline_alignment_error")
            m.log_means("DOCAS", ph, rows, "docas_test_rmse_mgdl", "alignment_error")
            out["horizons"][str(ph)] = {
                "horizon_min": ph, "metrics": metrics, "per_patient": rows}
        path = m.results_dir() / cohort_name
        m._write_json(path, {**out, **m._protocol_meta()})
        results[name] = out
    return results


def run_lags(*, datasets=None, patient_ids=None, backbones=None) -> dict:
    ds = tuple(datasets or LAG_DATASETS)
    return {d: run_lag_dataset(d, patient_ids=patient_ids, backbones=backbones) for d in ds}



def _ms(vals):
    a = np.asarray(vals, float)
    if len(a) == 0:
        return {"mean": float("nan"), "std": 0.0}
    if len(a) == 1:
        return {"mean": float(a[0]), "std": 0.0}
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std(ddof=1)), 2)}


def _tail(data: PatientData, n: int = CAP) -> PatientData:
    if len(data.y_train) <= n:
        return data
    return PatientData(
        data.X_train[-n:], data.y_train[-n:], data.X_test, data.y_test,
        data.feature_names, data.cgm_scaler, data.train_scalers, data.patient_id,
        data.insulin_idx, data.cgm_idx, data.lookback,
    )


def _aligner(data, span, ph):
    ji = int(data.insulin_idx)
    gi = data.feature_names.index("CGM")
    fit_kw, drop, u_ref, _, _ = docas_fit_kw(data, span, ph, cfg=m.STUDY_CFG)
    return DOCAS(
        fit_kw["target"], intervention_idx=ji, outcome_idx=gi,
        action_scaler=data.train_scalers[ji], level_scaler=data.train_scalers[gi],
        delta_scaler=data.cgm_scaler, span=span, tau_drop=drop, u_ref=u_ref,
        tau_mode=str(fit_kw["tau_mode"]),
        level_ref=str(fit_kw.get("level_ref", "current")), predicts_level=False,
    ), fit_kw


def _score(model, data, aligner):
    Xte = np.asarray(data.X_test, float)
    yte = np.asarray(data.y_test, float)
    return dict(
        test_rmse_mgdl=round(ab._rmse_mgdl(model, Xte, yte, data.cgm_scaler), 2),
        alignment_error=round(float(aligner.alignment(model, Xte)), 2),
    )


def _one_comparators(pid: str, ph: int) -> dict:
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset(DS)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    span = float(m.insulin_span_u(data, horizon_min=ph))
    DOCAS.SPAN = span
    sync_cfg(SEED=42)
    DOCAS.SEED = 42
    aligner, fit_kw = _aligner(data, span, ph)
    src = _tail(data)
    out = dict(patient_id=str(pid), horizon_min=int(ph), insulin_span_u=span, arms={})

    mono = train_monotone(
        data.X_train, data.y_train, insulin_idx=data.insulin_idx,
        model_kwargs={"random_seed": 42})
    out["arms"]["monotone"] = _score(mono, data, aligner)

    mlp_rows = []
    for seed in SEEDS5:
        mdl = train_mlp(src.X_train, src.y_train, model_kwargs={"seed": seed})
        mlp_rows.append(_score(mdl, data, aligner))
    out["arms"]["mlp_baseline"] = dict(
        test_rmse_mgdl=round(float(np.mean([r["test_rmse_mgdl"] for r in mlp_rows])), 2),
        alignment_error=round(float(np.mean([r["alignment_error"] for r in mlp_rows])), 2),
        n_seeds=len(SEEDS5),
    )

    pen = PenaltyMLPForecaster(aligner=aligner, seed=42).fit(
        src.X_train, src.y_train, n_factual=len(src.X_train))
    out["arms"]["penalty_mlp"] = _score(pen, data, aligner)

    mlp_docas_rows = []
    old = DOCAS.train_fn
    try:
        for seed in SEEDS5:
            def _mlp_train(X, y, model_kwargs=None, n_factual=None, _s=seed, **__):
                return train_mlp(X, y, model_kwargs={"seed": _s})
            DOCAS.train_fn = _mlp_train
            DOCAS.SEED = seed
            sync_cfg(SEED=seed)
            base = train_mlp(src.X_train, src.y_train, model_kwargs={"seed": seed})
            res = DOCAS(**{**fit_kw, "seed": seed}).fit(
                src.X_train, src.y_train, baseline_model=base)
            mlp_docas_rows.append(_score(res["model"], data, aligner))
    finally:
        DOCAS.train_fn = old
        DOCAS.SEED = 42
        sync_cfg(SEED=42)
    out["arms"]["mlp_docas"] = dict(
        test_rmse_mgdl=round(float(np.mean([r["test_rmse_mgdl"] for r in mlp_docas_rows])), 2),
        alignment_error=round(float(np.mean([r["alignment_error"] for r in mlp_docas_rows])), 2),
        n_seeds=len(SEEDS5),
    )

    # 12-lag CatBoost baseline + population DOCAS (seed 42)
    lag = copy.deepcopy(m.load_patient(
        str(m.DATA_ROOT), str(pid),
        horizon_steps=horizon_steps(ph), lookback=LAG_LOOKBACK))
    _patch_lag_layout(lag)
    lag_span = float(m.insulin_span_u(lag, horizon_min=ph))
    DOCAS.SPAN = lag_span
    lag_aligner, lag_fit = _aligner(lag, lag_span, ph)
    lag_base = train_baseline(lag.X_train, lag.y_train, model_kwargs={"random_seed": 42})
    old = DOCAS.train_fn
    DOCAS.train_fn = train_baseline
    try:
        lag_docas = DOCAS(**{**lag_fit, "seed": 42}).fit(
            lag.X_train, lag.y_train, baseline_model=lag_base)["model"]
    finally:
        DOCAS.train_fn = old
        DOCAS.SPAN = span
    out["arms"]["lags_baseline"] = _score(lag_base, lag, lag_aligner)
    out["arms"]["lags_docas"] = _score(lag_docas, lag, lag_aligner)
    print(
        f"  cmp {pid} PH{ph}: mono {out['arms']['monotone']['test_rmse_mgdl']:.1f} "
        f"mlp {out['arms']['mlp_baseline']['test_rmse_mgdl']:.1f} "
        f"pen {out['arms']['penalty_mlp']['test_rmse_mgdl']:.1f} "
        f"mlpD {out['arms']['mlp_docas']['test_rmse_mgdl']:.1f}",
        flush=True,
    )
    return out


def _load_payload(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _summarize_arms(rows):
    rows = [r for r in rows if "arms" in r]
    arms = sorted({a for r in rows for a in r["arms"]})
    return {
        a: {
            k: _ms([r["arms"][a][k] for r in rows if a in r["arms"]])
            for k in ("test_rmse_mgdl", "alignment_error")
        }
        for a in arms
    }


def _safe_cmp(pid: str, ph: int) -> dict:
    try:
        return _one_comparators(pid, ph)
    except Exception as exc:
        print(f"  cmp skip {pid} PH{ph}: {exc}", flush=True)
        return dict(patient_id=str(pid), horizon_min=int(ph), error=str(exc))


def run_comparators():
    pids = _cohort_pids()
    payload = _load_payload(OUT)
    for ph in (60, 30):
        key = str(ph)
        store = dict(payload.get(key) or {})
        done = {r["patient_id"] for r in store.get("patients", [])}
        todo = [p for p in pids if p not in done]
        patients = list(store.get("patients", []))
        print(f"[comparators] PH{ph} todo={len(todo)}/{len(pids)}", flush=True)
        step = 8
        for i in range(0, len(todo), step):
            batch = todo[i:i + step]
            got = Parallel(n_jobs=JOBS, backend="loky")(
                delayed(_safe_cmp)(pid, ph) for pid in batch
            )
            patients.extend(got)
            store["patients"] = patients
            store["summary"] = _summarize_arms(patients)
            payload[key] = store
            _save(OUT, payload)
            print(f"[comparators] PH{ph} {len(patients)}/{len(pids)} {store['summary']}", flush=True)
    return payload


def _cohort_pids():
    m.set_dataset(DS)
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in SOURCES]
    return [p for p in pids if not str(p).startswith("ReplaceBG|") or p in REPLACE_KEEP]


def _thetas():
    thetas = {}
    for path in (REV / "full_recent").glob("*.json"):
        d = json.loads(path.read_text())
        if d.get("source") not in SOURCES or "horizons" not in d:
            continue
        thetas[str(d["patient_id"])] = d["horizons"]["60"]["theta"]
    return thetas


def run_strata():
    thetas = _thetas()
    out = _load_payload(OUT_STRATA)
    done = set(out.get("per_patient", {}))
    per = dict(out.get("per_patient") or {})
    todo = [pid for pid in thetas if str(pid) not in done]
    print(f"[strata] todo={len(todo)}/{len(thetas)}", flush=True)
    step = 8
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        got = Parallel(n_jobs=JOBS, backend="loky")(
            delayed(_strata_one)(pid, DS, thetas[pid]) for pid in batch
        )
        for pid, cell in zip(batch, got):
            per[str(pid)] = cell
        keys = sorted({k for v in per.values() for k in v})
        summary = {}
        for k in keys:
            fields = sorted({f for pid in per if k in per[pid] for f in per[pid][k]})
            summary[k] = {
                f: round(float(np.mean([per[pid][k][f] for pid in per if k in per[pid]])), 2)
                for f in fields
            }
        out = dict(per_patient=per, summary=summary)
        _save(OUT_STRATA, out)
        print(f"[strata] {len(per)}/{len(thetas)}", flush=True)
    return out


def _metabonet_test_frame(pid: str):
    source, sid = _metabonet_key(pid)
    if DS == "metabonet_full":
        from preprocessing import _read_metabonet_full_raw, TRAIN_FRAC
        raw = _read_metabonet_full_raw(source, sid)
        raw = raw.iloc[int(len(raw) * TRAIN_FRAC):].reset_index(drop=True)
    else:
        raw = _read_metabonet_split(source, sid, "test")
    df = _frame_from_metabonet_split(raw, ph_steps=1, lookback=LOOKBACK)
    ts = raw["ts"].to_numpy(np.int64)
    if ts.size == 0 or df.empty:
        raise ValueError(f"empty test {pid}")
    t0, t1 = int(ts[0]), int(ts[-1])
    n = int(round((t1 - t0) / 300_000_000_000)) + 1
    grid = t0 + np.arange(n, dtype=np.int64) * 300_000_000_000
    bolus = _impulses(grid, ts[raw["bolus"].to_numpy(float) > 0],
                      raw.loc[raw["bolus"] > 0, "bolus"].to_numpy(float))
    meal = _impulses(grid, ts[raw["carb"].to_numpy(float) > 0],
                     raw.loc[raw["carb"] > 0, "carb"].to_numpy(float))
    # Align impulses to the (possibly bridged) feature frame length.
    if len(bolus) < len(df):
        pad_b = np.zeros(len(df))
        pad_m = np.zeros(len(df))
        pad_b[:len(bolus)] = bolus
        pad_m[:len(meal)] = meal
        bolus, meal = pad_b, pad_m
    else:
        bolus, meal = bolus[:len(df)], meal[:len(df)]
    return df, bolus, meal


STEPS = 24
MARKS = (6, 12, 18, 24)


def clarke_counts(ref, pred):
    r = np.asarray(ref, float).ravel()
    p = np.asarray(pred, float).ravel()
    ok = np.isfinite(r) & np.isfinite(p) & (r > 0)
    r, p = r[ok], p[ok]
    zone = np.full(r.size, "")
    zone[((r <= 70) & (p <= 70)) | ((p <= 1.2 * r) & (p >= 0.8 * r))] = "A"
    e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    zone[(zone == "") & e] = "E"
    c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | ((r >= 130) & (r <= 180) & (p <= (7 / 5) * r - 182))
    zone[(zone == "") & c] = "C"
    d = ((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 175 / 3) & (p <= 180) & (p >= 70)) | (
        (r >= 175 / 3) & (r <= 70) & (p >= (6 / 5) * r)
    )
    zone[(zone == "") & d] = "D"
    zone[zone == ""] = "B"
    return {z: round(100.0 * float(np.mean(zone == z)), 2) for z in "ABCDE"} | {"n": int(zone.size)}


def _fit_three(data, span, horizon):
    _, R_tab = ab._pop_R(horizon, span, m.STUDY_CFG)
    theta = ab._search_theta(data, span, horizon, R_tab, seed=42)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    fit_kw, *_ = docas_fit_kw(data, span, horizon, cfg=m.STUDY_CFG)
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    pop = DOCAS(**{**fit_kw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    ctx, _tau = ab._fit_ctx(
        data, span, horizon, theta, R_tab, seed=42, base_mdl=base, X_eval=data.X_test,
    )
    return base, pop, ctx


def _long_endpoint(pid, dataset, horizon):
    m.set_dataset(dataset)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon))
    span = float(m.insulin_span_u(data, horizon_min=horizon))
    base, pop, ctx = _fit_three(data, span, horizon)
    Xt = np.asarray(data.X_test, float)
    yt = np.asarray(data.y_test, float).ravel()
    row = dict(patient_id=str(pid), horizon_min=int(horizon))
    for name, model in (("baseline", base), ("pop", pop), ("ctx", ctx)):
        g, _, _, dy = ab._cols(data, Xt, yt)
        ref = g + dy
        pred = g + ab._inv(data.cgm_scaler, model.predict(Xt))
        row[f"{name}_rmse"] = ab._rmse(model, Xt, yt, data.cgm_scaler)
        row[f"{name}_clarke"] = clarke_counts(ref, pred)
    return row


def _action_kernel(kind):
    if kind == "iob":
        dmax = int(_BOLUS_LOOKBACK_MIN) // 5
        return bolus_iob_fraction(np.arange(dmax + 1) * 5.0)
    return _carb_irf(np.arange(48 * 12 + 1) * 5.0)


def _future_known(imp, kernel, origins, steps):
    total = np.convolve(imp, kernel, mode="full")[: len(imp)]
    out = np.zeros((len(origins), steps))
    for k in range(1, steps + 1):
        t = origins + k
        extra = np.zeros(len(origins))
        for d in range(min(k, len(kernel))):
            extra += imp[t - d] * kernel[d]
        out[:, k - 1] = np.maximum(total[t] - extra, 0.0)
    return out


def _roll(model, data, cgm, iob, cho, origins):
    g = cgm[origins].astype(float).copy()
    pred = np.empty((len(origins), STEPS))
    sc = data.train_scalers
    for k in range(STEPS):
        raw = np.column_stack([g, iob[:, k], cho[:, k]])
        xs = np.column_stack([sc[i].transform(raw[:, i:i + 1]).ravel() for i in range(3)])
        g = g + ab._inv(data.cgm_scaler, model.predict(xs))
        pred[:, k] = g
    obs = np.column_stack([cgm[origins + k] for k in range(1, STEPS + 1)])
    return obs, pred


def _block_ms(rows, keys):
    return {k: _ms([r[k] for r in rows]) for k in keys}


def _strata_one(pid, dataset, theta):
    m.set_dataset(dataset)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    span = float(m.insulin_span_u(data, horizon_min=60))
    _, R_tab = ab._pop_R(60, span, m.STUDY_CFG)
    fit_kw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG)
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    pop = DOCAS(**{**fit_kw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    ctx, _ = ab._fit_ctx(data, span, 60, theta, R_tab, seed=42, base_mdl=base)
    C_te = ab._build_drc21(data, span, data.X_test, R_tab, theta, base_mdl=base)
    X = np.asarray(data.X_test, float)
    y = np.asarray(data.y_test, float).ravel()
    ji = data.insulin_idx
    yi = data.feature_names.index("CGM")
    ci = data.feature_names.index("CHO")
    g = data.train_scalers[yi].inverse_transform(X[:, yi: yi + 1]).ravel()
    cho = data.train_scalers[ci].inverse_transform(X[:, ci: ci + 1]).ravel()
    iob = data.train_scalers[ji].inverse_transform(X[:, ji: ji + 1]).ravel()
    trend = np.full(len(g), np.nan)
    step = np.abs(np.diff(g, prepend=np.nan))
    ok = np.array([i >= 3 and np.all(step[i - 2: i + 1] <= 15.0) for i in range(len(g))])
    idx = np.flatnonzero(ok)
    if len(idx):
        trend[idx] = g[idx] - g[idx - 3]
    q = np.quantile(iob, [1 / 3, 2 / 3])
    masks = {
        "cho_on": cho > 1.0,
        "cho_off": cho <= 1.0,
        "rising": trend > 0,
        "falling": trend <= 0,
        "iob_lo": iob <= q[0],
        "iob_mid": (iob > q[0]) & (iob <= q[1]),
        "iob_hi": iob > q[1],
    }
    pop_scorer = DOCAS(**{**fit_kw, "seed": 42})
    arms = {"baseline": base, "pop": pop, "ctx": ctx}
    out = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, bool)
        if name in ("rising", "falling"):
            mask = mask & np.isfinite(trend)
        if int(mask.sum()) < 20:
            continue
        Xm, ym = X[mask], y[mask]
        cell = {}
        for arm, model in arms.items():
            cell[f"{arm}_rmse"] = round(ab._rmse(model, Xm, ym, data.cgm_scaler), 2)
        cell["baseline_ae"] = round(float(pop_scorer.alignment(arms["baseline"], Xm)), 2)
        cell["pop_ae"] = round(float(pop_scorer.alignment(arms["pop"], Xm)), 2)
        th_m = {**theta, "span": span}
        tau_m = ab._make_tau(R_tab, cho[mask], C_mat=C_te[mask], **th_m)
        ctx_scorer = DOCAS(**{**fit_kw, "seed": 42, "tau_exact": tau_m})
        cell["ctx_ae"] = round(float(ctx_scorer.alignment(arms["ctx"], Xm)), 2)
        out[name] = cell
    return out


def _one_endpoint(pid: str, ph: int) -> dict:
    return _long_endpoint(pid, DS, ph)


def _one_trajectory(pid: str):
    m.set_dataset(DS)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=1)
    try:
        df, bolus, meal = _metabonet_test_frame(pid)
    except Exception as exc:
        print(f"  trajectory skip {pid}: frame {exc}", flush=True)
        return None
    cgm = df["CGM"].to_numpy(float)
    keep = _valid_feature_rows(df, LOOKBACK, 1)
    origins = keep[(keep + STEPS) < len(cgm)]
    finite = np.all(np.isfinite(cgm[origins[:, None] + np.arange(STEPS + 1)]), axis=1)
    origins = origins[finite]
    if len(origins) < 30 or len(keep) != len(data.X_test):
        print(
            f"  trajectory skip {pid}: origins={len(origins)} keep={len(keep)} X={len(data.X_test)}",
            flush=True,
        )
        return None
    iob = _future_known(bolus, _action_kernel("iob"), origins, STEPS)
    cho = _future_known(meal, _action_kernel("cho"), origins, STEPS)
    span = float(m.insulin_span_u(data, horizon_min=5))
    base, pop, ctx = _fit_three(data, span, 5)
    row = dict(patient_id=str(pid), n_origins=int(len(origins)))
    for name, model in (("baseline", base), ("pop", pop), ("ctx", ctx)):
        obs, pred = _roll(model, data, cgm, iob, cho, origins)
        marks = {}
        for step in MARKS:
            err = pred[:, step - 1] - obs[:, step - 1]
            marks[str(step * 5)] = dict(
                rmse=round(float(np.sqrt(np.mean(err ** 2))), 2),
                clarke=clarke_counts(obs[:, step - 1], pred[:, step - 1]),
            )
        row[name] = marks
    print(
        f"  trajectory {pid}: 60-min path RMSE "
        f"{row['baseline']['60']['rmse']:.1f} / {row['pop']['60']['rmse']:.1f} / {row['ctx']['60']['rmse']:.1f}",
        flush=True,
    )
    return row


def run_long():
    m.set_dataset(DS)
    pids = _cohort_pids()
    payload = _load_payload(OUT_LONG)
    payload.setdefault("endpoint", {})
    payload.setdefault("trajectory", {})
    for ph in (90, 120):
        key = str(ph)
        store = dict(payload["endpoint"].get(key) or {})
        done = {r["patient_id"] for r in store.get("per_patient", [])}
        todo = [p for p in pids if p not in done]
        rows = list(store.get("per_patient") or [])
        print(f"[endpoint] PH{ph} todo={len(todo)}/{len(pids)}", flush=True)
        step = 8
        for i in range(0, len(todo), step):
            batch = todo[i:i + step]
            got = Parallel(n_jobs=JOBS, backend="loky")(
                delayed(_one_endpoint)(pid, ph) for pid in batch
            )
            rows.extend(got)
            block = _block_ms(rows, ("baseline_rmse", "pop_rmse", "ctx_rmse"))
            for arm in ("baseline", "pop", "ctx"):
                block[f"{arm}_A"] = _ms([r[f"{arm}_clarke"]["A"] for r in rows])
            block["per_patient"] = rows
            payload["endpoint"][key] = block
            _save(OUT_LONG, payload)
            print(f"[endpoint] PH{ph} {len(rows)}/{len(pids)}", flush=True)

    tstore = dict(payload.get("trajectory") or {})
    done = {r["patient_id"] for r in tstore.get("per_patient", [])}
    todo = [p for p in pids if p not in done]
    trows = list(tstore.get("per_patient") or [])
    print(f"[trajectory] todo={len(todo)}/{len(pids)}", flush=True)
    step = 6
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        got = Parallel(n_jobs=JOBS, backend="loky")(
            delayed(_one_trajectory)(pid) for pid in batch
        )
        trows.extend(r for r in got if r is not None)
        traj = {}
        for minute in ("30", "60", "90", "120"):
            traj[minute] = {
                arm: _ms([r[arm][minute]["rmse"] for r in trows])
                for arm in ("baseline", "pop", "ctx")
            }
        traj["per_patient"] = trows
        payload["trajectory"] = traj
        _save(OUT_LONG, payload)
        print(f"[trajectory] {len(trows)} kept / {len(done) + i + len(batch)} attempted", flush=True)
    return payload


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "lags":
        datasets, pids = [], []
        for a in sys.argv[2:]:
            if a.lower().replace("-", "_") in LAG_DATASETS:
                datasets.append(a.lower().replace("-", "_"))
            else:
                pids.append(a)
        run_lags(datasets=datasets or None, patient_ids=pids or None)
        return
    run_comparators()
    run_strata()
    run_long()
    print("done", flush=True)


if __name__ == "__main__":
    main()
