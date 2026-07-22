"""Paired ReplayBG validation (Prendin et al. 2023 CIB protocol)."""
from __future__ import annotations
import json, os, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from callbacks.callback import ReplayCallback
from data.single_meal_t1d_data import SingleMealT1DData
from distributions import Gamma, LogNormal, Normal, Uniform
from model.single_meal_t1d import SingleMealT1DModel
from replaybg import ReplayBG
from utils.load_results import load_results
from utils.numba_dicts import to_typed_f64_dict
from ohio_t1dm_eval import (
    DATA_ROOT, FORECAST_HORIZONS, OHIO_BACKEND, RESULTS, docas_ready, load_models, mean_std,
    replaybg_response, set_backend,
)
from ohio_t1dm_preprocessing import CGM_STEP_MIN, TIMESTAMP_COL, bolus_action_fraction, list_patients, load_glucose_trace, load_patient_meta, paper_feat_arrays, seg_to_replaybg_df
from docas import DOCAS
_SINGLE_MEAL_PRIORS = {
    "Gb": {"prior": Normal(mu=119.13, sigma=7.11), "min": 70, "max": 150},
    "SG": {"prior": LogNormal(mu=-3.8, sigma=0.5), "min": 0, "max": 0.5},
    "p2": {"prior": Normal(mu=0.11, sigma=0.004), "min": 0, "max": 0.5},
    "f": {"prior": Normal(mu=0.8, sigma=0.05), "min": 0, "max": 1},
    "ka2": {"prior": LogNormal(mu=-4.2875, sigma=0.4274), "min": 0, "max": 0.5},
    "kd": {"prior": LogNormal(mu=-3.5090, sigma=0.6187), "min": 0, "max": 0.5},
    "kempt": {"prior": LogNormal(mu=-1.9646, sigma=0.7069), "min": 0, "max": 0.75},
    "SI": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
    "kabs": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
    "beta": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
}
TWIN_N_STARTS = 16
REPLAY_HORIZONS = FORECAST_HORIZONS
REPLAY_RESULTS = RESULTS / "replay"
REPLAY_WORKSPACE = REPLAY_RESULTS / "workspace"
CIB_TARGET_MGDL = 120.0
HYPER_TRIGGER_MGDL = 180.0
CIB_DELAY_MIN = 120.0
CIB_GAP_MIN = 120.0
CIB_PENALTY_U2 = 10.0
CIB_MIN_U = 0.5
CIB_BOLUS_MAX_U = 10.0
CIB_BOLUS_STEP_U = 0.5
CIB_BOLUS_GRID = np.arange(0.0, CIB_BOLUS_MAX_U + 0.5 * CIB_BOLUS_STEP_U, CIB_BOLUS_STEP_U)
REPLAY_WINDOW_H = 8
MAX_WINDOWS_PER_PATIENT = 5
TWIN_RMSE_PRIMARY_MGDL = 25.0
_CPU = max(1, os.cpu_count() or 4)
TWIN_N_JOBS = max(1, _CPU - 1)
REPLAY_PATIENT_WORKERS = max(1, min(12, _CPU - 1))
YTS_MIN = 5
TIR_LO, TIR_HI = 70.0, 180.0
TABLE2_PATIENT = "588"
_RBG = None
_TWINNED: dict[str, dict] = {}
_SEG: dict[tuple, object] = {}
_RBG_DATA: dict[tuple, object] = {}


def _sync_replay_paths():
    global REPLAY_RESULTS
    name = "replay" if OHIO_BACKEND == "lgbm" else f"replay_{OHIO_BACKEND}"
    REPLAY_RESULTS = RESULTS / name


def replay_cohort_path(scope: str = "") -> Path:
    _sync_replay_paths()
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"cohort{scope}.json"


def replay_horizon_payload(cohort: dict, horizon_min: int) -> dict:
    return cohort["horizons"][str(horizon_min)]


def replay_json_default(value):
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value))


def glycemic_metrics(glucose) -> dict:
    g = np.asarray(glucose, float)
    g = g[np.isfinite(g)]
    return {"tir": 100. * float(np.mean((g >= TIR_LO) & (g <= TIR_HI))), "tbr": 100. * float(np.mean(g < TIR_LO)), "tar": 100. * float(np.mean(g > TIR_HI))}


def _future_cgm_curve(model, X):
    X = np.asarray(X, float)
    current = X[:, 0]
    if hasattr(model, "predict_delta"):
        return current + np.asarray(model.predict_delta(X), float).ravel()
    return np.asarray(model.predict(X), float).ravel()


def _predicts_glucose_lowering(curve: np.ndarray) -> bool:
    curve = np.asarray(curve, float)
    return len(curve) > 1 and float(np.min(curve[1:])) < float(curve[0])


def _dose_and_curve(model, row, insulin_idx, cfg, grid_u, *, span):
    grid_u = np.asarray(grid_u, float)
    row = np.asarray(row, float).ravel()
    if hasattr(model, "future_dose_curve"):
        curve = np.asarray(model.future_dose_curve(row, grid_u, span=span), float)
    else:
        X = np.tile(row, (len(grid_u), 1))
        X[:, insulin_idx] += grid_u
        curve = _future_cgm_curve(model, X)
    objective = (curve - cfg["target"]) ** 2 + cfg["penalty"] * grid_u ** 2
    if not _predicts_glucose_lowering(curve):
        return 0.0, curve
    best = int(np.argmin(objective))
    if best == 0 or objective[best] >= objective[0]:
        return 0.0, curve
    dose = float(grid_u[best])
    return (dose if dose >= cfg["min_u"] else 0.0), curve


def _choose_cib_dose(g, iob, cho, model, cols, cfg, grid):
    j_ins = cols.index("insulin")
    row = np.array([g, iob, cho], float)
    grid = np.asarray(grid, float)
    raw, curve = _dose_and_curve(model, row, j_ins, cfg, grid, span=float(DOCAS.SPAN))
    delivered = CIB_BOLUS_STEP_U * round(raw * cfg["dose_gain"] / CIB_BOLUS_STEP_U)
    delivered = delivered if delivered >= cfg["min_u"] else 0.
    return delivered, raw, curve, grid.tolist()


class PrendinCIB(ReplayCallback):
    name = "prendin_cib"

    def __init__(self, *, model, cfg, grid, cols, cho_trace, observed_cgm, arm):
        self.forecaster, self.cfg, self.grid, self.cols = model, cfg, np.asarray(grid, float), cols
        self.cho_trace, self.observed_cgm, self.arm = np.asarray(cho_trace, float), np.asarray(observed_cgm, float), arm
        self.last_cib_idx, self.decisions, self.doses = None, [], []
        self._bolus_idx, self._bolus_events = None, []

    def action(self, ctx):
        cfg = self.cfg
        k = int(ctx.k)
        if (k + 1) % YTS_MIN or k < cfg["delay_min"]: return
        g = float(ctx.measurement)
        if g <= cfg["trigger"] or self.last_cib_idx is not None and k - self.last_cib_idx < cfg["gap_min"]: return
        slot = min(k // YTS_MIN, len(self.cho_trace) - 1)
        bw = float(self.rbg_data.body_weight)
        if self._bolus_idx is None:
            self._bolus_idx = next(i for i, name in ctx.data_to_input.items() if name == "bolus")
            hist = np.asarray(ctx.input_history[:k, self._bolus_idx], float) * (bw / 1000.0)
            self._bolus_events.extend((int(i), float(hist[i])) for i in np.flatnonzero(hist > 0))
        times, units = zip(*self._bolus_events) if self._bolus_events else ((), ())
        iob = float(np.dot(units, bolus_action_fraction(k - 1 - np.asarray(times)))) if times else 0.
        delivered, raw, curve, grid_u = _choose_cib_dose(g, iob, float(self.cho_trace[slot]), self.forecaster, self.cols, cfg, self.grid)
        self.decisions.append({"time_min": k, "arm": self.arm, "observed_cgm": float(self.observed_cgm[slot]), "simulated_cgm": g, "model_cgm": g, "iob_u": iob, "cho": float(self.cho_trace[slot]), "grid_u": grid_u, "future_cgm_curve": curve.tolist(), "raw_dose_u": raw, "dose_gain": cfg["dose_gain"], "delivered_dose_u": delivered, "clipped": delivered != raw})
        if delivered < cfg["min_u"]:
            return
        ctx.add_input("bolus", delivered * (1000.0 / bw))
        ctx.log(cb_u=delivered, arm=self.arm)
        self.last_cib_idx = k
        self.doses.append((k, delivered))
        self._bolus_events.append((k, delivered))


def cib_handler_params(feat, model, cfg, grid, cols, *, arm):
    return {"model": model, "cfg": cfg, "grid": np.asarray(grid, float), "cols": cols,
            "cho_trace": feat["CHO"], "observed_cgm": feat["CGM"], "arm": arm}


def _cgm_5min(out, n: int) -> np.ndarray:
    cgm = np.asarray(out["output"], float)
    if len(cgm) > n: cgm = cgm[::YTS_MIN]
    return cgm[:n]


def _rbg() -> ReplayBG:
    global _RBG
    if _RBG is None:
        REPLAY_WORKSPACE.mkdir(parents=True, exist_ok=True)
        _RBG = ReplayBG(ts=1, seed=1, plot_mode=False, verbose=False)
    return _RBG


def _seg_df(seg):
    key = (seg.height, str(seg[TIMESTAMP_COL][0]))
    if key not in _SEG:
        _SEG[key] = seg_to_replaybg_df(seg, window_h=REPLAY_WINDOW_H, yts_min=YTS_MIN)
    return _SEG[key]


def _rbg_data(seg, *, bw, u2ss=None):
    key = (id(seg), float(bw), None if u2ss is None else float(u2ss))
    if key not in _RBG_DATA:
        data = SingleMealT1DData(data=_seg_df(seg), body_weight=float(bw), environment=_rbg().environment)
        if u2ss is not None: data.u2ss = float(u2ss)
        _RBG_DATA[key] = data
    return _RBG_DATA[key]


def _twin(seg, *, bw, name, u2ss):
    if name in _TWINNED:
        return _TWINNED[name]
    path = REPLAY_WORKSPACE / f"twin_{name}.pkl"
    if path.is_file():
        twin = load_results(str(REPLAY_WORKSPACE), name, prefix="twin")
    else:
        rbg_data = _rbg_data(seg, bw=bw, u2ss=u2ss)
        model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)
        twin = _rbg().twin(rbg_data=rbg_data, model=model, unknown_parameters_prior=_SINGLE_MEAL_PRIORS, parallelize=TWIN_N_JOBS > 1, n_jobs=TWIN_N_JOBS, n_starts=TWIN_N_STARTS, path=str(REPLAY_WORKSPACE), save_name=name)
    _TWINNED[name] = twin
    return twin


def _replay(seg, *, bw, name, u2ss, suffix, cib_params=None):
    twin = _twin(seg, bw=bw, name=name, u2ss=u2ss)
    rbg_data = _rbg_data(seg, bw=bw, u2ss=u2ss)
    np.random.seed(zlib.crc32(f"{name}{suffix}".encode()) & 0xFFFFFFFF)
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, theta0=to_typed_f64_dict(twin["theta"]))
    cb = PrendinCIB(**cib_params) if cib_params is not None else None
    # Skip pickle I/O: cohort JSON already stores CGM/decisions.
    out = _rbg().replay(rbg_data=rbg_data, model=model, callbacks=[cb] if cb else None)
    return _cgm_5min(out, seg.height), out, cb.decisions if cb else [], cb.doses if cb else []


def replay_windows(trace):
    trace = trace.sort(TIMESTAMP_COL)
    cho = "meal_carbs" if "meal_carbs" in trace.columns else "cho"
    steps = int(REPLAY_WINDOW_H * 60 / CGM_STEP_MIN)
    cho_vals = trace[cho].to_numpy().astype(float)
    bolus = trace["bolus_dose"].to_numpy().astype(float)
    windows = []
    for start in np.where(cho_vals > 0)[0]:
        end = min(int(start) + steps, trace.height)
        if end - start < steps // 2: continue
        if (cho_vals[start + 1 : end] > 0).any() or (bolus[start + 1 : end] > 0).any(): continue
        seg = trace.slice(int(start), end - int(start))
        windows.append({"trace": seg, "source_start": int(start), "observed_tar": glycemic_metrics(seg["glucose"].to_numpy())["tar"]})
    windows.sort(key=lambda w: (-w["observed_tar"], w["source_start"]))
    return [{**w, "source_index": i} for i, w in enumerate(windows[:MAX_WINDOWS_PER_PATIENT])]


def cib_cfg(horizon_min):
    dose_gain = float(np.sqrt(replaybg_response(1., min(REPLAY_HORIZONS)) / replaybg_response(1., horizon_min)))
    return {"loop": "closed", "target": CIB_TARGET_MGDL, "trigger": HYPER_TRIGGER_MGDL, "delay_min": CIB_DELAY_MIN, "gap_min": CIB_GAP_MIN, "penalty": CIB_PENALTY_U2, "min_u": CIB_MIN_U, "dose_gain": dose_gain}


def eval_replay_window(seg, pid, horizon_min, baseline, docas, cols, bw, u2ss, cfg, grid, *, source_start, source_index, observed_tar):
    feat = paper_feat_arrays(seg)
    name = f"paper_v2_{pid}_start{source_start}"
    observed = seg["glucose"].to_numpy().astype(float)
    no_dss, _, _, _ = _replay(seg, bw=bw, name=name, u2ss=u2ss, suffix=f"_ph{horizon_min}_none")
    twin_rmse = float(np.sqrt(np.mean((no_dss - observed) ** 2)))
    base_g, _, base_dec, base_doses = _replay(seg, bw=bw, name=name, u2ss=u2ss, suffix=f"_ph{horizon_min}_base", cib_params=cib_handler_params(feat, baseline, cfg, grid, cols, arm="baseline"))
    docas_g, _, doc_dec, doc_doses = _replay(seg, bw=bw, name=name, u2ss=u2ss, suffix=f"_ph{horizon_min}_docas", cib_params=cib_handler_params(feat, docas, cfg, grid, cols, arm="docas"))
    base_m, docas_m = glycemic_metrics(base_g), glycemic_metrics(docas_g)
    return {"patient_id": pid, "horizon_min": horizon_min, "window_index": source_index, "source_start": source_start, "observed_tar": observed_tar, "twin_rmse": twin_rmse, "no_dss": glycemic_metrics(no_dss), "baseline": base_m, "docas": docas_m, "delta_tir": docas_m["tir"] - base_m["tir"], "delta_tbr": docas_m["tbr"] - base_m["tbr"], "delta_tar": docas_m["tar"] - base_m["tar"], "baseline_doses": [{"time_min": t, "units": u} for t, u in base_doses], "docas_doses": [{"time_min": t, "units": u} for t, u in doc_doses], "baseline_decisions": base_dec, "docas_decisions": doc_dec, "trajectories": {"observed": observed, "no_dss": no_dss, "baseline_dss": base_g, "docas_dss": docas_g}}


def format_replay_patient_line(rows) -> str:
    if not rows: return "no windows"
    tir = np.mean([r["docas"]["tir"] for r in rows])
    tbr = np.mean([r["docas"]["tbr"] for r in rows])
    tar = np.mean([r["docas"]["tar"] for r in rows])
    bol = np.mean([len(r["docas_doses"]) for r in rows])
    return f"n{len(rows)} TIR {tir:.0f}% TBR {tbr:.0f}% TAR {tar:.0f}% CIB {bol:.1f} boluses"


def _table2_from_windows(horizon_min: int, windows: list) -> dict:
    patient_windows = [r for r in windows if r["patient_id"] == TABLE2_PATIENT and r["twin_rmse"] <= TWIN_RMSE_PRIMARY_MGDL]
    summary = {}
    for arm in ("no_dss", "baseline", "docas"):
        if arm == "no_dss":
            metrics = [glycemic_metrics(r["trajectories"]["no_dss"]) for r in patient_windows]
            doses, boluses = [0.0] * len(patient_windows), [0] * len(patient_windows)
        else:
            metrics = [r[arm] for r in patient_windows]
            doses = [sum(d["units"] for d in r[f"{arm}_doses"]) for r in patient_windows]
            boluses = [len(r[f"{arm}_doses"]) for r in patient_windows]
        summary[arm] = {"tir": mean_std([m["tir"] for m in metrics]), "tbr": mean_std([m["tbr"] for m in metrics]), "tar": mean_std([m["tar"] for m in metrics]), "insulin_u": mean_std(doses), "n_boluses": mean_std(boluses)}
    return {"horizon_min": horizon_min, "patient_id": TABLE2_PATIENT, "summary": summary, "paired_docas_vs_baseline": {"mean_delta_tir_pp": float(np.mean([r["delta_tir"] for r in patient_windows])), "mean_delta_tbr_pp": float(np.mean([r["delta_tbr"] for r in patient_windows])), "n_windows": len(patient_windows)}}


def format_table2_line(table2: dict) -> str:
    s, n = table2["summary"], table2["paired_docas_vs_baseline"]["n_windows"]
    ms = lambda arm, key: f"{s[arm][key]['mean']:.2f}±{s[arm][key]['std']:.2f}"
    return (f"  Table2 PH{table2['horizon_min']} patient {table2['patient_id']} n={n}: "
            f"No DS TIR {ms('no_dss', 'tir')} | Baseline TIR {ms('baseline', 'tir')} | DOCAS TIR {ms('docas', 'tir')}")


def _patient_windows(pid: str):
    windows = replay_windows(load_glucose_trace(str(DATA_ROOT), pid, split="test"))
    if not windows:
        return [], None, None
    meta = load_patient_meta(str(DATA_ROOT), pid)
    return windows, float(meta["weight_kg"]), meta.get("u2ss")


def _pre_twin_patients(pids, horizon_min: int) -> None:
    """Twin all windows once with full CPU parallelism before patient-parallel replay."""
    missing = []
    for pid in pids:
        if not docas_ready(horizon_min, pid):
            continue
        windows, bw, u2ss = _patient_windows(pid)
        for w in windows:
            name = f"paper_v2_{pid}_start{w['source_start']}"
            if name in _TWINNED or (REPLAY_WORKSPACE / f"twin_{name}.pkl").is_file():
                continue
            missing.append((w["trace"], bw, name, u2ss))
    if not missing:
        return
    print(f"  twinning {len(missing)} windows · {TWIN_N_STARTS} starts × {TWIN_N_JOBS} jobs", flush=True)
    for i, (seg, bw, name, u2ss) in enumerate(missing, 1):
        _twin(seg, bw=bw, name=name, u2ss=u2ss)
        print(f"    [{i}/{len(missing)}] {name}", flush=True)


def _eval_patient_horizon(pid: str, horizon_min: int) -> list:
    if not docas_ready(horizon_min, pid):
        return []
    baseline, docas, cols = load_models(horizon_min, pid)
    windows, bw, u2ss = _patient_windows(pid)
    if not windows:
        return []
    cfg, grid = cib_cfg(horizon_min), CIB_BOLUS_GRID
    return [eval_replay_window(
        w["trace"], pid, horizon_min, baseline, docas, cols, bw, u2ss, cfg, grid,
        source_start=w["source_start"], source_index=w["source_index"], observed_tar=w["observed_tar"],
    ) for w in windows]


def run_horizon(horizon_min: int, *, patient_ids=None) -> dict:
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    rows = []
    print(f"[Replay PH{horizon_min} · {OHIO_BACKEND}] {len(pids)} patients", flush=True)
    _pre_twin_patients(pids, horizon_min)
    workers = REPLAY_PATIENT_WORKERS
    if workers > 1 and len(pids) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_patient_horizon, pid, horizon_min): pid for pid in pids}
            done = 0
            for future in as_completed(futures):
                pid = futures[future]
                done += 1
                patient_rows = future.result()
                rows.extend(patient_rows)
                if patient_rows:
                    print(f"  [{done}/{len(pids)}] {pid} {format_replay_patient_line(patient_rows)}", flush=True)
    else:
        for i, pid in enumerate(pids, 1):
            patient_rows = _eval_patient_horizon(pid, horizon_min)
            if not patient_rows:
                continue
            rows.extend(patient_rows)
            print(f"  [{i}/{len(pids)}] {pid} {format_replay_patient_line(patient_rows)}", flush=True)
    table2 = _table2_from_windows(horizon_min, rows)
    print(format_table2_line(table2), flush=True)
    return {"horizon_min": horizon_min, "dss_config": cib_cfg(horizon_min), "patient_ids": list(pids), "windows": rows, "table2": table2}


def run(*, patient_ids=None, backend=None) -> dict:
    if backend is not None:
        set_backend(backend)
    _sync_replay_paths()
    pids = tuple(patient_ids or list_patients(str(DATA_ROOT)))
    all_patients = tuple(list_patients(str(DATA_ROOT)))
    scope = "" if pids == all_patients else "_patients-" + "-".join(pids)
    horizons = {str(ph): run_horizon(ph, patient_ids=pids) for ph in REPLAY_HORIZONS}
    path = replay_cohort_path(scope)
    payload = {"patient_ids": list(pids), "backend": OHIO_BACKEND, "horizons": horizons}
    path.write_text(json.dumps(payload, indent=2, default=replay_json_default))
    return {"path": str(path), **payload}

