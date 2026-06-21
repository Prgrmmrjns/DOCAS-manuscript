"""Regenerate manuscript replay figure (patient 588) with LightGBM labels."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import ohio_t1dm as oh
from cib import CIB_DELAY_MIN, CIB_MIN_GAP_MIN, CIB_GRID_U, cib_dose, predict_abs_glucose
from lib import normalize_features, run_ishap_pipeline
from visuals import save_replay_figure

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATIENT = "588"
WINDOW_INDEX = 5  # Prendin-faithful window shown in manuscript
WINDOW_HOURS = 8
EXCLUDE = {4}


class _NormedModel:
    def __init__(self, model, lo: np.ndarray, hi: np.ndarray):
        self._model, self._lo, self._hi = model, lo, hi

    def predict(self, X, **kwargs):
        return self._model.predict(normalize_features(X, self._lo, self._hi), **kwargs)


def _postprandial_windows(trace: pd.DataFrame) -> list[dict]:
    trace = trace.sort_values(oh.TIMESTAMP_COL).reset_index(drop=True)
    cho = "meal_carbs" if "meal_carbs" in trace.columns else "cho"
    step = int(WINDOW_HOURS * 60 / 5)
    windows = []
    for i0 in trace.index[trace[cho].astype(float) > 0].tolist():
        seg = trace.iloc[i0 : min(i0 + step, len(trace))]
        if len(seg) < step // 2:
            continue
        if (seg[cho].iloc[1:].astype(float) > 0).any():
            continue
        if (seg["bolus_dose"].iloc[1:].astype(float) > 0).any():
            continue
        windows.append({
            "meal_time": seg[oh.TIMESTAMP_COL].iloc[0],
            "meal_bolus_u": float(seg["bolus_dose"].iloc[0]),
            "trace": seg.reset_index(drop=True),
            "source_index": len(windows),
        })
    return [w for w in windows if w["source_index"] not in EXCLUDE]


def _correction_factor(model, row: np.ndarray, j: int) -> float | None:
    x0 = np.asarray(row, float).reshape(1, -1)
    x1 = x0.copy()
    u_ref = float(CIB_GRID_U[-1])
    x1[0, j] = x1[0, j] + u_ref
    drop = float(model.predict(x0)[0]) - float(model.predict(x1)[0])
    if not np.isfinite(drop) or drop <= 1e-6:
        return None
    return drop / u_ref


def _insulin_lowers(model, row: np.ndarray, j: int) -> bool:
    cf = _correction_factor(model, row, j)
    return cf is not None and cf > 1e-6


def _dss_boluses(
    model, seg: pd.DataFrame, feat: pd.DataFrame, cols: list[str],
) -> list[tuple[pd.Timestamp, int]]:
    j = cols.index("insulin")
    X = feat[cols].to_numpy(float)
    boluses: list[tuple[pd.Timestamp, int]] = []
    given: list[tuple[pd.Timestamp, float]] = []
    meal_t = pd.Timestamp(seg[oh.TIMESTAMP_COL].iloc[0])
    last: pd.Timestamp | None = None
    for k in range(len(seg)):
        t = pd.Timestamp(seg[oh.TIMESTAMP_COL].iloc[k])
        if (t - meal_t).total_seconds() / 60 < CIB_DELAY_MIN:
            continue
        if last is not None and (t - last).total_seconds() / 60 < CIB_MIN_GAP_MIN:
            continue
        iob = float(sum(
            db * oh._bolus_action_fraction(np.array([(t - tb).total_seconds() / 60.0]))[0]
            for tb, db in given if (t - tb).total_seconds() > 0
        ))
        row = X[k].copy()
        row[j] += iob
        if not _insulin_lowers(model, row, j):
            continue
        u = cib_dose(model, row, j, float(seg["glucose"].iloc[k]), iob_u=iob,
                     correction_factor=_correction_factor(model, row, j))
        if u >= 1:
            boluses.append((t, int(u)))
            given.append((t, float(u)))
            last = t
    return boluses


def _ishap_glucose_trace(
    model,
    seg: pd.DataFrame,
    feat: pd.DataFrame,
    cols: list[str],
    boluses: list[tuple[pd.Timestamp, int]],
) -> np.ndarray:
    g_obs = seg["glucose"].to_numpy(float)
    if not boluses:
        return g_obs.copy()
    j = cols.index("insulin")
    X = feat[cols].to_numpy(float)
    times = pd.to_datetime(seg[oh.TIMESTAMP_COL])
    bolus_times = [(pd.Timestamp(t), float(d)) for t, d in boluses]

    def iob_at(i: int) -> float:
        t = times.iloc[i]
        return float(sum(
            db * oh._bolus_action_fraction(np.array([(t - tb).total_seconds() / 60.0]))[0]
            for tb, db in bolus_times if (t - tb).total_seconds() > 0
        ))

    k0 = min(int(np.argmin(np.abs(times - tb))) for tb, _ in bolus_times)
    g_sim = g_obs.copy()
    for i in range(k0, len(g_sim)):
        row = X[i].copy()
        row[j] += iob_at(i)
        prev = float(g_sim[i - 1]) if i > 0 else float(g_obs[i - 1])
        g_sim[i] = float(predict_abs_glucose(model, row.reshape(1, -1), np.array([prev]))[0])
    return np.clip(g_sim, 20.0, None)


def _trace_to_dl4bgp(seg: pd.DataFrame) -> pd.DataFrame:
    """Prendin / interpretable-DL4BGP inputs on the 5-min grid."""
    return pd.DataFrame({
        "glucose": seg["glucose"].astype(float),
        "insulin": seg["basal_rate"].astype(float) + seg["bolus_dose"].astype(float),
        "cho": seg["meal_carbs"].astype(float),
    })


def main() -> None:
    trace_train = oh.load_glucose_trace(ROOT, PATIENT, split="train")
    trace_test = oh.load_glucose_trace(ROOT, PATIENT, split="test")
    X_tr = _trace_to_dl4bgp(trace_train)
    y_tr = trace_train[oh.TARGET].reset_index(drop=True)
    X_te = _trace_to_dl4bgp(trace_test)
    y_te = trace_test[oh.TARGET].reset_index(drop=True)
    cols = list(X_tr.columns)
    res = run_ishap_pipeline(
        X_tr, y_tr, oh.TARGET, root=ROOT, name=f"dl4bgp/{PATIENT}",
        X_test=X_te, y_test=y_te, random_state=42,
    )
    base = _NormedModel(res["model_before"], res["norm_lo"], res["norm_hi"])
    aligned = _NormedModel(res["model_after"], res["norm_lo"], res["norm_hi"])

    trace = trace_test.copy()
    windows = _postprandial_windows(trace)
    win = next(w for w in windows if w["source_index"] == WINDOW_INDEX)
    seg = win["trace"]
    feat = _trace_to_dl4bgp(seg)
    g = seg["glucose"].to_numpy(float)

    bol_b = _dss_boluses(base, seg, feat, cols)
    bol_i = _dss_boluses(aligned, seg, feat, cols)
    g_base = g.copy()  # baseline issues no boluses → observed CGM
    g_ishap = _ishap_glucose_trace(aligned, seg, feat, cols, bol_i)

    out = os.path.join(ROOT, "manuscript", "images", "replay_588.png")
    save_replay_figure(
        times=seg[oh.TIMESTAMP_COL],
        glucose=g,
        sim_baseline_dss=g_base,
        sim_ishap_dss=g_ishap,
        meal_bolus_u=win["meal_bolus_u"],
        bolus_baseline=bol_b,
        bolus_ishap=bol_i,
        out_path=out,
        title=f"Patient {PATIENT} · postprandial window {WINDOW_INDEX + 1} (index {WINDOW_INDEX})",
    )
    print(f"Wrote {out}  (baseline CIB={len(bol_b)}, iSHAP CIB={len(bol_i)})")


if __name__ == "__main__":
    main()
