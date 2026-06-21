"""Ohio T1DM: build ``datasets/ohio_t1dm.csv`` from XML; load patient splits.

Insulin feature (``insulin``)
-----------------------------
Each 5-minute CGM row gets **active insulin in units (U)** — how much insulin
is still driving glucose uptake at that moment, combining pump basal and recent
boluses (Humalog, per patient XML).

1. **Bolus insulin on board (IOB)** — for every bolus in the last 48 h:
   ``dose_U × action(lag_min)``. ``action`` is a two-compartment curve
   (rise τ=15 min, fall τ=120 min) peaking at ~36 min, normalized so **1 U
   delivered contributes at most 1 U active** at peak. Biologically: rapid-
   acting insulin increases peripheral glucose uptake (GLUT4); higher IOB
   lowers predicted future glucose (our SCM monotonicity rule).

2. **Basal contribution** — ``BASAL_ACTIVE_HOURS × basal_rate_U_per_hr``.
   Pump basal (piecewise schedule from XML) is scaled to an active-U
   equivalent over a 2.5 h horizon — background insulin present between boluses.

Raw XML:
- ``<glucose_level event>`` — current CGM (mg/dL) at each 5-min row
- ``<bolus event dose ts_begin>`` — meal/correction boluses (U)
- ``<basal event value ts>`` — scheduled basal rate (U/hr)
- ``<meal event carbs>`` — separate ``carbs`` feature (g, CHO absorption IRF)

Target ``future glucose`` = glucose(t+30 min) − glucose(t) in mg/dL.
"""

from __future__ import annotations

import glob
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

PREDICTION_HORIZON_MIN = 30  # delete datasets/ohio_t1dm.csv to rebuild after change
HORIZON_STEPS = PREDICTION_HORIZON_MIN // 5
TARGET = "future glucose"
CSV = "ohio_t1dm.csv"
PATIENT_COL = "patient_id"
SPLIT_COL = "data_split"
TEST_SIZE = 0.25
NAME = "ohio_t1dm"
TIMESTAMP_COL = "timestamp"
# Prendin et al. (2023): current CGM, active insulin, carb absorption.
FEATURE_COLUMNS = ("glucose", "insulin", "carbs")

SCM_FEATURES = ("insulin",)

# Rapid-acting bolus PK (Humalog-like): onset ~15 min, duration ~4 h.
_BOLUS_TAU_RISE_MIN = 15.0
_BOLUS_TAU_FALL_MIN = 120.0
_BOLUS_LOOKBACK_MIN = 48 * 60
_d_peak = np.log(_BOLUS_TAU_FALL_MIN / _BOLUS_TAU_RISE_MIN) / (
    1 / _BOLUS_TAU_RISE_MIN - 1 / _BOLUS_TAU_FALL_MIN
)
_BOLUS_ACTION_PEAK = (
    np.exp(-_d_peak / _BOLUS_TAU_FALL_MIN) - np.exp(-_d_peak / _BOLUS_TAU_RISE_MIN)
) / (1 / _BOLUS_TAU_RISE_MIN - 1 / _BOLUS_TAU_FALL_MIN)
# Effective active units from current basal rate (U/hr × hours).
_BASAL_ACTIVE_HOURS = 2.5

_NS = int(60e9)  # nanoseconds per minute
_CSV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s.strip(), "%d-%m-%Y %H:%M:%S"))


def _ns(events: list) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([t.value for t, _ in events], np.int64),
        np.array([v for _, v in events], np.float64),
    )


def _irf_sum(grid: np.ndarray, et: np.ndarray, ev: np.ndarray, irf) -> np.ndarray:
    dt = (grid[:, None] - et[None, :]) / _NS
    m = (dt > 0) & (dt <= 48 * 60)
    return (irf(np.where(m, dt, 0.0)) * ev * m).sum(1)


def _bolus_action_fraction(lag_min: np.ndarray) -> np.ndarray:
    """Share of a 1 U bolus still acting after ``lag_min`` (0→1, peak≈1 at ~36 min)."""
    lag = np.asarray(lag_min, float)
    out = np.zeros_like(lag)
    m = lag > 0
    if not np.any(m):
        return out
    t1, t2 = _BOLUS_TAU_RISE_MIN, _BOLUS_TAU_FALL_MIN
    d = lag[m]
    out[m] = (np.exp(-d / t2) - np.exp(-d / t1)) / (1 / t1 - 1 / t2) / _BOLUS_ACTION_PEAK
    return np.maximum(0.0, out)


def _bolus_insulin_on_board(grid: np.ndarray, bolus_times: np.ndarray, bolus_doses: np.ndarray) -> np.ndarray:
    """IOB from boluses in U: Σ dose_U × action_fraction(lag)."""
    dt = (grid[:, None] - bolus_times[None, :]) / _NS
    m = (dt > 0) & (dt <= _BOLUS_LOOKBACK_MIN)
    return (_bolus_action_fraction(np.where(m, dt, 0.0)) * bolus_doses * m).sum(1)


def _basal_rate_at(grid: np.ndarray, basal_times: np.ndarray, basal_rates: np.ndarray) -> np.ndarray:
    """Current pump basal rate (U/hr) at each CGM timestamp.

    Basal events in the Ohio XML do not cover every CGM timestamp (gaps before the
    first event and long pump-schedule intervals). Leading/zero gaps are filled with
    the patient's median positive basal rate so that ReplayBG twinning always sees a
    physiological background insulin infusion (a zero-basal twin is degenerate).
    """
    idx = np.searchsorted(basal_times, grid, "right") - 1
    out = np.zeros(grid.size, float)
    ok = idx >= 0
    out[ok] = basal_rates[idx[ok]]
    pos = basal_rates[basal_rates > 0]
    if pos.size:
        fill = float(np.median(pos))
        out = np.where(out > 0, out, fill)
    return out


def _carb_action_fraction(lag_min: np.ndarray) -> np.ndarray:
    """Share of meal carbs still absorbing (gamma-like, peak ~48 min)."""
    lag = np.asarray(lag_min, float)
    out = np.zeros_like(lag)
    m = lag > 0
    if not np.any(m):
        return out
    d = lag[m]
    out[m] = (d / 48) ** 2 * np.exp(-d / 48) / (4 * np.exp(-2))
    return np.maximum(0.0, out)


def _xml_paths(ohio: str) -> list[str]:
    paths = []
    for year in ("2018", "2020"):
        for split in ("train", "test"):
            paths.extend(sorted(glob.glob(os.path.join(ohio, year, split, "*-ws-*.xml"))))
    return paths


def _parse(path: str) -> dict:
    r = ET.parse(path).getroot()
    stem = os.path.splitext(os.path.basename(path))[0]
    pid = str(r.attrib.get("id", stem.split("-")[0]))
    weight = float(r.attrib.get("weight") or 75.0)
    ev = lambda tag: [(_ts(x.attrib["ts"]), float(x.attrib["value"])) for x in r.findall(f".//{tag}/event")]
    gluc = ev("glucose_level")
    bolus = [(_ts(x.attrib["ts_begin"]), float(x.attrib["dose"])) for x in r.findall(".//bolus/event") if float(x.attrib.get("dose") or 0) > 0]
    meal = [(_ts(x.attrib["ts"]), float(x.attrib.get("carbs") or 0)) for x in r.findall(".//meal/event")]
    basal = [(_ts(x.attrib["ts"]), float(x.attrib.get("value") or 0)) for x in r.findall(".//basal/event")]
    for s in (gluc, bolus, meal, basal):
        s.sort(key=lambda e: e[0])
    return dict(pid=pid, weight=weight, gluc=gluc, bolus=bolus, meal=meal, basal=basal)


def _frame(pat: dict) -> pl.DataFrame:
    g = np.array([v for _, v in pat["gluc"]], float)
    i0, i1 = 24, g.size - HORIZON_STEPS
    grid = np.array([t.value for t, _ in pat["gluc"][i0:i1]], np.int64)
    times = [t for t, _ in pat["gluc"][i0:i1]]

    mt, mv = _ns([(t, v) for t, v in pat["meal"] if v > 0])
    bt, bv = _ns(pat["bolus"])
    bat, bav = _ns(pat["basal"])
    bolus_iob = _bolus_insulin_on_board(grid, bt, bv)
    basal_rate = _basal_rate_at(grid, bat, bav)
    active_insulin = bolus_iob + _BASAL_ACTIVE_HOURS * basal_rate
    carb_absorption = _irf_sum(grid, mt, mv, _carb_action_fraction)

    return pl.DataFrame({
        PATIENT_COL: [pat["pid"]] * grid.size,
        TIMESTAMP_COL: times,
        "glucose": g[i0:i1],
        "insulin": active_insulin,
        "carbs": carb_absorption,
        TARGET: g[i0 + HORIZON_STEPS : i1 + HORIZON_STEPS] - g[i0:i1],
    })


def _trace_frame(pat: dict) -> pl.DataFrame:
    """5-min grid with absolute glucose and pump events (for DSS replay figures)."""
    g = np.array([v for _, v in pat["gluc"]], float)
    i0, i1 = 24, g.size - HORIZON_STEPS
    grid = np.array([t.value for t, _ in pat["gluc"][i0:i1]], np.int64)
    times = [t for t, _ in pat["gluc"][i0:i1]]

    mt, mv = _ns([(t, v) for t, v in pat["meal"] if v > 0])
    bt, bv = _ns(pat["bolus"])
    bat, bav = _ns(pat["basal"])
    bolus_iob = _bolus_insulin_on_board(grid, bt, bv)
    basal_rate = _basal_rate_at(grid, bat, bav)
    active_insulin = bolus_iob + _BASAL_ACTIVE_HOURS * basal_rate
    carb_absorption = _irf_sum(grid, mt, mv, _carb_action_fraction)

    meal_carbs = np.zeros(grid.size, float)
    if mt.size:
        for t_ns, v in zip(mt, mv):
            idx = np.searchsorted(grid, t_ns)
            if 0 <= idx < grid.size:
                meal_carbs[idx] = float(v)
    bolus_dose = np.zeros(grid.size, float)
    if bt.size:
        for t_ns, v in zip(bt, bv):
            idx = np.searchsorted(grid, t_ns)
            if 0 <= idx < grid.size:
                bolus_dose[idx] = float(v)

    return pl.DataFrame({
        PATIENT_COL: [pat["pid"]] * grid.size,
        TIMESTAMP_COL: times,
        "glucose": g[i0:i1],
        "insulin": active_insulin,
        "carbs": carb_absorption,
        "meal_carbs": meal_carbs,
        "bolus_dose": bolus_dose,
        "basal_rate": basal_rate,
        TARGET: g[i0 + HORIZON_STEPS : i1 + HORIZON_STEPS] - g[i0:i1],
    })


def load_glucose_trace(root: str, patient_id: str, *, split: str | None = None) -> pd.DataFrame:
    """Absolute glucose time series with pump events (from XML)."""
    ohio = _ohio_root(root)
    pid = str(patient_id)
    frames = []
    for p in _xml_paths(ohio):
        pat = _parse(p)
        if str(pat["pid"]) != pid:
            continue
        sp = "test" if "/test/" in p.lower() else "train"
        if split is not None and sp != split:
            continue
        frames.append(_trace_frame(pat).with_columns(pl.lit(sp).alias(SPLIT_COL)).to_pandas())
    if not frames:
        raise FileNotFoundError(f"No XML trace for patient {pid} split={split}")
    return pd.concat(frames, ignore_index=True)


def _ohio_root(root: str) -> str:
    root = os.path.abspath(root)
    for rel in (("datasets", "OhioT1DM"), ("OhioT1DM",)):
        ohio = os.path.join(root, *rel)
        if os.path.isdir(ohio):
            return ohio
    raise FileNotFoundError("OhioT1DM directory not found")


def build_ohio_t1dm_dataframe(ohio: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for p in _xml_paths(ohio):
        try:
            pat = _parse(p)
            sp = "test" if "/test/" in p.lower() else "train"
            frames.append(_frame(pat).with_columns(pl.lit(sp).alias(SPLIT_COL)))
        except (ET.ParseError, KeyError, ValueError, OSError):
            continue
    if not frames:
        raise RuntimeError(f"No rows built from XML under {ohio}")
    return pl.concat(frames, how="vertical_relaxed")


def _csv_has_features(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return all(c in header for c in FEATURE_COLUMNS)


def ensure_ohio_csv(root: str) -> str:
    root = os.path.abspath(root)
    out = os.path.join(root, "datasets", CSV)
    if os.path.isfile(out) and _csv_has_features(out):
        return out
    for rel in (("datasets", "OhioT1DM"), ("OhioT1DM",)):
        ohio = os.path.join(root, *rel)
        if os.path.isdir(ohio):
            break
    else:
        raise FileNotFoundError("OhioT1DM directory not found under datasets/ or repo root")
    t0 = time.perf_counter()
    df = build_ohio_t1dm_dataframe(ohio)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.write_csv(out)
    _CSV_CACHE.pop(root, None)
    print(f"Built {out} horizon={PREDICTION_HORIZON_MIN}min rows={df.height} in {time.perf_counter() - t0:.1f}s")
    return out


def _load_csv(root: str) -> pd.DataFrame:
    root = os.path.abspath(root)
    path = os.path.join(root, "datasets", CSV)
    mtime = os.path.getmtime(path)
    cached = _CSV_CACHE.get(root)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    df = pd.read_csv(path)
    _CSV_CACHE[root] = (mtime, df)
    return df


def _patient_sub(root: str, patient_id: str) -> pd.DataFrame:
    ensure_ohio_csv(root)
    df = _load_csv(root)
    return df[df[PATIENT_COL].astype(str) == str(patient_id)].copy()


def _features(sub: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    X = sub.drop(columns=[c for c in (TARGET, PATIENT_COL, SPLIT_COL) if c in sub.columns], errors="ignore")
    return X.reindex(columns=list(cols)).reset_index(drop=True)


def patient_ids(root: str) -> list[str]:
    ensure_ohio_csv(root)
    return sorted(_load_csv(root)[PATIENT_COL].astype(str).unique().tolist())


def load_patient(root: str, patient_id: str):
    return load_train_test(root, patient_id)


def load_train_test(root: str, patient_id: str):
    sub = _patient_sub(root, patient_id)
    cols = tuple(c for c in FEATURE_COLUMNS if c in sub.columns)
    if SPLIT_COL not in sub.columns:
        X, y = _features(sub, cols), sub[TARGET].reset_index(drop=True)
        return X, y, X.iloc[:0].copy(), y.iloc[:0].copy()
    tr, te = sub[sub[SPLIT_COL] == "train"], sub[sub[SPLIT_COL] == "test"]
    X, y = _features(tr, cols), tr[TARGET].reset_index(drop=True)
    return (X, y, _features(te, cols), te[TARGET].reset_index(drop=True)) if len(te) else (X, y, X.iloc[:0].copy(), y.iloc[:0].copy())


if __name__ == "__main__":
    ensure_ohio_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
