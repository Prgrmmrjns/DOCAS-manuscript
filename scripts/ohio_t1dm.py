"""Ohio T1DM: build ``datasets/ohio_t1dm.csv`` from XML when missing; load + SCM rules."""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import polars as pl

from lib import MonotonicRelationship

# --- dataset module metadata ---
NAME = "ohio_t1dm"
TASK = "regression"
TARGET = "future glucose"
CSV = "ohio_t1dm.csv"
PATIENT_COL = "patient_id"
SPLIT_COL = "data_split"
MULTI_PATIENT = True
TEST_SIZE = 0.25

PA_SENSOR_COLUMNS = ("pa_steps", "pa_hr", "pa_accel", "pa_exercise")
METABOLIC_COLUMNS = ("insulin", "carbs")
GLUCOSE_DYNAMICS_COLUMNS = (
    "glucose_roc_5m", "glucose_roc_30m", "glucose_accel", "glucose_context_m30", "glucose_std_1h",
)
FEATURE_COLUMNS = METABOLIC_COLUMNS + PA_SENSOR_COLUMNS + GLUCOSE_DYNAMICS_COLUMNS

HIDDEN_CONFOUNDER_CANDIDATES = tuple(f"confounder_{i}" for i in range(1, 11))
LATENT_COLUMNS = HIDDEN_CONFOUNDER_CANDIDATES

SCM_METABOLIC_COLUMNS = METABOLIC_COLUMNS
SCM_ACTIVITY_COLUMNS = PA_SENSOR_COLUMNS
SCM_GLUCOSE_DYNAMICS_COLUMNS = GLUCOSE_DYNAMICS_COLUMNS
SCM_CORE_COLUMNS = METABOLIC_COLUMNS + PA_SENSOR_COLUMNS + GLUCOSE_DYNAMICS_COLUMNS
SCM_OBSERVED_CONFOUNDER_COLUMNS: tuple[str, ...] = ()
SCM_FLEXIBLE_EDGES = tuple((c, TARGET) for c in GLUCOSE_DYNAMICS_COLUMNS)
LATENT_HIDDEN_EDGES: tuple[tuple[str, str], ...] = ()

_BASE_SCM_RULES = [
    MonotonicRelationship("insulin", TARGET, increasing=False).as_rule(),
    MonotonicRelationship("carbs", TARGET, increasing=True).as_rule(),
    *[MonotonicRelationship(c, TARGET, increasing=False).as_rule() for c in PA_SENSOR_COLUMNS],
]
SCM_RULES = _BASE_SCM_RULES

# --- preprocess constants ---
TS_FMT = "%d-%m-%Y %H:%M:%S"
INS_ALPHA, INS_BETA = 1.0 / 15.0, 1.0 / 120.0
CARB_TAU = 48.0
BASAL_TO_EFFECT = 2.5
MAX_PAST_MIN = 48.0 * 60.0
HORIZON_STEPS = 6
MIN_INDEX = 24
ACTIVITY_WINDOW_MIN = 45.0
EXERCISE_TAU = 60.0
_Z_CLIP = 5.0
_NS_PER_MIN = 60.0 * 1e9

BUILD_PATIENT_IDS: tuple[str, ...] | None = None
WRITE_PREPROCESS_PLOTS = False


@dataclass(frozen=True)
class _Event:
    t: pd.Timestamp
    value: float


@dataclass(frozen=True)
class _ExerciseEvent:
    t: pd.Timestamp
    intensity: float
    duration_min: float


@dataclass(frozen=True)
class _SignalStats:
    median: float
    sigma: float


@dataclass(frozen=True)
class _PatientData:
    pid: str
    glucose: list[_Event]
    bolus: list[_Event]
    meal: list[_Event]
    basal: list[_Event]
    steps: list[_Event]
    heart_rate: list[_Event]
    accel: list[_Event]
    exercise: list[_ExerciseEvent]
    steps_stats: _SignalStats
    hr_stats: _SignalStats
    accel_stats: _SignalStats


def _parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(str(s).strip(), TS_FMT))


def _parse_ts_safe(s: str | None) -> pd.Timestamp | None:
    if not s or not str(s).strip():
        return None
    try:
        return _parse_ts(s)
    except (ValueError, TypeError):
        return None


def _robust_stats(values: list[float]) -> _SignalStats:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _SignalStats(0.0, 1.0)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(arr) or 1.0)
    return _SignalStats(med, sigma if np.isfinite(sigma) and sigma > 1e-9 else 1.0)


def _iter_xml_files(ohio_root: Path) -> Iterator[Path]:
    for year in ("2018", "2020"):
        for split in ("train", "test"):
            sdir = ohio_root / year / split
            if sdir.is_dir():
                yield from sorted(sdir.glob("*-ws-*.xml"))


def _events_to_ns(events: list[_Event]) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return (
        np.array([e.t.value for e in events], dtype=np.int64),
        np.array([e.value for e in events], dtype=np.float64),
    )


def _exercise_to_ns(exercise: list[_ExerciseEvent]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not exercise:
        z = np.empty(0, dtype=np.int64)
        return z, z, z
    return (
        np.array([e.t.value for e in exercise], dtype=np.int64),
        np.array([e.intensity for e in exercise], dtype=np.float64),
        np.array([e.duration_min for e in exercise], dtype=np.float64),
    )


def _insulin_irf_vec(dt_min: np.ndarray) -> np.ndarray:
    dt = np.asarray(dt_min, dtype=np.float64)
    out = np.zeros_like(dt)
    pos = dt > 0
    if not np.any(pos):
        return out
    a, b = INS_ALPHA, INS_BETA
    out[pos] = np.maximum(0.0, (np.exp(-b * dt[pos]) - np.exp(-a * dt[pos])) / (a - b))
    return out


def _carb_irf_vec(dt_min: np.ndarray) -> np.ndarray:
    dt = np.asarray(dt_min, dtype=np.float64)
    out = np.zeros_like(dt)
    pos = dt > 0
    if not np.any(pos):
        return out
    x = dt[pos] / CARB_TAU
    peak = 4.0 * np.exp(-2.0)
    out[pos] = np.maximum(0.0, (x * x) * np.exp(-x) / peak) if peak > 1e-12 else 0.0
    return out


def carb_irf(dt_min: float) -> float:
    return float(_carb_irf_vec(np.array([dt_min], dtype=np.float64))[0])


def _irf_sum_batch(
    t_grid_ns: np.ndarray,
    evt_t_ns: np.ndarray,
    evt_v: np.ndarray,
    irf_vec_fn,
    *,
    max_past_min: float = MAX_PAST_MIN,
) -> np.ndarray:
    if evt_t_ns.size == 0:
        return np.zeros(t_grid_ns.shape[0], dtype=np.float64)
    dt = (t_grid_ns[:, None] - evt_t_ns[None, :]).astype(np.float64) / _NS_PER_MIN
    mask = (dt > 0) & (dt <= max_past_min)
    return np.sum(np.where(mask, irf_vec_fn(dt) * evt_v[None, :], 0.0), axis=1)


def _basal_batch(t_grid_ns: np.ndarray, basal_t_ns: np.ndarray, basal_v: np.ndarray) -> np.ndarray:
    if basal_t_ns.size == 0:
        return np.zeros(t_grid_ns.shape[0], dtype=np.float64)
    idx = np.searchsorted(basal_t_ns, t_grid_ns, side="right") - 1
    out = np.zeros(t_grid_ns.shape[0], dtype=np.float64)
    valid = idx >= 0
    out[valid] = basal_v[idx[valid]]
    return out


def _windowed_zmean_batch(
    t_grid_ns: np.ndarray,
    evt_t_ns: np.ndarray,
    evt_v: np.ndarray,
    stats: _SignalStats,
    window_min: float,
) -> np.ndarray:
    n = int(t_grid_ns.size)
    if evt_t_ns.size == 0 or stats.sigma <= 1e-9:
        return np.zeros(n, dtype=np.float64)
    z = np.clip((evt_v - stats.median) / stats.sigma, -_Z_CLIP, _Z_CLIP)
    z = np.where(np.isfinite(z), z, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(z)))
    cn = np.concatenate(([0.0], np.cumsum(np.ones_like(z))))
    lo_ns = t_grid_ns - int(window_min * _NS_PER_MIN)
    hi = np.searchsorted(evt_t_ns, t_grid_ns, side="right")
    lo = np.searchsorted(evt_t_ns, lo_ns, side="left")
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, 0.0)


def _windowed_mean_batch(
    t_grid_ns: np.ndarray,
    evt_t_ns: np.ndarray,
    evt_v: np.ndarray,
    window_min: float,
) -> np.ndarray:
    """Windowed mean of raw sensor values (no robust z-scoring)."""
    n = int(t_grid_ns.size)
    if evt_t_ns.size == 0:
        return np.zeros(n, dtype=np.float64)
    v = np.asarray(evt_v, dtype=np.float64)
    v = np.where(np.isfinite(v), v, np.nan)
    cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(v, nan=0.0))))
    cn = np.concatenate(([0.0], np.cumsum(np.isfinite(v).astype(np.float64))))
    lo_ns = t_grid_ns - int(window_min * _NS_PER_MIN)
    hi = np.searchsorted(evt_t_ns, t_grid_ns, side="right")
    lo = np.searchsorted(evt_t_ns, lo_ns, side="left")
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, 0.0)


def _exercise_load_batch(
    t_grid_ns: np.ndarray,
    ex_t_ns: np.ndarray,
    ex_inten: np.ndarray,
    ex_dur_min: np.ndarray,
    *,
    max_min: float = 360.0,
) -> np.ndarray:
    if ex_t_ns.size == 0:
        return np.zeros(t_grid_ns.shape[0], dtype=np.float64)
    dt = (t_grid_ns[:, None] - ex_t_ns[None, :]).astype(np.float64) / _NS_PER_MIN
    mask = (dt > 0) & (dt <= max_min)
    decay = np.zeros_like(dt)
    np.exp(-dt[mask] / EXERCISE_TAU, out=decay[mask])
    load = ex_inten[None, :] * (ex_dur_min[None, :] / 60.0) * decay
    return np.sum(load * mask, axis=1)


def _insulin_batch(t_grid_ns: np.ndarray, pat: _PatientData) -> np.ndarray:
    bolus_t, bolus_v = _events_to_ns(pat.bolus)
    basal_t, basal_v = _events_to_ns(pat.basal)
    ins = _irf_sum_batch(t_grid_ns, bolus_t, bolus_v, _insulin_irf_vec)
    return ins + BASAL_TO_EFFECT * _basal_batch(t_grid_ns, basal_t, basal_v)


def insulin_feature_at(t: pd.Timestamp, bolus: list[_Event], basal: list[_Event]) -> float:
    pat = _PatientData(
        pid="", glucose=[], bolus=bolus, meal=[], basal=basal,
        steps=[], heart_rate=[], accel=[], exercise=[],
        steps_stats=_SignalStats(0, 1), hr_stats=_SignalStats(0, 1), accel_stats=_SignalStats(0, 1),
    )
    return float(_insulin_batch(np.array([t.value], dtype=np.int64), pat)[0])


def physical_activity_at(t: pd.Timestamp, pat: _PatientData) -> float:
    t_ns = np.array([t.value], dtype=np.int64)
    steps_t, steps_v = _events_to_ns(pat.steps)
    hr_t, hr_v = _events_to_ns(pat.heart_rate)
    accel_t, accel_v = _events_to_ns(pat.accel)
    ex_t, ex_i, ex_d = _exercise_to_ns(pat.exercise)
    return float(
        _windowed_zmean_batch(t_ns, steps_t, steps_v, pat.steps_stats, ACTIVITY_WINDOW_MIN)[0]
        + _windowed_zmean_batch(t_ns, hr_t, hr_v, pat.hr_stats, ACTIVITY_WINDOW_MIN)[0]
        + _windowed_zmean_batch(t_ns, accel_t, accel_v, pat.accel_stats, ACTIVITY_WINDOW_MIN)[0]
        + _exercise_load_batch(t_ns, ex_t, ex_i, ex_d)[0]
    )


_carb_irf = carb_irf


def _read_value_events(root: ET.Element, tag: str) -> list[_Event]:
    out: list[_Event] = []
    for ev in root.findall(f".//{tag}/event"):
        ts = _parse_ts_safe(ev.attrib.get("ts"))
        if ts is None:
            continue
        try:
            v = float(ev.attrib.get("value", "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(_Event(ts, v))
    return out


def _read_exercise_events(root: ET.Element) -> list[_ExerciseEvent]:
    out: list[_ExerciseEvent] = []
    for ev in root.findall(".//exercise/event"):
        ts = _parse_ts_safe(ev.attrib.get("ts"))
        if ts is None:
            continue
        try:
            inten = float(ev.attrib.get("intensity") or "nan")
            dur = float(ev.attrib.get("duration") or "nan")
        except (TypeError, ValueError):
            continue
        if not np.isfinite(inten) or inten <= 0:
            continue
        if not np.isfinite(dur) or dur <= 0:
            dur = 30.0
        out.append(_ExerciseEvent(ts, inten, dur))
    return out


def _parse_patient_xml(path: Path) -> _PatientData:
    root = ET.parse(path).getroot()
    pid = str(root.attrib.get("id", path.stem.split("-")[0]))

    gluc, bolus, meal, basal = [], [], [], []
    for gl in root.findall(".//glucose_level/event"):
        gluc.append(_Event(_parse_ts(gl.attrib["ts"]), float(gl.attrib["value"])))
    for bl in root.findall(".//bolus/event"):
        dose = float(bl.attrib.get("dose", 0) or 0)
        if dose > 0:
            bolus.append(_Event(_parse_ts(bl.attrib["ts_begin"]), dose))
    for me in root.findall(".//meal/event"):
        meal.append(_Event(_parse_ts(me.attrib["ts"]), float(me.attrib.get("carbs", 0) or 0)))
    for ba in root.findall(".//basal/event"):
        basal.append(_Event(_parse_ts(ba.attrib["ts"]), float(ba.attrib.get("value", 0) or 0)))

    steps = _read_value_events(root, "basis_steps")
    hr = _read_value_events(root, "basis_heart_rate")
    accel = _read_value_events(root, "acceleration")
    exercise = _read_exercise_events(root)

    for stream in (gluc, bolus, meal, basal, steps, hr, accel):
        stream.sort(key=lambda e: e.t)
    exercise.sort(key=lambda e: e.t)

    return _PatientData(
        pid=pid, glucose=gluc, bolus=bolus, meal=meal, basal=basal,
        steps=steps, heart_rate=hr, accel=accel, exercise=exercise,
        steps_stats=_robust_stats([e.value for e in steps]),
        hr_stats=_robust_stats([e.value for e in hr]),
        accel_stats=_robust_stats([e.value for e in accel]),
    )


def _patient_dataframe(pat: _PatientData) -> pl.DataFrame:
    n = len(pat.glucose)
    if n < MIN_INDEX + HORIZON_STEPS + 1:
        return pl.DataFrame()

    g = np.array([e.value for e in pat.glucose], dtype=np.float64)
    t_all_ns = np.array([e.t.value for e in pat.glucose], dtype=np.int64)
    i0, i1 = MIN_INDEX, n - HORIZON_STEPS
    t_ns = t_all_ns[i0:i1]

    meal_t, meal_v = _events_to_ns([m for m in pat.meal if m.value > 0])
    steps_t, steps_v = _events_to_ns(pat.steps)
    hr_t, hr_v = _events_to_ns(pat.heart_rate)
    accel_t, accel_v = _events_to_ns(pat.accel)
    ex_t, ex_i, ex_d = _exercise_to_ns(pat.exercise)

    roc1_by_i = (g[1:] - g[:-1]) / 5.0
    roc6_by_i = (g[6:] - g[:-6]) / 30.0
    csum = np.cumsum(g)
    ctx_by_i = (csum[6:] - csum[:-6]) / 6.0 - g[6:]
    std1h_by_i = np.std(np.lib.stride_tricks.sliding_window_view(g, 12), axis=1)

    return pl.DataFrame({
        "patient_id": [pat.pid] * int(t_ns.size),
        "insulin": _insulin_batch(t_ns, pat),
        "carbs": _irf_sum_batch(t_ns, meal_t, meal_v, _carb_irf_vec),
        "pa_steps": _windowed_zmean_batch(t_ns, steps_t, steps_v, pat.steps_stats, ACTIVITY_WINDOW_MIN),
        "pa_hr": _windowed_zmean_batch(t_ns, hr_t, hr_v, pat.hr_stats, ACTIVITY_WINDOW_MIN),
        "pa_accel": _windowed_zmean_batch(t_ns, accel_t, accel_v, pat.accel_stats, ACTIVITY_WINDOW_MIN),
        "pa_exercise": _exercise_load_batch(t_ns, ex_t, ex_i, ex_d),
        "glucose_roc_5m": roc1_by_i[i0 - 1 : i1 - 1],
        "glucose_roc_30m": roc6_by_i[i0 - 6 : i1 - 6],
        "glucose_accel": roc1_by_i[i0 - 1 : i1 - 1] - roc1_by_i[i0 - 2 : i1 - 2],
        "glucose_context_m30": ctx_by_i[i0 - 6 : i1 - 6],
        "glucose_std_1h": std1h_by_i[i0 - 12 : i1 - 12],
        "future glucose": g[i0 + HORIZON_STEPS : i1 + HORIZON_STEPS] - g[i0:i1],
    })


def _busiest_day_events(glucose: list[_Event], *, min_events: int = 5) -> list[_Event]:
    if not glucose:
        return []
    counts = Counter(e.t.normalize().date() for e in glucose)
    day = counts.most_common(1)[0][0]
    day_events = [e for e in glucose if e.t.normalize().date() == day]
    if len(day_events) >= min_events:
        return day_events
    day0 = glucose[0].t.normalize().date()
    return [e for e in glucose if e.t.normalize().date() == day0]


def _events_on_day(events: list[_Event], day) -> list[_Event]:
    return [e for e in events if e.t.normalize().date() == day]


def _pa_accel_on_grid(pat: _PatientData, t_ns: np.ndarray) -> np.ndarray:
    """Absolute windowed mean acceleration for overview plots (raw sensor units)."""
    accel_t, accel_v = _events_to_ns(pat.accel)
    return _windowed_mean_batch(t_ns, accel_t, np.abs(accel_v), ACTIVITY_WINDOW_MIN)


def _pa_active_fraction(pa: np.ndarray, *, min_delta: float = 0.015) -> int:
    pa = np.asarray(pa, dtype=np.float64)
    if not pa.size:
        return 0
    return int(np.sum(np.abs(pa - np.nanmedian(pa)) > min_delta))


def _pick_overview_day(pat: _PatientData, *, min_pa_std: float = 0.02) -> object | None:
    glucose_days = Counter(e.t.normalize().date() for e in pat.glucose)
    meal_days = {e.t.normalize().date() for e in pat.meal if e.value > 0}
    bolus_days = {e.t.normalize().date() for e in pat.bolus if e.value > 0}
    candidates: list[tuple[float, int, object]] = []
    for day, cnt in glucose_days.items():
        if day not in meal_days or day not in bolus_days or cnt < 40:
            continue
        devents = _events_on_day(pat.glucose, day)
        t_ns = np.array([e.t.value for e in devents], dtype=np.int64)
        pa = _pa_accel_on_grid(pat, t_ns)
        std = float(np.nanstd(pa))
        if std < min_pa_std or _pa_active_fraction(pa) < 20:
            continue
        candidates.append((std, int(cnt), day))
    if candidates:
        return max(candidates)[2]
    joint = [(d, c) for d, c in glucose_days.items() if d in meal_days and d in bolus_days]
    if joint:
        return max(joint, key=lambda x: x[1])[0]
    if glucose_days:
        return glucose_days.most_common(1)[0][0]
    return None


def _trim_to_pa_active_window(
    day_events: list[_Event],
    pa: np.ndarray,
    *,
    min_points: int = 72,
    max_points: int = 180,
    min_pa_std: float = 0.02,
) -> tuple[list[_Event], np.ndarray]:
    """Keep a contiguous slice where pa_accel varies (not flat throughout)."""
    n = int(len(day_events))
    if n <= min_points:
        return day_events, pa
    full_std = float(np.nanstd(pa))
    if full_std >= min_pa_std and _pa_active_fraction(pa) >= max(20, n // 4):
        return day_events, pa
    best_i, best_w, best_score = 0, min_points, -1.0
    for w in range(min_points, min(max_points, n) + 1):
        for i in range(0, n - w + 1):
            chunk = pa[i : i + w]
            std = float(np.nanstd(chunk))
            nz = _pa_active_fraction(chunk)
            score = std + 0.01 * nz
            if score > best_score:
                best_score, best_i, best_w = score, i, w
    if best_score < 0:
        return day_events, pa
    j = best_i + best_w
    return day_events[best_i:j], pa[best_i:j]


def _save_dataset_overview_plot(
    path: Path,
    *,
    title: str,
    times: list[pd.Timestamp],
    glucose: list[float],
    insulin: list[float],
    carbs: list[float],
    pa_accel: list[float],
    meal_events: list[_Event],
    bolus_events: list[_Event],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    panels = (
        (axes[0], glucose, "#1f77b4", "Glucose (mg/dL)", "CGM"),
        (axes[1], insulin, "#c0392b", "Insulin feature", "Insulin IRF + basal"),
        (axes[2], carbs, "#e67e22", "Carbs feature", "Carbs IRF"),
        (axes[3], pa_accel, "#27ae60", "PA accel (absolute)", "pa_accel"),
    )
    meal_times = [e.t for e in meal_events if e.value > 0]
    bolus_times = [e.t for e in bolus_events if e.value > 0]
    for i, (ax, y, color, ylabel, _label) in enumerate(panels):
        ax.plot(times, y, color=color, lw=1.6)
        for mt in meal_times:
            ax.axvline(mt, color="#2ca02c", ls=":", lw=1.1, alpha=0.65, zorder=1)
        for bt in bolus_times:
            ax.axvline(bt, color="#c0392b", ls="--", lw=1.1, alpha=0.65, zorder=1)
        ax.set_ylabel(ylabel, fontsize=12, color=color)
        ax.tick_params(axis="y", labelcolor=color, labelsize=10)
        ax.grid(True, color="#eee", linewidth=0.5)
    axes[0].legend(
        handles=[Line2D([0], [0], color="#1f77b4", lw=1.6, label="CGM")],
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.suptitle(title, fontsize=10, y=1.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _first_xml_per_patient(ohio_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in _iter_xml_files(ohio_root):
        try:
            pid = str(ET.parse(p).getroot().attrib.get("id", p.stem.split("-")[0]))
        except ET.ParseError:
            continue
        out.setdefault(pid, p)
    return out


def find_overview_patient_xml(ohio_root: Path, preferred_ids: tuple[str, ...] = ("552", "540", "559", "570", "591")) -> tuple[str, Path] | None:
    """Patient + XML with overview day that has meals, boluses, and varying pa_accel."""
    by_pid = _first_xml_per_patient(ohio_root)
    best: tuple[float, str, Path] | None = None
    for pid in preferred_ids:
        xmlp = by_pid.get(pid)
        if xmlp is None:
            continue
        try:
            pat = _parse_patient_xml(xmlp)
        except (ET.ParseError, KeyError, ValueError, OSError):
            continue
        day = _pick_overview_day(pat)
        if day is None:
            continue
        devents = _events_on_day(pat.glucose, day)
        t_ns = np.array([e.t.value for e in devents], dtype=np.int64)
        pa = _pa_accel_on_grid(pat, t_ns)
        std = float(np.nanstd(pa))
        if std < 0.02:
            continue
        score = std + 0.001 * len(devents)
        if best is None or score > best[0]:
            best = (score, pid, xmlp)
    if best:
        return best[1], best[2]
    for pid, xmlp in sorted(by_pid.items(), key=lambda x: x[0]):
        if pid in preferred_ids:
            continue
        try:
            pat = _parse_patient_xml(xmlp)
            if _pick_overview_day(pat) is not None and float(np.nanstd(_pa_accel_on_grid(
                pat, np.array([e.t.value for e in _events_on_day(pat.glucose, _pick_overview_day(pat))], dtype=np.int64),
            ))) >= 0.02:
                return pid, xmlp
        except (ET.ParseError, KeyError, ValueError, OSError):
            continue
    return None


def _save_example_plot(
    path: Path, *, title: str, times: list[pd.Timestamp],
    left: tuple[list[float], str, str], right: tuple[list[float], str, str],
) -> None:
    fig, ax_l = plt.subplots(figsize=(11, 4.5))
    ax_r = ax_l.twinx()
    ax_l.plot(times, left[0], color="#1f77b4", lw=1.8, label=left[1])
    ax_r.plot(times, right[0], color="#d62728", lw=1.5, alpha=0.9, label=right[1])
    ax_l.set_ylabel(left[2], color="#1f77b4")
    ax_r.set_ylabel(right[2], color="#d62728")
    ax_l.set_xlabel("Time")
    ax_l.set_title(title)
    ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_example_insulin_glucose_plot(
    root: Path, ohio_root: Path, *, xml_path: Path, patient_subdir: str,
) -> Path | None:
    try:
        pat = _parse_patient_xml(xml_path)
        day = _pick_overview_day(pat)
        if day is None:
            return None
        day_events = _events_on_day(pat.glucose, day)
        if len(day_events) < 5:
            day_events = _busiest_day_events(pat.glucose)
            if len(day_events) < 5:
                return None
            day = day_events[0].t.normalize().date()
        times = [e.t for e in day_events]
        t_ns = np.array([e.t.value for e in day_events], dtype=np.int64)
        meal_t, meal_v = _events_to_ns([m for m in pat.meal if m.value > 0])
        pa_accel = _pa_accel_on_grid(pat, t_ns)
        day_events, pa_accel = _trim_to_pa_active_window(day_events, pa_accel)
        times = [e.t for e in day_events]
        t_ns = np.array([e.t.value for e in day_events], dtype=np.int64)
        t0, t1 = times[0].strftime("%H:%M"), times[-1].strftime("%H:%M")
        out = root / "manuscript" / "images" / "ohio_t1dm" / str(patient_subdir) / "preprocess_example_insulin_glucose.png"
        _save_dataset_overview_plot(
            out,
            title=f"Ohio T1DM — patient {pat.pid} — {day} ({t0}–{t1})",
            times=times,
            glucose=[e.value for e in day_events],
            insulin=_insulin_batch(t_ns, pat).tolist(),
            carbs=_irf_sum_batch(t_ns, meal_t, meal_v, _carb_irf_vec).tolist(),
            pa_accel=pa_accel.tolist(),
            meal_events=[m for m in _events_on_day(pat.meal, day) if times[0] <= m.t <= times[-1]],
            bolus_events=[b for b in _events_on_day(pat.bolus, day) if times[0] <= b.t <= times[-1]],
        )
        return out
    except Exception:
        plt.close("all")
        return None


def save_example_physical_activity_plot(root: Path, pat: _PatientData) -> Path | None:
    day_events = _busiest_day_events(pat.glucose)
    if len(day_events) < 5:
        return None
    times = [e.t for e in day_events]
    t_ns = np.array([e.t.value for e in day_events], dtype=np.int64)
    steps_t, steps_v = _events_to_ns(pat.steps)
    hr_t, hr_v = _events_to_ns(pat.heart_rate)
    accel_t, accel_v = _events_to_ns(pat.accel)
    ex_t, ex_i, ex_d = _exercise_to_ns(pat.exercise)
    pa = (
        _windowed_zmean_batch(t_ns, steps_t, steps_v, pat.steps_stats, ACTIVITY_WINDOW_MIN)
        + _windowed_zmean_batch(t_ns, hr_t, hr_v, pat.hr_stats, ACTIVITY_WINDOW_MIN)
        + _windowed_zmean_batch(t_ns, accel_t, accel_v, pat.accel_stats, ACTIVITY_WINDOW_MIN)
        + _exercise_load_batch(t_ns, ex_t, ex_i, ex_d)
    )
    day = times[0].normalize().date()
    out = root / "manuscript" / "images" / "ohio_t1dm" / str(pat.pid) / "preprocess_example_physical_activity.png"
    _save_example_plot(
        out,
        title=f"Ohio T1DM — patient {pat.pid} — {day}\nActivity sensors (sum of PA columns)",
        times=times,
        left=([e.value for e in day_events], "CGM (mg/dL)", "Glucose (mg/dL)"),
        right=(pa.tolist(), "PA (sum)", "Activity"),
    )
    return out


def save_all_example_plots(root: Path, ohio_root: Path) -> tuple[list[Path], list[Path]]:
    insulin_out, pa_out = [], []
    for pid, xmlp in sorted(
        _first_xml_per_patient(ohio_root).items(),
        key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x[0]),
    ):
        try:
            pat = _parse_patient_xml(xmlp)
        except (ET.ParseError, KeyError, ValueError, OSError):
            continue
        p = save_example_insulin_glucose_plot(root, ohio_root, xml_path=xmlp, patient_subdir=pid)
        if p:
            insulin_out.append(p)
        p = save_example_physical_activity_plot(root, pat)
        if p:
            pa_out.append(p)
    return insulin_out, pa_out


def _xml_data_split(path: Path) -> str:
    return "test" if "test" in {p.lower() for p in path.parts} else "train"


def build_ohio_t1dm_dataframe(ohio_root: Path, *, patient_ids: set[str] | None = None) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for xml_path in _iter_xml_files(ohio_root):
        try:
            pat = _parse_patient_xml(xml_path)
            if patient_ids is not None and pat.pid not in patient_ids:
                continue
            df = _patient_dataframe(pat)
            if not df.is_empty():
                frames.append(df.with_columns(pl.lit(_xml_data_split(xml_path)).alias("data_split")))
        except (ET.ParseError, KeyError, ValueError, OSError):
            continue
    if not frames:
        raise RuntimeError(f"No rows built from XML under {ohio_root}")
    return pl.concat(frames, how="vertical_relaxed")


def _ohio_t1dm_xml_dir(root: Path) -> Path:
    for sub in (root / "datasets" / "OhioT1DM", root / "OhioT1DM"):
        if sub.is_dir():
            return sub
    raise FileNotFoundError(f"OhioT1DM not found under {root}")


def ensure_ohio_csv(
    root: str | Path,
    *,
    force: bool = False,
    write_plots: bool | None = None,
) -> Path:
    """Build ``datasets/ohio_t1dm.csv`` from XML when the file is missing (or *force*)."""
    root = Path(root)
    out = root / "datasets" / CSV
    if out.is_file() and not force:
        return out

    ohio = _ohio_t1dm_xml_dir(root)
    pids = set(BUILD_PATIENT_IDS) if BUILD_PATIENT_IDS is not None else None
    t0 = time.perf_counter()
    df = build_ohio_t1dm_dataframe(ohio, patient_ids=pids)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    print(
        f"Built {out} rows={df.height} patients={df.select(pl.col('patient_id').n_unique()).item()} "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    plots = WRITE_PREPROCESS_PLOTS if write_plots is None else write_plots
    if plots:
        ins_plots, pa_plots = save_all_example_plots(root, ohio)
        if ins_plots:
            print(f"Insulin/CGM plots: {len(ins_plots)} under manuscript/images/ohio_t1dm/<id>/")
        if pa_plots:
            print(f"Activity plots: {len(pa_plots)} under manuscript/images/ohio_t1dm/<id>/")
    return out


# --- load API ---

def _column_has_signal(series: pd.Series, *, atol: float = 1e-12) -> bool:
    v = np.asarray(series, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    if float(np.max(np.abs(v))) <= atol:
        return False
    return float(np.std(v)) > atol


def feature_columns_for(sub: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c for c in FEATURE_COLUMNS
        if c in sub.columns and _column_has_signal(sub[c])
    )


def inactive_feature_columns(sub: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in FEATURE_COLUMNS if c not in feature_columns_for(sub))


def scm_rules_for(sub: pd.DataFrame) -> list[dict]:
    active = set(feature_columns_for(sub))
    return [r for r in _BASE_SCM_RULES if str(r["start"]) in active]


def scm_core_columns_for(sub: pd.DataFrame) -> tuple[str, ...]:
    feats = feature_columns_for(sub)
    return tuple(c for c in feats if c in METABOLIC_COLUMNS or c in PA_SENSOR_COLUMNS or c in GLUCOSE_DYNAMICS_COLUMNS)


def _csv_path(root: str) -> str:
    return os.path.join(root, "datasets", CSV)


def _patient_sub(root: str, patient_id: str) -> pd.DataFrame:
    ensure_ohio_csv(root)
    df = pd.read_csv(_csv_path(root))
    sub = df.loc[df[PATIENT_COL].astype(str) == str(patient_id)]
    if sub.empty:
        raise ValueError(f"No rows for patient_id={patient_id!r}")
    return sub.copy()


def _features(sub: pd.DataFrame, *, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    drop = [c for c in (TARGET, PATIENT_COL, SPLIT_COL) if c in sub.columns]
    X = sub.drop(columns=drop, errors="ignore").reindex(columns=list(feature_cols)).reset_index(drop=True)
    for c in HIDDEN_CONFOUNDER_CANDIDATES:
        X[c] = float("nan")
    return X


def patient_ids(root: str) -> list[str]:
    ensure_ohio_csv(root)
    ids = sorted(pd.read_csv(_csv_path(root), usecols=[PATIENT_COL])[PATIENT_COL].astype(str).unique())
    if not ids:
        raise ValueError("No patients in CSV")
    return ids


def load(root: str, patient_id: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = _patient_sub(root, patient_id)
    feat_cols = feature_columns_for(sub)
    return _features(sub, feature_cols=feat_cols), sub[TARGET].reset_index(drop=True)


def load_train_test(
    root: str, patient_id: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
    sub = _patient_sub(root, patient_id)
    feat_cols = feature_columns_for(sub)
    if SPLIT_COL not in sub.columns:
        return (*load(root, patient_id), None, None)
    tr = sub.loc[sub[SPLIT_COL].astype(str) == "train"]
    te = sub.loc[sub[SPLIT_COL].astype(str) == "test"]
    X_train, y_train = _features(tr, feature_cols=feat_cols), tr[TARGET].reset_index(drop=True)
    if te.empty:
        return X_train, y_train, None, None
    return X_train, y_train, _features(te, feature_cols=feat_cols), te[TARGET].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parent.parent
    force = "--force" in sys.argv
    ensure_ohio_csv(repo, force=force, write_plots=True)
