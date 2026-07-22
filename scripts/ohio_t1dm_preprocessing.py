"""Ohio T1DM XML loaders: Prendin protocol matrices and ReplayBG traces."""
from __future__ import annotations
import glob, os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np, pandas as pd, polars as pl
from sklearn.preprocessing import MinMaxScaler

CGM_STEP_MIN = 5


def horizon_steps(horizon_min: int) -> int:
    return horizon_min // CGM_STEP_MIN


HORIZON_STEPS = horizon_steps(30)
TARGET = "change in blood glucose"
PATIENT_COL = "patient_id"
SPLIT_COL = "data_split"
TIMESTAMP_COL = "timestamp"
FEATURE_COLUMNS = ("CGM", "insulin", "CHO")
# ``insulin`` = active insulin on board (U); ``CHO`` = cumulative carb absorption (g).
LOOKBACK, HORIZON = 1, 6
_GAP_STEPS = 6
_STEP_NS = int(5 * 60 * 1e9)
_BOLUS_TAU_RISE_MIN, _BOLUS_TAU_FALL_MIN = 15.0, 120.0
_BOLUS_LOOKBACK_MIN = 48 * 60
_BASAL_ACTIVE_HOURS = 2.5
_NS = int(60e9)
_d_peak = np.log(_BOLUS_TAU_FALL_MIN / _BOLUS_TAU_RISE_MIN) / (1 / _BOLUS_TAU_RISE_MIN - 1 / _BOLUS_TAU_FALL_MIN)
_BOLUS_ACTION_PEAK = (np.exp(-_d_peak / _BOLUS_TAU_FALL_MIN) - np.exp(-_d_peak / _BOLUS_TAU_RISE_MIN)) / (1 / _BOLUS_TAU_RISE_MIN - 1 / _BOLUS_TAU_FALL_MIN)
_SPLIT_XML = {"train": "training", "test": "testing"}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s.strip(), "%d-%m-%Y %H:%M:%S"))


def _ns(events: list) -> tuple[np.ndarray, np.ndarray]:
    return np.array([t.value for t, _ in events], np.int64), np.array([v for _, v in events], np.float64)


def bolus_action_fraction(lag_min: np.ndarray) -> np.ndarray:
    lag = np.asarray(lag_min, float)
    out = np.zeros_like(lag)
    m = lag > 0
    d, t1, t2 = lag[m], _BOLUS_TAU_RISE_MIN, _BOLUS_TAU_FALL_MIN
    out[m] = np.maximum(0.0, (np.exp(-d / t2) - np.exp(-d / t1)) / (1 / t1 - 1 / t2) / _BOLUS_ACTION_PEAK)
    return out


def _bolus_iob(grid: np.ndarray, bolus_times: np.ndarray, bolus_doses: np.ndarray) -> np.ndarray:
    dt = (grid[:, None] - bolus_times[None, :]) / _NS
    m = (dt > 0) & (dt <= _BOLUS_LOOKBACK_MIN)
    return (bolus_action_fraction(np.where(m, dt, 0.0)) * bolus_doses * m).sum(1)


def _basal_rate_at(grid: np.ndarray, basal_times: np.ndarray, basal_rates: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(basal_times, grid, "right") - 1
    out = np.zeros(grid.size, float)
    out[idx >= 0] = basal_rates[idx[idx >= 0]]
    pos = basal_rates[basal_rates > 0]
    return np.where(out > 0, out, float(np.median(pos))) if pos.size else out


def _carb_irf(lag_min: np.ndarray) -> np.ndarray:
    lag = np.asarray(lag_min, float)
    out = np.zeros_like(lag)
    m = lag > 0
    d = lag[m]
    out[m] = np.maximum(0.0, (d / 48) ** 2 * np.exp(-d / 48) / (4 * np.exp(-2)))
    return out


def _irf_sum(grid: np.ndarray, et: np.ndarray, ev: np.ndarray, irf) -> np.ndarray:
    dt = (grid[:, None] - et[None, :]) / _NS
    m = (dt > 0) & (dt <= 48 * 60)
    return (irf(np.where(m, dt, 0.0)) * ev * m).sum(1)


def _ohio_root(root: str) -> str:
    return f"{root}/OhioT1DM"


def _xml_paths(ohio: str) -> list[str]:
    paths = []
    for year in ("2018", "2020"):
        for split in ("train", "test"):
            paths.extend(sorted(glob.glob(os.path.join(ohio, year, split, "*-ws-*.xml"))))
    return paths


def _patient_xml(root: str, patient_id: str, split: str) -> str:
    return sorted(glob.glob(os.path.join(_ohio_root(root), "*", split, f"{patient_id}-ws-{_SPLIT_XML[split]}.xml")))[0]


def _parse(path: str) -> dict:
    r = ET.parse(path).getroot()
    stem = os.path.splitext(os.path.basename(path))[0]
    pid = str(r.attrib.get("id", stem.split("-")[0]))
    ev = lambda tag: [(_ts(x.attrib["ts"]), float(x.attrib["value"])) for x in r.findall(f".//{tag}/event")]
    gluc = ev("glucose_level")
    bolus = [(_ts(x.attrib["ts_begin"]), float(x.attrib["dose"])) for x in r.findall(".//bolus/event") if float(x.attrib.get("dose") or 0) > 0]
    meal = [(_ts(x.attrib["ts"]), float(x.attrib.get("carbs") or 0)) for x in r.findall(".//meal/event")]
    basal = [(_ts(x.attrib["ts"]), float(x.attrib.get("value") or 0)) for x in r.findall(".//basal/event")]
    for s in (gluc, bolus, meal, basal):
        s.sort(key=lambda e: e[0])
    return dict(pid=pid, weight=float(r.attrib.get("weight") or 75.0),
                gluc=gluc, bolus=bolus, meal=meal, basal=basal)


def _impulses(grid: np.ndarray, et: np.ndarray, ev: np.ndarray) -> np.ndarray:
    out = np.zeros(grid.size, float)
    if not et.size:
        return out
    step = int(grid[1] - grid[0]) if grid.size > 1 else _STEP_NS
    t0 = grid[0]
    for t_ns, v in zip(et, ev):
        j = int(round((t_ns - t0) / step))
        if 0 <= j < out.size:
            out[j] = float(v)
    return out


def _uniform_grid(gluc: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    times = [t for t, _ in gluc]
    vals = np.array([v for _, v in gluc], float)
    t0 = times[0].value
    grid = np.arange(t0, times[-1].value + _STEP_NS, _STEP_NS, dtype=np.int64)
    cgm = np.full(grid.size, np.nan)
    for t_ns, v in zip([t.value for t in times], vals):
        j = int(round((t_ns - t0) / _STEP_NS))
        if 0 <= j < cgm.size:
            cgm[j] = v
    return grid, cgm


def _interp_short_gaps(cgm: np.ndarray, max_gap: int) -> np.ndarray:
    x, isnan = cgm.copy(), np.isnan(cgm)
    if not isnan.any():
        return x
    i = 0
    while i < x.size:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < x.size and isnan[j]:
            j += 1
        if j - i <= max_gap and i > 0 and j < x.size:
            x[i:j] = np.linspace(x[i - 1], x[j], j - i + 2)[1:-1]
        i = j
    return x


def cumulative_insulin_cho_at_grid(grid: np.ndarray, bolus_times: np.ndarray, bolus_doses: np.ndarray, basal_times: np.ndarray, basal_rates: np.ndarray, meal_times: np.ndarray, meal_carbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack overlapping bolus IOB and meal carb absorption at each grid time."""
    basal_rate = _basal_rate_at(grid, basal_times, basal_rates)
    insulin = _bolus_iob(grid, bolus_times, bolus_doses) + _BASAL_ACTIVE_HOURS * basal_rate
    cho = _irf_sum(grid, meal_times, meal_carbs, _carb_irf)
    return insulin, cho


def _frame_from_xml(path: str, *, interpolate_cgm: bool) -> pd.DataFrame:
    pat = _parse(path)
    grid, cgm = _uniform_grid(pat["gluc"])
    if interpolate_cgm:
        cgm = _interp_short_gaps(cgm, _GAP_STEPS)
    mt, mv = _ns([(t, v) for t, v in pat["meal"] if v > 0])
    bt, bv = _ns(pat["bolus"])
    bat, bav = _ns(pat["basal"])
    insulin, cho = cumulative_insulin_cho_at_grid(grid, bt, bv, bat, bav, mt, mv)
    return pd.DataFrame({"CGM": cgm, "insulin": insulin, "CHO": cho})


def _roll_features(x: np.ndarray, lb: int, ph: int) -> np.ndarray:
    n = len(x) - lb - ph + 1
    out = np.zeros([n, lb])
    for i in range(lb):
        out[:, i] = np.roll(x, -i)[:- lb - ph + 1]
    return out


def _target_delta(cgm: np.ndarray, lb: int, ph: int) -> np.ndarray:
    """Future minus current CGM at the prediction horizon (mg/dL)."""
    future = cgm[lb + ph - 1 :].astype(float)
    current = cgm[: len(future)].astype(float)
    return future - current


def _scaled_matrices(data: pd.DataFrame, labels: tuple[str, ...], lb: int, ph: int, *, feature_scalers: list[MinMaxScaler] | None = None, target_scaler: MinMaxScaler | None = None):
    y_raw = _target_delta(data["CGM"].values, lb, ph)
    if target_scaler is None:
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(y_raw.reshape(-1, 1))
    y = target_scaler.transform(y_raw.reshape(-1, 1)).ravel()
    xs, scalers = [], [] if feature_scalers is None else feature_scalers
    for i, col in enumerate(labels):
        raw = data[col].values.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1)) if feature_scalers is None else feature_scalers[i]
        if feature_scalers is None:
            sc.fit(raw)
            scalers.append(sc)
        xs.append(_roll_features(sc.transform(raw).flatten(), lb, ph))
    X = np.concatenate(xs, axis=1)
    nan = np.union1d(np.argwhere(np.isnan(X))[:, 0], np.argwhere(np.isnan(y))[:, 0])
    return np.delete(X, nan, 0), np.delete(y, nan, 0), scalers, target_scaler

@dataclass
class OhioPatientData:
    X_train: np.ndarray; y_train: np.ndarray; X_test: np.ndarray; y_test: np.ndarray
    feature_names: tuple[str, ...]; cgm_scaler: MinMaxScaler; train_scalers: list[MinMaxScaler]; patient_id: str
    insulin_idx: int = 1


def list_patients(root: str) -> list[str]:
    ohio = _ohio_root(root)
    ids: set[str] = set()
    for year in ("2018", "2020"):
        for path in glob.glob(os.path.join(ohio, year, "train", "*-ws-training.xml")):
            ids.add(os.path.basename(path).split("-")[0])
    return sorted(ids, key=int)


def load_patient(root: str, patient_id: str, *, horizon_steps: int = HORIZON_STEPS) -> OhioPatientData:
    labels = FEATURE_COLUMNS
    train_df = _frame_from_xml(_patient_xml(root, patient_id, "train"), interpolate_cgm=True)
    test_df = _frame_from_xml(_patient_xml(root, patient_id, "test"), interpolate_cgm=False)
    X_train, y_train, scalers, target_scaler = _scaled_matrices(train_df, labels, LOOKBACK, horizon_steps)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(test_df, labels, LOOKBACK, horizon_steps, feature_scalers=scalers, target_scaler=target_scaler)
    return OhioPatientData(X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers, str(patient_id), insulin_idx=1)


def rmse_mgdl(y_scaled: np.ndarray, yhat_scaled: np.ndarray, cgm_scaler: MinMaxScaler) -> float:
    y = cgm_scaler.inverse_transform(np.asarray(y_scaled, float).reshape(-1, 1)).flatten()
    yh = cgm_scaler.inverse_transform(np.asarray(yhat_scaled, float).reshape(-1, 1)).flatten()
    ok = ~np.isnan(y)
    return float(np.sqrt(np.mean((y[ok] - yh[ok]) ** 2)))


class PaperGlucoseModel:
    def __init__(self, model, train_scalers: list[MinMaxScaler], cgm_scaler: MinMaxScaler):
        self._model = model
        self._scalers = train_scalers
        self._cgm_scaler = cgm_scaler

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X, np.column_stack([s.transform(X[:, i:i + 1]).ravel() for i, s in enumerate(self._scalers)])

    def predict_delta(self, X: np.ndarray, **_) -> np.ndarray:
        """Predicted change in CGM (mg/dL) over the forecast horizon."""
        X, scaled = self._scale(X)
        y_scaled = self._model.predict(scaled)
        return self._cgm_scaler.inverse_transform(np.asarray(y_scaled, float).reshape(-1, 1)).ravel()

    def predict(self, X: np.ndarray, **_) -> np.ndarray:
        """Future CGM (mg/dL) = current CGM + predicted ΔCGM."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X[:, 0] + self.predict_delta(X)

    def future_dose_curve(self, row, doses_u, *, span: float):
        """Future CGM under added correction doses (U). Uses intervention feature when present."""
        row = np.asarray(row, float).ravel()
        doses_u = np.asarray(doses_u, float).ravel()
        _, scaled = self._scale(row.reshape(1, -1))
        m = self._model
        if hasattr(m, "predict_intervention"):
            y = m.predict_intervention(scaled, doses_u / float(span), span).ravel()
            delta = self._cgm_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
            return float(row[0]) + delta
        X = np.tile(row, (len(doses_u), 1))
        X[:, 1] = row[1] + doses_u
        return self.predict(X)


def _trace_frame(pat: dict) -> pl.DataFrame:
    g = np.array([v for _, v in pat["gluc"]], float)
    i0, i1 = 24, g.size - HORIZON_STEPS
    grid = np.array([t.value for t, _ in pat["gluc"][i0:i1]], np.int64)
    times = [t for t, _ in pat["gluc"][i0:i1]]
    mt, mv = _ns([(t, v) for t, v in pat["meal"] if v > 0])
    bt, bv = _ns(pat["bolus"])
    bat, bav = _ns(pat["basal"])
    basal_rate = _basal_rate_at(grid, bat, bav)
    insulin, carbs = cumulative_insulin_cho_at_grid(grid, bt, bv, bat, bav, mt, mv)
    return pl.DataFrame({PATIENT_COL: [pat["pid"]] * grid.size, TIMESTAMP_COL: times, "glucose": g[i0:i1], "insulin": insulin, "carbs": carbs, "meal_carbs": _impulses(grid, mt, mv), "bolus_dose": _impulses(grid, bt, bv), "basal_rate": basal_rate, TARGET: g[i0 + HORIZON_STEPS:i1 + HORIZON_STEPS] - g[i0:i1]})


def replay_u2ss_mU_kg_min(mean_basal_u_hr: float, weight_kg: float) -> float:
    """Convert pump basal U/hour to ReplayBG steady-state mU/(kg·min)."""
    return float(mean_basal_u_hr) / 60.0 * 1000.0 / float(weight_kg)


# Per-patient τ knobs (optional). Merged into load_patient_meta()["target"].
# Keys: tau, SI (ReplayBG). Amplitude follows R_h(1)·G_ref from ReplayBG (no search).
# Example: PATIENT_TARGET = {"588": {"tau": 2.0, "SI": 6e-4}}
PATIENT_TARGET: dict[str, dict] = {}


def load_patient_meta(root: str, patient_id: str) -> dict:
    ohio, pid = _ohio_root(root), str(patient_id)
    weights, basal_rates = [], []
    for p in _xml_paths(ohio):
        pat = _parse(p)
        if str(pat["pid"]) != pid:
            continue
        weights.append(pat["weight"])
        basal_rates.extend(v for _, v in pat["basal"] if v > 0)
    weight = float(weights[0] if weights else 75.0)
    mean_basal_u_hr = float(np.mean(basal_rates)) if basal_rates else 0.0
    return {
        "weight_kg": weight,
        "u2ss": replay_u2ss_mU_kg_min(mean_basal_u_hr, weight),
        "target": dict(PATIENT_TARGET.get(pid, {})),
    }


def load_glucose_trace(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    ohio, pid = _ohio_root(root), str(patient_id)
    frames = []
    for p in _xml_paths(ohio):
        pat = _parse(p)
        if str(pat["pid"]) != pid:
            continue
        sp = "test" if "/test/" in p.lower() else "train"
        if split is not None and sp != split:
            continue
        frames.append(_trace_frame(pat).with_columns(pl.lit(sp).alias(SPLIT_COL)))
    return pl.concat(frames) if frames else pl.DataFrame()


def paper_feat_arrays(seg: pl.DataFrame) -> dict[str, np.ndarray]:
    cho = "carbs" if "carbs" in seg.columns else "CHO"
    return {"CGM": seg["glucose"].to_numpy().astype(float), "insulin": seg["insulin"].to_numpy().astype(float), "CHO": seg[cho].to_numpy().astype(float)}


def seg_to_replaybg_df(seg: pl.DataFrame, *, window_h: int = 8, yts_min: int = 5) -> pd.DataFrame:
    """Polars window segment → pandas frame for ReplayBG.twin/replay."""
    n = min(seg.height, int(window_h * 60 / yts_min))
    seg = seg.head(n)
    t0 = seg[TIMESTAMP_COL][0]
    if hasattr(t0, "to_pydatetime"):
        t0 = t0.to_pydatetime()
    meal_col = "meal_carbs" if "meal_carbs" in seg.columns else "cho"
    meal = seg[meal_col].to_numpy().astype(float)
    bolus = seg["bolus_dose"].to_numpy().astype(float)
    basal = seg["basal_rate"].to_numpy().astype(float)
    cho = np.where(meal > 0, meal / yts_min, 0.0)
    bolus_rate = np.where(bolus > 0, bolus / yts_min, 0.0)
    out = pd.DataFrame({
        "t": pd.to_datetime([t0 + timedelta(minutes=yts_min * i) for i in range(n)]),
        "glucose": seg["glucose"].to_numpy().astype(float),
        "cho": cho, "cho_label": [""] * n,
        "bolus": bolus_rate, "bolus_label": [""] * n,
        "basal": basal / 60.0,
    })
    if (out["cho"] > 0).any():
        out.loc[out.index[out["cho"] > 0][0], "cho_label"] = "M"
    if (out["bolus"] > 0).any():
        out.loc[out.index[out["bolus"] > 0][0], "bolus_label"] = "C"
    return out
