"""T1D cohort loaders: OhioT1DM + AZT1D → shared CGM/IOB/CHO matrices."""
from __future__ import annotations
import glob, json, os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
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
# Tabular ``insulin`` / ``CHO`` = causal values at the *prediction horizon*:
# from boluses/meals with time ≤ now, evaluated at now+PH (no future events).
# Insulin = ReplayBG action-remaining bolus IOB (U). Basal excluded.
LOOKBACK, HORIZON = 1, 6
_GAP_STEPS = 6
_STEP_NS = int(5 * 60 * 1e9)
_NS = int(60e9)
# ReplayBG DSS bolus grid ceiling (U); DOCAS span = per-patient max horizon IOB.
REPLAY_CIB_MAX_U = 5.0
CIB_MAX_U = REPLAY_CIB_MAX_U
# Library SQ insulin rates (single_meal_t1d defaults) + modal SI for action.
_RBG_KD, _RBG_KA2, _RBG_KE = 0.026, 0.014, 0.127
_RBG_VI, _RBG_BW, _RBG_P2 = 0.135, 75.0, 0.11
_RBG_SI = 2.3 * 5e-4  # modal SI (action peak ~60–90 min for rapid analogs)
_SPLIT_XML = {"train": "training", "test": "testing"}


def _rbg_pk_trajectories(n_min: int = 12 * 60) -> dict[str, np.ndarray]:
    """1 U bolus: mass IOB, peak-norm action X, action-remaining IOB fraction."""
    dose = 1000.0 / _RBG_BW
    Isc1 = dose / (1.0 + _RBG_KD)
    Isc2 = (_RBG_KD * Isc1) / (1.0 + _RBG_KA2)
    Ip = (_RBG_KA2 * Isc2) / (1.0 + _RBG_KE)
    X = (_RBG_P2 * (_RBG_SI / _RBG_VI) * Ip) / (1.0 + _RBG_P2)
    mass = np.empty(n_min + 1, float)
    act = np.empty(n_min + 1, float)
    mass[0] = (Isc1 + Isc2 + Ip * _RBG_VI) * _RBG_BW / 1000.0
    act[0] = X
    for t in range(1, n_min + 1):
        Isc1 = Isc1 / (1.0 + _RBG_KD)
        Isc2 = (Isc2 + _RBG_KD * Isc1) / (1.0 + _RBG_KA2)
        Ip = (Ip + _RBG_KA2 * Isc2) / (1.0 + _RBG_KE)
        X = (X + _RBG_P2 * (_RBG_SI / _RBG_VI) * Ip) / (1.0 + _RBG_P2)
        mass[t] = (Isc1 + Isc2 + Ip * _RBG_VI) * _RBG_BW / 1000.0
        act[t] = X
    mass = mass / mass[0]
    act_n = act / act.max()
    # Remaining AUC of action = fraction of total glucose-lowering still ahead.
    auc_tail = np.cumsum(act[::-1])[::-1]
    act_iob = auc_tail / auc_tail[0]
    cut = int(np.flatnonzero(act_iob < 0.01)[0]) if np.any(act_iob < 0.01) else n_min
    return dict(mass=mass[: cut + 1], action=act_n[: cut + 1], act_iob=act_iob[: cut + 1])


_RBG_PK = _rbg_pk_trajectories()
_RBG_IOB_LUT = _RBG_PK["act_iob"]  # feature = action-remaining IOB
_RBG_MASS_LUT = _RBG_PK["mass"]
_RBG_ACTION_LUT = _RBG_PK["action"]
_BOLUS_LOOKBACK_MIN = len(_RBG_IOB_LUT) - 1


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s.strip(), "%d-%m-%Y %H:%M:%S"))

def _ns(events: list) -> tuple[np.ndarray, np.ndarray]:
    return np.array([t.value for t, _ in events], np.int64), np.array([v for _, v in events], np.float64)

def bolus_iob_fraction(lag_min: np.ndarray) -> np.ndarray:
    """Action-remaining IOB fraction vs minutes since bolus (1 at delivery)."""
    lag = np.asarray(lag_min, float)
    out = np.zeros_like(lag, float)
    m = (lag >= 0) & (lag <= _BOLUS_LOOKBACK_MIN)
    if not np.any(m):
        return out
    ages = np.clip(np.rint(lag[m]).astype(int), 0, _BOLUS_LOOKBACK_MIN)
    out[m] = _RBG_IOB_LUT[ages]
    return out


# Alias used by ReplayBG ForecasterCIB.
bolus_action_fraction = bolus_iob_fraction


def _bolus_iob(grid: np.ndarray, bolus_times: np.ndarray, bolus_doses: np.ndarray) -> np.ndarray:
    dt = (grid[:, None] - bolus_times[None, :]) / _NS
    m = (dt >= 0) & (dt <= _BOLUS_LOOKBACK_MIN)
    return (bolus_iob_fraction(np.where(m, dt, 0.0)) * bolus_doses * m).sum(1)

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

def _patient_xml(root: str, patient_id: str, split: str) -> str:
    hits = sorted(glob.glob(os.path.join(
        _ohio_root(root), "*", split, f"{patient_id}-ws-{_SPLIT_XML[split]}.xml")))
    if not hits:
        raise FileNotFoundError(f"Ohio XML missing: {patient_id} {split}")
    return hits[0]

@lru_cache(maxsize=None)
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
    """Current-time ReplayBG IOB + CHO (for ReplayBG traces / twinning).

    ``basal_times`` / ``basal_rates`` kept for call-site compatibility.
    """
    del basal_times, basal_rates
    insulin = _bolus_iob(grid, bolus_times, bolus_doses)
    cho = _irf_sum(grid, meal_times, meal_carbs, _carb_irf)
    return insulin, cho


def _causal_horizon_iob(
    grid: np.ndarray, bolus_times: np.ndarray, bolus_doses: np.ndarray, ph_steps: int,
) -> np.ndarray:
    """iob[c] = IOB at grid[c+ph] from boluses with time ≤ grid[c] (no leakage)."""
    n = grid.size
    out = np.zeros(n, float)
    if not bolus_times.size or ph_steps <= 0:
        return out
    c_max = n - ph_steps
    for tb, dose in zip(bolus_times, bolus_doses):
        c0 = int(np.searchsorted(grid, tb, side="left"))
        if c0 >= c_max:
            continue
        ages = (grid[c0 + ph_steps: c_max + ph_steps] - tb) / _NS
        out[c0:c_max] += float(dose) * bolus_iob_fraction(ages)
    return out


def _causal_horizon_cho(
    grid: np.ndarray, meal_times: np.ndarray, meal_carbs: np.ndarray, ph_steps: int,
) -> np.ndarray:
    """cho[c] = CHO IRF at grid[c+ph] from meals with time ≤ grid[c] (no leakage)."""
    n = grid.size
    out = np.zeros(n, float)
    if not meal_times.size or ph_steps <= 0:
        return out
    c_max = n - ph_steps
    for tm, carbs in zip(meal_times, meal_carbs):
        c0 = int(np.searchsorted(grid, tm, side="left"))
        if c0 >= c_max:
            continue
        ages = (grid[c0 + ph_steps: c_max + ph_steps] - tm) / _NS
        m = (ages > 0) & (ages <= 48 * 60)
        contrib = np.zeros_like(ages)
        if np.any(m):
            contrib[m] = _carb_irf(ages[m]) * float(carbs)
        out[c0:c_max] += contrib
    return out


def _frame_from_xml(path: str, *, interpolate_cgm: bool, ph_steps: int) -> pd.DataFrame:
    """Forecast features: CGM now; causal IOB/CHO at prediction horizon."""
    pat = _parse(path)
    grid, cgm = _uniform_grid(pat["gluc"])
    if interpolate_cgm:
        cgm = _interp_short_gaps(cgm, _GAP_STEPS)
    mt, mv = _ns([(t, v) for t, v in pat["meal"] if v > 0])
    bt, bv = _ns(pat["bolus"])
    insulin = _causal_horizon_iob(grid, bt, bv, int(ph_steps))
    cho = _causal_horizon_cho(grid, mt, mv, int(ph_steps))
    return pd.DataFrame({"CGM": cgm, "insulin": insulin, "CHO": cho})

def _roll_features(x: np.ndarray, lb: int, ph: int) -> np.ndarray:
    n = len(x) - lb - ph + 1
    out = np.zeros([n, lb])
    for i in range(lb):
        out[:, i] = np.roll(x, -i)[:- lb - ph + 1]
    return out

def _target_delta(cgm: np.ndarray, lb: int, ph: int) -> np.ndarray:
    """Future minus CGM at the end of the lookback window (mg/dL)."""
    n = len(cgm) - lb - ph + 1
    if n <= 0:
        return np.zeros(0, float)
    future = cgm[lb + ph - 1: lb + ph - 1 + n].astype(float)
    current = cgm[lb - 1: lb - 1 + n].astype(float)
    return future - current

def _fit_unit_scaler(raw: np.ndarray) -> MinMaxScaler:
    finite = np.asarray(raw, float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite values to scale")
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(finite.reshape(-1, 1))
    return sc


def _scaled_matrices(data: pd.DataFrame, labels: tuple[str, ...], lb: int, ph: int, *, feature_scalers: list[MinMaxScaler] | None = None, target_scaler: MinMaxScaler | None = None):
    y_raw = _target_delta(data["CGM"].values, lb, ph)
    if target_scaler is None:
        target_scaler = _fit_unit_scaler(y_raw)
    y = target_scaler.transform(y_raw.reshape(-1, 1)).ravel()
    xs, scalers = [], [] if feature_scalers is None else feature_scalers
    for i, col in enumerate(labels):
        raw = data[col].values.reshape(-1, 1)
        sc = _fit_unit_scaler(raw) if feature_scalers is None else feature_scalers[i]
        if feature_scalers is None:
            scalers.append(sc)
        xs.append(_roll_features(sc.transform(raw).flatten(), lb, ph))
    X = np.concatenate(xs, axis=1)
    nan = np.union1d(np.argwhere(np.isnan(X))[:, 0], np.argwhere(np.isnan(y))[:, 0])
    return np.delete(X, nan, 0), np.delete(y, nan, 0), scalers, target_scaler


def _valid_feature_rows(data: pd.DataFrame, lb: int, ph: int) -> np.ndarray:
    """Row indices into the length-(len-lb-ph+1) feature matrix that survive NaN drop.

    With ``lb=1``, feature row ``r`` reads CGM/IOB/CHO at grid index ``r``.
    """
    y_raw = _target_delta(data["CGM"].values, lb, ph)
    xs = [_roll_features(data[col].values.astype(float), lb, ph) for col in FEATURE_COLUMNS]
    X = np.column_stack(xs)
    nan = np.union1d(np.argwhere(np.isnan(X))[:, 0], np.argwhere(np.isnan(y_raw))[:, 0])
    keep = np.ones(len(y_raw), bool)
    keep[nan] = False
    return np.flatnonzero(keep)

@dataclass
class PatientData:
    X_train: np.ndarray; y_train: np.ndarray; X_test: np.ndarray; y_test: np.ndarray
    feature_names: tuple[str, ...]; cgm_scaler: MinMaxScaler; train_scalers: list[MinMaxScaler]; patient_id: str
    insulin_idx: int = 1
    cgm_idx: int = 0
    lookback: int = 1

OhioPatientData = PatientData  # backward-compatible alias

def list_patients_ohio(root: str) -> list[str]:
    ohio = _ohio_root(root)
    ids: set[str] = set()
    for year in ("2018", "2020"):
        for path in glob.glob(os.path.join(ohio, year, "train", "*-ws-training.xml")):
            ids.add(os.path.basename(path).split("-")[0])
    return sorted(ids, key=int)

def _pk_split_context(
    df: pd.DataFrame, *, grid_ns: np.ndarray, hours: np.ndarray,
    pk: dict, keep_rows: np.ndarray, lb: int,
) -> dict[str, np.ndarray]:
    """Align PK state / hour / IOB / CGM with NaN-filtered feature rows."""
    # Feature row r ↔ grid index r + lb - 1
    gidx = keep_rows + int(lb) - 1
    gidx = np.clip(gidx, 0, len(df) - 1)
    return dict(
        Isc1=pk["Isc1"][gidx], Isc2=pk["Isc2"][gidx],
        Ip=pk["Ip"][gidx], X=pk["X"][gidx],
        g=df["CGM"].values.astype(float)[gidx],
        iob_u=df["insulin"].values.astype(float)[gidx],
        cho=df["CHO"].values.astype(float)[gidx],
        hours=hours[gidx],
        grid_idx=gidx.astype(int),
    )


def load_patient_pk_context(
    root: str, patient_id: str, *, horizon_min: int = 60, lookback: int | None = None,
    SI: float | None = None, dataset: str = "ohio_t1dm",
) -> dict:
    """PatientData-aligned causal PK states for state-dependent τ experiments."""
    from core import causal_pk_states  # local import: core imports forecasters only

    lb = int(LOOKBACK if lookback is None else lookback)
    ph = int(horizon_steps(horizon_min))
    si = float(SI if SI is not None else MODAL_SI)
    if dataset == "azt1d":
        path = _subject_csv(root, patient_id)
        raw = _normalize_columns(_read_events(path))
        raw["EventDateTime"] = pd.to_datetime(raw["EventDateTime"])
        raw = raw.sort_values("EventDateTime").reset_index(drop=True)
        t0, t1 = raw["EventDateTime"].iloc[0], raw["EventDateTime"].iloc[-1]
        grid_ts = pd.date_range(t0.floor("5min"), t1.ceil("5min"), freq="5min")
        grid = grid_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        hours = grid_ts.hour.to_numpy() + grid_ts.minute.to_numpy() / 60.0
        bmask = raw["TotalBolusInsulinDelivered"].fillna(0) > 0
        bolus_t = raw.loc[bmask, "EventDateTime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        bolus_v = raw.loc[bmask, "TotalBolusInsulinDelivered"].astype(float).to_numpy()
        df = _frame_from_csv(path, interpolate_cgm=True, ph_steps=ph)
        ok = ~np.isnan(df["CGM"].values)
        i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
        df = df.iloc[i0:i1].reset_index(drop=True)
        grid, hours = grid[i0:i1], hours[i0:i1]
        # PK over the trimmed grid; boluses still absolute timestamps.
        pk = causal_pk_states(grid, bolus_t, bolus_v, SI=si)
        n = len(df)
        cut = max(int(n * TRAIN_FRAC), lb + ph + 50)
        cut = min(cut, n - (lb + ph + 20))
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        train_df["CGM"] = _interp_short_gaps(train_df["CGM"].values, _GAP_STEPS)
        train_keep = _valid_feature_rows(train_df, lb, ph)
        test_keep = _valid_feature_rows(test_df, lb, ph)
        # PK/hour arrays for train/test slices of the trimmed grid.
        pk_tr = {k: v[:cut] for k, v in pk.items() if k != "basal"}
        pk_te = {k: v[cut:] for k, v in pk.items() if k != "basal"}
        tr = _pk_split_context(train_df, grid_ns=grid[:cut], hours=hours[:cut],
                               pk=pk_tr, keep_rows=train_keep, lb=lb)
        te = _pk_split_context(test_df, grid_ns=grid[cut:], hours=hours[cut:],
                               pk=pk_te, keep_rows=test_keep, lb=lb)
    else:
        train_path = _patient_xml(root, patient_id, "train")
        test_path = _patient_xml(root, patient_id, "test")
        train_df = _frame_from_xml(train_path, interpolate_cgm=True, ph_steps=ph)
        test_df = _frame_from_xml(test_path, interpolate_cgm=False, ph_steps=ph)
        out = {}
        for split, path, df in (("train", train_path, train_df), ("test", test_path, test_df)):
            pat = _parse(path)
            grid, cgm = _uniform_grid(pat["gluc"])
            # Match _frame_from_xml length (full grid).
            hours = np.array([
                pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0
                for t in pd.to_datetime(grid)
            ], float)
            bt, bv = _ns(pat["bolus"])
            pk = causal_pk_states(grid, bt, bv, SI=si)
            keep = _valid_feature_rows(df, lb, ph)
            out[split] = _pk_split_context(
                df, grid_ns=grid, hours=hours, pk=pk, keep_rows=keep, lb=lb)
        tr, te = out["train"], out["test"]
    meta = (
        load_patient_meta_azt1d(root, patient_id) if dataset == "azt1d"
        else load_patient_meta_ohio(root, patient_id))
    return dict(train=tr, test=te, meta=meta, SI=si, patient_id=str(patient_id))


@lru_cache(maxsize=None)
def _load_patient_ohio_cached(
    root: str, patient_id: str, horizon_steps: int, lookback: int,
) -> PatientData:
    labels = FEATURE_COLUMNS
    ph = int(horizon_steps)
    train_df = _frame_from_xml(
        _patient_xml(root, patient_id, "train"), interpolate_cgm=True, ph_steps=ph)
    test_df = _frame_from_xml(
        _patient_xml(root, patient_id, "test"), interpolate_cgm=False, ph_steps=ph)
    X_train, y_train, scalers, target_scaler = _scaled_matrices(
        train_df, labels, lookback, ph)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(
        test_df, labels, lookback, ph,
        feature_scalers=scalers, target_scaler=target_scaler)
    return PatientData(
        X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers,
        str(patient_id), insulin_idx=lookback + (lookback - 1),
        cgm_idx=lookback - 1, lookback=lookback)

def load_patient_ohio(
    root: str,
    patient_id: str,
    *,
    horizon_steps: int = HORIZON_STEPS,
    lookback: int | None = None,
) -> PatientData:
    lb = int(LOOKBACK if lookback is None else lookback)
    return _load_patient_ohio_cached(
        str(root), str(patient_id), int(horizon_steps), lb)

def rmse_mgdl(y_scaled: np.ndarray, yhat_scaled: np.ndarray, cgm_scaler: MinMaxScaler) -> float:
    y = cgm_scaler.inverse_transform(np.asarray(y_scaled, float).reshape(-1, 1)).flatten()
    yh = cgm_scaler.inverse_transform(np.asarray(yhat_scaled, float).reshape(-1, 1)).flatten()
    ok = ~np.isnan(y)
    return float(np.sqrt(np.mean((y[ok] - yh[ok]) ** 2)))

class PaperGlucoseModel:
    def __init__(self, model, train_scalers: list[MinMaxScaler], cgm_scaler: MinMaxScaler,
                 *, predicts_level: bool = False, level_scaler: MinMaxScaler | None = None):
        self._model = model
        self._scalers = train_scalers
        self._cgm_scaler = cgm_scaler
        self._predicts_level = bool(predicts_level)
        self._level_scaler = level_scaler if level_scaler is not None else train_scalers[0]

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X, np.column_stack([s.transform(X[:, i:i + 1]).ravel() for i, s in enumerate(self._scalers)])

    def predict_delta(self, X: np.ndarray, **_) -> np.ndarray:
        """Predicted change in CGM (mg/dL) over the forecast horizon."""
        X, scaled = self._scale(X)
        y_scaled = self._model.predict(scaled)
        if self._predicts_level:
            level = self._level_scaler.inverse_transform(
                np.asarray(y_scaled, float).reshape(-1, 1)).ravel()
            cgm0 = self._level_scaler.inverse_transform(
                scaled[:, 0:1]).ravel()
            return level - cgm0
        return self._cgm_scaler.inverse_transform(np.asarray(y_scaled, float).reshape(-1, 1)).ravel()

    def predict(self, X: np.ndarray, **_) -> np.ndarray:
        """Future CGM (mg/dL) = current CGM + predicted ΔCGM."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X[:, 0] + self.predict_delta(X)

    def predict_intervention(self, X, u, span):
        """Scaled ΔCGM under replace probe ``u·span``."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = len(self._scalers)
        _, scaled = self._scale(X[:, :n])
        m = self._model
        if hasattr(m, "predict_intervention"):
            return np.asarray(m.predict_intervention(scaled, u, span), float)
        u = np.asarray(u, float).ravel()
        rows = np.repeat(scaled, u.size, 0)
        ji = 1
        rows[:, ji] = self._scalers[ji].transform(
            np.tile(u * float(span), len(X)).reshape(-1, 1)).ravel()
        pred = np.asarray(m.predict(rows), float).reshape(len(X), u.size)
        if self._predicts_level:
            # Convert level predictions → scaled ΔCGM for future() / CIB.
            level = self._level_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            cgm0 = self._level_scaler.inverse_transform(scaled[:, 0:1]).ravel()[:, None]
            delta = level - cgm0
            return self._cgm_scaler.transform(delta.reshape(-1, 1)).reshape(pred.shape)
        return pred

    def future_dose_curve(self, row, doses_u, *, span: float):
        row = np.asarray(row, float).ravel()
        doses_u = np.asarray(doses_u, float).ravel()
        X = np.tile(row, (len(doses_u), 1))
        ji = 1
        X[:, ji] = self._scalers[ji].transform(np.clip(doses_u, 0.0, float(span)).reshape(-1, 1)).ravel()
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

# OhioT1DM XML weight="99" is a documented placeholder (Marling & Bunescu 2020),
# not measured body weight. Use the same fixed 75 kg ReplayBG prior as AZT1D.
DEFAULT_WEIGHT_KG = 75.0
# Modal SI and reference TDD for 1800-rule personalization (U/day).
MODAL_SI = 2.3 * 5e-4
REF_TDD_U = 40.0  # typical adult TDD corresponding to modal SI


def _tdd_from_events(bolus, basal, duration_days: float) -> float:
    """Total daily dose (U/day) ≈ mean basal×24 + bolus_total/days."""
    if duration_days <= 0:
        return float("nan")
    bolus_u = float(sum(v for _, v in bolus if v > 0))
    rates = [v for _, v in basal if v > 0]
    mean_basal = float(np.mean(rates)) if rates else 0.0
    return mean_basal * 24.0 + bolus_u / float(duration_days)


def si_from_tdd(tdd_u: float, *, modal_si: float = MODAL_SI, ref_tdd: float = REF_TDD_U) -> float:
    """Scale modal SI by ref_TDD / patient_TDD (1800-rule proxy: higher TDD → lower SI)."""
    if not np.isfinite(tdd_u) or tdd_u <= 0:
        return float(modal_si)
    return float(modal_si) * float(ref_tdd) / float(tdd_u)


@lru_cache(maxsize=None)
def load_patient_meta_ohio(root: str, patient_id: str) -> dict:
    """Cached; only this patient's train/test XMLs (not the full cohort)."""
    pid = str(patient_id)
    basal_rates = []
    bolus_all, basal_all = [], []
    t_min, t_max = None, None
    for split in ("train", "test"):
        try:
            pat = _parse(_patient_xml(root, pid, split))
        except FileNotFoundError:
            continue
        basal_rates.extend(v for _, v in pat["basal"] if v > 0)
        bolus_all.extend(pat["bolus"])
        basal_all.extend(pat["basal"])
        for t, _ in pat["gluc"]:
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)
    weight = float(DEFAULT_WEIGHT_KG)
    mean_basal_u_hr = float(np.mean(basal_rates)) if basal_rates else 0.0
    duration_days = (
        (t_max - t_min).total_seconds() / 86400.0
        if t_min is not None and t_max is not None else float("nan"))
    tdd = _tdd_from_events(bolus_all, basal_all, duration_days)
    si_pers = si_from_tdd(tdd)
    target = dict(PATIENT_TARGET.get(pid, {}))
    if "SI" not in target and np.isfinite(tdd) and tdd > 0:
        target = {**target, "SI": si_pers, "tdd_u": float(tdd)}
    return {
        "weight_kg": weight,
        "u2ss": replay_u2ss_mU_kg_min(mean_basal_u_hr, weight),
        "tdd_u": float(tdd) if np.isfinite(tdd) else None,
        "SI_personal": float(si_pers),
        "target": target,
    }

def load_glucose_trace(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    frames = []
    for sp in (("train", "test") if split is None else (split,)):
        try:
            pat = _parse(_patient_xml(root, patient_id, sp))
        except FileNotFoundError:
            continue
        frames.append(_trace_frame(pat).with_columns(pl.lit(sp).alias(SPLIT_COL)))
    return pl.concat(frames) if frames else pl.DataFrame()


ohio_replay_trace = load_glucose_trace  # alias for visuals / ReplayBG windows


def load_glucose_trace_azt1d(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    """5-min CGM/bolus/meal/basal trace for ReplayBG from AZT1D subject CSV."""
    raw = _normalize_columns(_read_events(_subject_csv(root, patient_id)))
    raw["EventDateTime"] = pd.to_datetime(raw["EventDateTime"])
    raw = raw.sort_values("EventDateTime").reset_index(drop=True)

    def _ns_arr(series: pd.Series) -> np.ndarray:
        return series.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    t0, t1 = raw["EventDateTime"].iloc[0], raw["EventDateTime"].iloc[-1]
    grid_ts = pd.date_range(t0.floor("5min"), t1.ceil("5min"), freq="5min")
    grid = grid_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cgm = np.full(grid.size, np.nan)
    gdf = raw.loc[raw["CGM"].notna(), ["EventDateTime", "CGM"]]
    for t_ns, v in zip(_ns_arr(gdf["EventDateTime"]), gdf["CGM"]):
        j = int(round((int(t_ns) - int(grid[0])) / _STEP_NS))
        if 0 <= j < cgm.size:
            cgm[j] = float(v)
    cgm = _interp_short_gaps(cgm, _GAP_STEPS)
    bmask = raw["TotalBolusInsulinDelivered"].fillna(0) > 0
    bolus_t = _ns_arr(raw.loc[bmask, "EventDateTime"])
    bolus_v = raw.loc[bmask, "TotalBolusInsulinDelivered"].astype(float).to_numpy()
    mmask = raw["CarbSize"].fillna(0) > 0
    meal_t = _ns_arr(raw.loc[mmask, "EventDateTime"])
    meal_v = raw.loc[mmask, "CarbSize"].astype(float).to_numpy()
    basal_rate = np.zeros(grid.size, float)
    if "Basal" in raw.columns:
        bdf = raw.loc[raw["Basal"].fillna(0) > 0, ["EventDateTime", "Basal"]]
        if len(bdf):
            bt = _ns_arr(bdf["EventDateTime"])
            bv = bdf["Basal"].astype(float).to_numpy()
            # Step-hold last basal onto the 5-min grid.
            j = 0
            for i in range(grid.size):
                while j + 1 < len(bt) and int(bt[j + 1]) <= int(grid[i]):
                    j += 1
                if int(bt[j]) <= int(grid[i]):
                    basal_rate[i] = float(bv[j])
    ok = np.isfinite(cgm)
    if not ok.any():
        return pl.DataFrame()
    i0 = int(np.argmax(ok))
    i1_full = int(len(ok) - np.argmax(ok[::-1]))
    i1 = max(i0 + 1, i1_full - HORIZON_STEPS)
    g = cgm[i0:i1]
    grid_s = grid[i0:i1]
    times = pd.to_datetime(grid_s)
    meal_imp = _impulses(grid_s, meal_t, meal_v)
    bolus_imp = _impulses(grid_s, bolus_t, bolus_v)
    target = cgm[i0 + HORIZON_STEPS:i1 + HORIZON_STEPS] - cgm[i0:i1]
    n = len(g)
    cut = max(int(n * TRAIN_FRAC), 50)
    cut = min(cut, n - 50) if n > 100 else n
    split_col = np.array(["train"] * cut + ["test"] * (n - cut), dtype=object)
    frame = pl.DataFrame({
        PATIENT_COL: [str(patient_id)] * n,
        TIMESTAMP_COL: times.to_list(),
        "glucose": g.astype(float),
        "insulin": np.zeros(n, float),
        "carbs": np.zeros(n, float),
        "meal_carbs": meal_imp.astype(float),
        "bolus_dose": bolus_imp.astype(float),
        "basal_rate": basal_rate[i0:i1].astype(float),
        TARGET: target.astype(float),
        SPLIT_COL: split_col,
    })
    if split is None:
        return frame
    return frame.filter(pl.col(SPLIT_COL) == split)

def causal_horizon_feats_on_seg(
    seg: pl.DataFrame, ph_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal IOB/CHO at prediction horizon from bolus/meal impulses on a segment."""
    n = seg.height
    bolus = np.asarray(seg["bolus_dose"], float)
    meal_col = next(c for c in ("meal_carbs", "carbs", "CHO") if c in seg.columns)
    meal = np.asarray(seg[meal_col], float)
    iob = np.zeros(n, float)
    cho = np.zeros(n, float)
    ph = int(ph_steps)
    if ph <= 0:
        return iob, cho
    c_max = n - ph
    for t in np.flatnonzero(bolus > 0):
        if t >= c_max:
            continue
        cs = np.arange(int(t), c_max)
        ages = (cs + ph - int(t)) * CGM_STEP_MIN
        iob[cs] += float(bolus[t]) * bolus_iob_fraction(ages.astype(float))
    for t in np.flatnonzero(meal > 0):
        if t >= c_max:
            continue
        cs = np.arange(int(t), c_max)
        ages = (cs + ph - int(t)) * CGM_STEP_MIN
        cho[cs] += float(meal[t]) * _carb_irf(ages.astype(float))
    return iob, cho


def paper_feat_arrays(seg: pl.DataFrame, *, ph_steps: int | None = None) -> dict[str, np.ndarray]:
    """Forecast features on a replay segment: CGM now; causal IOB/CHO at horizon."""
    ph = HORIZON_STEPS if ph_steps is None else int(ph_steps)
    iob_h, cho_h = causal_horizon_feats_on_seg(seg, ph)
    return {
        "CGM": seg["glucose"].to_numpy().astype(float),
        "insulin": iob_h,
        "CHO": cho_h,
    }

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
        # us resolution: ReplayBG's tsteps setup calls .astype(datetime), which
        # yields ints (not datetimes) for pandas' default ns dtype.
        "t": pd.to_datetime([t0 + timedelta(minutes=yts_min * i) for i in range(n)]).as_unit("us"),
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

# ── AZT1D ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.8
# Subject 14 has no isolated CGM channel: CGM and BGM share one "Readings" column.
EXCLUDE_PATIENTS = frozenset({"14"})

def _azt1d_root(root: str) -> Path:
    return Path(root) / "azt1d" / "CGM Records"

def list_patients_azt1d(root: str) -> list[str]:
    paths = sorted(glob.glob(str(_azt1d_root(root) / "Subject *" / "Subject *.csv")))
    ids = []
    for p in paths:
        try:
            ids.append(str(int(Path(p).stem.split()[-1])))
        except ValueError:
            continue
    return sorted((set(ids) - EXCLUDE_PATIENTS), key=int)

def _read_events(path: Path) -> pd.DataFrame:
    """Subject CSV without repeated events.

    The release repeats event rows (up to 26x), sometimes differing only in the
    Basal rate; summing them would stack the same bolus and meal.
    """
    raw = pd.read_csv(path)
    return raw.drop_duplicates(subset=[c for c in raw.columns if c != "Basal"])


def _subject_csv(root: str, patient_id: str) -> Path:
    sid = int(patient_id)
    path = _azt1d_root(root) / f"Subject {sid}" / f"Subject {sid}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def azt1d_aid_profile(root: str = ".") -> dict:
    """Quantify how much of AZT1D therapy the pump delivers on its own.

    Backs the decision to keep AZT1D out of the corrective-bolus evaluation
    (manuscript Section "In-silico decision support"): Control-IQ issues
    corrective boluses itself and modulates basal continuously, so a
    corrective-bolus DSS has no clean intervention to test there. Reads only
    the released CSV columns, so it needs no model or simulation.
    """
    rows = []
    for pid in list_patients_azt1d(root):
        raw = _read_events(_subject_csv(root, pid))[
            ["EventDateTime", "BolusType", "Basal", "CorrectionDelivered",
             "TotalBolusInsulinDelivered"]]
        bol = raw[raw["TotalBolusInsulinDelivered"].fillna(0) > 0]
        if not len(bol):
            continue
        kind = bol["BolusType"].astype(str).str.lower()
        basal = raw["Basal"].fillna(0).astype(float)
        # Basal is a U/h rate held until the next entry; approximate the
        # delivered amount on the release's own event spacing.
        dt_h = (raw["EventDateTime"].pipe(pd.to_datetime).diff().dt.total_seconds()
                / 3600.0).fillna(0.0).clip(lower=0.0, upper=1.0)
        basal_u = float((basal * dt_h).sum())
        bolus_u = float(bol["TotalBolusInsulinDelivered"].astype(float).sum())
        rows.append(dict(
            patient_id=str(pid), n_bolus_events=int(len(bol)),
            automatic_pct=100.0 * float(kind.str.contains("automat").mean()),
            correction_pct=100.0 * float(
                (bol["CorrectionDelivered"].fillna(0) > 0).mean()),
            basal_share_pct=100.0 * basal_u / max(basal_u + bolus_u, 1e-9),
            distinct_basal_rates=int(basal[basal > 0].round(3).nunique()),
        ))

    def q(key):
        a = np.asarray([r[key] for r in rows], float)
        return {"median": float(np.median(a)),
                "q1": float(np.percentile(a, 25)),
                "q3": float(np.percentile(a, 75)),
                "min": float(a.min()), "max": float(a.max())}

    out = {"n_patients": len(rows), "per_patient": rows,
           "automatic_pct": q("automatic_pct"),
           "correction_pct": q("correction_pct"),
           "basal_share_pct": q("basal_share_pct"),
           "distinct_basal_rates": q("distinct_basal_rates")}
    path = Path(root) / "results" / "azt1d" / "aid_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    return out


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map release variants (e.g. Subject 14) onto the CGM column used elsewhere."""
    if "CGM" not in raw.columns:
        for alt in ("Readings (CGM / BGM)", "Readings (CGM/BGM)", "CGM / BGM"):
            if alt in raw.columns:
                raw = raw.rename(columns={alt: "CGM"})
                break
    if "CGM" not in raw.columns:
        raise KeyError(f"No CGM column in {list(raw.columns)}")
    return raw

def _frame_from_csv(path: Path, *, interpolate_cgm: bool, ph_steps: int) -> pd.DataFrame:
    raw = _normalize_columns(_read_events(path))
    raw["EventDateTime"] = pd.to_datetime(raw["EventDateTime"])
    raw = raw.sort_values("EventDateTime").reset_index(drop=True)

    def _ns_arr(series: pd.Series) -> np.ndarray:
        return series.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    t0, t1 = raw["EventDateTime"].iloc[0], raw["EventDateTime"].iloc[-1]
    grid_ts = pd.date_range(t0.floor("5min"), t1.ceil("5min"), freq="5min")
    grid = grid_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cgm = np.full(grid.size, np.nan)
    gdf = raw.loc[raw["CGM"].notna(), ["EventDateTime", "CGM"]]
    for t_ns, v in zip(_ns_arr(gdf["EventDateTime"]), gdf["CGM"]):
        j = int(round((int(t_ns) - int(grid[0])) / _STEP_NS))
        if 0 <= j < cgm.size:
            cgm[j] = float(v)
    if interpolate_cgm:
        cgm = _interp_short_gaps(cgm, _GAP_STEPS)
    bmask = raw["TotalBolusInsulinDelivered"].fillna(0) > 0
    bolus_t = _ns_arr(raw.loc[bmask, "EventDateTime"])
    bolus_v = raw.loc[bmask, "TotalBolusInsulinDelivered"].astype(float).to_numpy()
    mmask = raw["CarbSize"].fillna(0) > 0
    meal_t = _ns_arr(raw.loc[mmask, "EventDateTime"])
    meal_v = raw.loc[mmask, "CarbSize"].astype(float).to_numpy()
    ph = int(ph_steps)
    insulin = _causal_horizon_iob(grid, bolus_t, bolus_v, ph)
    cho = _causal_horizon_cho(grid, meal_t, meal_v, ph)
    return pd.DataFrame({"CGM": cgm, "insulin": insulin, "CHO": cho})

@lru_cache(maxsize=None)
def _load_patient_azt1d_cached(
    root: str, patient_id: str, horizon_steps: int, lookback: int,
) -> PatientData:
    ph = int(horizon_steps)
    df = _frame_from_csv(
        _subject_csv(root, patient_id), interpolate_cgm=True, ph_steps=ph)
    ok = ~np.isnan(df["CGM"].values)
    if not ok.any():
        raise ValueError(f"AZT1D patient {patient_id}: no CGM")
    i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
    df = df.iloc[i0:i1].reset_index(drop=True)
    n = len(df)
    cut = max(int(n * TRAIN_FRAC), lookback + ph + 50)
    cut = min(cut, n - (lookback + ph + 20))
    train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    train_df["CGM"] = _interp_short_gaps(train_df["CGM"].values, _GAP_STEPS)
    labels = FEATURE_COLUMNS
    X_train, y_train, scalers, target_scaler = _scaled_matrices(
        train_df, labels, lookback, ph)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(
        test_df, labels, lookback, ph,
        feature_scalers=scalers, target_scaler=target_scaler)
    return PatientData(
        X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers,
        str(patient_id), insulin_idx=lookback + (lookback - 1),
        cgm_idx=lookback - 1, lookback=lookback)

def load_patient_azt1d(
    root: str,
    patient_id: str,
    *,
    horizon_steps: int = HORIZON_STEPS,
    lookback: int | None = None,
) -> PatientData:
    lb = int(LOOKBACK if lookback is None else lookback)
    return _load_patient_azt1d_cached(
        str(root), str(patient_id), int(horizon_steps), lb)

# Plausibility bounds for AZT1D insulin bookkeeping. The release's Basal column
# carries out-of-range values for some subjects (up to 2022 "U/h"), and two bolus
# logs total ~150 U/day, so basal rates above MAX_BASAL_U_HR are discarded and the
# implied total daily dose is clamped to a clinically usable band.
MAX_BASAL_U_HR = 5.0
TDD_BOUNDS_U = (18.0, 120.0)  # 1800 rule -> correction factor 15-100 mg/dL per U


def _azt1d_tdd_u(path: Path) -> float:
    """Total daily dose (U/day) = median plausible basal rate x 24 + bolus/day."""
    cols = {"EventDateTime", "Basal", "TotalBolusInsulinDelivered"}
    raw = _read_events(path)
    raw = raw[[c for c in raw.columns if c in cols]]
    if not {"EventDateTime", "TotalBolusInsulinDelivered"} <= set(raw.columns):
        return float("nan")
    t = pd.to_datetime(raw["EventDateTime"])
    days = (t.max() - t.min()).total_seconds() / 86400.0
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    bolus_per_day = float(
        raw["TotalBolusInsulinDelivered"].fillna(0).astype(float).sum()) / days
    basal_per_day = 0.0
    if "Basal" in raw.columns:
        b = raw["Basal"].fillna(0).astype(float)
        b = b[(b > 0) & (b <= MAX_BASAL_U_HR)]
        if len(b):
            basal_per_day = float(b.median()) * 24.0
    return float(np.clip(basal_per_day + bolus_per_day, *TDD_BOUNDS_U))


@lru_cache(maxsize=None)
def load_patient_meta_azt1d(root: str, patient_id: str) -> dict:
    path = _subject_csv(root, patient_id)
    raw = pd.read_csv(path, usecols=lambda c: c in {"Basal", "TotalBolusInsulinDelivered"})
    basal = raw["Basal"].dropna().astype(float) if "Basal" in raw.columns else pd.Series(dtype=float)
    mean_basal = float(basal[basal > 0].mean()) if len(basal[basal > 0]) else 0.0
    w = DEFAULT_WEIGHT_KG
    tdd = _azt1d_tdd_u(path)
    si_pers = si_from_tdd(tdd)
    return {
        "weight_kg": w,
        "u2ss": (mean_basal / 60.0 * 1000.0 / w) if mean_basal > 0 else 0.0,
        "tdd_u": float(tdd) if np.isfinite(tdd) else None,
        "SI_personal": float(si_pers),
        "target": {},
        "dataset": "azt1d",
    }

_DATA = Path(__file__).resolve().parents[1] / "datasets"
HUPA_DIR = _DATA / "HUPA-UCM"


def list_patients_hupa(root: str = ".") -> list[str]:
    del root
    return sorted(p.name[4:-5] for p in HUPA_DIR.glob("HUPA*P.csv"))


def _hupa_path(patient_id: str) -> Path:
    return HUPA_DIR / f"HUPA{patient_id}P.csv"


def _hupa_carb_grams(series: pd.Series) -> pd.Series:
    """HUPA-UCM logs carb_input in 10 g servings; 0002, 0004, 0022 left in grams."""
    c = pd.to_numeric(series, errors="coerce").fillna(0.0)
    pos = c[c > 0]
    scale = 1.0 if (len(pos) and float(pos.median()) > 20.0) else 10.0
    return c * scale


def _frame_from_hupa(path: Path, *, ph_steps: int) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";")
    raw["time"] = pd.to_datetime(raw["time"])
    raw["glucose"] = pd.to_numeric(raw["glucose"], errors="coerce")
    raw["bolus_volume_delivered"] = pd.to_numeric(raw["bolus_volume_delivered"], errors="coerce").fillna(0.0)
    raw["carb_input"] = _hupa_carb_grams(raw["carb_input"])
    g = raw.groupby("time", as_index=False).agg(
        glucose=("glucose", "mean"),
        bolus=("bolus_volume_delivered", "sum"),
        carb=("carb_input", "sum"),
    ).sort_values("time")
    grid = g["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cgm = g["glucose"].to_numpy(float)
    bolus = g["bolus"].to_numpy(float)
    carb = g["carb"].to_numpy(float)
    insulin = _causal_horizon_iob(grid, grid[bolus > 0], bolus[bolus > 0], int(ph_steps))
    cho = _causal_horizon_cho(grid, grid[carb > 0], carb[carb > 0], int(ph_steps))
    return pd.DataFrame({"CGM": cgm, "insulin": insulin, "CHO": cho})


@lru_cache(maxsize=None)
def _load_patient_hupa_cached(patient_id: str, horizon_steps: int, lookback: int) -> PatientData:
    ph = int(horizon_steps)
    lb = int(lookback)
    df = _frame_from_hupa(_hupa_path(patient_id), ph_steps=ph)
    ok = ~np.isnan(df["CGM"].values)
    i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
    df = df.iloc[i0:i1].reset_index(drop=True)
    n = len(df)
    cut = max(int(n * TRAIN_FRAC), lb + ph + 50)
    cut = min(cut, n - (lb + ph + 20))
    train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    train_df["CGM"] = _interp_short_gaps(train_df["CGM"].values, _GAP_STEPS)
    labels = FEATURE_COLUMNS
    X_train, y_train, scalers, target_scaler = _scaled_matrices(train_df, labels, lb, ph)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(
        test_df, labels, lb, ph, feature_scalers=scalers, target_scaler=target_scaler)
    return PatientData(
        X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers,
        str(patient_id), insulin_idx=lb + (lb - 1), cgm_idx=lb - 1, lookback=lb)


def load_patient_hupa(root: str, patient_id: str, *, horizon_steps: int = HORIZON_STEPS, lookback: int | None = None) -> PatientData:
    del root
    lb = int(LOOKBACK if lookback is None else lookback)
    return _load_patient_hupa_cached(str(patient_id), int(horizon_steps), lb)


def load_patient_meta_hupa(root: str, patient_id: str) -> dict:
    del root
    raw = pd.read_csv(_hupa_path(patient_id), sep=";")
    t = pd.to_datetime(raw["time"])
    days = max((t.max() - t.min()).total_seconds() / 86400.0, 1e-6)
    bolus = pd.to_numeric(raw["bolus_volume_delivered"], errors="coerce").fillna(0.0).sum()
    basal = pd.to_numeric(raw["basal_rate"], errors="coerce").fillna(0.0)
    basal = basal[(basal > 0) & (basal * 12.0 <= MAX_BASAL_U_HR)]
    mean_basal_u_hr = float(basal.median()) * 12.0 if len(basal) else 0.0
    basal_per_day = mean_basal_u_hr * 24.0
    tdd = float(np.clip(basal_per_day + float(bolus) / days, *TDD_BOUNDS_U))
    weight = float(DEFAULT_WEIGHT_KG)
    return {
        "weight_kg": weight,
        "u2ss": replay_u2ss_mU_kg_min(mean_basal_u_hr, weight),
        "tdd_u": tdd,
        "SI_personal": float(si_from_tdd(tdd)),
        "target": {"SI": float(si_from_tdd(tdd)), "tdd_u": tdd},
        "dataset": "hupa",
    }


def load_glucose_trace_hupa(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    """5-min CGM/bolus/meal/basal trace for ReplayBG from HUPA-UCM CSV."""
    del root
    raw = pd.read_csv(_hupa_path(patient_id), sep=";")
    raw["time"] = pd.to_datetime(raw["time"])
    raw["glucose"] = pd.to_numeric(raw["glucose"], errors="coerce")
    raw["bolus"] = pd.to_numeric(raw["bolus_volume_delivered"], errors="coerce").fillna(0.0)
    raw["carb"] = _hupa_carb_grams(raw["carb_input"])
    raw["basal"] = pd.to_numeric(raw["basal_rate"], errors="coerce").fillna(0.0) * 12.0
    g_df = raw.groupby("time", as_index=False).agg(
        glucose=("glucose", "mean"),
        bolus=("bolus", "sum"),
        carb=("carb", "sum"),
        basal=("basal", "mean"),
    ).sort_values("time")
    t0, t1 = g_df["time"].iloc[0], g_df["time"].iloc[-1]
    grid_ts = pd.date_range(t0.floor("5min"), t1.ceil("5min"), freq="5min")
    grid = grid_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cgm = np.full(grid.size, np.nan)
    bolus_imp = np.zeros(grid.size, float)
    meal_imp = np.zeros(grid.size, float)
    basal_rate = np.zeros(grid.size, float)
    t_ns = g_df["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for ts, gv, bv, cv, ba in zip(t_ns, g_df["glucose"], g_df["bolus"], g_df["carb"], g_df["basal"]):
        j = int(round((int(ts) - int(grid[0])) / _STEP_NS))
        if 0 <= j < grid.size:
            cgm[j] = float(gv)
            bolus_imp[j] += float(bv)
            meal_imp[j] += float(cv)
            basal_rate[j] = float(np.clip(ba, 0.0, MAX_BASAL_U_HR))
    cgm = _interp_short_gaps(cgm, _GAP_STEPS)
    ok = np.isfinite(cgm)
    if not ok.any():
        return pl.DataFrame()
    i0 = int(np.argmax(ok))
    i1_full = int(len(ok) - np.argmax(ok[::-1]))
    i1 = max(i0 + 1, i1_full - HORIZON_STEPS)
    g = cgm[i0:i1]
    times = pd.to_datetime(grid[i0:i1])
    target = cgm[i0 + HORIZON_STEPS:i1 + HORIZON_STEPS] - cgm[i0:i1]
    n = len(g)
    cut = max(int(n * TRAIN_FRAC), 50)
    cut = min(cut, n - 50) if n > 100 else n
    split_col = np.array(["train"] * cut + ["test"] * (n - cut), dtype=object)
    frame = pl.DataFrame({
        PATIENT_COL: [str(patient_id)] * n,
        TIMESTAMP_COL: times.to_list(),
        "glucose": g.astype(float),
        "insulin": np.zeros(n, float),
        "carbs": np.zeros(n, float),
        "meal_carbs": meal_imp[i0:i1].astype(float),
        "bolus_dose": bolus_imp[i0:i1].astype(float),
        "basal_rate": basal_rate[i0:i1].astype(float),
        TARGET: target.astype(float),
        SPLIT_COL: split_col,
    })
    if split is None:
        return frame
    return frame.filter(pl.col(SPLIT_COL) == split)


# MetaboNet v1.0 public release (Wolff et al., JDST 2026). Official files
# already apply the known-patient split: 20% of each source is held out as
# novel subjects, and 20% of the rest is cut chronologically 50/50 with a
# 24 h buffer. A per-patient model has training rows only for the known
# subjects (subject_split_across_traintest). Novel subjects stay out.
METABONET_TRAIN = _DATA / "train.parquet"
METABONET_TEST = _DATA / "test.parquet"
METABONET_CACHE = _DATA / "metabonet_known"
METABONET_MANIFEST = METABONET_CACHE / "manifest.json"
_MIN_TRAIN_BOLUS = 20
_MIN_TEST_CGM = 500


def _metabonet_slice(path: Path, split: str) -> pd.DataFrame:
    import pyarrow.dataset as ds
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=["source_file", "id", "date", "CGM", "bolus", "basal", "carbs",
                 "weight", "insulin_delivery_modality"],
        filter=ds.field("subject_split_across_traintest") == True,  # noqa: E712
    )
    df = table.to_pandas()
    df["split"] = split
    df["ts"] = pd.to_datetime(df["date"]).astype("int64")
    df["bolus"] = df["bolus"].fillna(0.0)
    df["basal"] = df["basal"].fillna(0.0)
    df["carb"] = df["carbs"].fillna(0.0)
    return df.rename(columns={
        "source_file": "source", "id": "sid", "CGM": "cgm",
        "insulin_delivery_modality": "modality",
    })[["source", "sid", "split", "ts", "cgm", "bolus", "basal", "carb", "weight", "modality"]]


def _ensure_metabonet_cache() -> list[dict]:
    if METABONET_MANIFEST.is_file():
        return json.loads(METABONET_MANIFEST.read_text())
    METABONET_CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.concat([
        _metabonet_slice(METABONET_TRAIN, "train"),
        _metabonet_slice(METABONET_TEST, "test"),
    ], ignore_index=True)
    manifest = []
    for (source, sid), g in df.groupby(["source", "sid"], sort=True):
        part = _metabonet_part(str(source), str(sid))
        part.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["source", "sid"]).to_parquet(part / "data.parquet", index=False)
        train = g["split"] == "train"
        w = g.loc[g["weight"] > 0, "weight"]
        nbol = int(((train) & (g["bolus"] > 0)).sum())
        ncarb = int(((train) & (g["carb"] > 0)).sum())
        ncgm = int(((~train) & g["cgm"].notna()).sum())
        modality = g["modality"].dropna()
        manifest.append(dict(
            key=f"{source}|{sid}", source=str(source), sid=str(sid),
            modality=None if modality.empty else str(modality.iloc[0]),
            weight=None if w.empty else float(w.median()),
            nbol=nbol, ncarb=ncarb, ncgm=ncgm,
            usable=bool(nbol >= _MIN_TRAIN_BOLUS and ncgm >= _MIN_TEST_CGM),
        ))
    METABONET_MANIFEST.write_text(json.dumps(manifest))
    return manifest


def _metabonet_manifest() -> list[dict]:
    return _ensure_metabonet_cache()


def list_patients_metabonet(root: str = ".") -> list[str]:
    del root
    return [r["key"] for r in _metabonet_manifest() if r["usable"]]


def list_patients_metabonet_sap(root: str = ".") -> list[str]:
    """Open-loop pump subjects. AID already delivers the corrective bolus."""
    del root
    return [
        r["key"] for r in _metabonet_manifest()
        if r["usable"] and r["modality"] == "SAP" and r["ncarb"] >= 10
    ]


def _metabonet_part(source: str, sid: str) -> Path:
    return METABONET_CACHE / f"source={source}" / f"sid={sid}"


def _read_metabonet_split(source: str, sid: str, split: str) -> pd.DataFrame:
    part = _metabonet_part(source, sid)
    files = sorted(part.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(part)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.loc[df["split"] == split].sort_values("ts")
    return df


def _metabonet_key(patient_id: str) -> tuple[str, str]:
    source, sid = str(patient_id).split("|", 1)
    return source, sid


def _segment_frame(ts: np.ndarray, cgm: np.ndarray, bolus: np.ndarray, carb: np.ndarray, ph: int) -> pd.DataFrame | None:
    if ts.size < ph + 8:
        return None
    t0, t1 = int(ts[0]), int(ts[-1])
    n = int(round((t1 - t0) / _STEP_NS)) + 1
    if n < ph + 8 or n > int(ts.size) * 3:
        return None
    grid = t0 + np.arange(n, dtype=np.int64) * _STEP_NS
    g = np.full(n, np.nan)
    b = np.zeros(n, float)
    c = np.zeros(n, float)
    j = np.rint((ts.astype(np.int64) - t0) / _STEP_NS).astype(int)
    ok = (j >= 0) & (j < n)
    g[j[ok]] = cgm[ok]
    np.add.at(b, j[ok], bolus[ok])
    np.add.at(c, j[ok], carb[ok])
    g = _interp_short_gaps(g, _GAP_STEPS)
    insulin = _causal_horizon_iob(grid, grid[b > 0], b[b > 0], int(ph))
    cho = _causal_horizon_cho(grid, grid[c > 0], c[c > 0], int(ph))
    return pd.DataFrame({"CGM": g, "insulin": insulin, "CHO": cho})


def _frame_from_metabonet_split(df: pd.DataFrame, *, ph_steps: int, lookback: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["CGM", "insulin", "CHO"])
    ts = df["ts"].to_numpy(np.int64)
    cgm = df["cgm"].to_numpy(float)
    bolus = df["bolus"].to_numpy(float)
    carb = df["carb"].to_numpy(float)
    cuts = np.flatnonzero(np.diff(ts) > 35 * 60 * 1_000_000_000) + 1
    parts = []
    bridge = None
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, ts.size]):
        seg = _segment_frame(ts[a:b], cgm[a:b], bolus[a:b], carb[a:b], int(ph_steps))
        if seg is None:
            continue
        if parts:
            if bridge is None:
                n = int(lookback) + int(ph_steps) + 2
                bridge = pd.DataFrame({
                    "CGM": np.full(n, np.nan),
                    "insulin": np.zeros(n),
                    "CHO": np.zeros(n),
                })
            parts.append(bridge)
        parts.append(seg)
    if not parts:
        return pd.DataFrame(columns=["CGM", "insulin", "CHO"])
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=None)
def _load_patient_metabonet_cached(patient_id: str, horizon_steps: int, lookback: int) -> PatientData:
    source, sid = _metabonet_key(patient_id)
    ph, lb = int(horizon_steps), int(lookback)
    train_df = _frame_from_metabonet_split(
        _read_metabonet_split(source, sid, "train"), ph_steps=ph, lookback=lb)
    test_df = _frame_from_metabonet_split(
        _read_metabonet_split(source, sid, "test"), ph_steps=ph, lookback=lb)
    if len(train_df) < lb + ph + 30 or len(test_df) < lb + ph + 10:
        raise ValueError(f"MetaboNet {patient_id}: train {len(train_df)} test {len(test_df)}")
    train_df = train_df.copy()
    train_df["CGM"] = _interp_short_gaps(train_df["CGM"].to_numpy(float), _GAP_STEPS)
    labels = FEATURE_COLUMNS
    X_train, y_train, scalers, target_scaler = _scaled_matrices(train_df, labels, lb, ph)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(
        test_df, labels, lb, ph, feature_scalers=scalers, target_scaler=target_scaler)
    if len(y_train) < 50 or len(y_test) < 20:
        raise ValueError(f"MetaboNet {patient_id}: rows train {len(y_train)} test {len(y_test)}")
    return PatientData(
        X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers,
        str(patient_id), insulin_idx=lb + (lb - 1), cgm_idx=lb - 1, lookback=lb)


def load_patient_metabonet(root: str, patient_id: str, *, horizon_steps: int = HORIZON_STEPS, lookback: int | None = None) -> PatientData:
    del root
    lb = int(LOOKBACK if lookback is None else lookback)
    return _load_patient_metabonet_cached(str(patient_id), int(horizon_steps), lb)


# Full records, one model per person. Ignores MetaboNet's known-patient flag
# and its preset train/test cut. Each subject's train and test rows are joined
# and cut chronologically at TRAIN_FRAC, as for AZT1D in the first submission.
METABONET_FULL = _DATA / "metabonet_full"
METABONET_FULL_INDEX = METABONET_FULL / "index.json"


def _full_part(source: str, sid: str) -> Path:
    return METABONET_FULL / f"source={source}" / f"sid={sid}" / "raw.parquet"


def build_metabonet_full_index() -> list[dict]:
    import pyarrow.dataset as ds
    METABONET_FULL.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in (METABONET_TRAIN, METABONET_TEST):
        table = ds.dataset(str(path), format="parquet").to_table(columns=["source_file", "id"])
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows = [
        {"source": str(s), "id": str(i), "key": f"{s}|{i}"}
        for s, i in zip(df["source_file"], df["id"])
    ]
    rows.sort(key=lambda r: (r["source"], r["id"]))
    METABONET_FULL_INDEX.write_text(json.dumps(rows))
    return rows


def _metabonet_full_index() -> list[dict]:
    if METABONET_FULL_INDEX.is_file():
        return json.loads(METABONET_FULL_INDEX.read_text())
    return build_metabonet_full_index()


def list_patients_metabonet_full(root: str = ".") -> list[str]:
    del root
    return [r["key"] for r in _metabonet_full_index()]


def _read_metabonet_full_raw(source: str, sid: str) -> pd.DataFrame:
    cache = _full_part(source, sid)
    if cache.is_file():
        return pd.read_parquet(cache)
    import pyarrow.dataset as ds
    frames = []
    for path in (METABONET_TRAIN, METABONET_TEST):
        table = ds.dataset(str(path), format="parquet").to_table(
            columns=["date", "CGM", "bolus", "basal", "carbs", "weight", "insulin_delivery_modality"],
            filter=(ds.field("source_file") == source) & (ds.field("id") == sid),
        )
        if table.num_rows:
            frames.append(table.to_pandas())
    if not frames:
        raise KeyError(f"{source}|{sid}")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["date"]).astype("int64")
    df["bolus"] = df["bolus"].fillna(0.0)
    df["basal"] = df["basal"].fillna(0.0)
    df["carb"] = df["carbs"].fillna(0.0)
    df = df.rename(columns={"CGM": "cgm", "insulin_delivery_modality": "modality"})
    df = df.sort_values("ts")
    keep = ["ts", "cgm", "bolus", "basal", "carb", "weight", "modality"]
    df = df[keep]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_patient_metabonet_full(
    root: str, patient_id: str, *, horizon_steps: int = HORIZON_STEPS, lookback: int | None = None,
) -> PatientData:
    """One person, full record, chronological 80/20. No MetaboNet preset split."""
    del root
    source, sid = _metabonet_key(patient_id)
    ph = int(horizon_steps)
    lb = int(LOOKBACK if lookback is None else lookback)
    raw = _read_metabonet_full_raw(source, sid)
    frame = _frame_from_metabonet_split(raw, ph_steps=ph, lookback=lb)
    n = len(frame)
    if n < lb + ph + 80:
        raise ValueError(f"{patient_id}: series {n}")
    cut = max(int(n * TRAIN_FRAC), lb + ph + 50)
    cut = min(cut, n - (lb + ph + 20))
    train_df, test_df = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    # Recent history only. The test tail stays the last 20% of the full record.
    # 8,000 rows is the same cap the MLP arms use (~28 days at 5 min).
    keep = 8000 + ph + lb
    if len(train_df) > keep:
        train_df = train_df.iloc[-keep:].copy()
    train_df["CGM"] = _interp_short_gaps(train_df["CGM"].to_numpy(float), _GAP_STEPS)
    labels = FEATURE_COLUMNS
    X_train, y_train, scalers, target_scaler = _scaled_matrices(train_df, labels, lb, ph)
    X_test, y_test, _, cgm_scaler = _scaled_matrices(
        test_df, labels, lb, ph, feature_scalers=scalers, target_scaler=target_scaler)
    if len(y_train) < 50 or len(y_test) < 20:
        raise ValueError(f"{patient_id}: rows train {len(y_train)} test {len(y_test)}")
    return PatientData(
        X_train, y_train, X_test, y_test, tuple(labels), cgm_scaler, scalers,
        str(patient_id), insulin_idx=lb + (lb - 1), cgm_idx=lb - 1, lookback=lb)


def load_patient_meta_metabonet_full(root: str, patient_id: str) -> dict:
    del root
    source, sid = _metabonet_key(patient_id)
    df = _read_metabonet_full_raw(source, sid)
    ts = df["ts"].to_numpy(np.int64)
    days = max((int(ts[-1]) - int(ts[0])) / 1e9 / 86400.0, 1e-6) if ts.size else 1.0
    bolus_per_day = float(df["bolus"].sum()) / days
    basal_hr = df["basal"].to_numpy(float) * 12.0
    ok = (basal_hr > 0) & (basal_hr <= MAX_BASAL_U_HR)
    mean_basal = float(np.median(basal_hr[ok])) if ok.any() else 0.0
    tdd = float(np.clip(mean_basal * 24.0 + bolus_per_day, *TDD_BOUNDS_U))
    w = df.loc[df["weight"] > 0, "weight"]
    weight = float(w.median()) if len(w) else float(DEFAULT_WEIGHT_KG)
    modality = df["modality"].dropna()
    return {
        "weight_kg": weight,
        "u2ss": replay_u2ss_mU_kg_min(mean_basal, weight) if mean_basal > 0 else 0.0,
        "tdd_u": tdd,
        "SI_personal": float(si_from_tdd(tdd)),
        "target": {"SI": float(si_from_tdd(tdd)), "tdd_u": tdd},
        "dataset": "metabonet_full",
        "source": source,
        "modality": None if modality.empty else str(modality.iloc[0]),
    }


def _metabonet_record(patient_id: str) -> dict:
    key = str(patient_id)
    for r in _metabonet_manifest():
        if r["key"] == key:
            return r
    raise KeyError(key)


def load_patient_meta_metabonet(root: str, patient_id: str) -> dict:
    del root
    source, sid = _metabonet_key(patient_id)
    rec = _metabonet_record(patient_id)
    df = _read_metabonet_split(source, sid, "train")
    ts = df["ts"].to_numpy(np.int64)
    days = max((int(ts[-1]) - int(ts[0])) / 1e9 / 86400.0, 1e-6) if ts.size else 1.0
    bolus_per_day = float(df["bolus"].sum()) / days
    basal = df["basal"].to_numpy(float)
    basal_hr = basal * 12.0
    ok = (basal_hr > 0) & (basal_hr <= MAX_BASAL_U_HR)
    mean_basal = float(np.median(basal_hr[ok])) if ok.any() else 0.0
    tdd = float(np.clip(mean_basal * 24.0 + bolus_per_day, *TDD_BOUNDS_U))
    weight = float(rec["weight"]) if rec.get("weight") else float(DEFAULT_WEIGHT_KG)
    return {
        "weight_kg": weight,
        "u2ss": replay_u2ss_mU_kg_min(mean_basal, weight) if mean_basal > 0 else 0.0,
        "tdd_u": tdd,
        "SI_personal": float(si_from_tdd(tdd)),
        "target": {"SI": float(si_from_tdd(tdd)), "tdd_u": tdd},
        "dataset": "metabonet",
        "source": source,
        "modality": rec.get("modality"),
    }


def load_glucose_trace_metabonet_full(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    """Full record, last 20% tagged test. Same grid as the known-patient trace."""
    del root
    source, sid = _metabonet_key(patient_id)
    raw = _read_metabonet_full_raw(source, sid)
    cut = int(len(raw) * TRAIN_FRAC)
    frames = []
    for name, part in (("train", raw.iloc[:cut]), ("test", raw.iloc[cut:])):
        if part.empty:
            continue
        frames.append(_metabonet_trace_segments(patient_id, part.reset_index(drop=True), name))
    frames = [f for f in frames if f.height]
    if not frames:
        return pl.DataFrame()
    frame = pl.concat(frames, how="vertical")
    if split is None:
        return frame
    return frame.filter(pl.col(SPLIT_COL) == split)


def _metabonet_trace_segments(patient_id: str, df: pd.DataFrame, name: str) -> pl.DataFrame:
    ts = df["ts"].to_numpy(np.int64)
    if ts.size == 0:
        return pl.DataFrame()
    cgm = df["cgm"].to_numpy(float)
    bolus = df["bolus"].to_numpy(float)
    carb = df["carb"].to_numpy(float)
    basal = np.clip(df["basal"].to_numpy(float) * 12.0, 0.0, MAX_BASAL_U_HR)
    parts = []
    cuts = np.flatnonzero(np.diff(ts) > 35 * 60 * 1_000_000_000) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, ts.size]):
        t0, t1 = int(ts[a]), int(ts[b - 1])
        n = int(round((t1 - t0) / _STEP_NS)) + 1
        if n < 8 or n > int(b - a) * 3:
            continue
        grid = t0 + np.arange(n, dtype=np.int64) * _STEP_NS
        g = np.full(n, np.nan)
        bv = np.zeros(n, float)
        cv = np.zeros(n, float)
        ba = np.zeros(n, float)
        j = np.rint((ts[a:b].astype(np.int64) - t0) / _STEP_NS).astype(int)
        m = (j >= 0) & (j < n)
        g[j[m]] = cgm[a:b][m]
        np.add.at(bv, j[m], bolus[a:b][m])
        np.add.at(cv, j[m], carb[a:b][m])
        ba[j[m]] = basal[a:b][m]
        g = _interp_short_gaps(g, _GAP_STEPS)
        parts.append(pl.DataFrame({
            PATIENT_COL: [str(patient_id)] * n,
            TIMESTAMP_COL: pd.to_datetime(grid).to_list(),
            "glucose": g.astype(float),
            "insulin": np.zeros(n, float),
            "carbs": np.zeros(n, float),
            "meal_carbs": cv.astype(float),
            "bolus_dose": bv.astype(float),
            "basal_rate": ba.astype(float),
            TARGET: np.zeros(n, float),
            SPLIT_COL: [name] * n,
        }))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical")


def load_glucose_trace_metabonet(root: str, patient_id: str, *, split: str | None = None) -> pl.DataFrame:
    """5-min trace. Basal in the release is units per 5-min bin; stored here as U/h."""
    del root
    source, sid = _metabonet_key(patient_id)
    parts = []
    for name in ("train", "test"):
        df = _read_metabonet_split(source, sid, name)
        if df.empty:
            continue
        ts = df["ts"].to_numpy(np.int64)
        cgm = df["cgm"].to_numpy(float)
        bolus = df["bolus"].to_numpy(float)
        carb = df["carb"].to_numpy(float)
        basal = np.clip(df["basal"].to_numpy(float) * 12.0, 0.0, MAX_BASAL_U_HR)
        cuts = np.flatnonzero(np.diff(ts) > 35 * 60 * 1_000_000_000) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, ts.size]):
            t0, t1 = int(ts[a]), int(ts[b - 1])
            n = int(round((t1 - t0) / _STEP_NS)) + 1
            if n < 8 or n > int(b - a) * 3:
                continue
            grid = t0 + np.arange(n, dtype=np.int64) * _STEP_NS
            g = np.full(n, np.nan)
            bv = np.zeros(n, float)
            cv = np.zeros(n, float)
            ba = np.zeros(n, float)
            j = np.rint((ts[a:b].astype(np.int64) - t0) / _STEP_NS).astype(int)
            m = (j >= 0) & (j < n)
            g[j[m]] = cgm[a:b][m]
            np.add.at(bv, j[m], bolus[a:b][m])
            np.add.at(cv, j[m], carb[a:b][m])
            ba[j[m]] = basal[a:b][m]
            g = _interp_short_gaps(g, _GAP_STEPS)
            parts.append(pl.DataFrame({
                PATIENT_COL: [str(patient_id)] * n,
                TIMESTAMP_COL: pd.to_datetime(grid).to_list(),
                "glucose": g.astype(float),
                "insulin": np.zeros(n, float),
                "carbs": np.zeros(n, float),
                "meal_carbs": cv.astype(float),
                "bolus_dose": bv.astype(float),
                "basal_rate": ba.astype(float),
                TARGET: np.zeros(n, float),
                SPLIT_COL: [name] * n,
            }))
    if not parts:
        return pl.DataFrame()
    frame = pl.concat(parts, how="vertical")
    if split is None:
        return frame
    return frame.filter(pl.col(SPLIT_COL) == split)


# Default API = Ohio (primary cohort).
list_patients = list_patients_ohio
load_patient = load_patient_ohio
load_patient_meta = load_patient_meta_ohio
