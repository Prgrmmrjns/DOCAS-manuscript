"""ReplayBG DSS: CatBoost baseline vs DOCAS.

Primary: OhioT1DM + AZT1D at PH60 (Prendin window screen + twin validity).
Patient 588 remains the Ohio showcase figure only.
`revision` runs the MetaboNet window screen and the Table 4--6 arms.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

from py_replay_bg import ReplayBG
from py_replay_bg.callbacks import ReplayCallback
from py_replay_bg.data.single_meal_t1d_data import SingleMealT1DData
from py_replay_bg.distributions import Gamma, LogNormal, Normal, Uniform
from py_replay_bg.model.single_meal_t1d import SingleMealT1DModel
from py_replay_bg.utils.load_results import load_results
from py_replay_bg.utils.numba_dicts import to_typed_f64_dict
from preprocessing import (
    CGM_STEP_MIN, REPLAY_CIB_MAX_U, TIMESTAMP_COL, PaperGlucoseModel, bolus_action_fraction,
    horizon_steps, load_glucose_trace, load_glucose_trace_hupa, load_glucose_trace_metabonet,
    list_patients_ohio, list_patients_hupa, list_patients_metabonet_sap,
    paper_feat_arrays, seg_to_replaybg_df,
)
from core import docas_fit_kw
from forecasters import train_baseline

# Single-meal twinning priors of the ReplayBG example (examples/twin_single_meal.py).
UNKNOWN_PARAMETERS_PRIOR = {
    'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
    'SG': {'prior': LogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': .5},
    'p2': {'prior': Normal(mu=0.11, sigma=0.004), 'min': 0, 'max': .5},
    'f': {'prior': Normal(mu=0.8, sigma=0.05), 'min': 0, 'max': 1},
    'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
    'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
    'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
    'SI': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
    'kabs': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
    'beta': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
}

RUN_REPLAY = True
PH = 60
SHOWCASE_PATIENT = "588"  # Ohio figure only
# OhioT1DM only. AZT1D runs Control-IQ, an automated corrective-bolus system, so
# a corrective-bolus DSS cannot be evaluated there: the pump has already issued
# the intervention under test (median 35.5% of its bolus events are automatic
# corrections) and its continuous basal modulation leaves bolus SI unidentifiable
# in an 8 h window. AZT1D remains a prediction / alignment cohort.
REPLAY_DATASETS = ("ohio_t1dm",)
REPLAY_HORIZONS = (PH,)
PID = SHOWCASE_PATIENT
BACKBONES = ("catboost",)
TABLE2_PATIENT = SHOWCASE_PATIENT  # alias for visuals.py
TWIN_N_STARTS = 24
TWIN_N_JOBS = 4
# Prendin CIB: trigger 180, target 120, 0–5 U in 0.5 U steps, quadratic dose penalty.
# Re-trigger gap 120 min matches their Fig. 2 (second CIB ~2 h after the first).
CIB_TARGET_MGDL, HYPER_TRIGGER_MGDL = 120.0, 180.0
CIB_GAP_MIN = 120.0
CIB_DOSE_PENALTY = 100.0  # shared across baseline / DOCAS / τ-wrapper
PENALTY_SWEEP = (10.0, 30.0, 100.0, 300.0)  # report TIR sensitivity
HYPO_FLOOR_MGDL = 70.0
# The floor + 2×-SI margin is ours, not Prendin's. Set False for a Prendin-faithful
# sensitivity that scores only the quadratic cost (see run_unguarded).
APPLY_HYPO_GUARD = True
# Prendin et al. score each candidate dose on the prediction at the horizon, and
# that is what we report. Only ~20% of a rapid-acting bolus has acted at 60 min,
# so the horizon objective over-doses relative to the eventual nadir; judging
# each arm on the nadir its own audit curve implies at NADIR_MIN is kept as a
# sensitivity (PROJECT_NADIR = True).
PROJECT_NADIR, NADIR_MIN = False, 180.0
TWIN_RMSE_PRIMARY_MGDL = 25.0
# Per-patient τ is off: SI is not identifiable from these windows. Twinning
# patient 588's training windows returns screened SI estimates spanning
# 1.4e-5 to 1.8e-3 (126x) with a median at 0.19x the population mode, while the
# test-window twins run to 1.3x it. Within one postprandial window meal
# absorption and insulin action are confounded, so the fit is good at almost any
# SI. patient_si() records this; the reported τ uses the population prior.
PERSONAL_TAU, SI_TRAIN_WINDOWS = False, 12
# The guard cannot know a patient's true sensitivity, so it is conservative
# instead of accurate: a candidate must stay above the floor even if the patient
# responds SAFETY_SI_FACTOR times more strongly than τ assumes.
SAFETY_SI_FACTOR = 2.0
# Plant-validity screen, evaluated before any controller runs: a 1 U test bolus
# on the identified twin must produce a correction factor inside the clinically
# plausible 1800-rule range. Windows whose twinning data contain no insulin
# leave SI unidentified, and those twins can crash glucose by >200 mg/dL per U.
TWIN_PROBE_U, TWIN_PROBE_AT_MIN = 1.0, 180
TWIN_MAX_CF_MGDL_PER_U = 100.0
# The same probe also has to leave room for the therapy under test. A corrective
# bolus algorithm searches from 0.5 U upward, so a twin that the 1 U probe alone
# drives into hypoglycaemia has no admissible dose at all: such a window scores
# caution rather than dose quality. Closed-loop cohorts produce these windows
# because the pump's own basal modulation has already corrected the excursion
# with insulin that no bolus log records.
TWIN_MIN_PROBE_G_MGDL = 70.0
# A participant whose windows never trigger the DSS never invokes a controller
# and contributes only ties to every contrast; drop the participant, not the
# individual window, so no window is selected on its outcome.
REQUIRE_TRIGGERING_PATIENT = True
YTS_MIN, REPLAY_WINDOW_H = 5, 8
RESULTS = ROOT / "results"
DATA_ROOT = ROOT
REPLAY_RESULTS = RESULTS / "replay"
REPLAY_WORKSPACE = REPLAY_RESULTS / "workspace"
GRID = np.arange(0.0, REPLAY_CIB_MAX_U + 0.5, 0.5)  # half-unit increments (pen-compatible)
_RBG, _TWIN, _CTRL = None, {}, {}

_SCOPE = {"ohio_t1dm": "", "hupa": "_hupa", "metabonet": "_metabonet"}


def _study():
    import main as m
    return m


def _patients(dataset: str) -> tuple[str, ...]:
    if dataset == "ohio_t1dm":
        return tuple(list_patients_ohio(str(ROOT)))
    if dataset == "hupa":
        return tuple(list_patients_hupa(str(ROOT)))
    if dataset == "metabonet":
        return tuple(list_patients_metabonet_sap(str(ROOT)))
    raise ValueError(dataset)


def replay_cohort_path(scope: str = "") -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"cohort{scope}.json"


def replay_horizon_payload(cohort: dict, horizon_min: int) -> dict:
    return cohort["horizons"][str(horizon_min)]


def _load_trace(pid: str, *, dataset: str = "ohio_t1dm", split: str | None = "test"):
    if dataset == "ohio_t1dm":
        return load_glucose_trace(str(DATA_ROOT), pid, split=split)
    if dataset == "hupa":
        return load_glucose_trace_hupa(str(DATA_ROOT), pid, split=split)
    if dataset == "metabonet":
        return load_glucose_trace_metabonet(str(DATA_ROOT), pid, split=split)
    if dataset == "metabonet_full":
        from preprocessing import load_glucose_trace_metabonet_full
        return load_glucose_trace_metabonet_full(str(DATA_ROOT), pid, split=split)
    raise ValueError(f"unknown dataset {dataset!r}")


def _json(v):
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating):
        x = float(v)
        return None if not np.isfinite(x) else round(x, 4)
    if isinstance(v, np.ndarray):
        return [_json(x) for x in v.tolist()]
    raise TypeError(type(v))


def _tir_tbr_tar(g):
    g = np.asarray(g, float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"tir": float("nan"), "tbr": float("nan"), "tar": float("nan")}
    return {"tir": 100. * float(np.mean((g >= 70) & (g <= 180))),
            "tbr": 100. * float(np.mean(g < 70)),
            "tar": 100. * float(np.mean(g > 180))}


def _post_gly(g, delay):
    i0 = int(delay) // YTS_MIN
    return {k: round(float(v), 2) for k, v in _tir_tbr_tar(np.asarray(g, float)[i0:]).items()}


def pending_action_ratio(horizon_min, nadir_min=None):
    """Insulin action still to arrive between the horizon and the nadir,
    as a multiple of the action already delivered at the horizon."""
    nadir_min = NADIR_MIN if nadir_min is None else nadir_min
    acted = lambda m: 1.0 - float(bolus_action_fraction(np.array([float(m)]))[0])
    a_ph = acted(horizon_min)
    a_end = 1.0 if not np.isfinite(nadir_min) else acted(nadir_min)
    return (a_end - a_ph) / a_ph


class ForecasterCIB(ReplayCallback):
    """Prendin-style grid search using a tabular forecaster (baseline or DOCAS)."""
    name = "forecaster_cib"

    def __init__(self, model, cols, cho, cgm, delay_min, gap_min=None,
                 *, target=None, trigger=None, dose_penalty=None, horizon_min=None,
                 project_nadir=None, safety_si=None, lookback=1):
        self.model, self.cols = model, cols
        self.cho, self.cgm = np.asarray(cho, float), np.asarray(cgm, float)
        self.delay_min = int(delay_min)
        self.gap_min = int(CIB_GAP_MIN if gap_min is None else gap_min)
        self.horizon_min = int(PH if horizon_min is None else horizon_min)
        self.lookback = int(lookback)
        self._cgm_hist = {}
        self._f_h = float(bolus_action_fraction(np.array([self.horizon_min]))[0])
        self.target = float(CIB_TARGET_MGDL if target is None else target)
        self.trigger = float(HYPER_TRIGGER_MGDL if trigger is None else trigger)
        self.dose_penalty = float(CIB_DOSE_PENALTY if dose_penalty is None else dose_penalty)
        self.project_nadir = PROJECT_NADIR if project_nadir is None else bool(project_nadir)
        # The cost function is scored where `project_nadir` says, but the
        # hypoglycaemia guard always looks ahead to the nadir: a bolus chosen at
        # the horizon is still acting long after it.
        self._guard_pending = pending_action_ratio(self.horizon_min)
        self._pending = self._guard_pending if self.project_nadir else 0.0
        self.safety_si = float(SAFETY_SI_FACTOR if safety_si is None else safety_si)
        self.last_k, self.decisions, self.doses = None, [], []
        self._events, self._bolus_i = [], None

    def _iob_at(self, k, events):
        if not events:
            return 0.0
        t = np.asarray([e[0] for e in events], float)
        u = np.asarray([e[1] for e in events], float)
        ages_h = (k - 1 + self.horizon_min) - t
        return float(np.dot(u, bolus_action_fraction(ages_h)))

    def _lag_row(self, slot, g, iob):
        """Oldest-to-newest CGM, horizon-IOB, and CHO. The last column of each block is now."""
        lb = self.lookback
        slots = [slot - (lb - 1 - i) for i in range(lb)]
        cgm = np.empty(lb, float)
        ins = np.empty(lb, float)
        cho = np.empty(lb, float)
        for i, s in enumerate(slots):
            if s == slot:
                cgm[i] = g
                ins[i] = iob
            elif s in self._cgm_hist:
                cgm[i] = self._cgm_hist[s]
            elif 0 <= s < len(self.cgm):
                cgm[i] = float(self.cgm[s])
            else:
                cgm[i] = g
            if s != slot:
                k_s = int(s) * YTS_MIN + (YTS_MIN - 1)
                past = [
                    e for e in self._events
                    if e[0] < k_s or (e[0] == k_s and len(e) < 3)
                ]
                ins[i] = self._iob_at(k_s, past)
            cho[i] = float(self.cho[s]) if 0 <= s < len(self.cho) else 0.0
        return np.concatenate([cgm, ins, cho])

    def action(self, ctx):
        k = int(ctx.k)
        if (k + 1) % YTS_MIN == 0:
            self._cgm_hist[k // YTS_MIN] = float(ctx.measurement)
        if (k + 1) % YTS_MIN or k < self.delay_min:
            return
        g = float(ctx.measurement)
        if g <= self.trigger or (self.last_k is not None and k - self.last_k < self.gap_min):
            return
        bw = float(self.rbg_data.body_weight)
        if self._bolus_i is None:
            self._bolus_i = next(i for i, n in ctx.data_to_input.items() if n == "bolus")
            hist = np.asarray(ctx.input_history[:k, self._bolus_i], float) * (bw / 1000.0)
            self._events.extend((int(i), float(hist[i])) for i in np.flatnonzero(hist > 0))
        iob = self._iob_at(k, self._events)
        slot = min(k // YTS_MIN, len(self.cho) - 1)
        ji = self.cols.index("insulin")
        if self.lookback <= 1:
            row = np.array([g, iob, float(self.cho[slot])], float)
        else:
            row = self._lag_row(slot, g, iob)
        grid = GRID
        X = np.tile(row, (len(grid), 1))
        X[:, ji] = iob + grid * self._f_h
        curve_g = (g + np.asarray(self.model.predict_delta(X), float).ravel()
                   if hasattr(self.model, "predict_delta")
                   else np.asarray(self.model.predict(X), float).ravel())
        # _pending is 0 under the reported horizon objective. When the nadir
        # sensitivity is enabled it charges each candidate with the insulin
        # action still to arrive after the horizon.
        judged = curve_g - (curve_g[0] - curve_g) * self._pending
        nadir = curve_g - (curve_g[0] - curve_g) * self._guard_pending
        # Worst-case nadir: the same curve read as if the patient responded
        # `safety_si` times more strongly than the target assumes.
        nadir_k = curve_g[0] - self.safety_si * (1.0 + self._guard_pending) * (curve_g[0] - curve_g)
        obj = (judged - self.target) ** 2 + self.dose_penalty * grid ** 2
        # Constrain the search rather than veto the winner: a candidate is
        # admissible only if the glucose it commits to, once the bolus has
        # finished acting, stays above the floor. Dropping the argmin without
        # this would fall back to no bolus at all instead of a smaller safe one.
        # APPLY_HYPO_GUARD=False recovers Prendin's unconstrained cost argmin.
        if APPLY_HYPO_GUARD:
            admissible = np.asarray(nadir_k, float) >= HYPO_FLOOR_MGDL
            admissible[0] = True
            best = int(np.argmin(np.where(admissible, obj, np.inf)))
            dose = (float(grid[best]) if best and obj[best] < obj[0]
                    and dose_ok(judged, best, nadir_k) else 0.0)
        else:
            admissible = np.ones(len(grid), dtype=bool)
            best = int(np.argmin(obj))
            dose = float(grid[best]) if best and obj[best] < obj[0] else 0.0
        self.decisions.append({
            "time_min": k, "observed_cgm": round(float(self.cgm[slot]), 1),
            "simulated_cgm": round(g, 1), "iob_u": round(iob, 2),
            "delivered_dose_u": round(dose, 2), "raw_dose_u": round(dose, 2),
            "pred_at_dose": round(float(curve_g[best]), 1), "clipped": False,
            "grid_u": [round(float(x), 2) for x in grid],
            "iob_probe_u": [round(float(iob + x * self._f_h), 2) for x in grid],
            "future_cgm_curve": [None if not np.isfinite(float(x)) else round(float(x), 1) for x in curve_g],
            "judged_curve": [round(float(x), 1) for x in judged],
            "nadir_curve": [round(float(x), 1) for x in nadir],
            "nadir_worst_case_curve": [round(float(x), 1) for x in nadir_k],
            "n_admissible": int(admissible.sum()),
            "cib": dict(target=self.target, trigger=self.trigger,
                        dose_penalty=self.dose_penalty,
                        project_nadir=self.project_nadir,
                        hypo_floor_mgdl=HYPO_FLOOR_MGDL if APPLY_HYPO_GUARD else None,
                        guard_nadir_min=NADIR_MIN,
                        safety_si_factor=self.safety_si if APPLY_HYPO_GUARD else None,
                        apply_hypo_guard=bool(APPLY_HYPO_GUARD)),
        })
        if dose < 0.5:
            return
        ctx.add_input("bolus", dose * (1000.0 / bw))
        self.last_k = k
        self.doses.append((k, dose))
        self._events.append((k, dose, True))


class TestBolus(ReplayCallback):
    """Deliver one fixed bolus; used to read the twin's own dose response."""
    name = "test_bolus"

    def __init__(self, at_min=TWIN_PROBE_AT_MIN, units=TWIN_PROBE_U):
        self.at_min, self.units, self.done = int(at_min), float(units), False
        self.decisions, self.doses = [], []

    def action(self, ctx):
        if self.done or int(ctx.k) < self.at_min:
            return
        bw = float(self.rbg_data.body_weight)
        ctx.add_input("bolus", self.units * (1000.0 / bw))
        self.done = True
        self.doses.append((int(ctx.k), self.units))


def dose_ok(curve, best, guard=None):
    """Require a lowering effect on the curve the dose was chosen against, and
    no predicted hypoglycaemia on `guard`, the glucose the bolus commits to once
    it has finished acting."""
    if len(curve) <= 1:
        return False
    c = np.asarray(curve, float)
    if float(np.min(c[1:])) >= float(c[0]):
        return False
    g = c if guard is None else np.asarray(guard, float)
    if float(g[best]) < float(HYPO_FLOOR_MGDL):
        return False
    return True


def _rbg():
    global _RBG
    if _RBG is None:
        REPLAY_WORKSPACE.mkdir(parents=True, exist_ok=True)
        _RBG = ReplayBG(plot_mode=False, verbose=False)  # ts=1, seed=1 defaults
    return _RBG


_REPLAY_DATA = {}  # name → (df, data) for reuse across no-DSS / baseline / DOCAS


def _twin(seg, bw, name, u2ss):
    if name in _TWIN:
        return _TWIN[name]
    path = REPLAY_WORKSPACE / f"twin_{name}.pkl"
    if path.is_file():
        try:
            twin = load_results(str(REPLAY_WORKSPACE), name, prefix="twin")
            if "theta" in twin:
                _TWIN[name] = twin
                return twin
        except Exception:
            pass
    df = seg_to_replaybg_df(seg, window_h=seg.height * YTS_MIN / 60, yts_min=YTS_MIN)
    data = SingleMealT1DData(data=df, body_weight=float(bw), environment=_rbg().environment)
    if u2ss is not None:
        data.u2ss = float(u2ss)
    model = SingleMealT1DModel(u2ss=data.u2ss, tsteps=data.tsteps)
    twin = _rbg().twin(
        rbg_data=data, model=model,
        unknown_parameters_prior=UNKNOWN_PARAMETERS_PRIOR,
        parallelize=TWIN_N_JOBS > 1, n_jobs=max(1, int(TWIN_N_JOBS)),
        n_starts=int(TWIN_N_STARTS),
        path=str(REPLAY_WORKSPACE), save_name=name,
    )
    _TWIN[name] = twin
    _REPLAY_DATA[name] = (df, data)
    return twin


def _replay_data(seg, bw, name, u2ss):
    if name in _REPLAY_DATA:
        return _REPLAY_DATA[name]
    df = seg_to_replaybg_df(seg, window_h=seg.height * YTS_MIN / 60, yts_min=YTS_MIN)
    data = SingleMealT1DData(data=df, body_weight=float(bw), environment=_rbg().environment)
    if u2ss is not None:
        data.u2ss = float(u2ss)
    _REPLAY_DATA[name] = (df, data)
    return df, data


def _replay(seg, bw, name, u2ss, suffix, cb=None):
    twin = _twin(seg, bw, name, u2ss)
    _, data = _replay_data(seg, bw, name, u2ss)
    model = SingleMealT1DModel(
        u2ss=data.u2ss, tsteps=data.tsteps,
        theta0=to_typed_f64_dict(dict(twin["theta"])))
    out = _rbg().replay(rbg_data=data, model=model, callbacks=[cb] if cb else None)
    cgm = np.asarray(out["output"], float).ravel()
    n = int(seg.height)
    # ReplayBG is 1-min; align to the 5-min segment length (off-by-one on some windows).
    if len(cgm) == n:
        pass
    elif len(cgm) >= n * YTS_MIN:
        cgm = cgm[::YTS_MIN][:n]
    elif len(cgm) > n:
        idx = np.linspace(0, len(cgm) - 1, n).round().astype(int)
        cgm = cgm[idx]
    else:
        pad = np.full(n, float(cgm[-1]) if len(cgm) else np.nan)
        pad[:len(cgm)] = cgm
        cgm = pad
    return cgm, (cb.decisions if cb else []), (cb.doses if cb else [])


def _windows(trace):
    """Postprandial 8 h windows: no later CHO and no corrective bolus (Prendin)."""
    trace = trace.sort(TIMESTAMP_COL)
    cho = "meal_carbs" if "meal_carbs" in trace.columns else "cho"
    steps, lead = int(REPLAY_WINDOW_H * 60 / CGM_STEP_MIN), int(30 / CGM_STEP_MIN)
    cho_v = np.asarray(trace[cho].to_numpy(), float)
    bolus = np.asarray(trace["bolus_dose"].to_numpy(), float)
    out = []
    for meal in np.where(cho_v > 0)[0]:
        meal, end = int(meal), min(int(meal) + steps, trace.height)
        if end - meal < steps:
            continue
        if (cho_v[meal + 1:end] > 0).any() or (bolus[meal + 1:end] > 0).any():
            continue
        pre = np.flatnonzero(bolus[max(0, meal - lead):meal] > 0)
        start = max(0, meal - lead) + int(pre[0]) if len(pre) else meal
        seg = trace.slice(start, end - start)
        out.append({"trace": seg, "source_start": start,
                    "meal_offset_min": (meal - start) * CGM_STEP_MIN,
                    "observed_tar": _tir_tbr_tar(seg["glucose"].to_numpy())["tar"]})
    out.sort(key=lambda w: (-w["observed_tar"], w["source_start"]))
    return [{**w, "source_index": i} for i, w in enumerate(out)]


def _select_typical_wins(wins, *, n=4, tar_lo=25.0, tar_hi=80.0):
    """Prefer moderate-to-high TAR postprandial windows in chronological order.

    Band matches the OhioT1DM primary-window regime (room for DSS improvement
    without selecting only extreme unbolused excursions). Falls back to the
    earliest finite-TAR windows when fewer than ``n`` lie in ``[tar_lo, tar_hi]``.
    """
    if not wins:
        return []
    by_time = sorted(
        (w for w in wins if np.isfinite(float(w["observed_tar"]))),
        key=lambda w: int(w["source_start"]))
    mid = [w for w in by_time if tar_lo <= float(w["observed_tar"]) <= tar_hi]
    if len(mid) >= n:
        pick = mid[:n]
    else:
        # Fill shortfall from the highest-TAR remaining windows, not the
        # earliest (which are often near-euglycaemic).
        mid_starts = {int(w["source_start"]) for w in mid}
        rest = sorted(
            (w for w in by_time if int(w["source_start"]) not in mid_starts),
            key=lambda w: -float(w["observed_tar"]))
        pick = mid + rest[: max(0, n - len(mid))]
        pick = sorted(pick, key=lambda w: int(w["source_start"]))
    return [{**w, "source_index": i} for i, w in enumerate(pick)]


def patient_si_path() -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / "patient_si.json"


def patient_si(pid, *, dataset: str = "ohio_t1dm", refresh: bool = False) -> dict:
    """Insulin sensitivity identified on the patient's own *training* windows.

    The replayed windows are test data, so twinning the training set gives a
    per-patient SI without letting the target see the windows it is scored on.
    SI is weakly identified in any single 8 h window, so the estimate is the
    median over windows whose twin reproduces the recorded glucose and whose
    correction factor is clinically plausible. Falls back to the modal prior
    when no window qualifies.
    """
    cache = json.loads(patient_si_path().read_text()) if patient_si_path().is_file() else {}
    key = f"{dataset}:{pid}"
    if not refresh and key in cache:
        return cache[key]
    st = _study()
    modal = float(st.SI_MODE)
    meta = st.load_patient_meta(str(DATA_ROOT), str(pid))
    bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
    wins = _windows(_load_trace(pid, dataset=dataset, split="train"))[:SI_TRAIN_WINDOWS]
    per_window = []
    for w in wins:
        seg = w["trace"]
        name = _twin_name(pid, "si", w["source_start"], dataset=dataset)
        try:
            twin = _twin(seg, bw, name, u2ss)
            fit, _, _ = _replay(seg, bw, name, u2ss, "_sifit")
            probe, _, _ = _replay(seg, bw, name, u2ss, "_siprobe", TestBolus())
        except Exception:
            continue
        observed = np.asarray(seg["glucose"].to_numpy(), float)[:len(fit)]
        rmse = float(np.sqrt(np.mean((fit[:len(observed)] - observed) ** 2)))
        i0 = TWIN_PROBE_AT_MIN // YTS_MIN
        cf = float(np.max(fit[i0:] - probe[i0:]) / TWIN_PROBE_U)
        si = float(np.asarray(twin["theta"]["SI"], float).ravel()[0])
        per_window.append({"source_start": int(w["source_start"]), "si": si,
                           "twin_rmse": round(rmse, 2), "cf_mgdl_per_u": round(cf, 2),
                           "ok": bool(rmse <= TWIN_RMSE_PRIMARY_MGDL
                                      and np.isfinite(cf) and cf <= TWIN_MAX_CF_MGDL_PER_U)})
    good = [r["si"] for r in per_window if r["ok"]]
    si_hat = float(np.median(good)) if good else modal
    out = {"patient_id": str(pid), "dataset": dataset, "SI": si_hat,
           "SI_ratio": si_hat / modal if modal > 0 else float("inf"),
           "n_windows": len(per_window), "n_ok": len(good),
           "fallback_modal": not good, "per_window": per_window}
    cache[key] = out
    patient_si_path().write_text(json.dumps(cache, separators=(",", ":"), default=_json))
    print(f"    SI {pid}: {si_hat:.5g} (x{out['SI_ratio']:.2f} modal) "
          f"from {len(good)}/{len(per_window)} training windows", flush=True)
    return out


def _controllers(pid, backbone="catboost", *, dataset: str = "ohio_t1dm",
                 clinical_tau: bool = False):
    key = (str(dataset), str(pid), str(backbone), bool(clinical_tau))
    if key in _CTRL:
        return _CTRL[key]
    st = _study()
    st.set_dataset(dataset)
    data = st.load_patient(str(DATA_ROOT), pid, horizon_steps=horizon_steps(PH))
    st.DOCAS.SPAN = span = float(st.insulin_span_u(data, horizon_min=PH))
    cfg = dict(st.STUDY_CFG)
    if clinical_tau:
        from ablation import clinical_cfg
        tdd = st.load_patient_meta(str(DATA_ROOT), pid).get("tdd_u")
        if not tdd:
            raise ValueError(f"no TDD for clinical τ ({pid})")
        cfg = clinical_cfg(float(tdd), cfg=cfg)
    elif PERSONAL_TAU:
        cfg["SI"] = float(patient_si(pid, dataset=dataset)["SI"])
    train = lambda X, y, model_kwargs=None, n_factual=None, **_: train_baseline(
        X, y, model_kwargs=model_kwargs, n_factual=n_factual)
    baseline = train_baseline(data.X_train, data.y_train)
    old = st.DOCAS.train_fn
    st.DOCAS.train_fn = train
    try:
        fit_kw, drop, u_ref, _, _ = docas_fit_kw(data, span, PH, cfg=cfg)
        result = st.DOCAS(**fit_kw).fit(
            data.X_train, data.y_train, baseline_model=baseline)
        docas = result["model"]
        predicts_level = bool(result.get("predicts_level", False))
    finally:
        st.DOCAS.train_fn = old
    # τ-wrapper controller (insulin-removed trunk + post-hoc τ)
    from ablation import TauWrapForecaster
    from core import tau_from_cfg
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    target, drop_w, u_ref_w, mode = tau_from_cfg(PH, span, cfg)
    wrap_model = TauWrapForecaster().fit(
        data.X_train, data.y_train, target_R=target, drop=drop_w, u_ref=u_ref_w,
        ji=ji, gi=gi, cgm_s=data.train_scalers[gi], tgt_s=data.cgm_scaler,
        tau_mode=mode, level_ref=str(cfg.get("level_ref", "current")),
        span=float(span))
    wrap_model._a_s = data.train_scalers[ji]

    def wrap(m, level=False):
        return PaperGlucoseModel(
            m, data.train_scalers, data.cgm_scaler,
            predicts_level=level, level_scaler=data.train_scalers[gi])

    # Last element describes the τ in use: correction factor for the pump target,
    # insulin sensitivity otherwise.
    tau_id = float(cfg["cf_mgdl_per_u"]) if clinical_tau else float(cfg["SI"])
    _CTRL[key] = (
        wrap(baseline, False), wrap(docas, predicts_level), wrap(wrap_model, False),
        list(data.feature_names), span, tau_id)
    return _CTRL[key]


def _twin_name(pid, backbone, source_start, *, dataset: str = "ohio_t1dm") -> str:
    """Cache key for an identified twin.

    A twin is fitted to the window's own trace, so the window length is part of
    its identity; the 8 h Prendin protocol keeps the original unsuffixed key so
    its cached fits stay valid. Dataset prefix separates Ohio vs AZT1D caches.
    """
    suffix = "" if REPLAY_WINDOW_H == 8 else f"_w{REPLAY_WINDOW_H:g}h"
    ds = "" if dataset == "ohio_t1dm" else f"{dataset}_"
    return f"paper_v2_{ds}{pid}_{backbone}_start{int(source_start)}{suffix}"


def _twin_cf(row) -> float:
    """Twin correction factor (mg/dL per U) from the 1 U probe; ∞ if missing."""
    cf = row.get("twin_cf_mgdl_per_u")
    return float("inf") if cf is None else float(cf)


def _twin_probe_min(row) -> float:
    """Lowest twin glucose under the 1 U probe; ∞ when the probe is unavailable."""
    g = row.get("twin_min_g_at_probe")
    return float("inf") if g is None else float(g)


def _triggered(row) -> bool:
    """Did any controller arm deliver a correction in this window?"""
    return any(row.get(f"{a}_doses") for a in ("baseline", "docas"))


def _primary_windows(rows, *, require_trigger=None):
    """Twin fit + plant validity + therapy-relevant No-DS hyperglycaemia.

    Prendin et al. kept all eligible postprandial windows for subject 588
    (no later CHO / corrective bolus); we additionally require No-DS TAR > 0,
    a twin RMSE fit, and a twin whose own 1 U response is both inside the
    1800-rule correction-factor range and clear of hypoglycaemia. Both probe
    screens are read off the plant before any controller runs and therefore
    apply to every arm alike.

    Participants whose simulated windows never cross the trigger threshold
    never invoke any controller, so they carry no information about the
    comparison and are dropped as a unit (see REQUIRE_TRIGGERING_PATIENT).
    """
    ok = [r for r in rows
          if float(r["twin_rmse"]) <= TWIN_RMSE_PRIMARY_MGDL
          and float(r["no_dss"]["tar"]) > 0.0
          and _twin_cf(r) <= TWIN_MAX_CF_MGDL_PER_U
          and _twin_probe_min(r) >= TWIN_MIN_PROBE_G_MGDL]
    if REQUIRE_TRIGGERING_PATIENT if require_trigger is None else require_trigger:
        live = {str(r["patient_id"]) for r in ok if _triggered(r)}
        ok = [r for r in ok if str(r["patient_id"]) in live]
    return ok


def _median_iqr(values) -> dict:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
    }


def _summarize(rows, *, suffix=""):
    """Cohort summary. `suffix="_post"` scores only from the first possible
    decision onward, the stretch a DSS can actually influence."""
    ms = _study().mean_std
    s = {}
    for arm in ("no_dss", "baseline", "docas", "tauwrap"):
        key = f"{arm}{suffix}"
        if arm == "no_dss":
            # Stored per-window metrics, not the rounded plotting trajectory, so
            # every arm's glycaemic summary comes off the same simulated values.
            mets = [r[key] for r in rows]
            doses, nbol = [0.0] * len(rows), [0] * len(rows)
        else:
            if not rows or key not in rows[0]:
                continue
            mets = [r[key] for r in rows]
            doses = [sum(d["units"] for d in r.get(f"{arm}_doses", [])) for r in rows]
            nbol = [len(r.get(f"{arm}_doses", [])) for r in rows]
        s[arm] = {k: {**ms([x[k] for x in mets]), **_median_iqr([x[k] for x in mets])}
                  for k in ("tir", "tbr", "tar")}
        s[arm]["insulin_u"] = {**ms(doses), **_median_iqr(doses)}
        s[arm]["n_boluses"] = {**ms(nbol), **_median_iqr(nbol)}
    return s


def _table2_from_windows(horizon_min, windows):
    rows = _primary_windows(windows)
    pids = sorted({str(r["patient_id"]) for r in rows})
    return {"horizon_min": horizon_min, "patient_ids": pids,
            "patient_id": (SHOWCASE_PATIENT if SHOWCASE_PATIENT in pids
                           else (pids[0] if pids else SHOWCASE_PATIENT)),
            "summary": _summarize(rows),
            "paired_docas_vs_baseline": {
                "mean_delta_tir_pp": float(np.mean([r["delta_tir"] for r in rows])) if rows else float("nan"),
                "mean_delta_tbr_pp": float(np.mean([r["delta_tbr"] for r in rows])) if rows else float("nan"),
                "n_windows": len(rows)}}


def _run_backbone(st, pids, backbone: str, *, dataset: str = "ohio_t1dm") -> tuple[list, dict, dict]:
    print(f"  Replay PH{PH} · {backbone} · {dataset} · patients {pids}", flush=True)
    rows = []
    n_eligible = 0
    for pid in pids:
        wins = _windows(_load_trace(pid, dataset=dataset, split="test"))
        # Replace-BG logs hundreds of isolated meals. Keep up to four typical
        # postprandial windows (moderate observed TAR, chronological) so the
        # screen matches the OhioT1DM glycaemic regime rather than extreme
        # unbolused hyperglycaemia.
        if dataset == "metabonet":
            wins = _select_typical_wins(wins, n=4)
        n_eligible += len(wins)
        if not wins:
            print(f"    {pid}: no eligible windows", flush=True)
            continue
        meta = st.load_patient_meta(str(DATA_ROOT), pid)
        bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
        need = []
        for w in wins:
            name = _twin_name(pid, backbone, w["source_start"], dataset=dataset)
            if name not in _TWIN and not (REPLAY_WORKSPACE / f"twin_{name}.pkl").is_file():
                need.append((w["trace"], name))
        if need:
            print(f"  twinning {len(need)} windows (n_starts={TWIN_N_STARTS}, jobs={TWIN_N_JOBS})",
                  flush=True)
            for seg, name in need:
                _twin(seg, bw, name, u2ss)
        baseline, docas, tauwrap, cols, span, _si = _controllers(
            pid, backbone=backbone, dataset=dataset)
        for w in wins:
            seg, name = w["trace"], _twin_name(
                pid, backbone, w["source_start"], dataset=dataset)
            feat = paper_feat_arrays(seg, ph_steps=horizon_steps(PH))
            observed = np.asarray(seg["glucose"].to_numpy(), float)
            delay = 120 + int(w["meal_offset_min"])
            twin = _twin(seg, bw, name, u2ss)
            no_dss, _, _ = _replay(seg, bw, name, u2ss, "_none")
            twin_rmse = float(np.sqrt(np.mean((no_dss - observed) ** 2)))
            th = twin["theta"]
            twin_si = float(np.asarray(th["SI"], float).ravel()[0])
            modal_si = float(_study().SI_MODE)
            twin_si_ratio = twin_si / modal_si if modal_si > 0 else float("inf")
            probe_g, _, _ = _replay(seg, bw, name, u2ss, "_probe1u", TestBolus())
            i0 = TWIN_PROBE_AT_MIN // YTS_MIN
            twin_cf = float(np.max(no_dss[i0:] - probe_g[i0:]) / TWIN_PROBE_U)
            window_bolus_u = float(np.asarray(seg["bolus_dose"].to_numpy(), float).sum())
            mk = lambda mod, pen=None: ForecasterCIB(
                mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                dose_penalty=pen, horizon_min=PH)
            # Matched dose penalty across all arms.
            base_g, base_dec, base_d = _replay(
                seg, bw, name, u2ss, f"_base_{backbone}", mk(baseline, CIB_DOSE_PENALTY))
            doc_g, doc_dec, doc_d = _replay(
                seg, bw, name, u2ss, f"_docas_{backbone}", mk(docas, CIB_DOSE_PENALTY))
            wrap_g, wrap_dec, wrap_d = _replay(
                seg, bw, name, u2ss, f"_wrap_{backbone}", mk(tauwrap, CIB_DOSE_PENALTY))
            bm, dm, wm = _tir_tbr_tar(base_g), _tir_tbr_tar(doc_g), _tir_tbr_tar(wrap_g)
            # Every arm is identical before the first decision, so that stretch
            # dilutes each contrast; score it separately as well.
            i0 = delay // YTS_MIN
            post = {f"{a}_post": {k: round(float(v), 2) for k, v in
                                  _tir_tbr_tar(g[i0:]).items()}
                    for a, g in (("no_dss", no_dss), ("baseline", base_g),
                                 ("docas", doc_g), ("tauwrap", wrap_g))}
            # Penalty sweep (DOCAS and baseline) for robustness reporting.
            sweep = {}
            for pen in PENALTY_SWEEP:
                bg, _, _ = _replay(seg, bw, name, u2ss, f"_base_p{pen}", mk(baseline, pen))
                dg, _, _ = _replay(seg, bw, name, u2ss, f"_docas_p{pen}", mk(docas, pen))
                sweep[str(pen)] = {
                    "baseline_tir": round(_tir_tbr_tar(bg)["tir"], 2),
                    "docas_tir": round(_tir_tbr_tar(dg)["tir"], 2),
                }
            rt = lambda a: [None if not np.isfinite(float(x)) else round(float(x), 1)
                            for x in np.asarray(a, float).ravel()]
            rows.append({
                "patient_id": str(pid), "horizon_min": PH, "backbone": backbone,
                "window_index": int(w["source_index"]), "source_start": int(w["source_start"]),
                "meal_offset_min": int(w["meal_offset_min"]),
                "observed_tar": round(float(w["observed_tar"]), 2),
                "twin_rmse": round(twin_rmse, 2),
                "twin_si": round(twin_si, 8),
                "twin_si_ratio": round(twin_si_ratio, 3),
                "twin_cf_mgdl_per_u": round(twin_cf, 2),
                "twin_min_g_at_probe": round(float(probe_g.min()), 1),
                "window_bolus_u": round(window_bolus_u, 2),
                "no_dss": {k: round(float(v), 2) for k, v in _tir_tbr_tar(no_dss).items()},
                "baseline": {k: round(float(v), 2) for k, v in bm.items()},
                "docas": {k: round(float(v), 2) for k, v in dm.items()},
                "tauwrap": {k: round(float(v), 2) for k, v in wm.items()},
                **post,
                "eval_from_min": int(delay),
                "delta_tir_post": round(post["docas_post"]["tir"]
                                        - post["baseline_post"]["tir"], 2),
                "delta_tir": round(dm["tir"] - bm["tir"], 2),
                "delta_tbr": round(dm["tbr"] - bm["tbr"], 2),
                "delta_tar": round(dm["tar"] - bm["tar"], 2),
                "baseline_doses": [{"time_min": int(t), "units": round(float(u), 2)} for t, u in base_d],
                "docas_doses": [{"time_min": int(t), "units": round(float(u), 2)} for t, u in doc_d],
                "tauwrap_doses": [{"time_min": int(t), "units": round(float(u), 2)} for t, u in wrap_d],
                "baseline_decisions": base_dec, "docas_decisions": doc_dec,
                "tauwrap_decisions": wrap_dec,
                "penalty_sweep": sweep,
                "cib_dose_penalty": float(CIB_DOSE_PENALTY),
                "cib_gap_min": float(CIB_GAP_MIN),
                "trajectories": {"observed": rt(observed), "no_dss": rt(no_dss),
                                 "baseline_dss": rt(base_g), "docas_dss": rt(doc_g),
                                 "tauwrap_dss": rt(wrap_g)},
            })
    twin_ok = [r for r in rows if float(r["twin_rmse"]) <= TWIN_RMSE_PRIMARY_MGDL]
    ok = _primary_windows(rows)
    table2 = _table2_from_windows(PH, rows)
    cohort = {"horizon_min": PH, "backbone": backbone,
              "n_eligible": int(n_eligible),
              "n_windows": len(ok), "n_twin_ok": len(twin_ok),
              "n_patients": len({r["patient_id"] for r in ok}),
              "patient_ids": sorted({r["patient_id"] for r in ok}),
              "summary": _summarize(ok),
              # Same windows, scored only from the first decision onward.
              "summary_post": _summarize(ok, suffix="_post"),
              "summary_post_dosing": _summarize(
                  [r for r in ok if _triggered(r)], suffix="_post"),
              "n_windows_dosing": len([r for r in ok if _triggered(r)]),
              # No hyperglycaemia screen: includes euglycaemic-start windows.
              "summary_all_twin_ok": _summarize(twin_ok),
              "cib_dose_penalty": float(CIB_DOSE_PENALTY),
              "cib_gap_min": float(CIB_GAP_MIN),
              "twin_max_cf_mgdl_per_u": float(TWIN_MAX_CF_MGDL_PER_U),
              "twin_min_probe_g_mgdl": float(TWIN_MIN_PROBE_G_MGDL),
              "penalty_sweep": list(PENALTY_SWEEP),
              # Sensitivity: same summary without the plant-validity screen.
              "summary_no_cf_screen": _summarize(
                  [r for r in twin_ok if float(r["no_dss"]["tar"]) > 0.0]),
              # Sensitivity: keeping participants that never trigger the DSS.
              "summary_all_patients": _summarize(
                  _primary_windows(rows, require_trigger=False)),
              "n_windows_all_patients": len(_primary_windows(rows, require_trigger=False)),
              "requires_triggering_patient": bool(REQUIRE_TRIGGERING_PATIENT),
              "screen": (f"twin_rmse<={TWIN_RMSE_PRIMARY_MGDL:g}, no_dss TAR>0, "
                         f"twin CF≤{TWIN_MAX_CF_MGDL_PER_U:g} mg/dL per U, "
                         f"twin ≥{TWIN_MIN_PROBE_G_MGDL:g} mg/dL under the "
                         f"{TWIN_PROBE_U:g} U probe, "
                         f"participant triggers the DSS at least once")}
    s, n = table2["summary"], table2["paired_docas_vs_baseline"]["n_windows"]
    n_pat = cohort["n_patients"]
    wrap_tir = s.get("tauwrap", {}).get("tir", {}).get("mean", float("nan"))
    print(f"  Table2 PH{PH} {backbone} n_pat={n_pat} n_win={n}/{n_eligible} "
          f"(twin_ok={len(twin_ok)}): "
          f"No DS TIR {s['no_dss']['tir']['mean']:.2f} | "
          f"Baseline TIR {s['baseline']['tir']['mean']:.2f} | "
          f"DOCAS TIR {s['docas']['tir']['mean']:.2f} | "
          f"τ-wrap TIR {wrap_tir:.2f}", flush=True)
    return rows, table2, cohort

def run_replay(*, patient_ids=None, dataset: str = "ohio_t1dm",
               scope: str | None = None) -> dict:
    if dataset not in _SCOPE:
        raise ValueError(f"unknown dataset {dataset!r}")
    st = _study()
    st.set_dataset(dataset)
    pids = tuple(patient_ids) if patient_ids is not None else _patients(dataset)
    scope = _SCOPE[dataset] if scope is None else scope
    by_bb = {}
    for bb in BACKBONES:
        rows, table2, cohort = _run_backbone(st, pids, bb, dataset=dataset)
        by_bb[bb] = dict(windows=rows, table2=table2, cohort=cohort)
    cat = by_bb["catboost"]
    path = replay_cohort_path(scope)
    payload = {
        "dataset": dataset,
        "patient_ids": list(pids),
        "backbones": list(BACKBONES),
        "n_grid_points": int(st.DOCAS.N_GRID_POINTS),
        "showcase_patient": SHOWCASE_PATIENT,
        "horizons": {str(PH): {
            "horizon_min": PH, "patient_ids": list(pids),
            "windows": cat["windows"], "table2": cat["table2"], "cohort": cat["cohort"],
            "by_backbone": by_bb,
        }},
    }
    path.write_text(json.dumps(payload, separators=(",", ":"), default=_json))
    return {"path": str(path), **payload}


DOSE_MATCH_PENALTIES = (3.0, 10.0, 30.0, 60.0, 100.0, 200.0, 400.0, 800.0)


def dose_matched_path(scope: str = "") -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"dose_matched{scope}.json"


def run_dose_matched(*, scope: str = "", dataset: str | None = None) -> dict:
    """Trace time in range against delivered insulin using the shared penalty.

    DOCAS doses more than the observational controller, so a single-penalty
    comparison cannot separate alignment from dose. Sweeping the penalty gives
    each arm a time-in-range-versus-insulin curve, and the arms can then be read
    off at matched insulin instead of at a matched penalty.
    """
    cohort = json.loads(replay_cohort_path(scope).read_text())
    dataset = str(dataset or cohort.get("dataset") or "ohio_t1dm")
    st = _study()
    st.set_dataset(dataset)
    keep = {(str(r["patient_id"]), int(r["source_start"]))
            for r in _primary_windows(replay_horizon_payload(cohort, PH)["windows"])}
    ms = st.mean_std

    def sweep(project):
        """Penalty sweep for both arms under one dose-selection objective."""
        curves = {"baseline": {}, "docas": {}}
        tag = "dm" if project else "dmh"
        for pid in sorted({p for p, _ in keep}):
            wins = _windows(_load_trace(pid, dataset=dataset, split="test"))
            meta = st.load_patient_meta(str(DATA_ROOT), pid)
            bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
            baseline, docas, _wrap, cols, _span, _si = _controllers(
                pid, backbone="catboost", dataset=dataset)
            for w in wins:
                if (str(pid), int(w["source_start"])) not in keep:
                    continue
                seg = w["trace"]
                name = _twin_name(pid, "catboost", w["source_start"], dataset=dataset)
                feat = paper_feat_arrays(seg, ph_steps=horizon_steps(PH))
                delay = 120 + int(w["meal_offset_min"])
                for arm, mod in (("baseline", baseline), ("docas", docas)):
                    for pen in DOSE_MATCH_PENALTIES:
                        cb = ForecasterCIB(mod, cols, feat["CHO"], feat["CGM"],
                                           delay_min=delay, dose_penalty=pen,
                                           horizon_min=PH, project_nadir=project)
                        g, _, d = _replay(seg, bw, name, u2ss, f"_{tag}_{arm}_{pen:g}", cb)
                        m = _tir_tbr_tar(g)
                        curves[arm].setdefault(str(pen), []).append(
                            {"patient_id": str(pid), "source_start": int(w["source_start"]),
                             **m, "u": float(sum(u for _, u in d))})
        front = {arm: [{"penalty": float(pen),
                        "tir": ms([r["tir"] for r in rows]),
                        "tbr": ms([r["tbr"] for r in rows]),
                        "insulin_u": ms([r["u"] for r in rows])}
                       for pen, rows in sorted(block.items(), key=lambda kv: float(kv[0]))]
                 for arm, block in curves.items()}
        return curves, front

    curves, frontier = sweep(PROJECT_NADIR)
    # Sensitivity: the same sweep under the other dose-selection objective.
    _ch, frontier_alt = sweep(not PROJECT_NADIR)

    def _at_insulin(arm, target_u):
        """Mean TIR of `arm` at a matched mean insulin dose.

        The penalty only spans the doses an arm is willing to give, so a target
        outside that span is reported at the arm's nearest attainable dose and
        flagged: the observational controller cannot reach DOCAS's dose at any
        penalty, which is itself part of the comparison.
        """
        pts = sorted(((p["insulin_u"]["mean"], p["tir"]["mean"]) for p in frontier[arm]))
        us, tirs = [x for x, _ in pts], [y for _, y in pts]
        t = float(target_u)
        return {"tir": float(np.interp(t, us, tirs)),
                "attainable": bool(us[0] <= t <= us[-1]),
                "nearest_u": float(min(us, key=lambda u: abs(u - t))),
                "u_range": [float(us[0]), float(us[-1])]}

    nominal = {arm: next(p for p in frontier[arm]
                         if abs(p["penalty"] - CIB_DOSE_PENALTY) < 1e-9)
               for arm in frontier}
    u_base = nominal["baseline"]["insulin_u"]["mean"]
    u_docas = nominal["docas"]["insulin_u"]["mean"]
    best = {arm: max(frontier[arm], key=lambda p: p["tir"]["mean"]) for arm in frontier}
    matched = {
        "baseline_u": u_base, "docas_u": u_docas,
        "at_docas_insulin": {"insulin_u": u_docas,
                             "baseline": _at_insulin("baseline", u_docas),
                             "docas_tir": nominal["docas"]["tir"]["mean"]},
        "at_baseline_insulin": {"insulin_u": u_base,
                                "baseline_tir": nominal["baseline"]["tir"]["mean"],
                                "docas": _at_insulin("docas", u_base)},
        "best_over_sweep": {arm: {"penalty": best[arm]["penalty"],
                                  "tir": best[arm]["tir"]["mean"],
                                  "insulin_u": best[arm]["insulin_u"]["mean"]}
                            for arm in best},
    }
    payload = {"horizon_min": PH, "scope": scope, "n_windows": len(keep),
               "penalties": list(DOSE_MATCH_PENALTIES), "frontier": frontier,
               "frontier_alt_objective": frontier_alt,
               "project_nadir": bool(PROJECT_NADIR), "nadir_min": float(NADIR_MIN),
               "matched": matched, "per_window": curves}
    dose_matched_path(scope).write_text(
        json.dumps(payload, separators=(",", ":"), default=_json))
    a, b = matched["at_docas_insulin"], matched["at_baseline_insulin"]
    flag = "" if a["baseline"]["attainable"] else f" (capped at {a['baseline']['nearest_u']:.2f} U)"
    print(f"  dose-matched (n={len(keep)}): at {a['insulin_u']:.2f} U baseline "
          f"{a['baseline']['tir']:.2f}{flag} vs DOCAS {a['docas_tir']:.2f}; "
          f"at {b['insulin_u']:.2f} U baseline {b['baseline_tir']:.2f} vs "
          f"DOCAS {b['docas']['tir']:.2f}", flush=True)
    return payload


def replay_paired_stats(*, scope: str = "") -> dict:
    """Paired DOCAS-minus-baseline contrasts, at window level and by patient.

    Windows are clustered within participants, so the patient-level test is the
    honest one; both are reported because the window count is small either way.
    """
    from scipy.stats import wilcoxon
    cohort = json.loads(replay_cohort_path(scope).read_text())
    rows = _primary_windows(replay_horizon_payload(cohort, PH)["windows"])
    d_win = np.array([float(r["docas"]["tir"]) - float(r["baseline"]["tir"]) for r in rows])
    by_pat = {}
    for r in rows:
        by_pat.setdefault(str(r["patient_id"]), []).append(
            float(r["docas"]["tir"]) - float(r["baseline"]["tir"]))
    d_pat = np.array([float(np.mean(v)) for v in by_pat.values()])
    rng = np.random.default_rng(0)
    pats = list(by_pat)
    boot = [float(np.mean(np.concatenate(
        [by_pat[pats[i]] for i in rng.integers(0, len(pats), len(pats))])))
        for _ in range(10000)]

    def _p(x):
        x = np.asarray(x, float)
        if len(x) < 2 or np.allclose(x, 0):
            return float("nan")
        return float(wilcoxon(x).pvalue)

    out = {"n_windows": int(len(d_win)), "n_patients": int(len(d_pat)),
           "window_mean_delta_tir": float(np.mean(d_win)),
           "window_p_wilcoxon": _p(d_win),
           "window_n_improved": int(np.sum(d_win > 0)),
           "window_n_worsened": int(np.sum(d_win < 0)),
           "patient_mean_delta_tir": float(np.mean(d_pat)),
           "patient_p_wilcoxon": _p(d_pat),
           "cluster_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))]}
    print(f"  paired ΔTIR: windows n={out['n_windows']} mean {out['window_mean_delta_tir']:+.2f} "
          f"(p={out['window_p_wilcoxon']:.3f}, {out['window_n_improved']}+/"
          f"{out['window_n_worsened']}-); patients n={out['n_patients']} "
          f"mean {out['patient_mean_delta_tir']:+.2f} (p={out['patient_p_wilcoxon']:.3f}); "
          f"cluster CI95 [{out['cluster_bootstrap_ci95'][0]:+.2f}, "
          f"{out['cluster_bootstrap_ci95'][1]:+.2f}]", flush=True)
    return out



def decision_point_alignment(*, scope: str = "") -> dict:
    """How far each arm's dose response sits from τ where the DSS actually doses.

    The τ-wrapper arm predicts τ by construction, so its own candidate curve at
    the same decision is the target for that context: comparing changes relative
    to the null dose removes the wrapper's different level. All arms share the
    trace up to the first decision, so that decision is the one context in which
    the three curves are strictly comparable.
    """
    cohort = json.loads(replay_cohort_path(scope).read_text())
    rows = _primary_windows(replay_horizon_payload(cohort, PH)["windows"])
    out = {"n_decisions": 0}
    stat = {a: {"ae": [], "at_max": [], "monotone": 0} for a in ("baseline", "docas")}
    for r in rows:
        dec = {a: (r.get(f"{a}_decisions") or [None])[0]
               for a in ("baseline", "docas", "tauwrap")}
        if any(d is None for d in dec.values()):
            continue
        if len({int(d["time_min"]) for d in dec.values()}) != 1:
            continue
        tau = np.asarray(dec["tauwrap"]["future_cgm_curve"], float)
        tau = tau - tau[0]
        out["n_decisions"] += 1
        for arm in ("baseline", "docas"):
            curve = np.asarray(dec[arm]["future_cgm_curve"], float)
            delta = curve - curve[0]
            stat[arm]["ae"].append(float(np.sqrt(np.mean((delta - tau) ** 2))))
            stat[arm]["at_max"].append(float(delta[-1]))
            stat[arm]["monotone"] += int(np.all(np.diff(delta) <= 1e-9))
        out.setdefault("tau_at_max", []).append(float(tau[-1]))
    for arm in ("baseline", "docas"):
        out[arm] = {"align_mgdl": float(np.mean(stat[arm]["ae"])),
                    "delta_at_max_dose": float(np.mean(stat[arm]["at_max"])),
                    "n_non_increasing": int(stat[arm]["monotone"])}
    out["tau_at_max"] = float(np.mean(out["tau_at_max"]))
    print(f"  decision-point dose response (n={out['n_decisions']}, 0-{GRID[-1]:g} U): "
          f"align baseline {out['baseline']['align_mgdl']:.2f} vs DOCAS "
          f"{out['docas']['align_mgdl']:.2f} mg/dL; Δ at max dose "
          f"{out['baseline']['delta_at_max_dose']:+.2f} / "
          f"{out['docas']['delta_at_max_dose']:+.2f} (τ {out['tau_at_max']:+.2f}); "
          f"non-increasing {out['baseline']['n_non_increasing']}/"
          f"{out['docas']['n_non_increasing']}", flush=True)
    return out


def print_replay_summary(*, dataset: str = "ohio_t1dm") -> None:
    scope = _SCOPE[dataset]
    p = json.loads(replay_cohort_path(scope).read_text())
    h = p["horizons"][str(PH)]
    by = h["by_backbone"]
    for bb, block in by.items():
        t2, co = block["table2"], block["cohort"]
        s, n = t2["summary"], t2["paired_docas_vs_baseline"]["n_windows"]

        def _line(arm):
            tir = s[arm]["tir"]
            u = s[arm]["insulin_u"]
            return (f"TIR {tir['mean']:.2f} U {u['mean']:.2f}")

        print(f"  Table2 PH{PH} {bb} {dataset} n_pat={co['n_patients']} n_win={n} "
              f"gap={CIB_GAP_MIN:g}min: "
              f"No DS {_line('no_dss')} | Baseline {_line('baseline')} | "
              f"DOCAS {_line('docas')}", flush=True)
        for tag, key in (("post-decision", "summary_post"),
                         ("post-decision, dosing windows", "summary_post_dosing")):
            sp = co.get(key)
            if not sp:
                continue
            g = lambda a: sp[a]["tir"]["mean"]
            print(f"    {tag} (n={co.get('n_windows_dosing') if 'dosing' in key else n}): "
                  f"No DS {g('no_dss'):.2f} | Baseline {g('baseline'):.2f} | "
                  f"DOCAS {g('docas'):.2f} (Δ{g('docas') - g('baseline'):+.2f})",
                  flush=True)


def unguarded_cohort_path(scope: str = "") -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"cohort_unguarded{scope}.json"


def run_unguarded(*, dataset: str = "ohio_t1dm") -> dict:
    """Replay the primary windows with Prendin's cost alone (no hypo floor).

    Reuses twins and the twin-validity screen of the guarded cohort so the
    comparison is only the DSS constraint. Writes a separate JSON; does not
    overwrite the reported guarded results.
    """
    global APPLY_HYPO_GUARD
    scope = _SCOPE[dataset]
    guarded = json.loads(replay_cohort_path(scope).read_text())
    keep = {(str(r["patient_id"]), int(r["source_start"]))
            for r in _primary_windows(replay_horizon_payload(guarded, PH)["windows"])}
    st = _study()
    st.set_dataset(dataset)
    old = APPLY_HYPO_GUARD
    APPLY_HYPO_GUARD = False
    rows = []
    try:
        print(f"\nUnguarded CIB ({dataset}; n_windows={len(keep)}; "
              f"no hypo floor / no 2× SI margin)", flush=True)
        for pid in sorted({p for p, _ in keep}):
            wins = _windows(_load_trace(pid, dataset=dataset, split="test"))
            meta = st.load_patient_meta(str(DATA_ROOT), pid)
            bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
            baseline, docas, _wrap, cols, _span, _si = _controllers(
                pid, backbone="catboost", dataset=dataset)
            for w in wins:
                if (str(pid), int(w["source_start"])) not in keep:
                    continue
                seg = w["trace"]
                name = _twin_name(pid, "catboost", w["source_start"], dataset=dataset)
                feat = paper_feat_arrays(seg, ph_steps=horizon_steps(PH))
                delay = 120 + int(w["meal_offset_min"])
                no_dss, _, _ = _replay(seg, bw, name, u2ss, "_none")
                mk = lambda mod: ForecasterCIB(
                    mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                    dose_penalty=CIB_DOSE_PENALTY, horizon_min=PH)
                base_g, base_dec, base_d = _replay(
                    seg, bw, name, u2ss, "_base_unguarded", mk(baseline))
                doc_g, doc_dec, doc_d = _replay(
                    seg, bw, name, u2ss, "_docas_unguarded", mk(docas))
                bm, dm = _tir_tbr_tar(base_g), _tir_tbr_tar(doc_g)
                post = {f"{a}_post": _post_gly(g, delay)
                        for a, g in (("no_dss", no_dss), ("baseline", base_g),
                                     ("docas", doc_g))}
                rows.append({
                    "patient_id": str(pid),
                    "source_start": int(w["source_start"]),
                    "no_dss": {k: round(float(v), 2) for k, v in _tir_tbr_tar(no_dss).items()},
                    "baseline": {k: round(float(v), 2) for k, v in bm.items()},
                    "docas": {k: round(float(v), 2) for k, v in dm.items()},
                    **post,
                    "delta_tir": round(dm["tir"] - bm["tir"], 2),
                    "delta_tbr": round(dm["tbr"] - bm["tbr"], 2),
                    "delta_tir_post": round(post["docas_post"]["tir"]
                                            - post["baseline_post"]["tir"], 2),
                    "delta_tbr_post": round(post["docas_post"]["tbr"]
                                            - post["baseline_post"]["tbr"], 2),
                    "baseline_doses": [{"time_min": int(t), "units": round(float(u), 2)}
                                       for t, u in base_d],
                    "docas_doses": [{"time_min": int(t), "units": round(float(u), 2)}
                                    for t, u in doc_d],
                    "baseline_insulin_u": round(float(sum(u for _, u in base_d)), 2),
                    "docas_insulin_u": round(float(sum(u for _, u in doc_d)), 2),
                    "baseline_decisions": base_dec,
                    "docas_decisions": doc_dec,
                })
    finally:
        APPLY_HYPO_GUARD = old

    # Enrich rows so _summarize / _table2_from_windows can read insulin_u.
    for r in rows:
        for arm in ("baseline", "docas"):
            r[arm]["insulin_u"] = float(r[f"{arm}_insulin_u"])
        r["no_dss"]["insulin_u"] = 0.0

    summary = _summarize(rows)
    summary_post = _summarize(rows, suffix="_post")
    g_coh = guarded["horizons"][str(PH)]["cohort"]
    g_sum = g_coh["summary"]
    g_post = g_coh.get("summary_post") or g_sum
    payload = {
        "dataset": dataset, "horizon_min": PH, "apply_hypo_guard": False,
        "n_windows": len(rows),
        "patient_ids": sorted({r["patient_id"] for r in rows}),
        "summary": summary,
        "summary_post": summary_post,
        "guarded_summary": g_sum,
        "guarded_summary_post": g_post,
        "windows": rows,
    }
    path = unguarded_cohort_path(scope)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=_json))
    s = summary
    print(f"  unguarded n={len(rows)}: "
          f"No DS TIR {s['no_dss']['tir']['mean']:.2f} "
          f"(TBR {s['no_dss']['tbr']['mean']:.2f}) | "
          f"Baseline TIR {s['baseline']['tir']['mean']:.2f} "
          f"U {s['baseline']['insulin_u']['mean']:.2f} "
          f"(TBR {s['baseline']['tbr']['mean']:.2f}) | "
          f"DOCAS TIR {s['docas']['tir']['mean']:.2f} "
          f"U {s['docas']['insulin_u']['mean']:.2f} "
          f"(TBR {s['docas']['tbr']['mean']:.2f}) | "
          f"ΔTIR {np.mean([r['delta_tir'] for r in rows]):+.2f} "
          f"ΔTBR {np.mean([r['delta_tbr'] for r in rows]):+.2f} | "
          f"post TIR No DSS {summary_post['no_dss']['tir']['mean']:.2f} "
          f"Base {summary_post['baseline']['tir']['mean']:.2f} "
          f"DOCAS {summary_post['docas']['tir']['mean']:.2f} "
          f"TBR {summary_post['docas']['tbr']['mean']:.2f}", flush=True)
    print(f"  guarded reference: "
          f"Baseline TIR {g_sum['baseline']['tir']['mean']:.2f} "
          f"U {g_sum['baseline']['insulin_u']['mean']:.2f} "
          f"(TBR {g_sum['baseline']['tbr']['mean']:.2f}) | "
          f"DOCAS TIR {g_sum['docas']['tir']['mean']:.2f} "
          f"U {g_sum['docas']['insulin_u']['mean']:.2f} "
          f"(TBR {g_sum['docas']['tbr']['mean']:.2f})", flush=True)
    print(f"  wrote {path}", flush=True)
    return payload


def floor_cohort_path(floor: float, scope: str = "") -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"cohort_floor{int(floor)}{scope}.json"


def run_floor(*, dataset: str = "ohio_t1dm", floor: float = 70.0) -> dict:
    """Same primary windows and 2× SI margin as the reported guard, floor only."""
    global HYPO_FLOOR_MGDL
    scope = _SCOPE[dataset]
    guarded = json.loads(replay_cohort_path(scope).read_text())
    keep = {(str(r["patient_id"]), int(r["source_start"]))
            for r in _primary_windows(replay_horizon_payload(guarded, PH)["windows"])}
    st = _study()
    st.set_dataset(dataset)
    old = HYPO_FLOOR_MGDL
    HYPO_FLOOR_MGDL = float(floor)
    tag = f"_floor{int(floor)}"
    rows = []
    try:
        print(f"\nFloor-{int(floor)} CIB ({dataset}; n_windows={len(keep)}; "
              f"{int(floor)} mg/dL floor, 2× SI margin kept)", flush=True)
        for pid in sorted({p for p, _ in keep}):
            wins = _windows(_load_trace(pid, dataset=dataset, split="test"))
            meta = st.load_patient_meta(str(DATA_ROOT), pid)
            bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
            baseline, docas, _wrap, cols, _span, _si = _controllers(
                pid, backbone="catboost", dataset=dataset)
            for w in wins:
                if (str(pid), int(w["source_start"])) not in keep:
                    continue
                seg = w["trace"]
                name = _twin_name(pid, "catboost", w["source_start"], dataset=dataset)
                feat = paper_feat_arrays(seg, ph_steps=horizon_steps(PH))
                delay = 120 + int(w["meal_offset_min"])
                no_dss, _, _ = _replay(seg, bw, name, u2ss, "_none")
                mk = lambda mod: ForecasterCIB(
                    mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                    dose_penalty=CIB_DOSE_PENALTY, horizon_min=PH)
                base_g, base_dec, base_d = _replay(
                    seg, bw, name, u2ss, f"_base{tag}", mk(baseline))
                doc_g, doc_dec, doc_d = _replay(
                    seg, bw, name, u2ss, f"_docas{tag}", mk(docas))
                bm, dm = _tir_tbr_tar(base_g), _tir_tbr_tar(doc_g)
                rows.append({
                    "patient_id": str(pid),
                    "source_start": int(w["source_start"]),
                    "no_dss": {k: round(float(v), 2) for k, v in _tir_tbr_tar(no_dss).items()},
                    "baseline": {k: round(float(v), 2) for k, v in bm.items()},
                    "docas": {k: round(float(v), 2) for k, v in dm.items()},
                    "delta_tir": round(dm["tir"] - bm["tir"], 2),
                    "delta_tbr": round(dm["tbr"] - bm["tbr"], 2),
                    "baseline_doses": [{"time_min": int(t), "units": round(float(u), 2)}
                                       for t, u in base_d],
                    "docas_doses": [{"time_min": int(t), "units": round(float(u), 2)}
                                    for t, u in doc_d],
                    "baseline_insulin_u": round(float(sum(u for _, u in base_d)), 2),
                    "docas_insulin_u": round(float(sum(u for _, u in doc_d)), 2),
                    "baseline_decisions": base_dec,
                    "docas_decisions": doc_dec,
                })
    finally:
        HYPO_FLOOR_MGDL = old

    for r in rows:
        for arm in ("baseline", "docas"):
            r[arm]["insulin_u"] = float(r[f"{arm}_insulin_u"])
        r["no_dss"]["insulin_u"] = 0.0

    summary = _summarize(rows)
    g_sum = guarded["horizons"][str(PH)]["cohort"]["summary"]
    payload = {
        "dataset": dataset, "horizon_min": PH, "apply_hypo_guard": True,
        "hypo_floor_mgdl": float(floor), "safety_si_factor": SAFETY_SI_FACTOR,
        "n_windows": len(rows),
        "patient_ids": sorted({r["patient_id"] for r in rows}),
        "summary": summary,
        "guarded_summary": g_sum,
        "windows": rows,
    }
    path = floor_cohort_path(floor, scope)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=_json))
    s = summary
    print(f"  floor {int(floor)} n={len(rows)}: "
          f"Baseline TIR {s['baseline']['tir']['mean']:.2f} "
          f"U {s['baseline']['insulin_u']['mean']:.2f} "
          f"(TBR {s['baseline']['tbr']['mean']:.2f}) | "
          f"DOCAS TIR {s['docas']['tir']['mean']:.2f} "
          f"U {s['docas']['insulin_u']['mean']:.2f} "
          f"(TBR {s['docas']['tbr']['mean']:.2f}) | "
          f"ΔTIR {np.mean([r['delta_tir'] for r in rows]):+.2f} "
          f"ΔTBR {np.mean([r['delta_tbr'] for r in rows]):+.2f}", flush=True)
    print(f"  cohort.json (then-default) reference: "
          f"Baseline TIR {g_sum['baseline']['tir']['mean']:.2f} "
          f"U {g_sum['baseline']['insulin_u']['mean']:.2f} "
          f"(TBR {g_sum['baseline']['tbr']['mean']:.2f}) | "
          f"DOCAS TIR {g_sum['docas']['tir']['mean']:.2f} "
          f"U {g_sum['docas']['insulin_u']['mean']:.2f} "
          f"(TBR {g_sum['docas']['tbr']['mean']:.2f})", flush=True)
    for r in rows:
        if float(r["docas"].get("tbr") or 0) > 0 or float(r["baseline"].get("tbr") or 0) > 0:
            print(f"  TBR>0  patient {r['patient_id']} start {r['source_start']}: "
                  f"DOCAS {r['docas_insulin_u']} U TBR {r['docas']['tbr']:.2f} "
                  f"(baseline {r['baseline_insulin_u']} U TBR {r['baseline']['tbr']:.2f})",
                  flush=True)
    print(f"  wrote {path}", flush=True)
    return payload


def clinical_tau_cohort_path(scope: str = "") -> Path:
    REPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    return REPLAY_RESULTS / f"cohort_clinical_tau{scope}.json"


def run_clinical_tau_replay(*, dataset: str = "ohio_t1dm") -> dict:
    """Same primary windows as Table tab:replay, DOCAS trained on clinical τ.

    Twins, screens, baseline and no-DSS arms are taken from the guarded cohort;
    only the DOCAS controller is retrained on the pump-settings τ (1800-rule
    correction factor times the insulin-action curve) and re-played.
    """
    scope = _SCOPE[dataset]
    guarded = json.loads(replay_cohort_path(scope).read_text())
    g_rows = _primary_windows(replay_horizon_payload(guarded, PH)["windows"])
    by_key = {(str(r["patient_id"]), int(r["source_start"])): r for r in g_rows}
    st = _study()
    st.set_dataset(dataset)
    rows = []
    print(f"\nClinical-τ CIB ({dataset}; n_windows={len(by_key)})", flush=True)
    for pid in sorted({p for p, _ in by_key}):
        wins = _windows(_load_trace(pid, dataset=dataset, split="test"))
        meta = st.load_patient_meta(str(DATA_ROOT), pid)
        bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
        _base, clin_docas, _wrap, cols, _span, clin_cf = _controllers(
            pid, backbone="catboost", dataset=dataset, clinical_tau=True)
        print(f"  {pid}: correction factor {clin_cf:.1f} mg/dL per U", flush=True)
        for w in wins:
            key = (str(pid), int(w["source_start"]))
            if key not in by_key:
                continue
            g = by_key[key]
            seg = w["trace"]
            name = _twin_name(pid, "catboost", w["source_start"], dataset=dataset)
            feat = paper_feat_arrays(seg, ph_steps=horizon_steps(PH))
            delay = 120 + int(w["meal_offset_min"])
            mk = ForecasterCIB(
                clin_docas, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                dose_penalty=CIB_DOSE_PENALTY, horizon_min=PH)
            clin_g, clin_dec, clin_d = _replay(
                seg, bw, name, u2ss, "_docas_clinical", mk)
            cm = _tir_tbr_tar(clin_g)
            bm = g["baseline"]
            sm = g["docas"]
            clin_post = _post_gly(clin_g, delay)
            rows.append({
                "patient_id": str(pid),
                "source_start": int(w["source_start"]),
                "clinical_cf_mgdl_per_u": float(clin_cf),
                "no_dss": dict(g["no_dss"]),
                "baseline": dict(bm),
                "docas_study": dict(sm),
                "docas": {k: round(float(v), 2) for k, v in cm.items()},
                "no_dss_post": dict(g.get("no_dss_post") or g["no_dss"]),
                "baseline_post": dict(g.get("baseline_post") or bm),
                "docas_study_post": dict(g.get("docas_post") or sm),
                "docas_post": clin_post,
                "delta_tir": round(cm["tir"] - bm["tir"], 2),
                "delta_tbr": round(cm["tbr"] - bm["tbr"], 2),
                "delta_tir_vs_study": round(cm["tir"] - sm["tir"], 2),
                "delta_tir_post": round(clin_post["tir"]
                                        - float((g.get("baseline_post") or bm)["tir"]), 2),
                "docas_doses": [{"time_min": int(t), "units": round(float(u), 2)}
                               for t, u in clin_d],
                "docas_insulin_u": round(float(sum(u for _, u in clin_d)), 2),
                "baseline_insulin_u": float(sum(d["units"] for d in g.get("baseline_doses", []))),
                "docas_study_insulin_u": float(sum(d["units"] for d in g.get("docas_doses", []))),
                "docas_decisions": clin_dec,
            })

    for r in rows:
        r["docas"]["insulin_u"] = float(r["docas_insulin_u"])
        r["baseline"]["insulin_u"] = float(r["baseline_insulin_u"])
        r["docas_study"]["insulin_u"] = float(r["docas_study_insulin_u"])
        r["no_dss"]["insulin_u"] = 0.0
        # _summarize expects baseline_doses / docas_doses for insulin when present
        r["baseline_doses"] = [{"units": r["baseline_insulin_u"]}] if r["baseline_insulin_u"] else []

    summary = _summarize(rows)
    summary_post = _summarize(rows, suffix="_post")
    g_coh = guarded["horizons"][str(PH)]["cohort"]
    gs = g_coh["summary"]
    gs_post = g_coh.get("summary_post") or gs
    summary["no_dss"] = gs["no_dss"]
    summary["baseline"] = gs["baseline"]
    summary["docas_study"] = gs["docas"]
    summary_post["no_dss"] = gs_post["no_dss"]
    summary_post["baseline"] = gs_post["baseline"]
    summary_post["docas_study"] = gs_post["docas"]

    payload = {
        "dataset": dataset, "horizon_min": PH, "tau": "clinical_1800",
        "n_windows": len(rows),
        "patient_ids": sorted({r["patient_id"] for r in rows}),
        "summary": summary,
        "summary_post": summary_post,
        "study_summary": gs,
        "study_summary_post": gs_post,
        "windows": rows,
    }
    path = clinical_tau_cohort_path(scope)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=_json))
    s = summary
    print(f"  clinical-τ n={len(rows)}: "
          f"Baseline TIR {s['baseline']['tir']['mean']:.2f} "
          f"U {s['baseline']['insulin_u']['mean']:.2f} | "
          f"DOCAS(study) TIR {s['docas_study']['tir']['mean']:.2f} "
          f"U {s['docas_study']['insulin_u']['mean']:.2f} | "
          f"DOCAS(clinical) TIR {s['docas']['tir']['mean']:.2f} "
          f"U {s['docas']['insulin_u']['mean']:.2f} "
          f"(TBR {s['docas']['tbr']['mean']:.2f}) | "
          f"ΔTIR vs baseline {np.mean([r['delta_tir'] for r in rows]):+.2f} | "
          f"post TIR Base {summary_post['baseline']['tir']['mean']:.2f} "
          f"study {summary_post['docas_study']['tir']['mean']:.2f} "
          f"pump {summary_post['docas']['tir']['mean']:.2f}",
          flush=True)
    print(f"  wrote {path}", flush=True)
    return payload


def maybe_run() -> None:
    if not RUN_REPLAY:
        print("  skip ReplayBG (off)", flush=True)
        return
    for dataset in REPLAY_DATASETS:
        pids = _patients(dataset)
        print(f"\nReplayBG DSS ({dataset} n={len(pids)}; "
              f"figure showcase {SHOWCASE_PATIENT})", flush=True)
        run_replay(dataset=dataset)
        print_replay_summary(dataset=dataset)
        run_dose_matched(scope=_SCOPE[dataset], dataset=dataset)
        replay_paired_stats(scope=_SCOPE[dataset])
        decision_point_alignment(scope=_SCOPE[dataset])



# Revision ReplayBG (Tables 4--6). uv run python scripts/replay_eval.py revision
import os
from collections import defaultdict
import catboost as cb  # noqa: F401
from joblib import Parallel, delayed
from scipy import stats
import ablation as cd
import main as m
import metabonet_comparators as mc
from core import DOCAS, sync_cfg
from forecasters import PenaltyMLPForecaster, train_mlp, train_monotone
from preprocessing import PatientData

OUT = ROOT / "results" / "revision" / "full_replay_open.json"
DS = "metabonet_full"
REPLACE_KEEP = {
    "ReplaceBG|2", "ReplaceBG|10", "ReplaceBG|11", "ReplaceBG|14", "ReplaceBG|15",
    "ReplaceBG|16", "ReplaceBG|17", "ReplaceBG|18", "ReplaceBG|19", "ReplaceBG|20",
}
CAP = 8000
ARMS = ("baseline", "mlp", "monotone", "penalty", "pop", "ctx")


def _tail(data: PatientData, n: int = CAP) -> PatientData:
    if len(data.y_train) <= n:
        return data
    return PatientData(
        data.X_train[-n:], data.y_train[-n:], data.X_test, data.y_test,
        data.feature_names, data.cgm_scaler, data.train_scalers, data.patient_id,
        data.insulin_idx, data.cgm_idx, data.lookback,
    )


def _wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        return round(float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue), 4)
    except ValueError:
        return 1.0


def _models(pid, theta):
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset(DS)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    span = float(m.insulin_span_u(data, horizon_min=60))
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    _, R_tab = cd._pop_R(60, span, m.STUDY_CFG)
    fit_kw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    mono = train_monotone(
        data.X_train, data.y_train, insulin_idx=data.insulin_idx,
        model_kwargs={"random_seed": 42})
    pop = DOCAS(**{**fit_kw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    ctx, _ = cd._fit_ctx(data, span, 60, theta, R_tab, seed=42, base_mdl=base)
    src = _tail(data)
    ji = int(data.insulin_idx)
    gi = data.feature_names.index("CGM")
    fit_kw2, drop2, u_ref2, _, _ = docas_fit_kw(src, span, 60, cfg=m.STUDY_CFG)
    aligner = DOCAS(
        fit_kw2["target"], intervention_idx=ji, outcome_idx=gi,
        action_scaler=src.train_scalers[ji], level_scaler=src.train_scalers[gi],
        delta_scaler=src.cgm_scaler, span=span, tau_drop=drop2, u_ref=u_ref2,
        tau_mode=str(fit_kw2["tau_mode"]),
        level_ref=str(fit_kw2.get("level_ref", "current")), predicts_level=False,
    )
    mlp = train_mlp(src.X_train, src.y_train, model_kwargs={"seed": 42})
    pen = PenaltyMLPForecaster(aligner=aligner, seed=42).fit(
        src.X_train, src.y_train, n_factual=len(src.X_train))
    return dict(
        data=data, span=span,
        baseline=base, mlp=mlp, monotone=mono, penalty=pen, pop=pop, ctx=ctx,
    )


def _twin_one(pid: str, w: dict, bw: float, u2ss: float) -> dict:
    try:
        return _twin_ok(pid, w, bw, u2ss)
    except Exception as exc:
        print(f"  twin skip {pid} start={w.get('source_start')}: {exc}", flush=True)
        return dict(patient_id=str(pid), source_start=int(w["source_start"]), ok=False, error=str(exc))


def _twin_ok(pid: str, w: dict, bw: float, u2ss: float) -> dict:
    TWIN_N_JOBS = 1
    seg = w["trace"]
    name = _twin_name(pid, "catboost", w["source_start"], dataset=DS)
    _twin(seg, bw, name, u2ss)
    observed = np.asarray(seg["glucose"].to_numpy(), float)
    no_dss, _, _ = _replay(seg, bw, name, u2ss, "_none")
    n = min(len(no_dss), len(observed))
    if n == 0:
        raise ValueError("empty twin overlap")
    twin_rmse = float(np.sqrt(np.mean((no_dss[:n] - observed[:n]) ** 2)))
    probe_g, _, _ = _replay(seg, bw, name, u2ss, "_probe1u", TestBolus())
    i0 = TWIN_PROBE_AT_MIN // YTS_MIN
    twin_cf = float(np.max(no_dss[i0:] - probe_g[i0:]) / TWIN_PROBE_U)
    tar0 = _tir_tbr_tar(no_dss)["tar"]
    min_p = float(probe_g.min())
    ok = (
        twin_rmse <= TWIN_RMSE_PRIMARY_MGDL
        and tar0 > 0.0
        and twin_cf <= TWIN_MAX_CF_MGDL_PER_U
        and min_p >= TWIN_MIN_PROBE_G_MGDL
    )
    print(
        f"  twin {pid} start={w['source_start']} tar_obs={w['observed_tar']:.1f} "
        f"rmse={twin_rmse:.1f} tar0={tar0:.1f} cf={twin_cf:.1f} ok={ok}",
        flush=True,
    )
    return dict(
        patient_id=str(pid),
        source_start=int(w["source_start"]),
        meal_offset_min=int(w["meal_offset_min"]),
        observed_tar=round(float(w["observed_tar"]), 2),
        twin_rmse=round(twin_rmse, 2),
        no_dss=_tir_tbr_tar(no_dss),
        twin_cf_mgdl_per_u=round(twin_cf, 2),
        twin_min_g_at_probe=round(min_p, 1),
        ok=bool(ok),
        source=str(pid).split("|", 1)[0],
    )


def _run_dss(cand_wins: list[dict], thetas: dict, *, guard: bool) -> list[dict]:
    old = APPLY_HYPO_GUARD
    APPLY_HYPO_GUARD = bool(guard)
    try:
        m.set_dataset(DS)
        pids = sorted({w["patient_id"] for w in cand_wins})
        by_pid = {p: {w["source_start"] for w in cand_wins if w["patient_id"] == p} for p in pids}
        rows = []
        for pid in pids:
            print(f"  replay metabonet {pid} guard={guard}", flush=True)
            pack = _models(pid, thetas[str(pid)])
            data = pack["data"]
            cols = list(data.feature_names)
            wrap = lambda mod: PaperGlucoseModel(mod, data.train_scalers, data.cgm_scaler)
            controllers = {arm: wrap(pack[arm]) for arm in ARMS}
            meta = m.load_patient_meta(str(m.DATA_ROOT), pid)
            bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
            wins = _windows(_load_trace(pid, dataset=DS, split="test"))
            for w in wins:
                if int(w["source_start"]) not in by_pid[pid]:
                    continue
                seg = w["trace"]
                name = _twin_name(pid, "catboost", w["source_start"], dataset=DS)
                feat = paper_feat_arrays(seg, ph_steps=horizon_steps(60))
                delay = 120 + int(w["meal_offset_min"])
                no_dss, _, _ = _replay(seg, bw, name, u2ss, "_none")
                row = {
                    "patient_id": str(pid),
                    "source": str(pid).split("|", 1)[0],
                    "source_start": int(w["source_start"]),
                    "observed_tar": round(float(w["observed_tar"]), 2),
                    "delay_min": int(delay),
                    "no_dss": _post_gly(no_dss, delay),
                }
                sfx = "" if guard else "_ug"
                for arm, mod in controllers.items():
                    cb = ForecasterCIB(
                        mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                        dose_penalty=CIB_DOSE_PENALTY, horizon_min=60,
                    )
                    g, _dec, doses = _replay(seg, bw, name, u2ss, f"_{arm}{sfx}", cb)
                    met = _post_gly(g, delay)
                    met["insulin_u"] = round(float(sum(u for _, u in doses)), 2)
                    row[arm] = met
                rows.append(row)
        return rows
    finally:
        APPLY_HYPO_GUARD = old


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {}
    for arm in ("no_dss",) + ARMS:
        keys = ("tir", "tbr", "tar") if arm == "no_dss" else ("tir", "tbr", "tar", "insulin_u")
        summary[arm] = {
            k: dict(
                mean=round(float(np.mean([r[arm][k] for r in rows])), 2),
                std=round(float(np.std([r[arm][k] for r in rows], ddof=1)), 2) if len(rows) > 1 else 0.0,
            )
            for k in keys
        }
    if rows:
        summary["p_pop_vs_baseline"] = _wilcoxon(
            [r["pop"]["tir"] for r in rows], [r["baseline"]["tir"] for r in rows])
        summary["p_ctx_vs_baseline"] = _wilcoxon(
            [r["ctx"]["tir"] for r in rows], [r["baseline"]["tir"] for r in rows])
        summary["p_baseline_vs_nodss"] = _wilcoxon(
            [r["baseline"]["tir"] for r in rows], [r["no_dss"]["tir"] for r in rows])
    return summary


def run_open_replay():
    m.set_dataset(DS)
    thetas = {}
    recent = ROOT / "results" / "revision" / "full_recent"
    for path in recent.glob("*.json"):
        d = json.loads(path.read_text())
        if d.get("source") not in {"HUPA-UCM", "T1D-UOM"} or "horizons" not in d:
            continue
        pid = str(d["patient_id"])
        thetas[pid] = d["horizons"]["60"]["theta"]

    tasks = []
    for pid in sorted(thetas):
        tr = _load_trace(pid, dataset=DS, split="test")
        if tr.height == 0 or float(tr["bolus_dose"].sum()) == 0.0:
            continue
        wins = _windows(tr)
        meta = m.load_patient_meta(str(m.DATA_ROOT), pid)
        for w in wins:
            tasks.append((pid, w, float(meta["weight_kg"]), float(meta.get("u2ss") or 0.0)))
    done = {}
    if OUT.is_file():
        old = json.loads(OUT.read_text())
        for row in old.get("screened") or []:
            done[(str(row["patient_id"]), int(row["source_start"]))] = row
    pending = [t for t in tasks if (t[0], int(t[1]["source_start"])) not in done]
    print(
        f"[metabonet] clean windows={len(tasks)} cached={len(tasks) - len(pending)} "
        f"to twin={len(pending)} across {len({t[0] for t in tasks})} patients",
        flush=True,
    )
    fresh = Parallel(n_jobs=max(1, (os.cpu_count() or 4) - 1), backend="loky")(
        delayed(_twin_one)(pid, w, bw, u2ss) for pid, w, bw, u2ss in pending
    ) if pending else []
    screened = [done[(t[0], int(t[1]["source_start"]))] for t in tasks if (t[0], int(t[1]["source_start"])) in done]
    screened.extend(fresh)
    # Plant validity only. The 35–75% TIR band was dropping valid hyperglycaemic twins.
    cand = [r for r in screened if r.get("ok")]
    print(
        f"[metabonet] plant-valid={len(cand)} across "
        f"{len({r['patient_id'] for r in cand})} patients "
        f"mean_obs_tar={np.mean([r['observed_tar'] for r in cand]) if cand else float('nan'):.1f} "
        f"mean_nodss_tir={np.mean([r['no_dss']['tir'] for r in cand]) if cand else float('nan'):.1f}",
        flush=True,
    )

    # Attach window traces for DSS by reloading typical wins keyed by source_start.
    win_map = {}
    for pid, w, _bw, _u2ss in tasks:
        win_map[(str(pid), int(w["source_start"]))] = dict(
            patient_id=str(pid),
            source_start=int(w["source_start"]),
            meal_offset_min=int(w["meal_offset_min"]),
            observed_tar=float(w["observed_tar"]),
            trace=w["trace"],
        )
    cand_wins = [win_map[(r["patient_id"], r["source_start"])] for r in cand
                 if (r["patient_id"], r["source_start"]) in win_map]

    guarded_all = _run_dss(cand_wins, thetas, guard=True)
    trig_pids = {
        r["patient_id"] for r in guarded_all
        if any(r[a]["insulin_u"] > 0 for a in ARMS)
    }
    guarded_rows = [r for r in guarded_all if r["patient_id"] in trig_pids]
    keep = {(r["patient_id"], r["source_start"]) for r in guarded_rows}
    cand_keep = [w for w in cand_wins if (w["patient_id"], w["source_start"]) in keep]
    unguarded_rows = _run_dss(cand_keep, thetas, guard=False)

    out = dict(
        selection="all_clean_8h_test_plant_valid",
        n_test_windows=len(tasks),
        n_plant_valid=len(cand),
        n_primary_windows=len(guarded_rows),
        n_patients=len(trig_pids),
        patient_ids=sorted(trig_pids),
        mean_observed_tar=round(float(np.mean([r["observed_tar"] for r in guarded_rows])), 2) if guarded_rows else None,
        mean_no_dss_tir=round(float(np.mean([r["no_dss"]["tir"] for r in guarded_rows])), 2) if guarded_rows else None,
        guarded=dict(summary=_summarize_rows(guarded_rows), windows=guarded_rows),
        unguarded=dict(summary=_summarize_rows(unguarded_rows), windows=unguarded_rows),
        screened=[{k: v for k, v in r.items()} for r in screened],
    )
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("METABONET REPLAY DONE:", json.dumps({
        "n_primary_windows": out["n_primary_windows"],
        "n_patients": out["n_patients"],
        "mean_observed_tar": out["mean_observed_tar"],
        "mean_no_dss_tir": out["mean_no_dss_tir"],
        "guarded": out["guarded"]["summary"],
    }, indent=2), flush=True)


def run_ohio_main_arms():
    """Dose the Table 3 forecasters on the 11 first-submission OhioT1DM windows.

    The plant, the windows, and the cached twins stay the first-submission
    ReplayBG run. The forecasters are the current per-person fits.
    """
    cohort = json.loads((ROOT / "results" / "replay" / "cohort.json").read_text())
    primary = _primary_windows(cohort["horizons"]["60"]["windows"])
    by: dict[str, list] = {}
    for row in primary:
        by.setdefault(str(row["patient_id"]), []).append(row)
    recent = ROOT / "results" / "revision" / "full_recent"
    thetas = {
        pid: json.loads((recent / f"OhioT1DM__{pid}.json").read_text())["horizons"]["60"]["theta"]
        for pid in by
    }
    rows = []
    for pid, wins_meta in by.items():
        print(f"  ohio arms {pid} n={len(wins_meta)}", flush=True)
        pack = _models(f"OhioT1DM|{pid}", thetas[pid])
        data = pack["data"]
        print(f"    features {data.feature_names} n_train {len(data.y_train)}", flush=True)
        cols = list(data.feature_names)
        controllers = {
            arm: PaperGlucoseModel(pack[arm], data.train_scalers, data.cgm_scaler)
            for arm in ARMS
        }
        m.set_dataset("ohio_t1dm")
        meta = m.load_patient_meta(str(m.DATA_ROOT), pid)
        bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
        trace = _load_trace(pid, dataset="ohio_t1dm", split="test")
        win_map = {int(w["source_start"]): w for w in _windows(trace)}
        for meta_w in wins_meta:
            start = int(meta_w["source_start"])
            seg = win_map[start]["trace"]
            name = _twin_name(pid, "catboost", start, dataset="ohio_t1dm")
            feat = paper_feat_arrays(seg, ph_steps=horizon_steps(60))
            delay = int(meta_w["eval_from_min"])
            row = {"patient_id": pid, "source_start": start, "delay_min": delay}
            for guard, sfx, key in ((True, "", "guarded"), (False, "_ug", "unguarded")):
                APPLY_HYPO_GUARD = guard
                block = {}
                for arm, mod in controllers.items():
                    cb = ForecasterCIB(
                        mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                        dose_penalty=CIB_DOSE_PENALTY, horizon_min=60,
                    )
                    g, _dec, doses = _replay(seg, bw, name, u2ss, f"_ohio_{arm}{sfx}", cb)
                    met = _post_gly(g, delay)
                    met["insulin_u"] = round(float(sum(u for _, u in doses)), 2)
                    block[arm] = met
                    print(
                        f"    {pid} {start} {arm} guard={guard} "
                        f"TIR {met['tir']} U {met['insulin_u']}",
                        flush=True,
                    )
                row[key] = block
            row["historical"] = {
                "no_dss": meta_w["no_dss_post"],
                "baseline": {
                    **meta_w["baseline_post"],
                    "insulin_u": round(sum(x["units"] for x in meta_w["baseline_doses"]), 2),
                },
                "pop": {
                    **meta_w["docas_post"],
                    "insulin_u": round(sum(x["units"] for x in meta_w["docas_doses"]), 2),
                },
            }
            rows.append(row)
    APPLY_HYPO_GUARD = True
    out = ROOT / "results" / "revision" / "ohio_replay_arms.json"
    out.write_text(json.dumps(rows, indent=2))
    print("OHIO ARMS DONE", out, flush=True)


def _fit_ablation_controllers(dataset: str, pid: str):
    import ablation as ab
    from core import tau_from_cfg
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset(dataset)
    data = m.load_patient(str(m.DATA_ROOT), pid, horizon_steps=horizon_steps(60))
    span = float(m.insulin_span_u(data, horizon_min=60))
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    gi, ji = data.feature_names.index("CGM"), int(data.insulin_idx)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})

    def wrap(mdl, level=False):
        return PaperGlucoseModel(
            mdl, data.train_scalers, data.cgm_scaler,
            predicts_level=level, level_scaler=data.train_scalers[gi])

    ctrls = {"baseline": wrap(base)}
    for b in (0.1, 1.0, 20.0, 100.0):
        src = _tail(data) if b >= 100.0 else data
        fkw, *_ = docas_fit_kw(src, span, 60, cfg=m.STUDY_CFG, synth_ratio=float(b))
        res = DOCAS(**{**fkw, "seed": 42}).fit(src.X_train, src.y_train, baseline_model=base)
        ctrls[f"b{b:g}"] = wrap(res["model"], bool(res.get("predicts_level", False)))
    fkw_u, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG, emp_frac=0.0)
    res_u = DOCAS(**{**fkw_u, "seed": 42}).fit(data.X_train, data.y_train, baseline_model=base)
    ctrls["uniform"] = wrap(res_u["model"], bool(res_u.get("predicts_level", False)))
    _, R_tab = cd._pop_R(60, span, m.STUDY_CFG)
    theta = cd._search_theta(data, span, 60, R_tab, seed=42)
    ctx_mdl, _ = cd._fit_ctx(data, span, 60, theta, R_tab, seed=42, base_mdl=base)
    ctrls["ctx"] = wrap(ctx_mdl)
    target, drop_w, u_ref_w, mode = tau_from_cfg(60, span, m.STUDY_CFG)
    wrap_mdl = ab.TauWrapForecaster().fit(
        data.X_train, data.y_train, target_R=target, drop=drop_w, u_ref=u_ref_w,
        ji=ji, gi=gi, cgm_s=data.train_scalers[gi], tgt_s=data.cgm_scaler,
        tau_mode=mode, level_ref=str(m.STUDY_CFG.get("level_ref", "current")),
        span=float(span))
    wrap_mdl._a_s = data.train_scalers[ji]
    ctrls["wrapper"] = wrap(wrap_mdl)
    ref_si = float(m.STUDY_CFG["SI"])
    for sc in (0.5, 2.0):
        cfg_s = {**m.STUDY_CFG, "SI": ref_si * sc}
        fkw_s, *_ = docas_fit_kw(data, span, 60, cfg=cfg_s)
        res_s = DOCAS(**{**fkw_s, "seed": 42}).fit(data.X_train, data.y_train, baseline_model=base)
        ctrls[f"si{sc:g}"] = wrap(res_s["model"], bool(res_s.get("predicts_level", False)))
    meta = m.load_patient_meta(str(m.DATA_ROOT), pid)
    tdd = float(meta.get("tdd_u") or 40.0)
    fkw_p, *_ = docas_fit_kw(data, span, 60, cfg=ab.clinical_cfg(tdd, cfg=dict(m.STUDY_CFG)))
    res_p = DOCAS(**{**fkw_p, "seed": 42}).fit(data.X_train, data.y_train, baseline_model=base)
    ctrls["pump"] = wrap(res_p["model"], bool(res_p.get("predicts_level", False)))
    return ctrls, list(data.feature_names), meta


def run_ablation_replay() -> dict:
    """ReplayBG across all 17 open-loop windows for every Table 5 ablation arm."""
    APPLY_HYPO_GUARD = True
    cohort = json.loads((ROOT / "results" / "replay" / "cohort.json").read_text())
    by_ohio: dict[str, list] = {}
    for r in _primary_windows(cohort["horizons"]["60"]["windows"]):
        by_ohio.setdefault(str(r["patient_id"]), []).append(r)
    open_rep = json.loads(OUT.read_text())
    by_open: dict[str, list] = {}
    for r in open_rep["guarded"]["windows"]:
        by_open.setdefault(str(r["patient_id"]), []).append(r)

    all_rows = []
    for ds_name, groups, no_key, d_key in (
        ("ohio_t1dm", by_ohio, "no_dss_post", "eval_from_min"),
        ("metabonet_full", by_open, "no_dss", "delay_min"),
    ):
        for pid, wins_meta in groups.items():
            ctrls, cols, meta = _fit_ablation_controllers(ds_name, pid)
            bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
            trace = _load_trace(pid, dataset=ds_name, split="test")
            win_map = {int(w["source_start"]): w for w in _windows(trace)}
            for mw in wins_meta:
                start = int(mw["source_start"])
                seg = win_map[start]["trace"]
                name = _twin_name(pid, "catboost", start, dataset=ds_name)
                feat = paper_feat_arrays(seg, ph_steps=horizon_steps(60))
                delay = int(mw[d_key])
                row = {
                    "patient_id": f"OhioT1DM|{pid}" if ds_name == "ohio_t1dm" else pid,
                    "source": "OhioT1DM" if ds_name == "ohio_t1dm" else pid.split("|")[0],
                    "source_start": start, "delay_min": delay, "no_dss": mw[no_key],
                }
                for arm, mod in ctrls.items():
                    cb_ctrl = ForecasterCIB(
                        mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                        dose_penalty=CIB_DOSE_PENALTY, horizon_min=60)
                    g, _, doses = _replay(seg, bw, name, u2ss, f"_abl_{arm}", cb_ctrl)
                    met = _post_gly(g, delay)
                    met["insulin_u"] = round(float(sum(u for _, u in doses)), 2)
                    row[arm] = met
                all_rows.append(row)

    arms_order = ("no_dss", "baseline", "b20", "b0.1", "b1", "b100", "uniform", "ctx", "wrapper", "si0.5", "si2", "pump")
    summary = {}
    for arm in arms_order:
        summary[arm] = {
            k: dict(
                mean=round(float(np.mean([r[arm].get(k, 0.0) for r in all_rows])), 2),
                std=round(float(np.std([r[arm].get(k, 0.0) for r in all_rows], ddof=1)), 2),
            )
            for k in ("tir", "tbr", "tar", "insulin_u")
        }
    dest = ROOT / "results" / "revision" / "ablation_replay.json"
    payload = {"n_windows": len(all_rows), "summary": summary, "windows": all_rows}
    dest.write_text(json.dumps(payload, indent=2))
    print("ABLATION REPLAY DONE", dest, flush=True)
    return payload


ALT_OUT = ROOT / "results" / "revision" / "alt_replay.json"
ALT_ARMS = ("monotone", "mlp", "mlp_docas", "penalty", "lags_baseline", "lags_docas")


def _alt_wrap(mdl, data, level=False):
    gi = int(data.cgm_idx)
    return PaperGlucoseModel(
        mdl, data.train_scalers, data.cgm_scaler,
        predicts_level=bool(level), level_scaler=data.train_scalers[gi],
    )


def _alt_fit(pid: str) -> dict:
    import copy
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    span = float(m.insulin_span_u(data, horizon_min=60))
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    aligner, fit_kw = mc._aligner(data, span, 60)
    src = mc._tail(data)
    mono = train_monotone(
        data.X_train, data.y_train, insulin_idx=data.insulin_idx,
        model_kwargs={"random_seed": 42})
    mlp = train_mlp(src.X_train, src.y_train, model_kwargs={"seed": 42})
    pen = PenaltyMLPForecaster(aligner=aligner, seed=42).fit(
        src.X_train, src.y_train, n_factual=len(src.X_train))
    old = DOCAS.train_fn
    try:
        def _mlp_train(X, y, model_kwargs=None, n_factual=None, **__):
            return train_mlp(X, y, model_kwargs={"seed": 42})
        DOCAS.train_fn = _mlp_train
        mlp_docas = DOCAS(**{**fit_kw, "seed": 42}).fit(
            src.X_train, src.y_train, baseline_model=mlp)
        level = bool(mlp_docas.get("predicts_level", False))
        mlp_docas = mlp_docas["model"]
    finally:
        DOCAS.train_fn = old
    lag = copy.deepcopy(m.load_patient(
        str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60), lookback=mc.LAG_LOOKBACK))
    mc._patch_lag_layout(lag)
    lag_span = float(m.insulin_span_u(lag, horizon_min=60))
    DOCAS.SPAN = lag_span
    _, lag_fit = mc._aligner(lag, lag_span, 60)
    lag_base = train_baseline(lag.X_train, lag.y_train, model_kwargs={"random_seed": 42})
    try:
        DOCAS.train_fn = train_baseline
        lag_res = DOCAS(**{**lag_fit, "seed": 42}).fit(
            lag.X_train, lag.y_train, baseline_model=lag_base)
        lag_level = bool(lag_res.get("predicts_level", False))
        lag_docas = lag_res["model"]
    finally:
        DOCAS.train_fn = old
        DOCAS.SPAN = span
        DOCAS.SEED = 42
        sync_cfg(SEED=42)
    cols = list(data.feature_names)
    lag_cols = list(lag.feature_names)
    return {
        "monotone": (_alt_wrap(mono, data), cols, 1),
        "mlp": (_alt_wrap(mlp, src), cols, 1),
        "mlp_docas": (_alt_wrap(mlp_docas, src, level), cols, 1),
        "penalty": (_alt_wrap(pen, src), cols, 1),
        "lags_baseline": (_alt_wrap(lag_base, lag), lag_cols, mc.LAG_LOOKBACK),
        "lags_docas": (_alt_wrap(lag_docas, lag, lag_level), lag_cols, mc.LAG_LOOKBACK),
    }


def _alt_one(pid: str, wins: list[dict]) -> list[dict]:
    print(f"fit {pid} n={len(wins)}", flush=True)
    arms = _alt_fit(pid)
    ds = "ohio_t1dm" if pid.startswith("OhioT1DM|") else "metabonet_full"
    short = pid.split("|", 1)[1] if ds == "ohio_t1dm" else pid
    m.set_dataset(ds)
    meta = m.load_patient_meta(str(m.DATA_ROOT), short)
    bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
    trace = _load_trace(short, dataset=ds, split="test")
    win_map = {int(w["source_start"]): w for w in _windows(trace)}
    APPLY_HYPO_GUARD = True
    rows = []
    for mw in wins:
        start = int(mw["source_start"])
        seg = win_map[start]["trace"]
        name = _twin_name(short, "catboost", start, dataset=ds)
        feat = paper_feat_arrays(seg, ph_steps=horizon_steps(60))
        delay = int(mw["delay_min"])
        row = {"patient_id": pid, "source": pid.split("|", 1)[0], "source_start": start, "delay_min": delay}
        for arm, (mod, cols, lb) in arms.items():
            cb = ForecasterCIB(
                mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                dose_penalty=CIB_DOSE_PENALTY, horizon_min=60, lookback=lb,
            )
            g, _, doses = _replay(seg, bw, name, u2ss, f"_alt_{arm}", cb)
            met = _post_gly(g, delay)
            met["insulin_u"] = round(float(sum(u for _, u in doses)), 2)
            row[arm] = met
        rows.append(row)
    return rows


def run_alt_replay() -> None:
    windows = ROOT / "results" / "revision" / "ablation_replay.json"
    payload = json.loads(windows.read_text())
    by = defaultdict(list)
    for w in payload["windows"]:
        by[str(w["patient_id"])].append(w)
    packed = Parallel(n_jobs=4, backend="loky")(
        delayed(_alt_one)(pid, wins) for pid, wins in sorted(by.items())
    )
    rows = [r for part in packed for r in part]

    def _summ(arm):
        out = {}
        for k in ("tir", "tbr", "tar", "insulin_u"):
            vals = [r[arm][k] for r in rows]
            out[k] = {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
            }
        return out

    summary = {arm: _summ(arm) for arm in ALT_ARMS}
    ALT_OUT.write_text(json.dumps({"n_windows": len(rows), "summary": summary, "windows": rows}, indent=2))
    print("ALT REPLAY DONE", flush=True)


def run_revision() -> None:
    run_open_replay()
    run_ablation_replay()
    run_alt_replay()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "unguarded":
        for dataset in REPLAY_DATASETS:
            run_unguarded(dataset=dataset)
    elif len(sys.argv) > 1 and sys.argv[1] == "floor70":
        for dataset in REPLAY_DATASETS:
            run_floor(dataset=dataset, floor=70.0)
    elif len(sys.argv) > 1 and sys.argv[1] == "clinical":
        for dataset in REPLAY_DATASETS:
            run_clinical_tau_replay(dataset=dataset)
    elif len(sys.argv) > 1 and sys.argv[1] == "revision":
        run_revision()
    else:
        maybe_run()
