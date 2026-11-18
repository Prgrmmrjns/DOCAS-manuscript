"""Ablations: context-adjusted DOCAS, budgets, wrapper, misspecification, sample weight.

uv run python scripts/ablation.py cohort          # Table 3 and context-adjusted DOCAS
uv run python scripts/ablation.py                 # Table 5 forecast arms
uv run python scripts/ablation.py weight          # sample weight 20 vs 20 copies
uv run python scripts/ablation.py weight-replay   # those two on the 25 windows
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from forecasters import CAT_KWARGS, CatBoostForecaster, train_baseline
from core import (abs_tau_drop, ambient_exact_tau, ape_selection_map,
                  calibrated_target, docas_fit_kw, frac_tau_drop,
                  INSULIN_DIA_MIN, INSULIN_PEAK_MIN, insulin_action_fraction,
                  select_by_ape, sync_cfg, tau_from_cfg)
import main as m
from preprocessing import PatientData, horizon_steps, rmse_mgdl
from main import (
    DATA_ROOT, DOCAS, RESULTS, STUDY_CFG, insulin_span_u, mean_std,
    set_dataset, train_baseline_horizon,
)

def budget_path(dataset: str = "ohio_t1dm") -> Path:
    tag = "ohio" if dataset == "ohio_t1dm" else dataset
    return RESULTS / "ablation" / f"budget_{tag}.json"


BUDGET_JSON = budget_path("ohio_t1dm")
TABLE_BUDGETS = (0.1, 1.0, 100.0)
EXACT_TAU_JSON = RESULTS / "ablation" / "exact_tau_ohio.json"
ABLATION_HORIZONS = (30, 60)
ABLATION_DATASETS = ("ohio_t1dm", "azt1d")
SI_SCALES = (0.5, 1.0, 2.0)
SYNTH_BUDGETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0)


def ablation_cohort_path(dataset: str = "ohio_t1dm") -> Path:
    tag = "ohio" if dataset == "ohio_t1dm" else dataset
    return RESULTS / "ablation" / f"{tag}_cohort.json"


def misspec_path(dataset: str = "ohio_t1dm") -> Path:
    tag = "ohio" if dataset == "ohio_t1dm" else dataset
    return RESULTS / "ablation" / f"misspec_{tag}.json"


# Ohio paths kept as module attributes for tables / main.
OUT_JSON = ablation_cohort_path("ohio_t1dm")
MISSPEC_JSON = misspec_path("ohio_t1dm")


def _rmse_mgdl(model, X, y, tgt_s, *, predicts_level=False, level_s=None, gi=0):
    X = np.asarray(X, float)
    pred = model.predict(X)
    if predicts_level:
        level = level_s.inverse_transform(np.asarray(pred, float).reshape(-1, 1)).ravel()
        cgm0 = level_s.inverse_transform(X[:, gi: gi + 1]).ravel()
        pred = tgt_s.transform((level - cgm0).reshape(-1, 1)).ravel()
    return float(rmse_mgdl(y, pred, tgt_s))


def _agg_metric_dicts(arms: list[dict], *, keys=("alignment_error", "rmse_delta_mgdl")) -> dict:
    return {k: mean_std([float(a[k]) for a in arms]) for k in keys}


def _align_error(model, data, span, horizon_min, *, cfg=None):
    cfg = dict(cfg or STUDY_CFG)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    target, drop, u_ref, mode = tau_from_cfg(int(horizon_min), span, cfg)
    a_s, dy_s, y_s = data.train_scalers[ji], data.cgm_scaler, data.train_scalers[gi]
    aligner = DOCAS(
        target,
        intervention_idx=ji, outcome_idx=gi, action_scaler=a_s, level_scaler=y_s,
        delta_scaler=dy_s, span=float(span), tau_drop=drop, u_ref=u_ref, tau_mode=mode,
        level_ref=str(cfg.get("level_ref", "current")), predicts_level=False,
    )
    return float(aligner.alignment(model, np.asarray(data.X_test, float)))


def _fit_variant(
    data, baseline, base_rmse, span, horizon_min, *,
    emp_frac=None, synth_ratio=None, cfg=None,
):
    cfg = dict(cfg or STUDY_CFG)
    over = {}
    if emp_frac is not None:
        over["emp_frac"] = emp_frac
    if synth_ratio is not None:
        over["synth_ratio"] = float(synth_ratio)
    fit_kw, _, u_ref, _, _ = docas_fit_kw(data, span, horizon_min, cfg=cfg, **over)
    result = DOCAS(**fit_kw).fit(data.X_train, data.y_train, baseline_model=baseline)
    model = result["model"]
    mode = str(fit_kw.get("tau_mode", cfg.get("tau_mode", "fractional")))
    predicts_level = bool(result.get("predicts_level", False))
    gi = data.feature_names.index("CGM")
    ae = _align_error(model, data, span, horizon_min, cfg=cfg)
    rm = _rmse_mgdl(
        model, data.X_test, data.y_test, data.cgm_scaler,
        predicts_level=predicts_level, level_s=data.train_scalers[gi], gi=gi)
    return dict(
        n_synth=int(result.get("n_synth", 0)), alignment_error=ae,
        rmse_delta_mgdl=rm - base_rmse, test_rmse_mgdl=rm, ape_mgdl=ae + rm,
        tau_mode=mode,
    )


class TauWrapForecaster:
    """No-insulin CatBoost trunk + post-hoc τ on the insulin feature / audit probe."""

    def fit(self, X, y, *, target_R, drop, u_ref, ji, gi, cgm_s, tgt_s,
            tau_mode="fractional", level_ref="current", span=1.0, **_):
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        self.ji, self.gi = int(ji), int(gi)
        self._cgm_s, self._tgt_s = cgm_s, tgt_s
        self.drop, self.u_ref = float(drop), float(u_ref)
        self.tau_mode = str(tau_mode).lower()
        self.level_ref = str(level_ref).lower()
        self.span = max(float(span), 1e-12)
        ug = np.linspace(0.0, 1.0, 1201)
        self._R_u = ug
        self._R = np.asarray(target_R(ug), float).ravel()
        self._r_ref = float(np.interp(self.u_ref, ug, self._R)) or 1.0
        self.keep = [c for c in range(X.shape[1]) if c != self.ji]
        self.base = CatBoostForecaster().fit(X[:, self.keep], y)
        return self

    def _ru(self, u):
        return float(np.interp(float(u), self._R_u, self._R)) / self._r_ref

    def _fut_level(self, g, d, u):
        """Future CGM level under τ at ambient g, trunk delta d, probe u."""
        f0 = g + d
        ru = self._ru(u)
        if not self.tau_mode.startswith("frac"):
            return f0 + self.drop * ru
        if self.level_ref.startswith("cur"):
            return f0 + g * self.drop * ru
        return f0 * (1.0 + self.drop * ru)

    def predict(self, X):
        """Scaled ΔCGM = τ-adjusted future − ambient (responds to insulin column)."""
        X = np.atleast_2d(np.asarray(X, float))
        d_s = self.base.predict(X[:, self.keep]).ravel()
        dy_s, dy_m = float(self._tgt_s.scale_[0]), float(self._tgt_s.min_[0])
        g_s, g_m = float(self._cgm_s.scale_[0]), float(self._cgm_s.min_[0])
        a_s = float(getattr(self, "_a_scale", 1.0))
        a_m = float(getattr(self, "_a_min", 0.0))
        d = (d_s - dy_m) / dy_s
        g = (X[:, self.gi] - g_m) / g_s
        # Insulin column is MinMax-scaled; invert with stored action scaler if set.
        if hasattr(self, "_a_s") and self._a_s is not None:
            iob = self._a_s.inverse_transform(X[:, self.ji: self.ji + 1]).ravel()
        else:
            iob = X[:, self.ji]  # assume already roughly in U if no scaler
        u = np.clip(iob / self.span, 0.0, 1.0)
        fut = np.array([self._fut_level(gi, di, ui) for gi, di, ui in zip(g, d, u)])
        return (fut - g) * dy_s + dy_m

    def predict_intervention(self, X, u, span):
        """Scaled ΔCGM per probe dose, for future(..., level=False)."""
        X = np.atleast_2d(np.asarray(X, float))
        u = np.asarray(u, float).ravel()
        d_s = self.base.predict(X[:, self.keep]).ravel()
        dy_s, dy_m = float(self._tgt_s.scale_[0]), float(self._tgt_s.min_[0])
        g_s, g_m = float(self._cgm_s.scale_[0]), float(self._cgm_s.min_[0])
        d = (d_s - dy_m) / dy_s
        g = (X[:, self.gi] - g_m) / g_s
        return np.stack([(self._fut_level(g, d, uu) - g) * dy_s + dy_m for uu in u], 1)


def _fit_tauwrap(data, base_rmse, span, horizon_min, *, cfg=None) -> dict:
    cfg = dict(cfg or STUDY_CFG)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    si, bw = float(cfg["SI"]), float(cfg["body_weight_kg"])
    mode = str(cfg.get("tau_mode", "fractional")).lower()
    drop_mode = str(cfg.get("drop_mode", "endpoint"))
    drop, u_ref = (
        frac_tau_drop(int(horizon_min), span, si=si, body_weight_kg=bw, drop_mode=drop_mode)
        if mode.startswith("frac") else
        abs_tau_drop(int(horizon_min), span, si=si, body_weight_kg=bw, drop_mode=drop_mode))
    target = calibrated_target(
        int(horizon_min), body_weight_kg=bw, SI=si, span=span, drop_mode=drop_mode)
    model = TauWrapForecaster().fit(
        data.X_train, data.y_train, target_R=target, drop=drop, u_ref=u_ref,
        ji=ji, gi=gi, cgm_s=data.train_scalers[gi], tgt_s=data.cgm_scaler,
        tau_mode=mode, level_ref=str(cfg.get("level_ref", "current")), span=float(span))
    model._a_s = data.train_scalers[ji]
    ae = _align_error(model, data, span, horizon_min, cfg=cfg)
    rm = _rmse_mgdl(model, data.X_test, data.y_test, data.cgm_scaler)
    return dict(
        n_synth=0, alignment_error=ae,
        rmse_delta_mgdl=rm - base_rmse, test_rmse_mgdl=rm, ape_mgdl=ae + rm,
        tau_mode=mode,
    )


def _ablation_one_patient(pid: str, horizon_min: int) -> dict:
    ph = int(horizon_min)
    data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    baseline = m.load_baseline_model(ph, pid)
    span = float(insulin_span_u(data, horizon_min=ph))
    DOCAS.SPAN = span
    base_rmse = _rmse_mgdl(baseline, data.X_test, data.y_test, data.cgm_scaler)
    default = _fit_variant(data, baseline, base_rmse, span, ph)
    return dict(
        patient_id=str(pid), horizon_min=ph,
        baseline_test_rmse_mgdl=base_rmse, insulin_span_u=span,
        default_docas=default,
        tauwrap=_fit_tauwrap(data, base_rmse, span, ph),
        uniform_only=_fit_variant(
            data, baseline, base_rmse, span, ph, emp_frac=0.0),
    )


def _ablation_horizon_metrics(per_patient: list[dict]) -> dict:
    keys = ("alignment_error", "rmse_delta_mgdl", "test_rmse_mgdl", "ape_mgdl")
    arms = ("default_docas", "tauwrap", "uniform_only")
    return {k: _agg_metric_dicts([r[k] for r in per_patient], keys=keys) for k in arms}


def _arm_cell(arm: dict, *, base_mean: float | None = None) -> str:
    a = arm["alignment_error"]["mean"]
    r = arm["test_rmse_mgdl"]["mean"] if "test_rmse_mgdl" in arm else base_mean + arm["rmse_delta_mgdl"]["mean"]
    return f"{a:.2f}/{r:.2f}"


def print_ablation_summary(payload: dict) -> None:
    for ph in ABLATION_HORIZONS:
        h = payload["horizons"][str(ph)]
        b = h["baseline_test_rmse_mgdl"]["mean"]
        met = h["metrics"]
        print(
            f"  PH{ph}  base {b:.2f} | DOCAS {_arm_cell(met['default_docas'], base_mean=b)} | "
            f"wrap {_arm_cell(met['tauwrap'])}",
            flush=True,
        )


def run_ablation(*, dataset: str = "ohio_t1dm") -> dict:
    set_dataset(dataset)
    pids = list(m.list_patients(str(DATA_ROOT)))
    out_path = ablation_cohort_path(dataset)
    print(f"  Ablation {dataset} n={len(pids)} horizons={ABLATION_HORIZONS}", flush=True)
    horizons_out = {}
    for ph in ABLATION_HORIZONS:
        per_patient = [_ablation_one_patient(pid, ph) for pid in pids]
        horizons_out[str(ph)] = dict(
            horizon_min=int(ph),
            baseline_test_rmse_mgdl=mean_std(
                [float(r["baseline_test_rmse_mgdl"]) for r in per_patient]),
            metrics=_ablation_horizon_metrics(per_patient),
            per_patient=per_patient,
        )
    payload = dict(
        dataset=dataset,
        patient_ids=[str(p) for p in pids], horizons=horizons_out,
        n_grid_points=int(DOCAS.N_GRID_POINTS),
        protocol=str(STUDY_CFG.get("tau_mode", "fractional")) + "_tau",
        tau_mode=str(STUDY_CFG.get("tau_mode", "fractional")),
    )
    m._write_json(out_path, payload)
    print_ablation_summary(payload)
    return payload


def run_ablations(*, datasets=None) -> None:
    for ds in (datasets or ABLATION_DATASETS):
        run_ablation(dataset=ds)


def run_uniform_only(*, dataset: str = "ohio_t1dm") -> dict:
    """Add an emp_frac=0 arm to an existing ablation cohort (no refit of the rest)."""
    from joblib import Parallel, delayed

    set_dataset(dataset)
    path = ablation_cohort_path(dataset)
    payload = json.loads(path.read_text())
    pids = list(payload["patient_ids"])

    def one(pid, ph):
        set_dataset(dataset)
        data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
        baseline = m.load_baseline_model(ph, pid)
        span = float(insulin_span_u(data, horizon_min=ph))
        DOCAS.SPAN = span
        base_rmse = _rmse_mgdl(baseline, data.X_test, data.y_test, data.cgm_scaler)
        return _fit_variant(data, baseline, base_rmse, span, ph, emp_frac=0.0)

    for ph in ABLATION_HORIZONS:
        rows = payload["horizons"][str(ph)]["per_patient"]
        extra = Parallel(n_jobs=4, backend="loky")(
            delayed(one)(r["patient_id"], ph) for r in rows)
        for r, uni in zip(rows, extra):
            r["uniform_only"] = uni
        payload["horizons"][str(ph)]["metrics"] = _ablation_horizon_metrics(rows)
        met = payload["horizons"][str(ph)]["metrics"]["uniform_only"]
        print(f"  uniform-only {dataset} PH{ph}: AE {met['alignment_error']['mean']:.2f} "
              f"RMSE {met['test_rmse_mgdl']['mean']:.2f}", flush=True)
    m._write_json(path, payload)
    return payload


def _budget_one_patient(
    pid: str, horizon_min: int, *, prev: dict | None = None,
    budgets=None, skip_val: bool = False, dataset: str | None = None,
) -> dict:
    """Synthetic-budget frontier: selection on a validation tail, frontier on test.

    `prev` is a previously computed row for the same patient/horizon; budgets it
    already holds are carried over instead of refitted.
    """
    ph = int(horizon_min)
    budgets = tuple(budgets if budgets is not None else SYNTH_BUDGETS)
    done = dict((prev or {}).get("budgets", {}))
    todo = [b for b in budgets if f"{b:g}x" not in done]
    if not todo and prev is not None:
        return prev
    if dataset:
        set_dataset(dataset)
    data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    span = float(insulin_span_u(data, horizon_min=ph))
    DOCAS.SPAN = span
    X, y = np.asarray(data.X_train, float), np.asarray(data.y_train, float).ravel()
    cut = max(1, int(round(0.8 * len(y))))
    if prev is not None:
        row = dict(prev)
    else:
        base = m.load_baseline_model(ph, pid)
        base_ae = _align_error(base, data, span, ph)
        base_rmse = _rmse_mgdl(base, data.X_test, data.y_test, data.cgm_scaler)
        row = dict(
            patient_id=str(pid), horizon_min=ph,
            baseline=dict(synth_ratio=0.0, alignment_error=base_ae,
                          test_rmse_mgdl=base_rmse, ape_mgdl=base_ae + base_rmse),
            budgets=done)
    for b in todo:
        fit_kw, _, _, _, _ = docas_fit_kw(data, span, ph, cfg=STUDY_CFG, synth_ratio=float(b))
        if skip_val:
            sel_ae = sel_rmse = float("nan")
        else:
            sel = DOCAS(**fit_kw).fit(X[:cut], y[:cut], baseline_model=None)["model"]
            sel_ae = _align_error_on(sel, data, span, ph, X[cut:])
            sel_rmse = _rmse_mgdl(sel, X[cut:], y[cut:], data.cgm_scaler)
        full = DOCAS(**fit_kw).fit(X, y, baseline_model=None)["model"]
        row["budgets"][f"{b:g}x"] = dict(
            synth_ratio=float(b),
            val_alignment_error=sel_ae, val_rmse_mgdl=sel_rmse,
            val_ape_mgdl=sel_ae + sel_rmse,
            alignment_error=_align_error(full, data, span, ph),
            test_rmse_mgdl=_rmse_mgdl(full, data.X_test, data.y_test, data.cgm_scaler),
        )
    for v in row["budgets"].values():
        v["ape_mgdl"] = v["alignment_error"] + v["test_rmse_mgdl"]
    return row


def _align_error_on(model, data, span, horizon_min, X):
    """Alignment error on an arbitrary context matrix (audit curves need no labels)."""
    cfg = dict(STUDY_CFG)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    si, bw = float(cfg["SI"]), float(cfg["body_weight_kg"])
    drop_mode = str(cfg.get("drop_mode", "endpoint"))
    drop, u_ref = frac_tau_drop(int(horizon_min), span, si=si, body_weight_kg=bw,
                                drop_mode=drop_mode)
    aligner = DOCAS(
        calibrated_target(int(horizon_min), body_weight_kg=bw, SI=si, span=span,
                          drop_mode=drop_mode),
        intervention_idx=ji, outcome_idx=gi, action_scaler=data.train_scalers[ji],
        level_scaler=data.train_scalers[gi], delta_scaler=data.cgm_scaler,
        span=float(span), tau_drop=drop, u_ref=u_ref,
        tau_mode=str(cfg.get("tau_mode", "fractional")),
        level_ref=str(cfg.get("level_ref", "current")), predicts_level=False,
    )
    return float(aligner.alignment(model, np.asarray(X, float)))


def _val_points(metrics: dict) -> dict:
    """Validation (RMSE, AE) per selectable budget; no test data enters the selection.

    Budgets above `main.BUDGET_CAP` are swept to draw the frontier but are not
    offered to the selector, which is a compute cap rather than a metric one.
    """
    return {b: (metrics[f"{b:g}x"]["val_rmse_mgdl"]["mean"],
                metrics[f"{b:g}x"]["val_alignment_error"]["mean"])
            for b in SYNTH_BUDGETS if b <= float(m.BUDGET_CAP)}


def select_budget(metrics: dict, *, ae_weight=None) -> float:
    """Synthetic budget minimising validation APE_λ = RMSE + λ·AE, for any λ ≥ 0."""
    return float(select_by_ape(_val_points(metrics), ae_weight=ae_weight))


def budget_selection_map(metrics: dict) -> list[dict]:
    """λ intervals and the budget each one selects, covering all λ ≥ 0."""
    return [dict(ae_weight_from=lo, ae_weight_to=hi, synth_ratio=float(b))
            for lo, hi, b in ape_selection_map(_val_points(metrics))]


def run_budget_sweep(*, dataset: str = "ohio_t1dm", budgets=None, skip_val: bool = False) -> dict:
    """Trace the RMSE-vs-alignment frontier over synthetic budgets.

    The preset is selected on a held-out validation tail of the training window,
    never on test; test numbers are reported only to draw the frontier. The
    sweep also stores the λ intervals over which each budget wins, so a user
    who weights alignment differently reads any λ off the same sweep.
    """
    from joblib import Parallel, delayed

    budgets = tuple(budgets if budgets is not None else SYNTH_BUDGETS)
    set_dataset(dataset)
    out_path = budget_path(dataset)
    pids = list(m.list_patients(str(DATA_ROOT)))
    old = json.loads(out_path.read_text()) if out_path.exists() else {"horizons": {}}
    horizons_out = {}
    for ph in ABLATION_HORIZONS:
        cached = {r["patient_id"]: r
                  for r in old["horizons"].get(str(ph), {}).get("per_patient", [])}
        rows = Parallel(n_jobs=4, backend="loky")(
            delayed(_budget_one_patient)(
                pid, ph, prev=cached.get(str(pid)), budgets=budgets,
                skip_val=skip_val, dataset=dataset)
            for pid in pids)
        keys = ("val_alignment_error", "val_rmse_mgdl", "val_ape_mgdl",
                "alignment_error", "test_rmse_mgdl", "ape_mgdl")
        metrics = {
            f"{b:g}x": {k: mean_std([r["budgets"][f"{b:g}x"][k] for r in rows]) for k in keys}
            for b in budgets
        }
        best = select_budget(metrics) if not skip_val else float("nan")
        sel_map = budget_selection_map(metrics) if not skip_val else []
        baseline = {k: mean_std([r["baseline"][k] for r in rows])
                    for k in ("alignment_error", "test_rmse_mgdl", "ape_mgdl")}
        horizons_out[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=rows,
                                     baseline=baseline, selected_budget=best,
                                     ae_weight=float(m.AE_WEIGHT),
                                     selection_map=sel_map)
        print(f"  budget {dataset} PH{ph}: " + " | ".join(
            f"{b:g}x AE={metrics[f'{b:g}x']['alignment_error']['mean']:.2f}"
            f"/RMSE={metrics[f'{b:g}x']['test_rmse_mgdl']['mean']:.2f}"
            for b in budgets), flush=True)
    payload = dict(patient_ids=[str(p) for p in pids], horizons=horizons_out,
                   budgets=list(budgets), ae_weight=float(m.AE_WEIGHT),
                   **m._protocol_meta())
    m._write_json(out_path, payload)
    return payload


def run_exact_tau() -> dict:
    """Re-simulate τ at each row's ambient glucose instead of scaling the G0 response.

    The study target measures the bolus response once at G0 and rescales it in
    proportion to ambient glucose, but the plant responds sub-proportionally above
    its basal glucose and more strongly below it. Training and scoring on a target
    re-simulated per row separates the size of that modelling gap from its effect
    on the result; the study-trained model is also scored against the re-simulated
    target, which recovers the gap rather than any property of the fit.
    """
    set_dataset("ohio_t1dm")
    pids = list(m.list_patients(str(DATA_ROOT)))
    horizons_out = {}
    for ph in ABLATION_HORIZONS:
        rows = []
        for pid in pids:
            data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
            baseline = m.load_baseline_model(ph, pid)
            span = float(insulin_span_u(data, horizon_min=ph))
            DOCAS.SPAN = span
            gi = data.feature_names.index("CGM")
            ctx = np.asarray(data.X_test, float)
            tau_x = ambient_exact_tau(ph, span, si=float(STUDY_CFG["SI"]),
                                      body_weight_kg=float(STUDY_CFG["body_weight_kg"]))
            fit_kw, _, _, _, _ = docas_fit_kw(data, span, ph, cfg=STUDY_CFG)
            study, exact = DOCAS(**fit_kw), DOCAS(**fit_kw, tau_exact=tau_x)
            r_s = study.fit(data.X_train, data.y_train, baseline_model=baseline)
            r_x = exact.fit(data.X_train, data.y_train, baseline_model=baseline)
            rm = lambda res: _rmse_mgdl(
                res["model"], data.X_test, data.y_test, data.cgm_scaler,
                predicts_level=bool(res.get("predicts_level", False)),
                level_s=data.train_scalers[gi], gi=gi)
            rows.append(dict(
                patient_id=str(pid), horizon_min=ph,
                baseline_test_rmse_mgdl=_rmse_mgdl(
                    baseline, data.X_test, data.y_test, data.cgm_scaler),
                base_ae_study=study.alignment(baseline, ctx),
                base_ae_exact=exact.alignment(baseline, ctx),
                docas_ae_study=study.alignment(r_s["model"], ctx),
                docas_ae_exact=exact.alignment(r_x["model"], ctx),
                docas_ae_study_model_exact_target=exact.alignment(r_s["model"], ctx),
                docas_rmse_study=rm(r_s), docas_rmse_exact=rm(r_x),
                tau_gap=_tau_gap(study, tau_x, ctx)))
        keys = [k for k in rows[0] if k not in ("patient_id", "horizon_min")]
        metrics = {k: mean_std([r[k] for r in rows]) for k in keys}
        horizons_out[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=rows)
        print(f"  exact-τ PH{ph}: study {metrics['base_ae_study']['mean']:.2f}→"
              f"{metrics['docas_ae_study']['mean']:.2f} | exact "
              f"{metrics['base_ae_exact']['mean']:.2f}→{metrics['docas_ae_exact']['mean']:.2f} | "
              f"target gap {metrics['tau_gap']['mean']:.2f}", flush=True)
    payload = dict(patient_ids=[str(p) for p in pids], horizons=horizons_out)
    m._write_json(EXACT_TAU_JSON, payload)
    return payload


def _tau_gap(aligner, tau_exact, ctx) -> float:
    """RMS distance between the study target and the re-simulated one, same probe grid."""
    from core import _inv, _shape
    u = aligner.u_grid
    g = _inv(aligner.y_s, ctx[:, aligner.yi])
    r_ref = float(_shape(aligner.target, [aligner.u_ref])[0]) or 1.0
    study = (g * aligner.tau_drop)[:, None] * (_shape(aligner.target, u) / r_ref)[None, :]
    return float(np.sqrt(np.mean((study - tau_exact(u[None, :], g[:, None])) ** 2)))


def run_misspec(*, dataset: str = "ohio_t1dm") -> dict:
    """Train DOCAS at 0.5× / 1× / 2× SI; score AE against the reference (1×) τ."""
    set_dataset(dataset)
    pids = list(m.list_patients(str(DATA_ROOT)))
    out_path = misspec_path(dataset)
    print(f"  Misspec {dataset} n={len(pids)}", flush=True)
    ref_si = float(STUDY_CFG["SI"])
    horizons_out = {}
    for ph in ABLATION_HORIZONS:
        rows = []
        for pid in pids:
            data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
            baseline = m.load_baseline_model(ph, pid)
            span = float(insulin_span_u(data, horizon_min=ph))
            DOCAS.SPAN = span
            base_rmse = _rmse_mgdl(baseline, data.X_test, data.y_test, data.cgm_scaler)
            arm = {}
            gi = data.feature_names.index("CGM")
            for scale in SI_SCALES:
                cfg = {**STUDY_CFG, "SI": ref_si * float(scale)}
                fit_kw, _, _, _, _ = docas_fit_kw(data, span, ph, cfg=cfg)
                res = DOCAS(**fit_kw).fit(
                    data.X_train, data.y_train, baseline_model=baseline)
                model = res["model"]
                # AE against the target actually trained on: confirms the fit is
                # faithful to its own tau rather than merely bad at learning it.
                ae_own = _align_error(model, data, span, ph, cfg=cfg)
                ae_ref = _align_error(model, data, span, ph, cfg={**STUDY_CFG, "SI": ref_si})
                rm = _rmse_mgdl(
                    model, data.X_test, data.y_test, data.cgm_scaler,
                    predicts_level=bool(res.get("predicts_level", False)),
                    level_s=data.train_scalers[gi], gi=gi)
                arm[f"si_{scale:g}x"] = dict(
                    alignment_error=ae_ref, alignment_error_own_tau=ae_own,
                    test_rmse_mgdl=rm,
                    ape_mgdl=ae_ref + rm, si_scale=float(scale),
                    rmse_delta_mgdl=rm - base_rmse, n_synth=int(res.get("n_synth", 0)),
                )
            rows.append(dict(
                patient_id=str(pid), horizon_min=ph,
                baseline_test_rmse_mgdl=base_rmse, **arm))
        keys = ("alignment_error", "alignment_error_own_tau", "test_rmse_mgdl", "ape_mgdl")
        metrics = {
            f"si_{s:g}x": _agg_metric_dicts([r[f"si_{s:g}x"] for r in rows], keys=keys)
            for s in SI_SCALES
        }
        horizons_out[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=rows)
        print(f"  misspec PH{ph}: " + " | ".join(
            f"{s:g}x AE(ref)={metrics[f'si_{s:g}x']['alignment_error']['mean']:.2f}"
            f"/AE(own)={metrics[f'si_{s:g}x']['alignment_error_own_tau']['mean']:.2f}"
            for s in SI_SCALES), flush=True)
    payload = dict(dataset=dataset, patient_ids=[str(p) for p in pids],
                   horizons=horizons_out, ref_SI=ref_si, si_scales=list(SI_SCALES))
    m._write_json(out_path, payload)
    return payload


def run_misspec_all(*, datasets=None) -> None:
    for ds in (datasets or ABLATION_DATASETS):
        run_misspec(dataset=ds)


def run_clinical_tau_all(*, datasets=None) -> None:
    for ds in (datasets or ABLATION_DATASETS):
        run_clinical_tau(dataset=ds)


# --- Clinical τ from pump settings only ---------------------------------------
# The study τ reads insulin sensitivity off ReplayBG's prior, so the simulator
# supplies both the target and, in the replay, the simulated patient. This τ uses
# neither: its size is the patient's own 1800-rule correction factor and its time
# course is the insulin-action curve programmed in pumps, so the alignment result
# can be tested against a target with no UVa/Padova content at all.
CLINICAL_RULE_CONSTANT = 1800.0   # mg/dL·U per U/day (standard correction rule)


def clinical_tau_path(dataset: str = "ohio_t1dm") -> Path:
    tag = "ohio" if dataset == "ohio_t1dm" else dataset
    return RESULTS / "ablation" / f"clinical_tau_{tag}.json"


def clinical_cf(tdd_u: float) -> float:
    """1800-rule correction factor (mg/dL per U) from total daily insulin."""
    return CLINICAL_RULE_CONSTANT / float(tdd_u)


def clinical_cfg(tdd_u: float, *, cfg: dict | None = None) -> dict:
    """STUDY_CFG with τ replaced by the patient's pump-settings target."""
    return {**dict(cfg or STUDY_CFG), "tau_kind": "pump",
            "cf_mgdl_per_u": clinical_cf(tdd_u)}


def run_clinical_tau(*, dataset: str = "ohio_t1dm") -> dict:
    """Retrain DOCAS against each patient's pump-settings τ (no UVa/Padova)."""
    set_dataset(dataset)
    pids = list(m.list_patients(str(DATA_ROOT)))
    print(f"  Clinical τ {dataset} n={len(pids)}", flush=True)
    horizons_out = {}
    for ph in ABLATION_HORIZONS:
        rows = []
        act = float(insulin_action_fraction(ph))
        for pid in pids:
            meta = m.load_patient_meta(str(DATA_ROOT), str(pid))
            tdd = meta.get("tdd_u")
            if not tdd:
                continue  # no logged insulin total, so no clinical correction factor
            data = m.load_patient(str(DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
            baseline = m.load_baseline_model(ph, pid)
            span = float(insulin_span_u(data, horizon_min=ph))
            DOCAS.SPAN = span
            gi = data.feature_names.index("CGM")
            cf = clinical_cf(float(tdd))
            cfg = clinical_cfg(float(tdd))
            fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=cfg)
            res = DOCAS(**fit_kw).fit(data.X_train, data.y_train, baseline_model=baseline)
            model = res["model"]
            rows.append(dict(
                patient_id=str(pid), horizon_min=ph, tdd_u=float(tdd),
                clinical_cf_mgdl_per_u=float(cf),
                insulin_action_fraction=act,
                tau_drop_1u_mgdl=float(cf * act),
                alignment_error_own_tau=_align_error(model, data, span, ph, cfg=cfg),
                alignment_error=_align_error(model, data, span, ph, cfg=STUDY_CFG),
                test_rmse_mgdl=_rmse_mgdl(
                    model, data.X_test, data.y_test, data.cgm_scaler,
                    predicts_level=bool(res.get("predicts_level", False)),
                    level_s=data.train_scalers[gi], gi=gi),
                baseline_test_rmse_mgdl=_rmse_mgdl(
                    baseline, data.X_test, data.y_test, data.cgm_scaler),
            ))
        keys = ("alignment_error", "alignment_error_own_tau", "test_rmse_mgdl",
                "tau_drop_1u_mgdl")
        metrics = {"clinical": _agg_metric_dicts(rows, keys=keys)}
        horizons_out[str(ph)] = dict(horizon_min=ph, metrics=metrics, per_patient=rows)
        met = metrics["clinical"]
        print(f"  clinical τ PH{ph}: 1 U -> {met['tau_drop_1u_mgdl']['mean']:.2f} mg/dL | "
              f"AE(own)={met['alignment_error_own_tau']['mean']:.2f} | "
              f"AE(study τ)={met['alignment_error']['mean']:.2f} | "
              f"RMSE={met['test_rmse_mgdl']['mean']:.2f}", flush=True)
    payload = dict(dataset=dataset, patient_ids=[str(r["patient_id"]) for r in
                                                 horizons_out[str(ABLATION_HORIZONS[0])]["per_patient"]],
                   rule_constant=CLINICAL_RULE_CONSTANT, tau_kind="pump",
                   insulin_peak_min=INSULIN_PEAK_MIN, insulin_dia_min=INSULIN_DIA_MIN,
                   horizons=horizons_out)
    m._write_json(clinical_tau_path(dataset), payload)
    return payload


FULL_ABLATION_OUT = Path(__file__).resolve().parents[1] / "results" / "revision" / "full_ablation.json"
FULL_SOURCES = {"OhioT1DM", "AZT1D", "HUPA-UCM", "T1D-UOM"}
FULL_CAP = 8000
FULL_BUDGETS = (0.1, 1.0, 20.0, 100.0)


def _full_tail(data: PatientData, n: int = FULL_CAP) -> PatientData:
    if len(data.y_train) <= n:
        return data
    return PatientData(
        data.X_train[-n:], data.y_train[-n:], data.X_test, data.y_test,
        data.feature_names, data.cgm_scaler, data.train_scalers, data.patient_id,
        data.insulin_idx, data.cgm_idx, data.lookback,
    )


def _fit_against(data, base, base_rmse, span, ph, cfg):
    fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=cfg)
    res = DOCAS(**fit_kw).fit(data.X_train, data.y_train, baseline_model=base)
    model = res["model"]
    gi = data.feature_names.index("CGM")
    own = _align_error(model, data, span, ph, cfg=cfg)
    ref = _align_error(model, data, span, ph, cfg=m.STUDY_CFG)
    rm = _rmse_mgdl(
        model, data.X_test, data.y_test, data.cgm_scaler,
        predicts_level=bool(res.get("predicts_level", False)),
        level_s=data.train_scalers[gi], gi=gi)
    return dict(
        n_synth=int(res.get("n_synth", 0)),
        alignment_error=ref, alignment_error_own=own,
        rmse_delta_mgdl=rm - base_rmse, test_rmse_mgdl=rm, ape_mgdl=ref + rm,
    )


def _full_one(pid: str, ph: int) -> dict:
    import forecasters
    from forecasters import train_baseline
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    try:
        return _full_one_ok(pid, ph, train_baseline)
    except Exception as exc:
        print(f"  abl skip {pid} PH{ph}: {exc}", flush=True)
        return dict(patient_id=str(pid), horizon_min=ph, error=str(exc))


def _full_one_ok(pid: str, ph: int, train_baseline) -> dict:
    import time
    import tracemalloc
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    t0 = time.perf_counter()
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    base_s = time.perf_counter() - t0
    span = float(m.insulin_span_u(data, horizon_min=ph))
    m.DOCAS.SPAN = span
    base_rmse = _rmse_mgdl(base, data.X_test, data.y_test, data.cgm_scaler)
    arms = {}
    fit_s = peak_mb = None
    for b in FULL_BUDGETS:
        src = _full_tail(data) if b >= 100 else data
        timed = abs(float(b) - 20.0) < 1e-9
        if timed:
            tracemalloc.start()
            t1 = time.perf_counter()
        arms[f"b{b:g}"] = _fit_variant(src, base, base_rmse, span, ph, synth_ratio=float(b))
        if timed:
            fit_s = time.perf_counter() - t1
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 ** 2)
    arms["uniform"] = _fit_variant(data, base, base_rmse, span, ph, emp_frac=0.0)
    arms["wrapper"] = _fit_tauwrap(data, base_rmse, span, ph)
    ref_si = float(m.STUDY_CFG["SI"])
    for scale in (0.5, 2.0):
        cfg = {**m.STUDY_CFG, "SI": ref_si * scale}
        arms[f"si{scale:g}"] = _fit_against(data, base, base_rmse, span, ph, cfg)
    meta = m.load_patient_meta(str(m.DATA_ROOT), str(pid))
    tdd = float(meta["tdd_u"])
    arms["pump"] = _fit_against(data, base, base_rmse, span, ph, clinical_cfg(tdd))
    arms["pump"]["clinical_cf"] = clinical_cf(tdd)
    print(
        f"  abl {pid} PH{ph} base {base_rmse:.1f} 20x {arms['b20']['alignment_error']:.2f}/{arms['b20']['test_rmse_mgdl']:.1f}"
        f"  {base_s:.2f}s -> {fit_s:.2f}s",
        flush=True,
    )
    return dict(
        patient_id=str(pid), horizon_min=ph, insulin_span_u=span,
        n_train=int(len(data.y_train)),
        baseline_fit_s=round(base_s, 3),
        docas20_fit_s=None if fit_s is None else round(fit_s, 3),
        docas20_peak_mb=None if peak_mb is None else round(peak_mb, 2),
        baseline_test_rmse_mgdl=base_rmse, arms=arms,
    )


def _full_summarize(rows):
    rows = [r for r in rows if "arms" in r]
    keys = sorted({k for r in rows for k in r["arms"]})
    out = {}
    for k in keys:
        ae = [r["arms"][k]["alignment_error"] for r in rows]
        rm = [r["arms"][k]["test_rmse_mgdl"] for r in rows]
        out[k] = dict(
            alignment_error=dict(mean=float(np.mean(ae)), std=float(np.std(ae, ddof=1))),
            test_rmse_mgdl=dict(mean=float(np.mean(rm)), std=float(np.std(rm, ddof=1))),
        )
    base = [r["baseline_test_rmse_mgdl"] for r in rows]
    out["baseline"] = dict(
        test_rmse_mgdl=dict(mean=float(np.mean(base)), std=float(np.std(base, ddof=1))))
    return out


def run_full_ablation() -> None:
    from joblib import Parallel, delayed
    m.set_dataset("metabonet_full")
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in FULL_SOURCES]
    payload = json.loads(FULL_ABLATION_OUT.read_text()) if FULL_ABLATION_OUT.is_file() else {"patients": []}
    done = {(r["patient_id"], r["horizon_min"]) for r in payload["patients"]}
    todo = [(pid, ph) for ph in (30, 60) for pid in pids if (pid, ph) not in done]
    print(f"[ablation] todo {len(todo)} / {len(pids) * 2}", flush=True)
    step = 8
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        n_jobs = max(1, (os.cpu_count() or 4) - 1)
        got = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_full_one)(pid, ph) for pid, ph in batch)
        payload["patients"].extend(got)
        for ph in (30, 60):
            rows = [r for r in payload["patients"] if r["horizon_min"] == ph]
            if rows:
                payload.setdefault("summary", {})[str(ph)] = _full_summarize(rows)
        FULL_ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
        FULL_ABLATION_OUT.write_text(json.dumps(payload))
        print(f"[ablation] saved {len(payload['patients'])}", flush=True)


def _weight_mean(rows, key, *, rnd=None):
    v = np.array([r[key] for r in rows], float)
    out = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
    if rnd is not None:
        out = {k: round(val, rnd) for k, val in out.items()}
    return out


def _weight_one(pid: str) -> dict | None:
    """One synthetic row per real row, weighted 20, versus twenty unweighted copies."""
    CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    ph = 60
    try:
        data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    except Exception as exc:
        print(f"  skip {pid}: {exc}", flush=True)
        return None
    span = float(m.insulin_span_u(data, horizon_min=ph))
    m.DOCAS.SPAN = span
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    base_rmse = _rmse_mgdl(base, data.X_test, data.y_test, data.cgm_scaler)
    copies = _fit_variant(data, base, base_rmse, span, ph, synth_ratio=20.0)
    fit_kw, *_ = docas_fit_kw(data, span, ph, cfg=m.STUDY_CFG, synth_ratio=1.0)
    weighted = m.DOCAS(**fit_kw).fit(
        data.X_train, data.y_train, baseline_model=base, synth_weight=20.0)
    model = weighted["model"]
    row = dict(
        patient_id=str(pid),
        copies20_ae=copies["alignment_error"],
        copies20_rmse=copies["test_rmse_mgdl"],
        weight20_ae=_align_error(model, data, span, ph),
        weight20_rmse=_rmse_mgdl(model, data.X_test, data.y_test, data.cgm_scaler),
        n_synth_weight=int(weighted.get("n_synth", 0)),
        n_train=int(len(data.y_train)),
    )
    print(
        f"  {pid} copies {row['copies20_ae']:.2f}/{row['copies20_rmse']:.1f}"
        f"  weight {row['weight20_ae']:.2f}/{row['weight20_rmse']:.1f}",
        flush=True,
    )
    return row


def run_weight_budget() -> None:
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    out = RESULTS / "revision" / "weight20.json"
    m.set_dataset("metabonet_full")
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in FULL_SOURCES]
    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    rows = [r for r in Parallel(n_jobs=n_jobs, backend="loky")(delayed(_weight_one)(pid) for pid in pids) if r]
    payload = dict(
        n=len(rows),
        note="synth_ratio=1 with synthetic sample_weight=20, versus synth_ratio=20 at weight 1. Same total weight, fewer probed doses.",
        copies20_ae=_weight_mean(rows, "copies20_ae"),
        copies20_rmse=_weight_mean(rows, "copies20_rmse"),
        weight20_ae=_weight_mean(rows, "weight20_ae"),
        weight20_rmse=_weight_mean(rows, "weight20_rmse"),
        p_ae=float(wilcoxon([r["copies20_ae"] for r in rows], [r["weight20_ae"] for r in rows]).pvalue),
        p_rmse=float(wilcoxon([r["copies20_rmse"] for r in rows], [r["weight20_rmse"] for r in rows]).pvalue),
        patients=rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "patients"}, indent=2))


def _weight_replay_fit(dataset: str, pid: str):
    import forecasters
    from core import DOCAS, sync_cfg
    from preprocessing import PaperGlucoseModel
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset(dataset)
    data = m.load_patient(str(m.DATA_ROOT), pid, horizon_steps=horizon_steps(60))
    span = float(m.insulin_span_u(data, horizon_min=60))
    DOCAS.SPAN, DOCAS.SEED = span, 42
    sync_cfg(SEED=42)
    gi = data.feature_names.index("CGM")
    fkw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG, synth_ratio=1.0)
    res = DOCAS(**{**fkw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=None, synth_weight=20.0)
    mod = PaperGlucoseModel(
        res["model"], data.train_scalers, data.cgm_scaler,
        predicts_level=bool(res.get("predicts_level", False)),
        level_scaler=data.train_scalers[gi])
    meta = m.load_patient_meta(str(m.DATA_ROOT), pid)
    return mod, list(data.feature_names), meta


def run_weight_replay() -> None:
    """ReplayBG on the 25 study windows for CatBoost with synthetic weight 20."""
    from scipy.stats import wilcoxon
    import replay_eval as re
    stored_path = RESULTS / "revision" / "ablation_replay.json"
    out = RESULTS / "revision" / "weight20_replay.json"
    re.APPLY_HYPO_GUARD = True
    stored = json.loads(stored_path.read_text())["windows"]
    groups: dict[tuple[str, str], list] = {}
    for w in stored:
        if w["source"] == "OhioT1DM":
            key = ("ohio_t1dm", w["patient_id"].split("|", 1)[1])
        else:
            key = ("metabonet_full", w["patient_id"])
        groups.setdefault(key, []).append(w)
    rows = []
    for (dataset, pid), wins in groups.items():
        print(f"  weight replay {dataset} {pid} n={len(wins)}", flush=True)
        mod, cols, meta = _weight_replay_fit(dataset, pid)
        bw, u2ss = float(meta["weight_kg"]), meta.get("u2ss")
        trace = re._load_trace(pid, dataset=dataset, split="test")
        win_map = {int(w["source_start"]): w for w in re._windows(trace)}
        for w in wins:
            start = int(w["source_start"])
            seg = win_map[start]["trace"]
            name = re._twin_name(pid, "catboost", start, dataset=dataset)
            feat = re.paper_feat_arrays(seg, ph_steps=horizon_steps(60))
            delay = int(w["delay_min"])
            cb = re.ForecasterCIB(
                mod, cols, feat["CHO"], feat["CGM"], delay_min=delay,
                dose_penalty=re.CIB_DOSE_PENALTY, horizon_min=60)
            g, _, doses = re._replay(seg, bw, name, u2ss, "_w20", cb)
            met = re._post_gly(g, delay)
            met["insulin_u"] = round(float(sum(u for _, u in doses)), 2)
            rows.append(dict(
                patient_id=w["patient_id"], source_start=start,
                weight20=met, b20=w["b20"], baseline=w["baseline"],
            ))
            print(
                f"    {w['patient_id']} {start} weight {met['tir']:.1f}/{met['insulin_u']:.2f}"
                f"  copies {w['b20']['tir']:.1f}/{w['b20']['insulin_u']:.2f}",
                flush=True,
            )
    keys = ("tir", "tbr", "tar", "insulin_u")
    summary = dict(
        n=len(rows),
        weight20={k: _weight_mean([r["weight20"] for r in rows], k, rnd=2) for k in keys},
        copies20={k: _weight_mean([r["b20"] for r in rows], k, rnd=2) for k in keys},
        baseline={k: _weight_mean([r["baseline"] for r in rows], k, rnd=2) for k in keys},
        p_tir_vs_copies=float(wilcoxon(
            [r["weight20"]["tir"] for r in rows], [r["b20"]["tir"] for r in rows],
            zero_method="wilcox").pvalue),
        p_tir_vs_baseline=float(wilcoxon(
            [r["weight20"]["tir"] for r in rows], [r["baseline"]["tir"] for r in rows],
            zero_method="wilcox").pvalue),
        windows=rows,
    )
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "windows"}, indent=2))



OUT = RESULTS / "revision" / "context_docas.json"
PH = 60
SEEDS = (42, 43, 44, 45, 46)
US = np.linspace(0.0, 1.0, 41)
U21 = np.linspace(0.0, 1.0, 21)

DEFAULT_THETA = dict(
    k_probes=4,
    c_half=30.0,
    meal_cap=0.95,
    min_pop_slope=0.25,
    n0_eff=3.5,
    sig_iob=0.8,
    k_ctx=30,
    iso_anc=0.35,
    edge_pow=1.20,
    sig_g=6.0,
    sig_c=5.0,
    n_oof_blocks=4,
    spec="support_posterior_unweighted",
)


def _rmse(model, X, y, scaler) -> float:
    return float(rmse_mgdl(y, model.predict(X), scaler))


def _inv(s, a):
    return s.inverse_transform(np.asarray(a, float).reshape(-1, 1)).reshape(np.asarray(a).shape)


def _trend15(g):
    """15-minute CGM change (mg/dL) on a chronological series. Gaps stay at 0."""
    g = np.asarray(g, float)
    trend = np.zeros(len(g), float)
    if len(g) < 4:
        return trend
    d = np.abs(np.diff(g))
    ok = (d[:-2] <= 15.0) & (d[1:-1] <= 15.0) & (d[2:] <= 15.0)
    trend[3:][ok] = g[3:][ok] - g[:-3][ok]
    return trend


def _cgm_gate(g, trend, theta):
    """How far this glucose state may leave the population slope, in [0, 1].

    ``off`` leaves the carb × dose gate unchanged. ``level`` opens it as glucose
    rises away from hypoglycaemia. ``rise`` and ``band`` further close it when
    glucose is falling, so the population slope is kept on the way down.
    """
    mode = str(theta.get("g_mode", "off"))
    if mode == "off":
        return np.ones(len(g), float)
    g = np.asarray(g, float)
    tr = np.nan_to_num(np.asarray(trend, float), nan=0.0)
    g_half = float(theta.get("g_half", 150.0))
    z_level = (g * g) / (g * g + max(g_half, 1.0) ** 2)
    scale = max(float(theta.get("trend_scale", 12.0)), 1.0)
    z_keep = np.clip(2.0 / (1.0 + np.exp(-(tr + 6.0) / scale)), 0.0, 1.0)
    if mode == "level":
        return z_level
    if mode == "rise":
        return z_level * z_keep
    if mode == "band":
        center = float(theta.get("g_center", 170.0))
        sig = max(float(theta.get("g_sig", 40.0)), 1.0)
        z_band = np.exp(-0.5 * ((g - center) / sig) ** 2)
        return z_band * z_keep
    return np.ones(len(g), float)


def _cols(data, X, y=None):
    gi = data.feature_names.index("CGM")
    ci = data.feature_names.index("CHO")
    ji = data.insulin_idx
    g = _inv(data.train_scalers[gi], X[:, gi]).ravel()
    cho = np.maximum(_inv(data.train_scalers[ci], X[:, ci]).ravel(), 0.0)
    iob = np.maximum(_inv(data.train_scalers[ji], X[:, ji]).ravel(), 0.0)
    dy = None if y is None else _inv(data.cgm_scaler, y).ravel()
    return g, iob, cho, dy


def _pop_R(horizon, span, cfg):
    """Fractional lowering R(u) on US (positive); study τ from null is −g·R(u)."""
    target, _drop, _u_ref, _ = tau_from_cfg(int(horizon), span, cfg)
    R = np.asarray(target(US), float).ravel()
    return US, R


def _pav_dec_weighted(F, w):
    """Row-wise non-increasing least-squares projection.

    `w` can be either a 1D vector of length `M` shared across rows or a 2D
    matrix `(N, M)` of row-specific weights.
    """
    F = np.asarray(F, float)
    w = np.asarray(w, float)
    row_w = w.ndim == 2
    out = np.empty_like(F)
    for r, row in enumerate(F):
        wr = w[r] if row_w else w.ravel()
        vals, wts, lens = [], [], []
        for v, wk in zip(row, wr):
            vals.append(float(v))
            wts.append(float(wk))
            lens.append(1)
            while len(vals) > 1 and vals[-2] < vals[-1]:
                w_sum = wts[-2] + wts[-1]
                v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w_sum
                l2 = lens[-2] + lens[-1]
                vals[-2:], wts[-2:], lens[-2:] = [v2], [w_sum], [l2]
        out[r] = np.repeat(vals, lens)
    return out


def _emp_u_weights21(u_obs, power=1.8):
    edges = np.linspace(0.0, 1.0, len(U21) + 1)
    hist, _ = np.histogram(np.clip(u_obs, 0.0, 1.0), bins=edges)
    w = (hist.astype(float) / max(len(u_obs), 1)) ** float(power)
    return np.maximum(w, 1e-4)


def _row_iso_curves21(data, span, X_rows, base_mdl, w_u):
    X_rows = np.asarray(X_rows, float)
    ji = int(data.insulin_idx)
    grid = np.repeat(X_rows, len(U21), axis=0)
    grid[:, ji] = data.train_scalers[ji].transform(
        np.tile(U21 * float(span), len(X_rows)).reshape(-1, 1)).ravel()
    F = _inv(data.cgm_scaler, base_mdl.predict(grid)).reshape(len(X_rows), len(U21))
    F_iso = _pav_dec_weighted(F, w_u)
    return F_iso - F_iso[:, :1]


def _interp_rows21(u_vec, curve_mat):
    u_vec = np.clip(np.asarray(u_vec, float).ravel(), 0.0, 1.0)
    pos = u_vec * (len(U21) - 1)
    i0 = np.minimum(pos.astype(int), len(U21) - 2)
    frac = pos - i0
    idx = np.arange(len(u_vec))
    return curve_mat[idx, i0] * (1.0 - frac) + curve_mat[idx, i0 + 1] * frac


def _oof_predictions_and_ice21(data, span, X, y, *, n_blocks=4, seed=42):
    """Chronological block out-of-fold baseline predictions and raw ICEs on U21."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    ji = int(data.insulin_idx)
    n = len(y)
    n_b = max(2, min(int(n_blocks), n // 40))
    dy_oof = np.empty(n, float)
    F21_oof = np.empty((n, len(U21)), float)
    blocks = np.array_split(np.arange(n), n_b)
    for b_idx in range(n_b):
        va_b = blocks[b_idx]
        tr_b = np.concatenate([blocks[k] for k in range(n_b) if k != b_idx])
        m_b = train_baseline(X[tr_b], y[tr_b], model_kwargs={"random_seed": int(seed)})
        dy_oof[va_b] = _inv(data.cgm_scaler, m_b.predict(X[va_b])).ravel()
        g21 = np.repeat(X[va_b], len(U21), axis=0)
        g21[:, ji] = data.train_scalers[ji].transform(
            np.tile(U21 * float(span), len(va_b)).reshape(-1, 1)).ravel()
        F21_oof[va_b] = _inv(data.cgm_scaler, m_b.predict(g21)).reshape(len(va_b), len(U21))
    return dy_oof, F21_oof


def _build_drc21(
    data, span, X_rows, R_tab, theta, base_mdl=None,
    *, X_train_ref=None, F21_override=None, return_iso=False,
):
    """`(N, 21)` support-gated posterior target DRC on `U21`."""
    X_rows = np.asarray(X_rows, float)
    X_ref = np.asarray(data.X_train if X_train_ref is None else X_train_ref, float)
    g_ref, iob_ref, cho_ref, _ = _cols(data, X_ref)
    g_q, _, cho_q, _ = _cols(data, X_rows)

    sig_iob = float(theta.get("sig_iob", 0.8))
    n0_eff = float(theta.get("n0_eff", 3.5))
    c_half = float(theta.get("c_half", 25.0))
    meal_cap = float(theta.get("meal_cap", 0.95))
    k_ctx = min(int(theta.get("k_ctx", 30)), len(g_ref))

    ctx_tree = cKDTree(np.column_stack([g_ref / 20.0, cho_ref / 15.0]))
    _, ctx_nn = ctx_tree.query(np.column_stack([g_q / 20.0, cho_q / 15.0]), k=k_ctx)
    if ctx_nn.ndim == 1:
        ctx_nn = ctx_nn[:, None]
    iob_ctx = iob_ref[ctx_nn]

    u21_u = (U21 * float(span))[None, :, None]
    W_loc21 = np.exp(-0.5 * ((u21_u - iob_ctx[:, None, :]) / sig_iob) ** 2).sum(axis=2) + 1e-3

    if F21_override is not None:
        F21 = np.asarray(F21_override, float)
    else:
        if base_mdl is None:
            base_mdl = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
        ji = int(data.insulin_idx)
        grid = np.repeat(X_rows, len(U21), axis=0)
        grid[:, ji] = data.train_scalers[ji].transform(
            np.tile(U21 * float(span), len(X_rows)).reshape(-1, 1)).ravel()
        F21 = _inv(data.cgm_scaler, base_mdl.predict(grid)).reshape(len(X_rows), len(U21))

    F_iso = _pav_dec_weighted(F21, W_loc21)
    C_iso = F_iso - F_iso[:, :1]

    Ru21 = np.interp(U21, US, R_tab)[None, :]
    g_safe = np.maximum(g_q, 40.0)
    C_pop = -g_safe[:, None] * Ru21
    d_pop = np.diff(C_pop, axis=1)
    min_pop_slope = float(theta.get("min_pop_slope", 0.25))
    zc2 = (cho_q ** 2) / (cho_q ** 2 + max(c_half, 1e-6) ** 2)
    zg = _cgm_gate(g_q, _trend15(g_q), theta)
    d_raw = np.diff(C_iso, axis=1)
    if "meal_floor" in theta:
        trust = np.clip(zc2 * zg, 0.0, 1.0)
        floor = (1.0 - trust) * min_pop_slope + trust * float(theta["meal_floor"])
        d_iso = np.clip(d_raw, d_pop, floor[:, None] * d_pop)
    else:
        d_iso = np.clip(d_raw, d_pop, min_pop_slope * d_pop)

    u_mids_u = (0.5 * (U21[:-1] + U21[1:]) * float(span))[None, :, None]
    eff_n = np.exp(-0.5 * ((u_mids_u - iob_ctx[:, None, :]) / sig_iob) ** 2).sum(axis=2)
    rho = eff_n / (eff_n + n0_eff)
    rho_eff = rho * (meal_cap * zc2 * zg)[:, None]

    mode = str(theta.get("curve", "blend"))
    if mode == "individual":
        d_post = np.minimum(np.diff(C_iso, axis=1), 0.0)
    elif mode == "carb":
        c = np.maximum(cho_q, 0.0)
        mix = c ** 2 / (c ** 2 + 30.0 ** 2)
        d_post = d_pop * (1.0 - 0.75 * mix)[:, None]
    else:
        d_post = rho_eff * d_iso + (1.0 - rho_eff) * d_pop
    C_post = np.hstack([np.zeros((len(g_q), 1)), np.cumsum(d_post, axis=1)])
    return (C_post, F_iso) if return_iso else C_post


def _make_tau(R_tab, cho, *, C_mat, **_theta):
    del R_tab, cho
    C_mat = np.asarray(C_mat, float)
    n = len(C_mat)

    def tau(u, g, idx=None):
        del g
        if idx is None:
            idx = np.arange(n)
        idx = np.clip(np.asarray(idx, int).ravel(), 0, n - 1)
        u_arr = np.asarray(u, float)
        sub = C_mat[idx]
        if u_arr.ndim == 2:
            return np.column_stack([
                _interp_rows21(np.broadcast_to(u_arr[:, j], (len(idx),)), sub)
                for j in range(u_arr.shape[1])
            ])
        return _interp_rows21(u_arr, sub)

    return tau


# Train-tail grid. `carb` is the published 4-probe curve. The other rows change
# how many synthetic doses are added, and whether meal rows are included.
TUNE_GRID = (
    dict(id="carb"),
    dict(id="k1", k_probes=1),
    dict(id="k2", k_probes=2),
    dict(id="k1cgm", k_probes=1, cgm_share=0.60),
    dict(id="k2cgm", k_probes=2, cgm_share=0.60, cgm_where="meal"),
    dict(id="fast", k_meal=0),
    dict(id="fast8", k_meal=0, k_probes=8),
    dict(id="m1", k_meal=1, cgm_share=0.60, cgm_where="meal"),
    dict(id="half", shrink_meal=0.5),
    dict(id="halfk2", shrink_meal=0.5, k_probes=2, cgm_share=0.35, cgm_where="meal"),
    dict(id="frac25", synth_frac=0.25),
    dict(id="wide", k_probes=2, sig_g=20.0, sig_c=12.0, cgm_share=0.35),
)


def _search_theta(data, span, horizon, R_tab, *, seed=42):
    del data, span, horizon, R_tab, seed
    return {**DEFAULT_THETA, "span": round(float(span), 3)} if False else dict(DEFAULT_THETA)


def _span_drop(model, data, X, span, n=400, seed=0):
    """Mean change in predicted delta from 0 U to the training span, CGM and CHO fixed."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(X), size=min(int(n), len(X)), replace=False)
    Xs = X[take]
    ji = int(data.insulin_idx)
    sc = data.train_scalers[ji]
    X0, X1 = Xs.copy(), Xs.copy()
    X0[:, ji] = sc.transform(np.zeros((len(Xs), 1))).ravel()
    X1[:, ji] = sc.transform(np.full((len(Xs), 1), float(span))).ravel()
    d0 = _inv(data.cgm_scaler, model.predict(X0))
    d1 = _inv(data.cgm_scaler, model.predict(X1))
    return float(np.mean(d1 - d0))


def _pick_theta(scored):
    """Lowest train-tail RMSE that keeps at least half the carbohydrate-gate dose drop."""
    carb = next(r for r in scored if r["id"] == "carb")
    floor = 0.5 * min(float(carb["drop"]), -1.0)
    ok = [r for r in scored if float(r["drop"]) <= floor]
    pool = ok or [carb]
    safe = [r for r in pool if float(r["theta"].get("min_pop_slope", 0.25)) >= 0.15]
    return min(safe or pool, key=lambda r: r["val_rmse"])


def _tune_theta(data, span, R_tab, *, seed=42, horizon=60):
    """Choose the glucose × carbohydrate × dose gate on this patient's train tail."""
    X = np.asarray(data.X_train, float)
    y = np.asarray(data.y_train, float).ravel()
    n = len(y)
    k = max(100, int(round(0.2 * n)))
    if n - k < 200:
        th = dict(DEFAULT_THETA)
        return th, []
    Xf, yf = X[: n - k], y[: n - k]
    Xv, yv = X[n - k :], y[n - k :]
    oof = _oof_predictions_and_ice21(data, span, Xf, yf, n_blocks=4, seed=int(seed))
    scored = []
    for spec in TUNE_GRID:
        th = {**DEFAULT_THETA, **spec}
        model, _ = _fit_ctx(
            data, span, horizon, th, R_tab, seed=int(seed),
            X=Xf, y=yf, oof=oof, build_eval=False,
        )
        scored.append(dict(
            id=spec["id"],
            val_rmse=float(_rmse(model, Xv, yv, data.cgm_scaler)),
            drop=float(_span_drop(model, data, Xv, span, seed=int(seed))),
            theta=th,
        ))
    chosen = _pick_theta(scored)
    out = dict(chosen["theta"])
    out["tune_id"] = chosen["id"]
    out["tune_val_rmse"] = round(chosen["val_rmse"], 3)
    return out, scored


def _probes_for(sweep_idx, k, g, cho, u_obs, rng, sig_g, sig_c, edge_pow):
    """Jitter one context and draw ``k`` doses. ``k`` of 0 yields no rows."""
    n = len(sweep_idx)
    empty = (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0))
    if k <= 0 or n == 0:
        return empty
    cgm = np.clip(g[sweep_idx] + rng.normal(0.0, sig_g, size=n), 40.0, 400.0)
    cho_s = np.where(
        cho[sweep_idx] > 0.0,
        np.clip(cho[sweep_idx] + rng.normal(0.0, sig_c, size=n), 0.0, None),
        0.0,
    )
    u_mat = np.empty((n, k), float)
    u_mat[:, 0] = np.where(
        rng.random(n) < 0.5, 0.0, u_obs[rng.integers(0, len(u_obs), size=n)],
    )
    if k > 1:
        edges = np.linspace(0.0, 1.0, k) ** float(edge_pow)
        for j in range(1, k):
            lo, hi = float(edges[j - 1]), float(edges[j])
            u_mat[:, j] = lo if hi <= lo else rng.uniform(lo, hi, size=n)
    return (
        np.repeat(sweep_idx, k),
        np.repeat(cgm, k),
        np.repeat(cho_s, k),
        u_mat.ravel(),
    )


def _fit_ctx(
    data, span, horizon, theta, R_tab, *, seed=42,
    X=None, y=None, base_mdl=None, cho_sc=None, X_eval=None,
    oof=None, build_eval=True,
):
    del horizon
    X = np.asarray(data.X_train if X is None else X, float)
    y = np.asarray(data.y_train if y is None else y, float).ravel()
    g, iob, cho, dy = _cols(data, X, y)
    u_obs = np.clip(iob / max(float(span), 1e-12), 0.0, 1.0)

    th = {**DEFAULT_THETA, **(theta or {})}
    k_probes = int(th["k_probes"])
    iso_anc = float(th["iso_anc"])
    edge_pow = float(th["edge_pow"])
    sig_g = float(th["sig_g"])
    sig_c = float(th["sig_c"])
    n_oof = int(th["n_oof_blocks"])

    if oof is None:
        dy_oof, F21_oof = _oof_predictions_and_ice21(
            data, span, X, y, n_blocks=n_oof, seed=int(seed))
    else:
        dy_oof, F21_oof = oof
    C_tr, F_iso_tr = _build_drc21(
        data, span, X, R_tab, th, base_mdl=base_mdl, X_train_ref=X,
        F21_override=F21_oof, return_iso=True)
    y_iso_obs = _interp_rows21(u_obs, F_iso_tr)

    rng = np.random.default_rng(int(seed))
    if str(th.get("probe_mode", "all")) == "not_rise":
        keep = ~((cho > 1.0) & (_trend15(g) > 0.0))
        if not np.any(keep):
            keep = np.ones(len(y), dtype=bool)
        sweep_idx = np.flatnonzero(keep)
    else:
        sweep_idx = np.arange(len(y))
    frac = float(th.get("synth_frac", 1.0))
    if frac < 1.0 and len(sweep_idx):
        take = rng.random(len(sweep_idx)) < frac
        if not np.any(take):
            take[int(rng.integers(0, len(sweep_idx)))] = True
        sweep_idx = sweep_idx[take]
    if "k_meal" in th:
        meal_row = cho[sweep_idx] > 1.0
        parts = []
        if np.any(~meal_row):
            parts.append(_probes_for(
                sweep_idx[~meal_row], k_probes, g, cho, u_obs, rng, sig_g, sig_c, edge_pow))
        k_meal = int(th["k_meal"])
        if k_meal > 0 and np.any(meal_row):
            parts.append(_probes_for(
                sweep_idx[meal_row], k_meal, g, cho, u_obs, rng, sig_g, sig_c, edge_pow))
        parts = [p for p in parts if len(p[0])]
        if parts:
            idx = np.concatenate([p[0] for p in parts])
            cgm = np.concatenate([p[1] for p in parts])
            cho_s = np.concatenate([p[2] for p in parts])
            u = np.concatenate([p[3] for p in parts])
        else:
            idx = np.zeros(0, int)
            cgm = cho_s = u = np.zeros(0)
        cho_sw = cho[sweep_idx] if len(sweep_idx) else np.zeros(0)
        n_sweeps = len(sweep_idx)
    else:
        n_sweeps = len(sweep_idx)
        cgm_sw = np.clip(g[sweep_idx] + rng.normal(0.0, sig_g, size=n_sweeps), 40.0, 400.0)
        cho_sw = np.where(
            cho[sweep_idx] > 0.0,
            np.clip(cho[sweep_idx] + rng.normal(0.0, sig_c, size=n_sweeps), 0.0, None),
            0.0,
        )
        idx = np.repeat(sweep_idx, k_probes)
        cgm = np.repeat(cgm_sw, k_probes)
        cho_s = np.repeat(cho_sw, k_probes)
        u_mat = np.empty((n_sweeps, k_probes), float)
        u_mat[:, 0] = np.where(
            rng.random(n_sweeps) < 0.5,
            0.0,
            u_obs[rng.integers(0, len(y), size=n_sweeps)],
        )
        edges = np.linspace(0.0, 1.0, k_probes) ** edge_pow
        for j in range(1, k_probes):
            u_mat[:, j] = rng.uniform(edges[j - 1], edges[j], size=n_sweeps)
        u = u_mat.ravel()

    if len(idx) == 0:
        model = train_baseline(X, y, model_kwargs={"random_seed": int(seed)})
        if not build_eval:
            return model, None
        if base_mdl is None:
            base_mdl = model
        X_ref = data.X_test if X_eval is None else X_eval
        C_eval = _build_drc21(data, span, X_ref, R_tab, th, base_mdl=base_mdl, X_train_ref=X)
        cho_ref = cho if cho_sc is None else np.asarray(cho_sc, float).ravel()
        return model, _make_tau(R_tab, cho_ref, C_mat=C_eval, **th)

    C_sub = C_tr[idx] * (cgm / np.maximum(g[idx], 40.0))[:, None]
    phi_u = _interp_rows21(u, C_sub)
    phi_0 = _interp_rows21(u_obs[idx], C_sub)
    anc = iso_anc * y_iso_obs[idx] + (1.0 - iso_anc) * dy_oof[idx]
    obs_w = float(th.get("obs_anc", 0.0))
    if obs_w > 0.0:
        anc = obs_w * dy[idx] + (1.0 - obs_w) * anc
    dphi = phi_u - phi_0
    shrink = float(th.get("shrink_meal", 1.0))
    if shrink != 1.0 and len(idx):
        dphi = dphi * np.where(cho[idx] > 1.0, shrink, 1.0)
    dcho = np.zeros(len(idx), float)
    if str(th.get("cho_mode", "off")) == "match":
        meal = cho[idx] > 1.0
        per = float(th.get("cho_per_u", 0.0))
        cho_new = np.clip(cho_s + per * (u - u_obs[idx]) * float(span), 0.0, None)
        cho_new = np.where(meal, cho_new, cho_s)
        dcho = cho_new - cho_s
        cho_s = cho_new
    beta = float(th.get("cho_beta", 0.0))
    alpha = float(th.get("cgm_share", 0.0))
    if str(th.get("cgm_where", "all")) == "meal":
        alpha = np.where(cho[idx] > 1.0, alpha, 0.0)
    gamma = float(th.get("cho_cgm_share", 0.0))
    g_pre = cgm
    cgm = np.clip(g_pre + alpha * dphi + gamma * beta * dcho, 40.0, 400.0)
    # Future level = original current glucose + anchor + insulin curve + carb effect.
    lab = g_pre + anc + dphi + beta * dcho - cgm

    if str(th.get("cho_mode", "off")) == "probe":
        k_cho = int(th.get("k_cho", 2))
        cho_obs = cho_sw
        caps = np.maximum(cho_obs * 1.8, 25.0)
        cho_draw = rng.uniform(0.0, caps[:, None], size=(n_sweeps, k_cho))
        dcho_c = (cho_draw - cho_obs[:, None]).ravel()
        g_rep = np.repeat(cgm_sw, k_cho)
        anc_c = iso_anc * y_iso_obs[sweep_idx] + (1.0 - iso_anc) * dy_oof[sweep_idx]
        if obs_w > 0.0:
            anc_c = obs_w * dy[sweep_idx] + (1.0 - obs_w) * anc_c
        anc_rep = np.repeat(anc_c, k_cho)
        desired = g_rep + gamma * beta * dcho_c
        cgm_c = np.clip(desired, 40.0, 400.0)
        lab_c = g_rep + anc_rep + beta * dcho_c - cgm_c
        u_c = np.repeat(u_obs[sweep_idx], k_cho)
        idx_c = np.repeat(sweep_idx, k_cho)
        cho_c = cho_draw.ravel()
        cgm = np.concatenate([cgm, cgm_c])
        cho_s = np.concatenate([cho_s, cho_c])
        u = np.concatenate([u, u_c])
        lab = np.concatenate([lab, lab_c])
        idx = np.concatenate([idx, idx_c])

    gi = data.feature_names.index("CGM")
    ci = data.feature_names.index("CHO")
    ji = int(data.insulin_idx)
    Xs = X[idx].copy()
    Xs[:, gi] = data.train_scalers[gi].transform(cgm[:, None]).ravel()
    Xs[:, ci] = data.train_scalers[ci].transform(cho_s[:, None]).ravel()
    Xs[:, ji] = data.train_scalers[ji].transform((u * float(span))[:, None]).ravel()
    ys = data.cgm_scaler.transform(lab[:, None]).ravel()

    # Pure unweighted CatBoost fit — identical learner call to Population DOCAS!
    model = train_baseline(
        np.vstack([X, Xs]),
        np.concatenate([y, ys]),
        n_factual=len(y),
        model_kwargs={"random_seed": int(seed)},
    )
    if not build_eval:
        return model, None

    if base_mdl is None:
        base_mdl = train_baseline(X, y, model_kwargs={"random_seed": int(seed)})
    X_ref = data.X_test if X_eval is None else X_eval
    C_eval = _build_drc21(data, span, X_ref, R_tab, th, base_mdl=base_mdl, X_train_ref=X)
    cho_ref = cho if cho_sc is None else np.asarray(cho_sc, float).ravel()
    tau = _make_tau(R_tab, cho_ref, C_mat=C_eval, **th)
    return model, tau


def _score_patient(pid, dataset, horizon=PH, *, seeds=SEEDS, quiet=False):
    m.set_dataset(dataset)
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(horizon))
    span = float(m.insulin_span_u(data, horizon_min=horizon))
    _, R_tab = _pop_R(horizon, span, m.STUDY_CFG)
    theta = _search_theta(data, span, horizon, R_tab, seed=42)
    theta["span"] = round(span, 3)

    Xt = np.asarray(data.X_test, float)
    yt = np.asarray(data.y_test, float).ravel()
    _, _, cho_te, _ = _cols(data, Xt)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    fit_kw, *_ = docas_fit_kw(data, span, horizon, cfg=m.STUDY_CFG)
    pop_scorer = DOCAS(**{**fit_kw, "seed": 42})
    base_rmse = _rmse(base, Xt, yt, data.cgm_scaler)
    base_ae = float(pop_scorer.alignment(base, Xt))
    rows = []
    for seed in seeds:
        DOCAS.SPAN, DOCAS.SEED = span, int(seed)
        sync_cfg(SEED=int(seed))
        pop = DOCAS(**{**fit_kw, "seed": int(seed)}).fit(
            data.X_train, data.y_train, baseline_model=base)["model"]
        ctx_model, tau_te = _fit_ctx(
            data, span, horizon, theta, R_tab, seed=int(seed), base_mdl=base,
            cho_sc=cho_te, X_eval=Xt,
        )
        ctx_scorer_te = DOCAS(**{**fit_kw, "seed": int(seed), "tau_exact": tau_te})
        rows.append(dict(
            seed=int(seed),
            baseline_rmse=base_rmse,
            pop_rmse=_rmse(pop, Xt, yt, data.cgm_scaler),
            ctx_rmse=_rmse(ctx_model, Xt, yt, data.cgm_scaler),
            baseline_ae_pop=base_ae,
            pop_ae_pop=float(pop_scorer.alignment(pop, Xt)),
            ctx_ae_own=float(ctx_scorer_te.alignment(ctx_model, Xt)),
            ctx_ae_pop=float(pop_scorer.alignment(ctx_model, Xt)),
        ))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    if not quiet:
        print(
            f"  {dataset} {pid} ({horizon}m): Base={mean['baseline_rmse']:.2f} "
            f"Pop={mean['pop_rmse']:.2f} Ctx={mean['ctx_rmse']:.2f} "
            f"AE_pop={mean['ctx_ae_pop']:.2f} AE_ctx={mean['ctx_ae_own']:.2f}",
            flush=True,
        )

    # Diagnostic target curves at representative (CGM, CHO) contexts
    I = np.linspace(0.0, 5.0, 11)
    u = np.clip(I / max(span, 1e-12), 0.0, 1.0)
    gi = data.feature_names.index("CGM")
    ci = data.feature_names.index("CHO")
    ji = int(data.insulin_idx)
    contexts = {
        "fast": (110.0, 0.0),
        "meal": (140.0, 40.0),
        "high": (200.0, 5.0),
    }
    X_ctx = np.zeros((3, data.X_train.shape[1]), float)
    ctx_names = list(contexts.keys())
    for k_i, name in enumerate(ctx_names):
        gv, cv = contexts[name]
        X_ctx[k_i, gi] = data.train_scalers[gi].transform([[gv]])[0, 0]
        X_ctx[k_i, ci] = data.train_scalers[ci].transform([[cv]])[0, 0]
        X_ctx[k_i, ji] = data.train_scalers[ji].transform([[0.0]])[0, 0]
    C_diag = _build_drc21(data, span, X_ctx, R_tab, theta, base_mdl=base)
    curves, pop_curves = {}, {}
    for k_i, name in enumerate(ctx_names):
        gv, _cv = contexts[name]
        c_row = np.repeat(C_diag[k_i:k_i + 1], len(u), axis=0)
        curves[name] = [round(float(v), 2) for v in _interp_rows21(u, c_row)]
        Ru = np.interp(u, US, R_tab)
        pop_curves[name] = [round(float(v), 2) for v in (-gv * Ru)]

    return dict(
        patient_id=str(pid),
        dataset=dataset,
        span=round(span, 2),
        theta=theta,
        metrics=mean,
        per_seed=rows,
        I=[round(float(v), 2) for v in I],
        curves=curves,
        pop_curves=pop_curves,
    )


def _summarize(patients):
    keys = list(patients[0]["metrics"].keys())
    out = {}
    for k in keys:
        vals = [p["metrics"][k] for p in patients]
        out[k] = dict(
            mean=round(float(np.mean(vals)), 2),
            std=round(float(np.std(vals, ddof=1)), 2),
        )
    return out


def run(datasets=("ohio_t1dm", "azt1d"), *, seeds=SEEDS):
    from joblib import Parallel, delayed
    payload = json.loads(OUT.read_text()) if OUT.is_file() else {}
    for ds in datasets:
        m.set_dataset(ds)
        pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT))]
        print(f"[{ds}] n={len(pids)} seeds={seeds}", flush=True)
        pts = Parallel(n_jobs=6, backend="loky")(
            delayed(_score_patient)(pid, ds, PH, seeds=seeds) for pid in pids
        )
        payload[ds] = dict(summary=_summarize(pts), patients=pts)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2))
        print(f"[{ds}] summary:", payload[ds]["summary"], flush=True)
    return payload


RECENT = RESULTS / "revision" / "full_recent"
RECENT_SOURCES = ("OhioT1DM", "AZT1D", "HUPA-UCM", "T1D-UOM")


def _recent_path(pid: str) -> Path:
    return RECENT / f"{pid.replace('|', '__')}.json"


def _score_recent(pid: str) -> str:
    import warnings
    import forecasters
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    dest = _recent_path(pid)
    if dest.is_file():
        return "have"
    try:
        payload = {"patient_id": pid, "source": pid.split("|", 1)[0], "horizons": {}}
        for ph in (60, 30):
            payload["horizons"][str(ph)] = _score_patient(
                pid, "metabonet_full", ph, seeds=(42,), quiet=True,
            )
        dest.write_text(json.dumps(payload))
        met = payload["horizons"]["60"]["metrics"]
        print(
            f"{pid}  60 min  glucose error {met['baseline_rmse']:.1f} → {met['pop_rmse']:.1f} mg/dL"
            f"   alignment {met['baseline_ae_pop']:.1f} → {met['pop_ae_pop']:.1f}",
            flush=True,
        )
        return "ok"
    except Exception:
        dest.write_text(json.dumps({"patient_id": pid, "error": "skipped"}))
        print(f"{pid}  skipped, no usable glucose", flush=True)
        return "skip"


def run_full_cohort() -> None:
    """Table 3. One JSON per person on the latest 8,000 training rows."""
    from joblib import Parallel, delayed
    m.set_dataset("metabonet_full")
    pids = [p for p in m.list_patients(str(m.DATA_ROOT)) if p.split("|", 1)[0] in RECENT_SOURCES]
    RECENT.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pids if not _recent_path(p).is_file()]
    print(f"{len(pids) - len(todo)} already done, {len(todo)} left", flush=True)
    if not todo:
        return
    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    Parallel(n_jobs=n_jobs, backend="loky")(delayed(_score_recent)(pid) for pid in todo)


def carb_step_factor(cho, *, half=30.0, floor=0.25):
    """1 with no carbohydrate, halfway to ``floor`` near ``half`` grams."""
    cho = np.maximum(np.asarray(cho, float), 0.0)
    mix = cho ** 2 / (cho ** 2 + max(float(half), 1e-6) ** 2)
    floor = float(np.clip(floor, 0.0, 1.0))
    return 1.0 - (1.0 - floor) * mix


def _fit_carb(data, span, R_tab, *, seed=42, k=4, half=30.0, floor=0.25):
    """Population percentage curve, multiplied by a carbohydrate factor only."""
    CAT_KWARGS["thread_count"] = 1
    X = np.asarray(data.X_train, float)
    y = np.asarray(data.y_train, float).ravel()
    g, iob, cho, dy = _cols(data, X, y)
    u_obs = np.clip(iob / max(float(span), 1e-12), 0.0, 1.0)
    rng = np.random.default_rng(int(seed))
    n = len(y)
    cgm = np.maximum(g + rng.normal(0.0, 10.0, size=n), 40.0)
    cho_s = np.maximum(cho + rng.normal(0.0, 10.0, size=n), 0.0)
    u_mat = np.empty((n, k), float)
    u_mat[:, 0] = np.where(rng.random(n) < 0.5, 0.0, u_obs[rng.integers(0, n, size=n)])
    edges = np.linspace(0.0, 1.0, k)
    for j in range(1, k):
        u_mat[:, j] = rng.uniform(edges[j - 1], edges[j], size=n)
    u = u_mat.ravel()
    idx = np.repeat(np.arange(n), k)
    cgm = np.repeat(cgm, k)
    cho_s = np.repeat(cho_s, k)
    Ru = np.interp(u, US, R_tab)
    Ro = np.interp(u_obs[idx], US, R_tab)
    dphi = -cgm * (Ru - Ro) * carb_step_factor(cho_s, half=half, floor=floor)
    lab = dy[idx] + dphi
    gi = data.feature_names.index("CGM")
    ci = data.feature_names.index("CHO")
    ji = int(data.insulin_idx)
    Xs = X[idx].copy()
    Xs[:, gi] = data.train_scalers[gi].transform(cgm[:, None]).ravel()
    Xs[:, ci] = data.train_scalers[ci].transform(cho_s[:, None]).ravel()
    Xs[:, ji] = data.train_scalers[ji].transform((u * float(span))[:, None]).ravel()
    ys = data.cgm_scaler.transform(lab[:, None]).ravel()
    return train_baseline(
        np.vstack([X, Xs]), np.concatenate([y, ys]),
        n_factual=len(y), model_kwargs={"random_seed": int(seed)},
    )


def _carb_tau(data, X, R_tab, *, half=30.0, floor=0.25):
    g, _, cho, _ = _cols(data, X)
    factor = carb_step_factor(cho, half=half, floor=floor)
    C = -np.maximum(g, 40.0)[:, None] * np.interp(U21, US, R_tab)[None, :] * factor[:, None]

    def tau(u, g_query, idx=None):
        del g_query
        if idx is None:
            idx = np.arange(len(C))
        idx = np.clip(np.asarray(idx, int).ravel(), 0, len(C) - 1)
        u_arr = np.asarray(u, float)
        sub = C[idx]
        if u_arr.ndim == 2:
            return np.column_stack([
                _interp_rows21(np.broadcast_to(u_arr[:, j], (len(idx),)), sub)
                for j in range(u_arr.shape[1])
            ])
        return _interp_rows21(u_arr, sub)

    return tau


def _score_carb(pid: str) -> dict | None:
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    try:
        data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    except Exception as exc:
        print(f"  skip {pid}: {exc}", flush=True)
        return None
    span = float(m.insulin_span_u(data, horizon_min=60))
    _, R_tab = _pop_R(60, span, m.STUDY_CFG)
    Xt = np.asarray(data.X_test, float)
    yt = np.asarray(data.y_test, float).ravel()
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    fit_kw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG)
    pop_scorer = DOCAS(**{**fit_kw, "seed": 42})
    pop = DOCAS(**{**fit_kw, "seed": 42}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    carb = _fit_carb(data, span, R_tab, seed=42)
    own = DOCAS(**{**fit_kw, "seed": 42, "tau_exact": _carb_tau(data, Xt, R_tab)})
    row = dict(
        patient_id=str(pid),
        baseline_rmse=_rmse(base, Xt, yt, data.cgm_scaler),
        pop_rmse=_rmse(pop, Xt, yt, data.cgm_scaler),
        carb_rmse=_rmse(carb, Xt, yt, data.cgm_scaler),
        baseline_ae=float(pop_scorer.alignment(base, Xt)),
        pop_ae=float(pop_scorer.alignment(pop, Xt)),
        carb_ae_pop=float(pop_scorer.alignment(carb, Xt)),
        carb_ae_own=float(own.alignment(carb, Xt)),
    )
    print(
        f"  {pid}  base {row['baseline_rmse']:.1f}  pop {row['pop_rmse']:.1f}"
        f"  carb {row['carb_rmse']:.1f}  AE {row['carb_ae_own']:.1f}/{row['carb_ae_pop']:.1f}",
        flush=True,
    )
    return row


def run_carb() -> None:
    """Context arm with the population curve scaled by carbohydrate only."""
    import json
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    out = RESULTS / "revision" / "carb_only.json"
    m.set_dataset("metabonet_full")
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in FULL_SOURCES]
    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    rows = [r for r in Parallel(n_jobs=n_jobs, backend="loky")(delayed(_score_carb)(pid) for pid in pids) if r]
    keys = ("baseline_rmse", "pop_rmse", "carb_rmse", "baseline_ae", "pop_ae", "carb_ae_pop", "carb_ae_own")
    summary = {k: _weight_mean(rows, k, rnd=2) for k in keys}
    diff = np.array([r["carb_rmse"] - r["baseline_rmse"] for r in rows], float)
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(diff, len(diff), replace=True).mean()) for _ in range(2000)]
    payload = dict(
        n=len(rows),
        note="Population percentage-of-glucose curve, times a carbohydrate factor (1 with no carbohydrate, one quarter at a large meal, halfway near 30 g). Four probes per real row. Glucose does not change the factor.",
        summary=summary,
        carb_minus_baseline_rmse=dict(
            mean=round(float(diff.mean()), 2),
            ci=[round(float(np.quantile(boots, 0.025)), 2), round(float(np.quantile(boots, 0.975)), 2)],
            p=float(wilcoxon([r["carb_rmse"] for r in rows], [r["baseline_rmse"] for r in rows]).pvalue),
        ),
        carb_minus_pop_rmse=dict(
            mean=round(float(np.mean([r["carb_rmse"] - r["pop_rmse"] for r in rows])), 2),
            p=float(wilcoxon([r["carb_rmse"] for r in rows], [r["pop_rmse"] for r in rows]).pvalue),
        ),
        patients=rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "patients"}, indent=2))


def _score_cgm(pid: str) -> dict | None:
    """Population DOCAS, with the dose effect also written into the synthetic CGM."""
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    try:
        data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    except Exception as exc:
        print(f"  skip {pid}: {exc}", flush=True)
        return None
    span = float(m.insulin_span_u(data, horizon_min=60))
    Xt = np.asarray(data.X_test, float)
    yt = np.asarray(data.y_test, float).ravel()
    fit_kw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG)
    moved = DOCAS(**{**fit_kw, "seed": 42, "cgm_share": 1.0}).fit(
        data.X_train, data.y_train, baseline_model=None)["model"]
    ae = float(DOCAS(**{**fit_kw, "seed": 42}).alignment(moved, Xt))
    row = dict(
        patient_id=str(pid),
        cgm_rmse=_rmse(moved, Xt, yt, data.cgm_scaler),
        cgm_ae=ae,
    )
    print(f"  {pid}  cgm-share {row['cgm_rmse']:.1f}  AE {row['cgm_ae']:.1f}", flush=True)
    return row


def run_cgm_share() -> None:
    """Same 20x population curve, but a different dose also moves the row's current glucose."""
    import json
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    out = RESULTS / "revision" / "cgm_share.json"
    prev = {r["patient_id"]: r for r in json.loads((RESULTS / "revision" / "carb_only.json").read_text())["patients"]}
    m.set_dataset("metabonet_full")
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in FULL_SOURCES]
    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    fresh = [r for r in Parallel(n_jobs=n_jobs, backend="loky")(delayed(_score_cgm)(pid) for pid in pids) if r]
    rows = []
    for r in fresh:
        old = prev.get(r["patient_id"])
        if old is None:
            continue
        rows.append({**old, **r})
    diff = np.array([r["cgm_rmse"] - r["baseline_rmse"] for r in rows], float)
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(diff, len(diff), replace=True).mean()) for _ in range(2000)]
    keys = ("baseline_rmse", "pop_rmse", "cgm_rmse", "baseline_ae", "pop_ae", "cgm_ae")
    payload = dict(
        n=len(rows),
        note="Population DOCAS at 20 copies. The whole dose effect is moved into the synthetic current glucose, and the future level is unchanged.",
        summary={k: _weight_mean(rows, k, rnd=2) for k in keys},
        cgm_minus_baseline_rmse=dict(
            mean=round(float(diff.mean()), 2),
            ci=[round(float(np.quantile(boots, 0.025)), 2), round(float(np.quantile(boots, 0.975)), 2)],
            p=float(wilcoxon([r["cgm_rmse"] for r in rows], [r["baseline_rmse"] for r in rows]).pvalue),
        ),
        cgm_minus_pop_rmse=dict(
            mean=round(float(np.mean([r["cgm_rmse"] - r["pop_rmse"] for r in rows])), 2),
            p=float(wilcoxon([r["cgm_rmse"] for r in rows], [r["pop_rmse"] for r in rows]).pvalue),
        ),
        patients=rows,
    )
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "patients"}, indent=2))


def _arm(model, data, Xt, yt, scorer, *, tau=None, scale=1.0):
    fit_kw = scorer
    rmse = _rmse(model, Xt, yt, data.cgm_scaler)
    ae_pop = float(DOCAS(**{**fit_kw, "seed": 42}).alignment(model, Xt))
    if tau is not None:
        ae_own = float(DOCAS(**{**fit_kw, "seed": 42, "tau_exact": tau}).alignment(model, Xt))
    elif scale != 1.0:
        ae_own = float(DOCAS(**{**fit_kw, "seed": 42, "tau_drop": fit_kw["tau_drop"] * scale}).alignment(model, Xt))
    else:
        ae_own = ae_pop
    return dict(rmse=round(rmse, 2), ae_pop=round(ae_pop, 2), ae_own=round(ae_own, 2))


def _person_scale(data, fit_kw):
    """Pick 0.5, 1, or 1.5 on the train tail, then the caller refits on all rows."""
    X = np.asarray(data.X_train, float)
    y = np.asarray(data.y_train, float).ravel()
    k = max(50, int(round(0.2 * len(y))))
    if len(y) - k < 80:
        return 1.0
    best_s, best_v = 1.0, float("inf")
    for s in (0.5, 1.0, 1.5):
        model = DOCAS(**{**fit_kw, "seed": 42, "curve_scale": s}).fit(
            X[:-k], y[:-k], baseline_model=None)["model"]
        v = _rmse(model, X[-k:], y[-k:], data.cgm_scaler)
        if v < best_v:
            best_s, best_v = s, v
    return best_s


def _score_adjust(pid: str) -> dict | None:
    import forecasters
    forecasters.CAT_KWARGS["thread_count"] = 1
    m.set_dataset("metabonet_full")
    try:
        data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(60))
    except Exception as exc:
        print(f"  skip {pid}: {exc}", flush=True)
        return None
    span = float(m.insulin_span_u(data, horizon_min=60))
    Xt = np.asarray(data.X_test, float)
    yt = np.asarray(data.y_test, float).ravel()
    fit_kw, *_ = docas_fit_kw(data, span, 60, cfg=m.STUDY_CFG)
    base = train_baseline(data.X_train, data.y_train, model_kwargs={"random_seed": 42})
    arms = {}
    for name, kw in (
        ("pop_share25", dict(cgm_share=0.25)),
        ("pop_share50", dict(cgm_share=0.5)),
        ("pop_carb", dict(cho_floor=0.25)),
        ("pop_carb_share50", dict(cho_floor=0.25, cgm_share=0.5)),
    ):
        model = DOCAS(**{**fit_kw, "seed": 42, **kw}).fit(
            data.X_train, data.y_train, baseline_model=base)["model"]
        tau = _carb_tau(data, Xt, _pop_R(60, span, m.STUDY_CFG)[1]) if "carb" in name else None
        arms[name] = _arm(model, data, Xt, yt, fit_kw, tau=tau)
    scale = _person_scale(data, fit_kw)
    person = DOCAS(**{**fit_kw, "seed": 42, "curve_scale": scale}).fit(
        data.X_train, data.y_train, baseline_model=base)["model"]
    arms["pop_person"] = {**_arm(person, data, Xt, yt, fit_kw, scale=scale), "scale": scale}
    _, R_tab = _pop_R(60, span, m.STUDY_CFG)
    oof = _oof_predictions_and_ice21(data, span, data.X_train, data.y_train, n_blocks=4, seed=42)
    _, _, cho_te, _ = _cols(data, Xt)
    for name, extra in (
        ("ctx_share50", dict(cgm_share=0.5)),
        ("ctx_share100", dict(cgm_share=1.0)),
        ("ctx_carb", dict(curve="carb")),
        ("indiv", dict(curve="individual")),
        ("indiv_share50", dict(curve="individual", cgm_share=0.5)),
    ):
        model, tau = _fit_ctx(
            data, span, 60, {**DEFAULT_THETA, **extra}, R_tab, seed=42,
            base_mdl=base, oof=oof, cho_sc=cho_te, X_eval=Xt,
        )
        arms[name] = _arm(model, data, Xt, yt, fit_kw, tau=tau)
    row = dict(
        patient_id=str(pid),
        baseline_rmse=round(_rmse(base, Xt, yt, data.cgm_scaler), 2),
        arms=arms,
    )
    brief = " ".join(f"{k} {v['rmse']:.1f}/{v['ae_pop']:.1f}" for k, v in arms.items())
    print(f"  {pid}  {brief}", flush=True)
    return row


def run_adjust() -> None:
    """Partial glucose shift, carbohydrate curve, and a per-person curve."""
    import json
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    out = RESULTS / "revision" / "adjust_runs.json"
    payload = json.loads(out.read_text()) if out.is_file() else {"patients": []}
    done = {r["patient_id"] for r in payload["patients"]}
    m.set_dataset("metabonet_full")
    pids = [str(p) for p in m.list_patients(str(m.DATA_ROOT)) if str(p).split("|", 1)[0] in FULL_SOURCES and str(p) not in done]
    print(f"[adjust] todo {len(pids)}", flush=True)
    step = 4
    for i in range(0, len(pids), step):
        got = Parallel(n_jobs=step, backend="loky")(delayed(_score_adjust)(pid) for pid in pids[i:i + step])
        payload["patients"].extend(r for r in got if r)
        rows = payload["patients"]
        names = sorted({k for r in rows for k in r["arms"]})
        summary = {}
        for name in names:
            rm = [r["arms"][name]["rmse"] for r in rows if name in r["arms"]]
            ae = [r["arms"][name]["ae_pop"] for r in rows if name in r["arms"]]
            own = [r["arms"][name]["ae_own"] for r in rows if name in r["arms"]]
            base = [r["baseline_rmse"] for r in rows if name in r["arms"]]
            summary[name] = dict(
                rmse=_weight_mean([{"v": v} for v in rm], "v", rnd=2),
                ae_pop=_weight_mean([{"v": v} for v in ae], "v", rnd=2),
                ae_own=_weight_mean([{"v": v} for v in own], "v", rnd=2),
                minus_baseline=round(float(np.mean(np.array(rm) - np.array(base))), 2),
                p_rmse=float(wilcoxon(rm, base).pvalue),
            )
        payload["summary"] = summary
        payload["n"] = len(rows)
        out.write_text(json.dumps(payload))
        print(f"[adjust] {len(rows)} saved", flush=True)
    print(json.dumps(payload.get("summary", {}), indent=2))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"
    if cmd == "cohort":
        run_full_cohort()
    elif cmd == "carb":
        run_carb()
    elif cmd == "cgm":
        run_cgm_share()
    elif cmd == "adjust":
        run_adjust()
    elif cmd == "context":
        run()
    elif cmd == "weight":
        run_weight_budget()
    elif cmd == "weight-replay":
        run_weight_replay()
    else:
        run_full_ablation()
