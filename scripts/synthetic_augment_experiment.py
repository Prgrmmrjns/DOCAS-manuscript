"""SCM-guided synthetic augmentation for Ohio T1DM (no latent confounders)."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

import ohio_t1dm
from lib import compute_feasibility, holdout_rmse, plotted_scm_rules
from model import make_model, monotonic_constraints_for

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
PATIENT_ID = "540"
SEED = 42
N_TRIALS = 1000
SELECTION_RMSE_WEIGHT = 0.5
FEASIBILITY_GRID_SIZE = 10
DO_CURVE_MAX_SAMPLES = 100
N_SYNTH_ROWS = 20


def _drop_latent_columns(X: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in X.columns if c in ohio_t1dm.HIDDEN_CONFOUNDER_CANDIDATES]
    return X.drop(columns=drop, errors="ignore")


def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    rules: list[dict],
    target: str,
    *,
    random_state: int = SEED,
    sample_weight: np.ndarray | None = None,
    monotonic: bool = False,
) -> Any:
    mono = monotonic_constraints_for(feat_names, rules, target) if monotonic else None
    m = make_model(random_state=random_state, n_jobs=1, monotonic_constraints=mono)
    m.fit(
        np.ascontiguousarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64).ravel(),
        sample_weight=sample_weight,
    )
    return m


def _feas_score(
    model: Any,
    X_val: np.ndarray,
    cols: list[str],
    rules: list[dict],
    *,
    target: str,
    feas_kw: dict[str, Any],
) -> float:
    rep = compute_feasibility(model, X_val, cols, rules, target, latent_columns=(), **feas_kw)
    return float(rep["feasibility_score_0_1"])


def _make_scm_synth_batch(
    X_real: np.ndarray,
    y_real: np.ndarray,
    feat_names: list[str],
    rules: list[dict],
    target: str,
    *,
    n: int,
    rule_idx: int,
    grid_lo_q: float,
    grid_hi_q: float,
    y_gain: float,
    feature_jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual rows: intervene on one SCM feature from real contexts."""
    plotted = plotted_scm_rules(rules, feat_names, target)
    if not plotted or n <= 0:
        return np.empty((0, X_real.shape[1]), dtype=np.float64), np.empty(0, dtype=np.float64)

    rule = plotted[int(rule_idx) % len(plotted)]
    u_idx = feat_names.index(str(rule["start"]))
    fn = rule.get("relationship_fn")
    increasing = bool(getattr(fn, "increasing", True))

    u_ref = X_real[:, u_idx]
    u_ref = u_ref[np.isfinite(u_ref)]
    if u_ref.size < 2:
        return np.empty((0, X_real.shape[1]), dtype=np.float64), np.empty(0, dtype=np.float64)

    lo_q, hi_q = sorted((float(np.clip(grid_lo_q, 0.0, 1.0)), float(np.clip(grid_hi_q, 0.0, 1.0))))
    if hi_q <= lo_q + 1e-3:
        hi_q = min(1.0, lo_q + 0.1)
    q_lo, q_hi = float(np.quantile(u_ref, lo_q)), float(np.quantile(u_ref, hi_q))
    if q_hi <= q_lo + 1e-9:
        q_hi = q_lo + 1e-6

    grid = np.linspace(q_lo, q_hi, max(2, n))
    if not increasing:
        grid = grid[::-1]

    u_scale = max(float(np.std(u_ref)), 1e-6)
    y_lo, y_hi = np.percentile(y_real, [1.0, 99.0])
    ctx = rng.integers(0, len(X_real), size=n)

    X_out = np.empty((n, X_real.shape[1]), dtype=np.float64)
    y_out = np.empty(n, dtype=np.float64)
    for i, row_i in enumerate(ctx):
        x = X_real[int(row_i)].copy()
        y0 = float(y_real[int(row_i)])
        u0 = float(x[u_idx]) if np.isfinite(x[u_idx]) else float(np.mean(u_ref))
        u_new = float(grid[i % len(grid)])
        x[u_idx] = u_new
        if feature_jitter > 0:
            for j in range(x.size):
                if j != u_idx:
                    x[j] += float(rng.normal(0.0, feature_jitter))
        du = (u_new - u0) / u_scale
        dy = y_gain * du if increasing else -y_gain * du
        y_out[i] = float(np.clip(y0 + dy, y_lo, y_hi))
        X_out[i] = x
    return X_out, y_out


def run_synthetic_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feat_names: list[str],
    rules: list[dict],
    target: str,
    *,
    n_trials: int | None = None,
    max_rounds: int | None = None,
    selection_rmse_weight: float | None = None,
    random_state: int = SEED,
    feas_max_samples: int | None = None,
    feas_grid_size: int | None = None,
    synth_per_round: int | None = None,
    show_progress_bar: bool = True,
) -> Any:
    """NSGA-II over SCM counterfactual batches; return best trial model (or baseline if no gain)."""
    del max_rounds  # single-pass search; kept for call-site compatibility
    n_trials = int(n_trials if n_trials is not None else N_TRIALS)
    n_synth = int(synth_per_round if synth_per_round is not None else N_SYNTH_ROWS)
    feas_kw = dict(
        random_state=int(random_state),
        max_samples=int(feas_max_samples if feas_max_samples is not None else DO_CURVE_MAX_SAMPLES),
        grid_size=int(feas_grid_size if feas_grid_size is not None else FEASIBILITY_GRID_SIZE),
    )
    feas_kw_full = {**feas_kw, "max_samples": None}

    plotted = plotted_scm_rules(rules, feat_names, target)
    if not plotted:
        return _fit_model(X_train, y_train, feat_names, rules, target, random_state=random_state)

    baseline_m = _fit_model(X_train, y_train, feat_names, rules, target, random_state=random_state, monotonic=False)
    baseline_feas = _feas_score(baseline_m, X_val, feat_names, rules, target=target, feas_kw=feas_kw_full)
    baseline_rmse = holdout_rmse(baseline_m, X_val, y_val)

    best_m = baseline_m
    best_feas = baseline_feas
    best_rmse = baseline_rmse
    n_rules = len(plotted)

    def _train_from_params(params: dict[str, float], trial_no: int) -> tuple[Any, float, float]:
        rng = np.random.default_rng(int(random_state) + int(trial_no))
        X_syn, y_syn = _make_scm_synth_batch(
            X_train, y_train, feat_names, rules, target,
            n=n_synth,
            rule_idx=int(params["rule_idx"]),
            grid_lo_q=float(params["grid_lo_q"]),
            grid_hi_q=float(params["grid_hi_q"]),
            y_gain=float(params["y_gain"]),
            feature_jitter=float(params["feature_jitter"]),
            rng=rng,
        )
        if not y_syn.size:
            return baseline_m, baseline_rmse, baseline_feas
        synth_w = float(params["synth_weight"])
        X_fit = np.vstack([X_train, X_syn])
        y_fit = np.concatenate([y_train, y_syn])
        weights = np.concatenate([np.ones(len(y_train)), np.full(len(y_syn), synth_w)])
        m = _fit_model(
            X_fit, y_fit, feat_names, rules, target,
            random_state=random_state, sample_weight=weights, monotonic=True,
        )
        return (
            m,
            holdout_rmse(m, X_val, y_val),
            _feas_score(m, X_val, feat_names, rules, target=target, feas_kw=feas_kw_full),
        )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = {
            "rule_idx": trial.suggest_int("rule_idx", 0, n_rules - 1),
            "grid_lo_q": trial.suggest_float("grid_lo_q", 0.05, 0.45),
            "grid_hi_q": trial.suggest_float("grid_hi_q", 0.55, 0.95),
            "y_gain": trial.suggest_float("y_gain", 0.0, 20.0),
            "feature_jitter": trial.suggest_float("feature_jitter", 1e-3, 0.5, log=True),
            "synth_weight": trial.suggest_float("synth_weight", 0.15, 1.0),
        }
        _, rmse, feas = _train_from_params(params, trial.number)
        trial.set_user_attr("val_rmse", rmse)
        trial.set_user_attr("feasibility", feas)
        return rmse, feas

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=int(random_state), crossover_prob=0.9),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=max(1, min(os.cpu_count() or 1, 4)),
        show_progress_bar=show_progress_bar,
    )

    if study.best_trials:
        for t in study.best_trials:
            cand_m, cand_rmse, cand_feas = _train_from_params(t.params, int(t.number))
            if cand_feas <= baseline_feas + 1e-4:
                continue
            if cand_feas > best_feas + 1e-4 or (
                abs(cand_feas - best_feas) <= 1e-4 and cand_rmse < best_rmse - 1e-4
            ):
                best_m, best_feas, best_rmse = cand_m, cand_feas, cand_rmse

    return best_m


def _metrics(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cols: list[str],
    rules: list[dict],
    *,
    feas_kw: dict[str, Any],
) -> dict[str, float]:
    return {
        "val_rmse": holdout_rmse(model, X_val, y_val),
        "test_rmse": holdout_rmse(model, X_test, y_test),
        "feasibility": _feas_score(model, X_val, cols, rules, target=ohio_t1dm.TARGET, feas_kw=feas_kw),
    }


def main() -> None:
    sub = ohio_t1dm._patient_sub(str(ROOT), PATIENT_ID)
    scm_rules = list(ohio_t1dm.scm_rules_for(sub))
    X_all, y_all, X_test_df, y_test = ohio_t1dm.load_train_test(str(ROOT), PATIENT_ID)
    X_all = _drop_latent_columns(X_all)
    if X_test_df is not None:
        X_test_df = _drop_latent_columns(X_test_df)

    feat_names = list(X_all.columns)
    X_np = X_all.to_numpy(np.float64)
    y_np = y_all.to_numpy(np.float64)
    X_test = X_test_df.to_numpy(np.float64) if X_test_df is not None else None
    y_test_np = y_test.to_numpy(np.float64) if y_test is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_np, test_size=ohio_t1dm.TEST_SIZE, random_state=SEED,
    )
    if X_test is None:
        X_test, y_test_np = X_val, y_val

    feas_kw = dict(random_state=SEED, max_samples=DO_CURVE_MAX_SAMPLES, grid_size=FEASIBILITY_GRID_SIZE)
    t0 = time.perf_counter()
    baseline_m = _fit_model(X_train, y_train, feat_names, scm_rules, ohio_t1dm.TARGET)
    baseline = _metrics(baseline_m, X_val, y_val, X_test, y_test_np, feat_names, scm_rules, feas_kw=feas_kw)

    final_m = run_synthetic_augmentation(
        X_train, y_train, X_val, y_val, feat_names, scm_rules, ohio_t1dm.TARGET,
        n_trials=N_TRIALS,
        selection_rmse_weight=SELECTION_RMSE_WEIGHT,
        random_state=SEED,
        synth_per_round=N_SYNTH_ROWS,
    )
    final = _metrics(final_m, X_val, y_val, X_test, y_test_np, feat_names, scm_rules, feas_kw=feas_kw)

    payload = {
        "patient_id": PATIENT_ID,
        "approach": "scm_guided_synthetic_augmentation",
        "n_synth_rows_per_trial": N_SYNTH_ROWS,
        "n_trials": N_TRIALS,
        "selection_rmse_weight": SELECTION_RMSE_WEIGHT,
        "baseline": {k: round(v, 4) for k, v in baseline.items()},
        "final": {k: round(v, 4) for k, v in final.items()},
        "delta_feasibility": round(final["feasibility"] - baseline["feasibility"], 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    out_dir = ROOT / "results" / f"{ohio_t1dm.NAME}/{PATIENT_ID}" / "synthetic_augment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== Synthetic augmentation summary ===")
    print(f"Baseline  feas {baseline['feasibility']:.4f}  val_RMSE {baseline['val_rmse']:.4f}")
    print(f"Final     feas {final['feasibility']:.4f}  val_RMSE {final['val_rmse']:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
