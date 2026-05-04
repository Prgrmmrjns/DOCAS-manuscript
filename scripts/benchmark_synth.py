"""
Compare synthetic-data strategies: same splits, LGBM, val composite + test RMSE.

Defaults: 5 runs (different train/val/test splits), 500 Optuna trials per outer iter,
16 outer iterations (longer than main.py).

Run from repo root:
  python scripts/benchmark_synth.py
  python scripts/benchmark_synth.py --trials 500 --max-iters 16
  python scripts/benchmark_synth.py --runs 3 --trials 80 --fast  # shorter smoke test
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
import sys
import warnings
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor, early_stopping
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from lib import domain_target_pearson_objective, project_root  # noqa: E402

import main as main_mod  # noqa: E402

# Longer benchmark defaults (override main.py's n_trials=300, max_outer_iters=10).
BENCHMARK_DEFAULT_TRIALS = 500
BENCHMARK_DEFAULT_MAX_ITERS = 16
BENCHMARK_DEFAULT_RUNS = 5
RUN_SEED_STRIDE = 100_003


@dataclass
class Prep:
    dm: Any
    cols: list[str]
    sc_x: MinMaxScaler
    sc_y: MinMaxScaler
    X_train: np.ndarray
    y_train_s: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_val_orig: np.ndarray
    y_test_orig: np.ndarray
    y_val_s: np.ndarray
    X_val_df: pd.DataFrame
    baseline_rmse: float
    pearson0: float
    test0: float
    d_features: int
    shap_max_samples: int
    RULES: list
    random_state: int

def _prepare(dm: Any, *, random_state: int | None = None) -> Prep:
    rs = int(random_state) if random_state is not None else main_mod.random_state
    test_size, val_frac = main_mod.test_size, main_mod.val_frac
    RULES = list(dm.SCM_RULES)
    X, y = dm.load(project_root(__file__))
    X_train_pool, X_test_df, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs,
        stratify=y if getattr(dm, "TASK", "regression") == "classification" else None,
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=val_frac, random_state=rs, shuffle=True,
        stratify=y_train_pool if getattr(dm, "TASK", "regression") == "classification" else None,
    )
    cols = list(X_train_df.columns)
    d_features = len(cols)
    sc_x = MinMaxScaler((0.0, 1.0)).fit(X_train_df)
    sc_y = MinMaxScaler((0.0, 1.0)).fit(y_train.to_numpy(float).reshape(-1, 1))
    X_train = sc_x.transform(X_train_df).astype(np.float32, copy=False)
    X_val = sc_x.transform(X_val_df).astype(np.float32, copy=False)
    X_test = sc_x.transform(X_test_df).astype(np.float32, copy=False)
    y_train_s = sc_y.transform(y_train.to_numpy(float).reshape(-1, 1)).ravel().astype(np.float32, copy=False)
    y_val_s = sc_y.transform(y_val.to_numpy(float).reshape(-1, 1)).ravel().astype(np.float32, copy=False)
    y_val_orig = y_val.to_numpy(np.float64, copy=False)
    y_test_orig = y_test.to_numpy(np.float64, copy=False)
    X_val_pd = pd.DataFrame(X_val, columns=cols)
    shap_max = min(200, len(X_val_pd))

    def domain_pearson(model: Any, Xdf: pd.DataFrame) -> tuple[float, dict[str, Any]]:
        return domain_target_pearson_objective(
            model, Xdf, cols, RULES, dm.TARGET, max_samples=shap_max, random_state=rs,
        )

    def train_lgb(X_fit: np.ndarray, y_fit: np.ndarray) -> Any:
        m = LGBMRegressor(**main_mod.LGB_PARAMS)
        m.fit(
            X_fit, y_fit, eval_set=[(X_val, y_val_s)],
            callbacks=[early_stopping(main_mod.EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return m

    def fit_rmse_pearson(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[float, float, Any]:
        m = train_lgb(X_tr, y_tr)
        pred_o = sc_y.inverse_transform(m.predict(X_val).reshape(-1, 1)).ravel()
        rmse_v = sqrt(mean_squared_error(y_val_orig, pred_o))
        p01, _ = domain_pearson(m, X_val_pd)
        return rmse_v, p01, m

    baseline_rmse, pearson0, _ = fit_rmse_pearson(X_train, y_train_s)

    def test_rmse(X_tr: np.ndarray, y_tr: np.ndarray) -> float:
        m = train_lgb(X_tr, y_tr)
        pred_o = sc_y.inverse_transform(m.predict(X_test).reshape(-1, 1)).ravel()
        return sqrt(mean_squared_error(y_test_orig, pred_o))

    test0 = test_rmse(X_train, y_train_s)

    return Prep(
        dm=dm,
        cols=cols,
        sc_x=sc_x,
        sc_y=sc_y,
        X_train=X_train,
        y_train_s=y_train_s,
        X_val=X_val,
        X_test=X_test,
        y_val_orig=y_val_orig,
        y_test_orig=y_test_orig,
        y_val_s=y_val_s,
        X_val_df=X_val_pd,
        baseline_rmse=float(baseline_rmse),
        pearson0=float(pearson0),
        test0=float(test0),
        d_features=d_features,
        shap_max_samples=shap_max,
        RULES=RULES,
        random_state=rs,
    )


def _composite(rmse_v: float, p01: float, rmse_ref: float) -> float:
    return main_mod.composite_score(rmse_v, p01, rmse_ref=rmse_ref)


def _train_lgb_prep(p: Prep, X_fit: np.ndarray, y_fit: np.ndarray) -> Any:
    m = LGBMRegressor(**main_mod.LGB_PARAMS)
    m.fit(
        X_fit, y_fit, eval_set=[(p.X_val, p.y_val_s)],
        callbacks=[early_stopping(main_mod.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return m


def _fit_val_metrics(p: Prep, X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[float, float, Any]:
    m = _train_lgb_prep(p, X_tr, y_tr)
    pred_o = p.sc_y.inverse_transform(m.predict(p.X_val).reshape(-1, 1)).ravel()
    rmse_v = sqrt(mean_squared_error(p.y_val_orig, pred_o))
    p01, _ = domain_target_pearson_objective(
        m, p.X_val_df, p.cols, p.RULES, p.dm.TARGET, max_samples=p.shap_max_samples, random_state=p.random_state,
    )
    return rmse_v, p01, m


def _test_rmse(p: Prep, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    m = _train_lgb_prep(p, X_tr, y_tr)
    pred_o = p.sc_y.inverse_transform(m.predict(p.X_test).reshape(-1, 1)).ravel()
    return sqrt(mean_squared_error(p.y_test_orig, pred_o))


def mixup_batch(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, mix_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    d = X.shape[1]
    out_x = np.empty((batch_size, d), dtype=np.float32)
    out_y = np.empty(batch_size, dtype=np.float32)
    for k in range(batch_size):
        i, j = rng.integers(0, n, size=2)
        lam = float(rng.beta(mix_alpha, mix_alpha))
        out_x[k] = (lam * X[i] + (1.0 - lam) * X[j]).astype(np.float32)
        out_y[k] = np.float32(lam * float(y[i]) + (1.0 - lam) * float(y[j]))
    return out_x, out_y


def gaussian_batch(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, sigma_x: float, sigma_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(X), size=batch_size)
    noise_x = rng.normal(0.0, sigma_x, size=(batch_size, X.shape[1])).astype(np.float32)
    out_x = np.clip(X[idx].astype(np.float32) + noise_x, 0.0, 1.0)
    ny = rng.normal(0.0, sigma_y, size=batch_size).astype(np.float32)
    out_y = np.clip(y[idx].astype(np.float32) + ny, 0.0, 1.0)
    return out_x, out_y


def run_mixup_loop(
    p: Prep,
    *,
    batch_size: int,
    max_iters: int,
    n_trials_per_iter: int,
    mix_alpha: float,
    base_seed: int,
) -> dict[str, Any]:
    rmse_ref = p.baseline_rmse
    prev_score = _composite(p.baseline_rmse, p.pearson0, rmse_ref=rmse_ref)
    Xa, ya = p.X_train.copy(), p.y_train_s.copy()
    synth_added = 0
    for it in range(max_iters):
        best_c, best_rmse, best_p01 = -1.0, float("inf"), 0.0
        best_xb, best_yb = None, None
        for tr in range(n_trials_per_iter):
            rng = np.random.default_rng(base_seed + 10_003 * it + 97 * tr)
            xb, yb = mixup_batch(Xa, ya, batch_size, rng, mix_alpha)
            rmse_v, p01, _ = _fit_val_metrics(p, np.vstack((Xa, xb)), np.concatenate((ya, yb)))
            c = _composite(rmse_v, p01, rmse_ref=rmse_ref)
            if c > best_c:
                best_c, best_rmse, best_p01 = c, rmse_v, p01
                best_xb, best_yb = xb, yb
        if best_xb is None or best_c <= prev_score:
            break
        Xa = np.vstack((Xa, best_xb))
        ya = np.concatenate((ya, best_yb))
        synth_added += int(best_xb.shape[0])
        prev_score = best_c
    rmse_end, pearson_end, _ = _fit_val_metrics(p, Xa, ya)
    end_c = _composite(rmse_end, pearson_end, rmse_ref=rmse_ref)
    test1 = _test_rmse(p, Xa, ya)
    return {
        "method": "mixup",
        "val_rmse_end": float(rmse_end),
        "pearson_end": float(pearson_end),
        "composite_end": float(end_c),
        "test_rmse_end": float(test1),
        "synth_rows_added": synth_added,
        "baseline_rmse": float(p.baseline_rmse),
        "test_rmse_baseline": float(p.test0),
    }


def run_gaussian_loop(
    p: Prep,
    *,
    batch_size: int,
    max_iters: int,
    n_trials_per_iter: int,
    base_seed: int,
) -> dict[str, Any]:
    rmse_ref = p.baseline_rmse
    prev_score = _composite(p.baseline_rmse, p.pearson0, rmse_ref=rmse_ref)
    Xa, ya = p.X_train.copy(), p.y_train_s.copy()
    synth_added = 0
    for it in range(max_iters):
        best_c = -1.0
        best_bundle: tuple[np.ndarray, np.ndarray] | None = None
        best_metrics: tuple[float, float] | None = None
        for tr in range(n_trials_per_iter):
            rng = np.random.default_rng(base_seed + 20_009 * it + 131 * tr)
            sigma_x = float(rng.uniform(0.01, 0.12))
            sigma_y = float(rng.uniform(0.01, 0.10))
            xb, yb = gaussian_batch(Xa, ya, batch_size, rng, sigma_x, sigma_y)
            rmse_v, p01, _ = _fit_val_metrics(p, np.vstack((Xa, xb)), np.concatenate((ya, yb)))
            c = _composite(rmse_v, p01, rmse_ref=rmse_ref)
            if c > best_c:
                best_c = c
                best_bundle = (xb, yb)
                best_metrics = (rmse_v, p01)
        if best_bundle is None or best_c <= prev_score:
            break
        xb, yb = best_bundle
        Xa = np.vstack((Xa, xb))
        ya = np.concatenate((ya, yb))
        synth_added += int(xb.shape[0])
        prev_score = best_c
    rmse_end, pearson_end, _ = _fit_val_metrics(p, Xa, ya)
    end_c = _composite(rmse_end, pearson_end, rmse_ref=rmse_ref)
    test1 = _test_rmse(p, Xa, ya)
    return {
        "method": "gaussian_noise",
        "val_rmse_end": float(rmse_end),
        "pearson_end": float(pearson_end),
        "composite_end": float(end_c),
        "test_rmse_end": float(test1),
        "synth_rows_added": synth_added,
        "baseline_rmse": float(p.baseline_rmse),
        "test_rmse_baseline": float(p.test0),
    }


def run_bootstrap_loop(
    p: Prep,
    *,
    batch_size: int,
    max_iters: int,
    n_trials_per_iter: int,
    base_seed: int,
) -> dict[str, Any]:
    """Resample training rows with replacement (duplicate information)."""
    rmse_ref = p.baseline_rmse
    prev_score = _composite(p.baseline_rmse, p.pearson0, rmse_ref=rmse_ref)
    Xa, ya = p.X_train.copy(), p.y_train_s.copy()
    synth_added = 0
    n = len(Xa)
    for it in range(max_iters):
        best_c = -1.0
        best_xb, best_yb = None, None
        for tr in range(n_trials_per_iter):
            rng = np.random.default_rng(base_seed + 30_017 * it + 173 * tr)
            idx = rng.integers(0, n, size=batch_size)
            xb = Xa[idx].copy()
            yb = ya[idx].astype(np.float32, copy=True)
            rmse_v, p01, _ = _fit_val_metrics(p, np.vstack((Xa, xb)), np.concatenate((ya, yb)))
            c = _composite(rmse_v, p01, rmse_ref=rmse_ref)
            if c > best_c:
                best_c = c
                best_xb, best_yb = xb, yb
        if best_xb is None or best_c <= prev_score:
            break
        Xa = np.vstack((Xa, best_xb))
        ya = np.concatenate((ya, best_yb))
        synth_added += int(best_xb.shape[0])
        prev_score = best_c
    rmse_end, pearson_end, _ = _fit_val_metrics(p, Xa, ya)
    end_c = _composite(rmse_end, pearson_end, rmse_ref=rmse_ref)
    test1 = _test_rmse(p, Xa, ya)
    return {
        "method": "bootstrap_duplicate",
        "val_rmse_end": float(rmse_end),
        "pearson_end": float(pearson_end),
        "composite_end": float(end_c),
        "test_rmse_end": float(test1),
        "synth_rows_added": synth_added,
        "baseline_rmse": float(p.baseline_rmse),
        "test_rmse_baseline": float(p.test0),
    }


def run_all_methods_for_split(
    dm: Any,
    prep: Prep,
    *,
    n_trials: int,
    max_iters: int,
    batch_size: int,
    mix_alpha: float,
    augment_seed: int,
) -> list[dict[str, Any]]:
    """Run every benchmark method on one fixed split (``prep``)."""
    rs = prep.random_state
    rows: list[dict[str, Any]] = []

    s0 = main_mod.run_dataset(
        dm,
        max_outer_iters_override=0,
        n_trials_override=n_trials,
        skip_outputs=True,
        random_state_override=rs,
    )
    rows.append(
        {
            "method": "baseline_no_synth",
            "val_rmse_end": s0["val_rmse_end"],
            "pearson_end": s0["pearson_end"],
            "composite_end": s0["composite_end"],
            "test_rmse_end": s0["test_rmse_end"],
            "synth_rows_added": s0["synth_rows_added"],
            "baseline_rmse": s0["baseline_rmse"],
            "test_rmse_baseline": s0["test_rmse_baseline"],
        }
    )

    s1 = main_mod.run_dataset(
        dm,
        n_trials_override=n_trials,
        max_outer_iters_override=max_iters,
        skip_outputs=True,
        uniform_shap_weights=False,
        random_state_override=rs,
    )
    rows.append(
        {
            "method": "scm_optuna_shap",
            "val_rmse_end": s1["val_rmse_end"],
            "pearson_end": s1["pearson_end"],
            "composite_end": s1["composite_end"],
            "test_rmse_end": s1["test_rmse_end"],
            "synth_rows_added": s1["synth_rows_added"],
            "baseline_rmse": s1["baseline_rmse"],
            "test_rmse_baseline": s1["test_rmse_baseline"],
        }
    )

    s2 = main_mod.run_dataset(
        dm,
        n_trials_override=n_trials,
        max_outer_iters_override=max_iters,
        skip_outputs=True,
        uniform_shap_weights=True,
        random_state_override=rs,
    )
    rows.append(
        {
            "method": "scm_optuna_uniform_shap",
            "val_rmse_end": s2["val_rmse_end"],
            "pearson_end": s2["pearson_end"],
            "composite_end": s2["composite_end"],
            "test_rmse_end": s2["test_rmse_end"],
            "synth_rows_added": s2["synth_rows_added"],
            "baseline_rmse": s2["baseline_rmse"],
            "test_rmse_baseline": s2["test_rmse_baseline"],
        }
    )

    rows.append(
        run_mixup_loop(
            prep, batch_size=batch_size, max_iters=max_iters, n_trials_per_iter=n_trials,
            mix_alpha=mix_alpha, base_seed=augment_seed,
        )
    )
    rows.append(
        run_gaussian_loop(
            prep, batch_size=batch_size, max_iters=max_iters, n_trials_per_iter=n_trials, base_seed=augment_seed + 1,
        )
    )
    rows.append(
        run_bootstrap_loop(
            prep, batch_size=batch_size, max_iters=max_iters, n_trials_per_iter=n_trials, base_seed=augment_seed + 2,
        )
    )
    rows.append(
        run_smote_like_loop(
            prep, batch_size=batch_size, max_iters=max_iters, n_trials_per_iter=n_trials, base_seed=augment_seed + 3,
        )
    )
    return rows


def _aggregate_metrics(per_run: list[dict[str, Any]], metric_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """For each method, mean/std/min/max over runs."""
    methods: set[str] = set()
    for block in per_run:
        for r in block["rows"]:
            methods.add(r["method"])
    out: dict[str, dict[str, Any]] = {}
    for m in sorted(methods):
        out[m] = {}
        for k in metric_keys:
            vals = np.array([next(r[k] for r in block["rows"] if r["method"] == m) for block in per_run], dtype=np.float64)
            out[m][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": int(len(vals)),
            }
    return out


def _winner_counts(per_run: list[dict[str, Any]], key: str, *, reverse: bool) -> dict[str, int]:
    c: Counter[str] = Counter()
    for block in per_run:
        rows = block["rows"]
        best = sorted(rows, key=lambda r: r[key], reverse=reverse)[0]
        c[best["method"]] += 1
    return dict(c)


def run_smote_like_loop(
    p: Prep,
    *,
    batch_size: int,
    max_iters: int,
    n_trials_per_iter: int,
    base_seed: int,
) -> dict[str, Any]:
    """Interpolate one random feature dimension between two random training rows (SMOTE-like)."""
    rmse_ref = p.baseline_rmse
    prev_score = _composite(p.baseline_rmse, p.pearson0, rmse_ref=rmse_ref)
    Xa, ya = p.X_train.copy(), p.y_train_s.copy()
    synth_added = 0
    d = p.d_features
    for it in range(max_iters):
        best_c = -1.0
        best_xb, best_yb = None, None
        for tr in range(n_trials_per_iter):
            rng = np.random.default_rng(base_seed + 40_021 * it + 191 * tr)
            out_x = np.empty((batch_size, d), dtype=np.float32)
            out_y = np.empty(batch_size, dtype=np.float32)
            for k in range(batch_size):
                i, j = rng.integers(0, len(Xa), size=2)
                dim = int(rng.integers(0, d))
                t = float(rng.random())
                row = Xa[i].copy()
                row[dim] = np.float32((1.0 - t) * float(Xa[i, dim]) + t * float(Xa[j, dim]))
                out_x[k] = np.clip(row, 0.0, 1.0)
                out_y[k] = np.float32((1.0 - t) * float(ya[i]) + t * float(ya[j]))
            rmse_v, p01, _ = _fit_val_metrics(p, np.vstack((Xa, out_x)), np.concatenate((ya, out_y)))
            c = _composite(rmse_v, p01, rmse_ref=rmse_ref)
            if c > best_c:
                best_c = c
                best_xb, best_yb = out_x, out_y
        if best_xb is None or best_c <= prev_score:
            break
        Xa = np.vstack((Xa, best_xb))
        ya = np.concatenate((ya, best_yb))
        synth_added += int(best_xb.shape[0])
        prev_score = best_c
    rmse_end, pearson_end, _ = _fit_val_metrics(p, Xa, ya)
    end_c = _composite(rmse_end, pearson_end, rmse_ref=rmse_ref)
    test1 = _test_rmse(p, Xa, ya)
    return {
        "method": "smote_like_mixdim",
        "val_rmse_end": float(rmse_end),
        "pearson_end": float(pearson_end),
        "composite_end": float(end_c),
        "test_rmse_end": float(test1),
        "synth_rows_added": synth_added,
        "baseline_rmse": float(p.baseline_rmse),
        "test_rmse_baseline": float(p.test0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark synthetic row generators (multi-run).")
    ap.add_argument("--dataset", default="d1namo", help="Dataset module name under scripts/")
    ap.add_argument("--runs", type=int, default=BENCHMARK_DEFAULT_RUNS, help="Number of train/val/test splits")
    ap.add_argument("--trials", type=int, default=None, help=f"Optuna trials per outer iter (default {BENCHMARK_DEFAULT_TRIALS})")
    ap.add_argument("--max-iters", type=int, default=None, help=f"Max outer iterations (default {BENCHMARK_DEFAULT_MAX_ITERS})")
    ap.add_argument("--batch-size", type=int, default=None, help=f"default {main_mod.SYNTH_BATCH_SIZE}")
    ap.add_argument("--mix-alpha", type=float, default=0.4, help="Beta(alpha,alpha) for mixup")
    ap.add_argument("--seed", type=int, default=main_mod.random_state, help="Base RNG seed; split i uses seed + i * stride")
    ap.add_argument("--fast", action="store_true", help="Shorter inner search: 80 trials & 5 outer iters (still uses --runs)")
    ap.add_argument("--out", type=str, default="", help="JSON path (default results/{dataset}_synth_benchmark.json)")
    args = ap.parse_args()

    n_runs = args.runs
    if args.fast:
        n_trials = 80 if args.trials is None else args.trials
        max_iters = 5 if args.max_iters is None else args.max_iters
    else:
        n_trials = BENCHMARK_DEFAULT_TRIALS if args.trials is None else args.trials
        max_iters = BENCHMARK_DEFAULT_MAX_ITERS if args.max_iters is None else args.max_iters
    batch_size = args.batch_size if args.batch_size is not None else main_mod.SYNTH_BATCH_SIZE

    dm = importlib.import_module(args.dataset)
    metric_keys = ("test_rmse_end", "composite_end", "val_rmse_end", "pearson_end", "synth_rows_added")

    per_run: list[dict[str, Any]] = []
    for run_idx in range(n_runs):
        split_rs = int(args.seed + run_idx * RUN_SEED_STRIDE)
        print(f"\n--- Run {run_idx + 1}/{n_runs}  split_random_state={split_rs} ---", flush=True)
        prep = _prepare(dm, random_state=split_rs)
        augment_seed = split_rs + 9_001
        rows = run_all_methods_for_split(
            dm,
            prep,
            n_trials=n_trials,
            max_iters=max_iters,
            batch_size=batch_size,
            mix_alpha=args.mix_alpha,
            augment_seed=augment_seed,
        )
        per_run.append({"run_index": run_idx, "random_state": split_rs, "rows": rows})
        by_test = sorted(rows, key=lambda r: r["test_rmse_end"])
        by_comp = sorted(rows, key=lambda r: r["composite_end"], reverse=True)
        print(f"  best test_rmse this split: {by_test[0]['method']} ({by_test[0]['test_rmse_end']:.4f})")
        print(f"  best composite this split: {by_comp[0]['method']} ({by_comp[0]['composite_end']:.4f})")

    agg = _aggregate_metrics(per_run, metric_keys)
    win_test = _winner_counts(per_run, "test_rmse_end", reverse=False)
    win_comp = _winner_counts(per_run, "composite_end", reverse=True)

    by_mean_test = sorted(agg.items(), key=lambda kv: kv[1]["test_rmse_end"]["mean"])
    by_mean_comp = sorted(agg.items(), key=lambda kv: kv[1]["composite_end"]["mean"], reverse=True)

    print(f"\n{'=' * 80}")
    print(
        f"SUMMARY  dataset={dm.NAME}  runs={n_runs}  max_iters={max_iters}  n_trials={n_trials}  batch={batch_size}\n"
        f"Each run uses a different train/val/test split (sklearn random_state = seed + run_index * {RUN_SEED_STRIDE})."
    )
    hdr = f"{'method':<28} {'test_rmse':>24} {'composite':>24} {'synth_n':>18}"
    print(hdr)
    print("-" * len(hdr))
    for m, st in by_mean_test:
        te = st["test_rmse_end"]
        co = st["composite_end"]
        sn = st["synth_rows_added"]
        print(
            f"{m:<28} {te['mean']:8.4f} ± {te['std']:6.4f}   "
            f"{co['mean']:8.4f} ± {co['std']:6.4f}   "
            f"{sn['mean']:8.1f} ± {sn['std']:5.1f}"
        )

    print("\n--- Mean test RMSE (lower better) ---")
    for i, (m, st) in enumerate(by_mean_test[:7], 1):
        t = st["test_rmse_end"]
        print(f"  {i}. {m}: {t['mean']:.4f} ± {t['std']:.4f}  [{t['min']:.4f}, {t['max']:.4f}]")

    print("\n--- Mean val composite (higher better) ---")
    for i, (m, st) in enumerate(by_mean_comp[:7], 1):
        t = st["composite_end"]
        print(f"  {i}. {m}: {t['mean']:.4f} ± {t['std']:.4f}  [{t['min']:.4f}, {t['max']:.4f}]")

    print("\n--- # wins per method (best test_rmse on each split) ---")
    for m, cnt in sorted(win_test.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {m}: {cnt}/{n_runs}")

    print("\n--- # wins per method (best composite on each split) ---")
    for m, cnt in sorted(win_comp.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {m}: {cnt}/{n_runs}")

    overall_test_m, _ = by_mean_test[0]
    overall_comp_m, _ = by_mean_comp[0]
    print(f"\nOverall (mean test_rmse): {overall_test_m}")
    print(f"Overall (mean composite): {overall_comp_m}")

    out_path = args.out or str(ROOT / "results" / f"{dm.NAME}_synth_benchmark.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dm.NAME,
        "n_runs": n_runs,
        "seed_base": args.seed,
        "split_seed_stride": RUN_SEED_STRIDE,
        "max_outer_iters": max_iters,
        "n_trials_per_iter": n_trials,
        "batch_size": batch_size,
        "per_run": per_run,
        "aggregate_by_method": agg,
        "rank_mean_test_rmse": [m for m, _ in by_mean_test],
        "rank_mean_composite": [m for m, _ in by_mean_comp],
        "wins_best_test_rmse_per_run": win_test,
        "wins_best_composite_per_run": win_comp,
    }
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
