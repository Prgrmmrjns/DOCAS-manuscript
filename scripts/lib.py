from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict


import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import shap

from model import make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

def holdout_rmse(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(model.predict(X), dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((pred - yv) ** 2)))


class SCMRule(TypedDict):
    start: str
    end: str
    relationship_fn: Callable[[str, str, np.ndarray, np.ndarray], float]


_MONO_TOL = 1e-9


def monotonic_pair_violations(
    grid: np.ndarray,
    avg: np.ndarray,
    *,
    increasing: bool,
) -> tuple[int, int]:
    """Global monotonicity over all intervention pairs with smaller ``grid[i] < grid[j]``.

    Not adjacent-only: any pair of grid points with a lower and higher intervention
    level must satisfy the rule direction (non-strict; equal ``avg`` is allowed).
    """
    g = np.asarray(grid, dtype=np.float64).ravel()
    y = np.asarray(avg, dtype=np.float64).ravel()
    if g.size != y.size:
        raise ValueError("grid and avg must have the same length")
    mask = np.isfinite(g) & np.isfinite(y)
    g, y = g[mask], y[mask]
    if g.size < 2:
        return 0, 0
    order = np.argsort(g, kind="mergesort")
    g, y = g[order], y[order]
    n = int(g.size)
    iu, ju = np.triu_indices(n, k=1)
    valid = g[iu] < g[ju] - _MONO_TOL
    if not np.any(valid):
        return 0, 0
    if increasing:
        bad = y[ju] < y[iu] - _MONO_TOL
    else:
        bad = y[ju] > y[iu] + _MONO_TOL
    bad = bad & valid
    return int(bad.sum()), int(valid.sum())


def monotonic_step_violations(
    avg: np.ndarray,
    *,
    increasing: bool,
    grid: np.ndarray | None = None,
) -> tuple[int, int]:
    """Backward-compatible alias; pass ``grid`` for intervention-level global checks."""
    if grid is not None:
        return monotonic_pair_violations(grid, avg, increasing=increasing)
    g = np.arange(int(np.asarray(avg, dtype=np.float64).size), dtype=np.float64)
    return monotonic_pair_violations(g, avg, increasing=increasing)


def monotonic_aligned_fraction(
    avg: np.ndarray,
    *,
    increasing: bool,
    grid: np.ndarray | None = None,
) -> float:
    n_viol, n_pairs = monotonic_step_violations(avg, increasing=increasing, grid=grid)
    return float("nan") if n_pairs == 0 else float(1.0 - n_viol / n_pairs)


class MonotonicRelationship:
    __slots__ = ("start", "end", "increasing")

    def __init__(self, start: str, end: str, *, increasing: bool) -> None:
        self.start, self.end, self.increasing = str(start), str(end), bool(increasing)

    def as_rule(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "relationship_fn": self}

    def __call__(self, rs: str, re: str, _x: np.ndarray, y: np.ndarray) -> float:
        if rs != self.start or re != self.end:
            return float("nan")
        yy = np.asarray(y, dtype=np.float64).ravel()
        return float("nan") if yy.size < 2 or not np.isfinite(yy).all() else monotonic_aligned_fraction(yy, increasing=self.increasing)


def _latent_indices(cols: list[str], latent_columns: tuple[str, ...]) -> set[int]:
    return {cols.index(c) for c in latent_columns if c in cols}


def _policy_mask(X: np.ndarray, u: int, latent_idx: set[int]) -> np.ndarray:
    if u in latent_idx:
        return np.isfinite(X[:, u])
    obs = [j for j in range(X.shape[1]) if j not in latent_idx]
    return np.all(np.isfinite(X[:, obs]), axis=1) if obs else np.ones(X.shape[0], dtype=bool)


def _intervention_grid(values: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.empty(0)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo + 1e-12:
        return np.full(max(2, int(n)), lo, dtype=np.float64)
    return np.linspace(lo, hi, max(2, int(n)), dtype=np.float64)


def _do_curve(
    model: Any,
    X: np.ndarray,
    u: int,
    n_grid: int,
    v: int | None = None,
    *,
    latent_idx: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    latent_idx = latent_idx or set()
    X = np.ascontiguousarray(X, dtype=np.float64)
    mask = _policy_mask(X, u, latent_idx)
    Xp = X[mask] if int(mask.sum()) >= 3 else X
    grid = _intervention_grid(Xp[:, u], n_grid)
    if not grid.size:
        return grid, grid
    n = int(Xp.shape[0])
    ref = np.asarray(model.predict(Xp), dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    clip = tuple(np.percentile(ref, [1, 99])) if ref.size >= 10 else None
    g = len(grid)
    Xb = np.repeat(Xp, g, axis=0)
    Xb[:, u] = np.tile(grid, n)
    if v is None:
        pred = np.asarray(model.predict(Xb), dtype=np.float64).reshape(n, g).T
    else:
        raw = np.asarray(model.predict(Xb, pred_contrib=True), dtype=np.float64)
        pred = raw[:, v].reshape(n, g).T if raw.ndim == 2 and v < raw.shape[1] - 1 else np.full((g, n), np.nan)
    if clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return grid, np.nanmedian(pred, axis=1)


def do_curve_avg(
    model: Any,
    X: np.ndarray,
    u: int,
    v: int | None = None,
    *,
    grid_size: int = 31,
    cols: list[str] | None = None,
    latent_columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    li = _latent_indices(cols, latent_columns) if cols else set()
    return _do_curve(model, X, u, int(grid_size), v, latent_idx=li)


# Back-compat aliases
do_curve_avg_preds = lambda model, X, u, **kw: do_curve_avg(model, X, u, **kw)
do_curve_avg_feature_sv = lambda model, X, u, v, **kw: do_curve_avg(model, X, u, v, **kw)


def plotted_scm_rules(rules: list[dict[str, Any]], cols: list[str], target_name: str) -> list[dict[str, Any]]:
    t = str(target_name)
    return [
        r for r in rules
        if (u := str(r.get("start", ""))) in cols and u != (v := str(r.get("end", ""))) and (v == t or v in cols)
    ]


def rule_do_curve(
    model: Any,
    X_np: np.ndarray,
    cols: list[str],
    r: dict[str, Any],
    target_name: str,
    *,
    latent_columns: tuple[str, ...] = (),
    grid_size: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Same do-curve as save_do_curve_progress_figure (validation policy, plotted rules only)."""
    u, v = str(r["start"]), str(r["end"])
    kw = dict(grid_size=grid_size, cols=cols, latent_columns=latent_columns)
    if v == target_name:
        return do_curve_avg(model, X_np, cols.index(u), **kw)
    return do_curve_avg(model, X_np, cols.index(u), cols.index(v), **kw)


def compute_feasibility(
    model: Any,
    X: np.ndarray,
    cols: list[str],
    rules: list[dict[str, Any]],
    target: str,
    *,
    random_state: int = 42,
    max_samples: int | None = 72,
    grid_size: int = 31,
    latent_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    X_np = np.asarray(X, dtype=np.float64)
    if max_samples is not None and X_np.shape[0] > max_samples:
        X_np = X_np[np.random.default_rng(random_state).choice(X_np.shape[0], max_samples, replace=False)]
    items: list[dict[str, Any]] = []
    for r in plotted_scm_rules(rules, cols, target):
        fn = r.get("relationship_fn")
        u, v = str(r["start"]), str(r["end"])
        increasing = bool(getattr(fn, "increasing", True))
        grid, avg = rule_do_curve(
            model, X_np, cols, r, target,
            latent_columns=latent_columns, grid_size=grid_size,
        )
        n_viol, n_steps = monotonic_pair_violations(grid, avg, increasing=increasing)
        aligned = monotonic_aligned_fraction(avg, increasing=increasing, grid=grid)
        items.append({
            "rule": f"{u}->{v}",
            "ideal_monotonic": 1.0 if increasing else -1.0,
            "increasing": increasing,
            "n_grid_points": int(avg.size),
            "n_steps": n_steps,
            "n_violations": n_viol,
            "aligned_step_fraction": None if not np.isfinite(aligned) else round(aligned, 4),
            "feature": u,
            "grid": grid.tolist() if grid.size else [],
            "avg": avg.tolist() if avg.size else [],
        })
    total_viol = sum(int(x["n_violations"]) for x in items)
    total_steps = sum(int(x["n_steps"]) for x in items)
    feas = float(1.0 - total_viol / total_steps) if total_steps > 0 else 0.5
    feas = float(np.clip(feas, 0.0, 1.0))
    return {
        "feasibility_score_0_1": round(feas, 4),
        "n_violations_total": total_viol,
        "n_steps_total": total_steps,
        "aligned_step_fraction_global": round(feas, 4) if total_steps > 0 else None,
        "rules": items,
    }


def _sample_rows(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


_IMPUTER_COEF_RANGE = (-2.5, 2.5)
_IMPUTER_NOISE_RANGE = (1e-4, 0.12)


@dataclass(frozen=True)
class LatentImputerConfig:
    """Latent imputer: SHAP main + target terms; optional intercept and SHAP interactions."""

    use_intercept: bool = False
    use_shap_interactions: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LatentImputerConfig":
        if not d:
            return cls()
        return cls(
            use_intercept=bool(d.get("use_intercept", False)),
            use_shap_interactions=bool(d.get("use_shap_interactions", True)),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)))


def _zscore(v: np.ndarray, mu: float, sd: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.where(np.isfinite(v), (v - mu) / sd, 0.0)


def _scalar_stats(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    mu, sd = float(np.mean(v)), float(np.std(v))
    return mu, sd if sd > 1e-12 else 1.0

def _shap_feature_stats(shap_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std of SHAP interaction features (for z-scoring before imputation)."""
    n = shap_features.shape[1]
    if n == 0:
        return np.array([]), np.array([])
    feat_mean = np.zeros(n, dtype=np.float64)
    feat_std = np.ones(n, dtype=np.float64)
    for j in range(n):
        finite = np.asarray(shap_features[:, j], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        feat_mean[j] = float(np.mean(finite))
        sd = float(np.std(finite))
        if sd > 1e-12:
            feat_std[j] = sd
    return feat_mean, feat_std


def _latent_feat_names(
    predictor_cols: list[str], cfg: LatentImputerConfig | None = None,
) -> list[str]:
    cfg = cfg or LatentImputerConfig()
    names: list[str] = []
    if cfg.use_intercept:
        names.append("intercept")
    for c in predictor_cols:
        names.append(f"shap:{c}")
    if cfg.use_shap_interactions:
        for i, a in enumerate(predictor_cols):
            for b in predictor_cols[i + 1 :]:
                names.append(f"shap_inter:{a}*{b}")
    names.extend(["y", "y_hat", "y*y_hat"])
    return names


def _extract_shap_features(shap_interactions: np.ndarray, predictor_idx: list[int]) -> np.ndarray:
    inter = shap_interactions[:, predictor_idx, :][:, :, predictor_idx]
    K = len(predictor_idx)
    parts = []
    for i in range(K):
        parts.append(inter[:, i, i].reshape(-1, 1))
    for i in range(K):
        for j in range(i + 1, K):
            parts.append((inter[:, i, j] * 2).reshape(-1, 1))
    if parts:
        return np.hstack(parts)
    return np.empty((shap_interactions.shape[0], 0))


def _latent_imputer_design(
    shap_features: np.ndarray,
    y: np.ndarray | None,
    y_hat: np.ndarray,
    *,
    use_true_y: bool,
    shap_feat_mean: np.ndarray,
    shap_feat_std: np.ndarray,
    y_mean: float,
    y_std: float,
    yh_mean: float,
    yh_std: float,
    n_main: int,
    cfg: LatentImputerConfig | None = None,
) -> np.ndarray:
    """Row-wise design: optional intercept, SHAP main, optional SHAP interactions, y / y_hat / y*y_hat."""
    cfg = cfg or LatentImputerConfig()
    y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
    y_for = np.asarray(y, dtype=np.float64).ravel() if use_true_y and y is not None else y_hat
    zy = _zscore(y_for, y_mean, y_std)
    zyh = _zscore(y_hat, yh_mean, yh_std)
    n_rows = int(shap_features.shape[0])

    if shap_features.shape[1] > 0:
        shap_z = (shap_features - shap_feat_mean) / shap_feat_std
        shap_z = np.where(np.isfinite(shap_z), shap_z, 0.0)
    else:
        shap_z = shap_features

    n_main = max(0, int(n_main))
    main_z = shap_z[:, :n_main] if n_main > 0 else np.empty((n_rows, 0))
    inter_z = shap_z[:, n_main:] if shap_z.shape[1] > n_main else np.empty((n_rows, 0))

    parts: list[np.ndarray] = []
    if cfg.use_intercept:
        parts.append(np.ones((n_rows, 1)))
    if main_z.shape[1] > 0:
        parts.append(main_z)
    if cfg.use_shap_interactions and inter_z.shape[1] > 0:
        parts.append(inter_z)
    parts.extend([zy.reshape(-1, 1), zyh.reshape(-1, 1), (zy * zyh).reshape(-1, 1)])
    return np.hstack(parts)


def _impute_latent_column(
    X: np.ndarray,
    latent_idx: int,
    imputer_design: np.ndarray,
    coef: np.ndarray,
    lo: float,
    hi: float,
    *,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    slope: float = 1.0,
) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64).copy()
    mu_z = lo + (hi - lo) * _sigmoid(float(slope) * (imputer_design @ coef.ravel()))
    if noise_scale > 0.0 and rng is not None:
        mu_z = mu_z + rng.normal(0.0, noise_scale, size=mu_z.shape[0])
    out[:, latent_idx] = np.clip(mu_z, lo, hi)
    return out


def _nsga_scalarized_trial(study: optuna.Study, rmse_weight: float) -> optuna.trial.FrozenTrial:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    assert done, "NSGA-II: no complete trials"
    w = float(np.clip(rmse_weight, 0.0, 1.0))
    rmses, feass = [float(t.values[0]) for t in done], [float(t.values[1]) for t in done]
    r_lo, r_hi, f_lo, f_hi = min(rmses), max(rmses), min(feass), max(feass)
    best_t, best_score = done[0], -np.inf
    for t in done:
        r, f = float(t.values[0]), float(t.values[1])
        r_n = 0.5 if r_hi <= r_lo + 1e-15 else (r_hi - r) / (r_hi - r_lo)
        f_n = 0.5 if f_hi <= f_lo + 1e-15 else (f - f_lo) / (f_hi - f_lo)
        score = w * r_n + (1.0 - w) * f_n
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def optimize_joint_latents(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cols: list[str],
    latent_cols: list[str],
    hidden_bounds: dict[str, tuple[float, float]],
    n_trials: int,
    random_state: int,
    fit_model_fn: Callable[[np.ndarray, np.ndarray], Any],
    feasibility_fn: Callable[[Any, np.ndarray], float],
    selection_rmse_weight: float = 0.5,
    show_progress_bar: bool = True,
) -> tuple[dict[str, dict[str, Any]], float, float, int, Any, np.ndarray, np.ndarray, int]:
    cfg = LatentImputerConfig()
    coef_lo, coef_hi = _IMPUTER_COEF_RANGE
    noise_lo, noise_hi = _IMPUTER_NOISE_RANGE

    m1 = fit_model_fn(X_train, y_train)
    y_hat_train = np.asarray(m1.predict(X_train), dtype=np.float64).ravel()
    y_hat_val = np.asarray(m1.predict(X_val), dtype=np.float64).ravel()
    y_mean, y_std = _scalar_stats(y_train)
    yh_mean, yh_std = _scalar_stats(y_hat_train)
    y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
    explainer = shap.TreeExplainer(m1)
    shap_train = explainer.shap_interaction_values(X_train)
    shap_val = explainer.shap_interaction_values(X_val)

    specs: list[dict[str, Any]] = []
    for col in latent_cols:
        lo, hi = float(hidden_bounds[col][0]), float(hidden_bounds[col][1])
        predictors = [c for c in cols if c != col]
        predictor_idx = [cols.index(c) for c in predictors]
        shap_feat_train = _extract_shap_features(shap_train, predictor_idx)
        shap_feat_mean, shap_feat_std = _shap_feature_stats(shap_feat_train)
        specs.append({
            "col": col, "lo": lo, "hi": hi, "latent_idx": cols.index(col),
            "predictor_cols": predictors, "predictor_idx": predictor_idx,
            "n_main": len(predictor_idx),
            "feat_names": _latent_feat_names(predictors, cfg),
            "shap_feat_mean": shap_feat_mean, "shap_feat_std": shap_feat_std,
            "shap_feat_train": shap_feat_train,
            "shap_feat_val": _extract_shap_features(shap_val, predictor_idx),
        })

    def _impute_joint_local(
        X: np.ndarray, coefs: dict[str, np.ndarray], noises: dict[str, float], slopes: dict[str, float],
        *, y: np.ndarray | None, y_hat: np.ndarray, use_true_y: bool, is_train: bool,
    ) -> np.ndarray:
        out = np.asarray(X, dtype=np.float64).copy()
        for s in specs:
            col = str(s["col"])
            shap_feat = s["shap_feat_train"] if is_train else s["shap_feat_val"]
            imputer_design = _latent_imputer_design(
                shap_feat, y, y_hat, use_true_y=use_true_y,
                shap_feat_mean=s["shap_feat_mean"], shap_feat_std=s["shap_feat_std"],
                y_mean=y_mean, y_std=y_std, yh_mean=yh_mean, yh_std=yh_std,
                n_main=int(s["n_main"]), cfg=cfg,
            )
            out = _impute_latent_column(
                out, int(s["latent_idx"]), imputer_design, coefs[col], float(s["lo"]), float(s["hi"]),
                noise_scale=float(noises[col]), slope=float(slopes[col]),
            )
        return out

    def _build_params(
        coefs: dict[str, np.ndarray], noises: dict[str, float], slopes: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for s in specs:
            col = str(s["col"])
            out[col] = {
                "latent_col": col, "predictor_cols": list(s["predictor_cols"]),
                "feature_names": list(s["feat_names"]),
                "n_main": int(s["n_main"]),
                "lo": float(s["lo"]), "hi": float(s["hi"]),
                "coef": coefs[col].tolist(),
                "noise_scale": noises[col],
                "slope": float(slopes[col]),
                "shap_feat_mean": np.asarray(s["shap_feat_mean"]).tolist(),
                "shap_feat_std": np.asarray(s["shap_feat_std"]).tolist(),
                "y_mean": y_mean, "y_std": y_std, "yh_mean": yh_mean, "yh_std": yh_std,
                "imputer_config": cfg.to_dict(),
            }
        return out

    def _eval(
        coefs: dict[str, np.ndarray], noises: dict[str, float], slopes: dict[str, float],
    ) -> tuple[float, float, Any, np.ndarray, np.ndarray]:
        X_fit = _impute_joint_local(X_train, coefs, noises, slopes, y=y_train, y_hat=y_hat_train, use_true_y=True, is_train=True)
        X_pol = _impute_joint_local(X_val, coefs, noises, slopes, y=None, y_hat=y_hat_val, use_true_y=False, is_train=False)
        m2 = fit_model_fn(X_fit, y_train)
        return holdout_rmse(m2, X_pol, y_val_arr), float(feasibility_fn(m2, X_pol)), m2, y_hat_train, y_hat_val

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        coefs: dict[str, np.ndarray] = {}
        noises: dict[str, float] = {}
        slopes: dict[str, float] = {}
        for s in specs:
            col = str(s["col"])
            coefs[col] = np.array(
                [trial.suggest_float(f"{col}__w_{n}", coef_lo, coef_hi) for n in s["feat_names"]],
                dtype=np.float64,
            )
            noises[col] = trial.suggest_float(f"{col}__noise_scale", noise_lo, noise_hi, log=True)
            slopes[col] = 1.0
        rmse, feas, _, _, _ = _eval(coefs, noises, slopes)
        trial.set_user_attr("val_rmse", rmse)
        trial.set_user_attr("feasibility", feas)
        return rmse, feas

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=int(random_state), crossover_prob=0.9),
    )
    study.optimize(
        objective, n_trials=int(n_trials),
        n_jobs=max(1, min(os.cpu_count() or 1, 4)), show_progress_bar=show_progress_bar,
    )
    t = _nsga_scalarized_trial(study, selection_rmse_weight)
    coefs = {
        str(s["col"]): np.array([t.params[f"{s['col']}__w_{n}"] for n in s["feat_names"]], dtype=np.float64)
        for s in specs
    }
    noises = {str(s["col"]): float(t.params[f"{s['col']}__noise_scale"]) for s in specs}
    slopes = {str(s["col"]): 1.0 for s in specs}
    rmse_sel, feas, m2, _, _ = _eval(coefs, noises, slopes)

    params_by_col = _build_params(coefs, noises, slopes)
    for p in params_by_col.values():
        p.update(
            two_stage=True, optimizer="nsga2_joint_shap",
            selection_rmse_weight=float(selection_rmse_weight),
            search_trial=int(t.number), search_n_pareto=len(study.best_trials),
            joint_latent_cols=list(latent_cols),
        )
    return params_by_col, feas, rmse_sel, int(t.number), m1, y_hat_train, y_hat_val, len(study.best_trials)


def _pipeline_result(
    *,
    cols: list[str],
    obs_cols: list[str],
    target: str,
    model: Any,
    model_a: Any,
    X_eval_before: pd.DataFrame,
    X_eval_after: pd.DataFrame,
    X_val_before: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_policy_eval: pd.DataFrame,
    results_dir: str,
    do_snapshots: list,
    rmse_metrics: dict,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "cols": cols,
        "cols_before": obs_cols,
        "target": target,
        "model_before": model,
        "model_after": model_a,
        "X_eval_before": X_eval_before,
        "X_eval_after": X_eval_after,
        "X_val_before": X_val_before,
        "X_val": X_val,
        "y_val": y_val,
        "X_policy_eval": X_policy_eval,
        "results_dir": results_dir,
        "do_curve_snapshots": do_snapshots,
        "rmse_metrics": rmse_metrics,
        **extra,
    }


def _save_feasibility_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


# Tokens from _latent_feat_names — same for every dataset.
_LATENT_IMPUTER_LABELS: dict[str, str] = {
    "intercept": "bias",
    "y": "true target",
    "y_hat": "predicted target",
    "y*y_hat": "target × prediction",
    "y^2": "true target²",
    "y_hat^2": "predicted target²",
    "y^3": "true target³",
    "y_hat^3": "predicted target³",
}
_FEATURE_PREFIX_LABELS: tuple[tuple[str, str], ...] = (
    ("shap:", "SHAP "),
    ("shap_main:", "SHAP "),
    ("shap_sq:", "SHAP² "),
    ("shap_inter:", "SHAP "),
    ("shap_x_y:", "SHAP × true target "),
    ("shap_x_yh:", "SHAP × pred target "),
    ("x:", ""),
    ("x2:", ""),
    ("x*x:", ""),
)


def _confounder_feature_label(name: str, *, target_name: str | None = None) -> str:
    if name in _LATENT_IMPUTER_LABELS:
        if target_name is None:
            return _LATENT_IMPUTER_LABELS[name]
        t = str(target_name)
        if name == "y":
            return f"true {t}"
        if name == "y_hat":
            return f"predicted {t}"
        if name == "y*y_hat":
            return f"{t} × prediction"
        if name.startswith("y^"):
            return f"true {t}^{name[2:]}"
        if name.startswith("y_hat^"):
            return f"predicted {t}^{name[6:]}"
        return _LATENT_IMPUTER_LABELS[name]
    for prefix, head in _FEATURE_PREFIX_LABELS:
        if name.startswith(prefix):
            body = name[len(prefix) :].replace("*", " × ").replace("_", " ")
            return f"{head}{body}".strip()
    return name.replace("_", " ")


def _top_confounder_weights(
    params: dict[str, Any], k: int = 6, *, target_name: str | None = None,
) -> list[tuple[str, float]]:
    names = params["feature_names"]
    coef = np.asarray(params["coef"], dtype=np.float64)
    order = np.argsort(-np.abs(coef))
    out: list[tuple[str, float]] = []
    for i in order:
        out.append((_confounder_feature_label(names[i], target_name=target_name), float(coef[i])))
    return out


def _impute_joint_params(
    X: np.ndarray,
    cols: list[str],
    active: list[str],
    params_by_col: dict[str, dict[str, Any]],
    *,
    y: np.ndarray | None,
    y_hat: np.ndarray,
    use_true_y: bool,
    m1: Any,
) -> np.ndarray:
    """Impute all latent columns using pre-calculated params. (Generates SHAP interactions internally)."""
    out = np.asarray(X, dtype=np.float64).copy()
    if not active:
        return out

    explainer = shap.TreeExplainer(m1)
    shap_interactions = explainer.shap_interaction_values(X)

    for col in active:
        p = params_by_col[col]
        cfg = LatentImputerConfig.from_dict(p.get("imputer_config"))
        latent_idx = cols.index(col)
        predictor_idx = [cols.index(c) for c in p["predictor_cols"]]
        n_main = int(p.get("n_main", len(predictor_idx)))
        shap_feat = _extract_shap_features(shap_interactions, predictor_idx)

        imputer_design = _latent_imputer_design(
            shap_feat, y, y_hat, use_true_y=use_true_y,
            shap_feat_mean=np.asarray(p["shap_feat_mean"], dtype=np.float64),
            shap_feat_std=np.asarray(p["shap_feat_std"], dtype=np.float64),
            y_mean=float(p["y_mean"]), y_std=float(p["y_std"]),
            yh_mean=float(p["yh_mean"]), yh_std=float(p["yh_std"]),
            n_main=n_main, cfg=cfg,
        )
        out = _impute_latent_column(
            out, latent_idx, imputer_design, np.asarray(p["coef"], dtype=np.float64),
            float(p["lo"]), float(p["hi"]),
            slope=float(p.get("slope", 1.0)),
        )
    return out


class iSHAP:
    @staticmethod
    def run_pipeline(
        X: pd.DataFrame,
        y: pd.Series,
        rules: list[dict[str, Any]],
        target: str,
        root: str,
        name: str,
        *,
        random_state: int = 42,
        n_trials: int = 1000,
        selection_rmse_weight: float = 0.5,
        test_size: float = 0.25,
        feasibility_grid_size: int = 31,
        do_curve_max_samples: int = 200,
        n_hidden_confounders: int = 1,
        do_curve_figure_path: str | None = None,
        latent_confounder_figure_path: str | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        show_progress_bar: bool = True,
    ) -> dict[str, Any]:
        cfg = LatentImputerConfig()
        cols = list(X.columns)
        t = str(target)
        latent_candidates = [c for c in cols if X[c].isna().all()]
        assert latent_candidates, "no all-NaN latent confounder columns in X"
        n_z = max(1, min(int(n_hidden_confounders), len(latent_candidates)))
        active_hidden = latent_candidates[:n_z]
        inactive_hidden = set(latent_candidates[n_z:])
        if inactive_hidden:
            cols = [c for c in cols if c not in inactive_hidden]
        hidden_bounds = {c: (0.0, 1.0) for c in active_hidden}
        obs_cols = [c for c in cols if c not in hidden_bounds]
        obs_idx = np.array([cols.index(c) for c in obs_cols], dtype=np.intp)
        X_obs = lambda arr: arr if not hidden_bounds else arr[:, obs_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X[cols].to_numpy(np.float64), y.to_numpy(np.float64),
            test_size=test_size, random_state=random_state,
        )
        X_test, y_test = (
            (X_test[cols].to_numpy(np.float64), y_test.to_numpy(np.float64))
            if X_test is not None and y_test is not None else (X_val, y_val)
        )

        rng = np.random.default_rng(random_state)
        X_val_policy = X_val
        rmse_metrics: dict[str, dict[str, float]] = {"val": {}, "test": {}}
        # Subsampled version used inside the optimisation loop (speed).
        _feas_kw_fast = dict(random_state=random_state, max_samples=do_curve_max_samples, grid_size=feasibility_grid_size)
        _feas_kw_full = dict(random_state=random_state, max_samples=None, grid_size=feasibility_grid_size)

        def _feas_report(
            m: Any, X_policy: np.ndarray | None = None, *, observed_only: bool = False, full: bool = False,
        ) -> dict[str, Any]:
            Xp = X_val if X_policy is None else X_policy
            fc = obs_cols if observed_only else cols
            lat = () if observed_only else tuple(active_hidden)
            kw = _feas_kw_full if full else _feas_kw_fast
            return compute_feasibility(m, X_obs(Xp) if observed_only else Xp, fc, rules, t, latent_columns=lat, **kw)

        def _feas(m: Any, X_policy: np.ndarray | None = None, *, observed_only: bool = False, full: bool = False) -> float:
            return float(_feas_report(m, X_policy, observed_only=observed_only, full=full)["feasibility_score_0_1"])

        model = make_model(random_state=random_state)
        model.fit(X_obs(X_train), y_train)
        feas_report0 = _feas_report(model, observed_only=True, full=True)
        feas0 = float(feas_report0["feasibility_score_0_1"])
        rmse_metrics["val"]["before"] = holdout_rmse(model, X_obs(X_val), y_val)
        rmse_metrics["test"]["before"] = holdout_rmse(model, X_obs(X_test), y_test)

        do_snapshots: list[tuple[str, Any]] = [("Before", model)]
        results_dir = os.path.join(root, "results", name)

        def _save_do_curve() -> None:
            stem = Path(do_curve_figure_path).stem
            csv_p = Path(results_dir) / f"{stem}.csv"
            save_do_curve_progress_figure(
                cols=cols, cols_before=obs_cols, target_name=t, rules=rules,
                round_snapshots=do_snapshots,
                X_eval_df=pd.DataFrame(X_val_policy, columns=cols),
                X_eval_before_df=pd.DataFrame(X_obs(X_val), columns=obs_cols),
                out_path=do_curve_figure_path, out_data_path=str(csv_p),
                latent_columns=tuple(active_hidden),
                grid_size=feasibility_grid_size,
            )

        def _fit_fast(Xf: np.ndarray, yf: np.ndarray) -> Any:
            m = make_model(random_state=random_state, n_jobs=1)
            m.fit(Xf, yf)
            return m

        params_by_col, feas_s, rmse_s, trial_no, m_stage1, y_hat_train, y_hat_val, n_pareto = (
            optimize_joint_latents(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, cols=cols,
                latent_cols=active_hidden, hidden_bounds=hidden_bounds,
                n_trials=n_trials, random_state=random_state,
                selection_rmse_weight=selection_rmse_weight,
                fit_model_fn=_fit_fast, feasibility_fn=_feas,
                show_progress_bar=show_progress_bar,
            )
        )
        X_train_imp = _impute_joint_params(X_train, cols, active_hidden, params_by_col, y=y_train, y_hat=y_hat_train, use_true_y=True, m1=m_stage1)
        X_val_imp = _impute_joint_params(X_val, cols, active_hidden, params_by_col, y=None, y_hat=y_hat_val, use_true_y=False, m1=m_stage1)
        if latent_confounder_figure_path:
            save_joint_latent_confounders_figure(
                target_name=t,
                latent_names=list(active_hidden),
                cols=cols,
                params_by_col=params_by_col,
                X_train_imp=X_train_imp,
                X_val_imp=X_val_imp,
                y_train=y_train,
                y_val=y_val,
                y_hat_train=y_hat_train,
                y_hat_val=y_hat_val,
                out_path=latent_confounder_figure_path,
            )
        X_val_policy = X_val_imp
        model_a = make_model(random_state=random_state)
        model_a.fit(X_train_imp, y_train)
        feas_report1 = _feas_report(model_a, X_val_imp, full=True)
        feas1 = float(feas_report1["feasibility_score_0_1"])
        rmse_metrics["val"]["after"] = holdout_rmse(model_a, X_val_imp, y_val)
        y_hat_test = np.asarray(m_stage1.predict(X_test), dtype=np.float64).ravel()
        X_test_imp = _impute_joint_params(X_test, cols, active_hidden, params_by_col, y=None, y_hat=y_hat_test, use_true_y=False, m1=m_stage1)
        rmse_metrics["test"]["after"] = holdout_rmse(model_a, X_test_imp, y_test)
        do_snapshots.append(("After", model_a))
        if do_curve_figure_path:
            _save_do_curve()
        os.makedirs(results_dir, exist_ok=True)
        _save_feasibility_json(os.path.join(results_dir, "latent_confounder_models.json"), params_by_col)
        _save_feasibility_json(os.path.join(results_dir, "feasibility.json"), {
            "feasibility": {"before": round(feas0, 4), "after": round(feas1, 4)},
            "rmse": {k: {kk: round(vv, 4) for kk, vv in d.items()} for k, d in rmse_metrics.items()},
            "optimizer": "nsga2_joint",
            "n_hidden_confounders": n_z,
            "selection_rmse_weight": float(selection_rmse_weight),
            "active_latent_confounders": list(active_hidden),
            "imputer_config": cfg.to_dict(),
            "search_objectives": {"val_rmse": round(rmse_s, 4), "feasibility": round(feas_s, 4), "trial": int(trial_no)},
            "top_weights": {
                c: [{"feature": n, "weight": w} for n, w in _top_confounder_weights(params_by_col[c], k=8, target_name=t)]
                for c in active_hidden
            },
        })
        Xall = np.vstack([X_train_imp, X_val_imp])
        X_bg = _sample_rows(Xall, min(do_curve_max_samples, len(Xall)), rng)
        return _pipeline_result(
            cols=cols, obs_cols=obs_cols, target=t, model=model, model_a=model_a,
            X_eval_before=pd.DataFrame(np.vstack([X_obs(X_train), X_obs(X_val)]), columns=obs_cols),
            X_eval_after=pd.DataFrame(np.vstack([X_train_imp, X_val_imp]), columns=cols),
            X_val_before=pd.DataFrame(X_obs(X_val), columns=obs_cols),
            X_val=pd.DataFrame(X_val_imp, columns=cols),
            y_val=y_val, X_policy_eval=pd.DataFrame(X_bg, columns=cols),
            results_dir=results_dir, do_snapshots=do_snapshots, rmse_metrics=rmse_metrics,
            feasibility_report_before=feas_report0,
            feasibility_report_after=feas_report1,
            latent_confounder_params=params_by_col,
            active_latent_confounders=active_hidden,
        )

# Imported last: visuals pulls from lib; early import would cycle.
from visuals import save_do_curve_progress_figure, save_joint_latent_confounders_figure

