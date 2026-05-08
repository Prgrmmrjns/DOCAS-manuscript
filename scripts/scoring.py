from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombinedScorer:
    metric_weight: float
    metric_anchor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_weight", max(0.0, min(1.0, float(self.metric_weight))))
        object.__setattr__(self, "metric_anchor", max(float(self.metric_anchor), 1e-12))

    def score(self, metric_loss: float, feasibility: float) -> float:
        norm_loss = float(metric_loss) / self.metric_anchor
        return (1.0 - self.metric_weight) * (1.0 - norm_loss) + self.metric_weight * float(feasibility)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or x.size != y.size:
        return float("nan")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def _residualize(target_vec: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS residuals of target_vec ~ [1, Z]. Returns mean-centered vector if Z empty."""
    y = np.asarray(target_vec, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.size == 0 or Z.shape[1] == 0:
        return y - float(np.mean(y))
    Z1 = np.column_stack([np.ones(len(Z), dtype=np.float64), Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta


def _do_effect_to_target(
    model: Any,
    Xs_np: np.ndarray,
    feature_idx: int,
    *,
    grid_size: int,
) -> float:
    """Slope of E[Y | do(X_u = g)] across a grid of u values, expressed as Pearson(grid, mean_pred).

    For a fixed evaluation set Xs, intervene on column `feature_idx` by overwriting
    every row with grid value g and averaging model predictions. The Pearson
    correlation between grid values and the resulting mean predictions captures
    the isolated do-effect of u on the target induced by the model.
    """
    u_col = Xs_np[:, feature_idx]
    u_min = float(np.min(u_col))
    u_max = float(np.max(u_col))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min + 1e-12:
        return float("nan")
    grid = np.linspace(u_min, u_max, int(max(2, grid_size)), dtype=np.float64)
    avg_preds = np.empty(grid.size, dtype=np.float64)
    for k, g in enumerate(grid):
        Xint = Xs_np.copy()
        Xint[:, feature_idx] = g
        preds = np.asarray(model.predict(Xint), dtype=np.float64)
        avg_preds[k] = float(np.mean(preds)) if preds.size else float("nan")
    return _safe_pearson(grid, avg_preds)


def _monotonic_signed_score(y: np.ndarray) -> float:
    """Monotonicity score in [-1, 1] based on magnitude of differences.

    Computes (net change) / (total variation).
    +1 means perfectly monotonically increasing, -1 means perfectly decreasing.
    """
    yy = np.asarray(y, dtype=np.float64).ravel()
    if yy.size < 2:
        return float("nan")
    
    diffs = np.diff(yy)
    net_change = np.sum(diffs)
    total_var = np.sum(np.abs(diffs))
    
    if total_var < 1e-12:
        return 0.0
        
    return float(net_change / total_var)


def _do_curve_to_target(
    model: Any,
    Xs_np: np.ndarray,
    feature_idx: int,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    u_col = Xs_np[:, feature_idx]
    u_min = float(np.min(u_col))
    u_max = float(np.max(u_col))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min + 1e-12:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    grid = np.linspace(u_min, u_max, int(max(2, grid_size)), dtype=np.float64)
    avg_preds = np.empty(grid.size, dtype=np.float64)
    for k, g in enumerate(grid):
        Xint = Xs_np.copy()
        Xint[:, feature_idx] = g
        preds = np.asarray(model.predict(Xint), dtype=np.float64)
        avg_preds[k] = float(np.mean(preds)) if preds.size else float("nan")
    return grid, avg_preds


def _partial_correlation_feature_pair(
    Xs_np: np.ndarray,
    iu: int,
    iv: int,
) -> float:
    """Partial correlation between columns iu and iv, controlling for all other columns.

    This is the linear-Gaussian analog of a conditional independence test under
    a backdoor adjustment that includes every other observed feature. It removes
    the influence of common observed confounders.
    """
    n_cols = Xs_np.shape[1]
    other_idx = [j for j in range(n_cols) if j != iu and j != iv]
    if not other_idx:
        return _safe_pearson(Xs_np[:, iu], Xs_np[:, iv])
    Z = Xs_np[:, other_idx]
    ru = _residualize(Xs_np[:, iu], Z)
    rv = _residualize(Xs_np[:, iv], Z)
    return _safe_pearson(ru, rv)


def compute_feasibility(
    model: Any,
    X_df: pd.DataFrame,
    cols: list[str],
    rules: list[dict[str, Any]],
    target: str,
    *,
    random_state: int = 42,
    max_samples: int = 200,
    grid_size: int = 11,
) -> dict[str, Any]:
    """Feasibility score using do-calculus / isolated effects.

    For each rule (u -> v, edge):
      - If v is the target, the isolated effect is the slope of
        E[Y | do(X_u = g)] across a grid of u values, captured as a Pearson
        correlation between grid values and average model predictions.
      - If v is another feature, the isolated effect is the partial correlation
        between u and v after controlling for all other observed features
        (a conservative backdoor adjustment).

    The per-rule deviation from the ideal sign is aggregated into a [0, 1]
    feasibility score, identical in shape to the previous SHAP-correlation score.
    """
    t = str(target)
    col_idx = {c: i for i, c in enumerate(cols)}
    n_total = len(X_df)
    n = min(int(max_samples), n_total)
    if n_total > n:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(n_total, size=n, replace=False)
        Xs = X_df.iloc[idx].copy().reset_index(drop=True)
    else:
        Xs = X_df.reset_index(drop=True)
    Xs_np = Xs[cols].to_numpy(dtype=np.float64, copy=False)

    items: list[dict[str, Any]] = []
    w_sum = 0.0
    dev_sum = 0.0
    for r in rules:
        edge = float(r["edge"])
        u, v = str(r["start"]), str(r["end"])
        edge_mag = abs(edge)
        w = (edge_mag if edge_mag > 1e-12 else 1.0) * (0.55 if v != t else 1.0)
        ideal = 0.0 if edge_mag <= 1e-12 else (1.0 if edge > 0 else -1.0)

        if v == t:
            if u not in col_idx:
                continue
            grid, avg = _do_curve_to_target(model, Xs_np, col_idx[u], grid_size=grid_size)
            rho = _safe_pearson(grid, avg)
            mono = _monotonic_signed_score(avg)
            
            if np.isfinite(mono) and np.isfinite(rho):
                # Combine magnitude-based monotonicity and Pearson correlation
                # to enforce a "monotonic linear" relationship.
                eff = float((mono + rho) / 2.0)
            else:
                eff = float("nan")
            key = f"{u}->{t}"
        else:
            if u not in col_idx or v not in col_idx:
                continue
            rho = _partial_correlation_feature_pair(Xs_np, col_idx[u], col_idx[v])
            mono = None
            eff = rho
            key = f"{u}->{v}"

        if not np.isfinite(eff):
            dev, ro, diff = 2.0, None, None
        else:
            ro = float(eff)
            dev = abs(ro - ideal)
            diff = round(ro - ideal, 4)
        w_sum += w
        dev_sum += w * dev
        items.append(
            {
                "rule": key,
                "weight": round(w, 4),
                "pearson_r": None if ro is None else round(ro, 4),
                "do_effect_corr": None if v != t or not np.isfinite(rho) else round(float(rho), 4),
                "monotonic_signed_score": None if mono is None or not np.isfinite(mono) else round(float(mono), 4),
                "ideal_pearson": ideal,
                "pearson_r_diff": diff,
            }
        )

    feasibility = 0.5 if w_sum < 1e-18 else float(np.clip(1.0 - dev_sum / w_sum / 2.0, 0.0, 1.0))
    return {"feasibility_score_0_1": round(feasibility, 4), "rules": items}
