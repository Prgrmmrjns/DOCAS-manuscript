from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Mapping, Sequence, TypedDict

import shap
import numpy as np
import pandas as pd


class SCMRule(TypedDict):
    """start → end; edge in [-1,1]. ``end == target``: Pearson(SHAP(start), x_start). ``end`` a feature: Pearson(x_start, SHAP(end)) — e.g.\ negative edge ⇒ higher ``start`` should associate with lower SHAP on ``end``."""

    start: str
    end: str
    edge: float


def project_root(relative_to_file: str) -> str:
    """Repository root when `relative_to_file` lives under `<root>/scripts/`."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(relative_to_file)), ".."))


def scm_structural_parents(rules: Sequence[Mapping[str, Any]], *, target: str) -> dict[str, tuple[str, ...]]:
    """Child → parents from feature→feature rules only (end != target)."""
    t = str(target)
    g: dict[str, list[str]] = defaultdict(list)
    for r in rules:
        if str(r["end"]) == t:
            continue
        g[str(r["end"])].append(str(r["start"]))
    return {k: tuple(v) for k, v in g.items()}


def scm_topological_order(
    all_features: tuple[str, ...],
    rules: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> tuple[str, ...]:
    """Walk order: Kahn on structural edges; tie-break by position in all_features."""
    t = str(target)
    edges = [(str(r["start"]), str(r["end"])) for r in rules if str(r["end"]) != t]
    fs = tuple(all_features)
    idx = {f: i for i, f in enumerate(fs)}
    pred: dict[str, set[str]] = {f: set() for f in fs}
    for a, b in edges:
        if a in idx and b in idx:
            pred[b].add(a)
    done: set[str] = set()
    out: list[str] = []
    while len(done) < len(fs):
        ready = [n for n in fs if n not in done and pred[n] <= done]
        if not ready:
            out.extend(sorted((n for n in fs if n not in done), key=lambda x: idx[x]))
            break
        ready.sort(key=lambda x: idx[x])
        for n in ready:
            done.add(n)
            out.append(n)
    return tuple(out)


def scm_parent_effect_sign(rules: Sequence[Mapping[str, Any]], *, target: str) -> dict[str, dict[str, int]]:
    """Integer sign per structural edge for Graphviz (from rule edge)."""
    t = str(target)
    out: dict[str, dict[str, int]] = {}
    for r in rules:
        if str(r["end"]) == t:
            continue
        ed = float(r["edge"])
        sg = 0 if abs(ed) < 1e-12 else (1 if ed > 0 else -1)
        out.setdefault(str(r["end"]), {})[str(r["start"])] = sg
    return out


def val_subsample_for_shap_plot(X_val_df: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    """Same val subsample as SHAP beeswarm export in `visuals` / Pearson objective."""
    rng = np.random.default_rng(int(random_state))
    n = min(int(max_samples), len(X_val_df))
    idx = rng.choice(len(X_val_df), size=n, replace=False) if len(X_val_df) > n else np.arange(len(X_val_df))
    return X_val_df.iloc[idx].copy().reset_index(drop=True)


def tree_shap_df_sv_ev(model: Any, X_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """TreeSHAP on DataFrame: SHAP matrix (n, F) and scalar E[f(x)] (regression)."""
    ex = shap.TreeExplainer(model)
    try:
        sv = ex.shap_values(X_df, check_additivity=False)
    except TypeError:
        sv = ex.shap_values(X_df)
    if isinstance(sv, list):
        sv = np.asarray(sv[0])
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 1:
        sv = sv.reshape(len(X_df), -1)
    ev_arr = np.asarray(ex.expected_value).ravel()
    ev0 = float(ev_arr[0]) if ev_arr.size else 0.0
    return sv, ev0


def tree_shap_values_from_df(model: Any, X_df: pd.DataFrame) -> np.ndarray:
    sv, _ = tree_shap_df_sv_ev(model, X_df)
    return sv


def tree_shap_mean_abs_weights(
    model: Any,
    X_df: pd.DataFrame,
    *,
    max_samples: int = 512,
    random_state: int = 0,
    clip_min: float = 0.28,
    clip_max: float = 3.8,
) -> np.ndarray:
    """Per-feature weights ~ mean |TreeSHAP|, median-normalized (for scaling synthetic perturbations)."""
    n = len(X_df)
    if n < 2:
        return np.ones(X_df.shape[1], dtype=np.float32)
    m = min(int(max_samples), n)
    rng = np.random.default_rng(int(random_state))
    Xs = X_df.iloc[rng.choice(n, size=m, replace=False)].copy() if n > m else X_df
    try:
        sv = np.asarray(tree_shap_values_from_df(model, Xs), dtype=np.float64)
    except Exception:
        return np.ones(X_df.shape[1], dtype=np.float32)
    if sv.ndim != 2 or sv.shape[1] != X_df.shape[1]:
        return np.ones(X_df.shape[1], dtype=np.float32)
    ma = np.mean(np.abs(sv), axis=0)
    med = float(np.median(ma)) + 1e-12
    w = np.clip(ma / med, clip_min, clip_max).astype(np.float32)
    return w


def _pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r via centered dot product (faster than np.corrcoef 2×2)."""
    x = np.asarray(x, dtype=np.float64, order="C").ravel()
    y = np.asarray(y, dtype=np.float64, order="C").ravel()
    n = int(x.size)
    if n < 2 or n != int(y.size):
        return float("nan")
    mx = float(x.mean())
    my = float(y.mean())
    dx = x - mx
    dy = y - my
    denom = float(np.sqrt(float(dx @ dx) * float(dy @ dy)))
    if denom < 1e-18:
        return float("nan")
    return float((dx @ dy) / denom)


def domain_target_pearson_objective(
    model: Any,
    X_val_df: pd.DataFrame,
    cols: list[str],
    scm_rules: Sequence[Mapping[str, Any]],
    target_name: str,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[float, dict[str, Any]]:
    """
    Weighted Pearson alignment on the val subsample:
    - ``end == target``: Pearson(SHAP(main ``start``), x_start).
    - ``end`` is a feature (not target): Pearson(observed x_start, SHAP(``end``)) on the same rows
      (higher ``start`` should align with lower SHAP on ``end`` when ``edge<0``).
    Per-rule score = clip(0.5*(1 + sign(edge)*r), 0, 1); overall = |edge|-weighted mean.
    """
    t = str(target_name)
    col_idx = {c: i for i, c in enumerate(cols)}

    def _feature_interaction_rule(r: Mapping[str, Any]) -> bool:
        v2, u = str(r["end"]), str(r["start"])
        return v2 != t and u in col_idx and v2 in col_idx

    Xs = val_subsample_for_shap_plot(X_val_df, max_samples, random_state)
    try:
        sv = tree_shap_values_from_df(model, Xs)
    except Exception:
        return 0.5, {
            "per_feature_pearson_r": {},
            "per_feature_score_0_1": {},
            "per_interaction_pearson_r": {},
            "per_interaction_score_0_1": {},
        }
    per_r: dict[str, float] = {}
    per_sc: dict[str, float] = {}
    per_ir: dict[str, float] = {}
    per_isc: dict[str, float] = {}
    num, den = 0.0, 0.0
    for r in scm_rules:
        edge = float(r["edge"])
        w = abs(edge)
        if w < 1e-12:
            continue
        sgn = 1.0 if edge > 1e-12 else (-1.0 if edge < -1e-12 else 1.0)
        if _feature_interaction_rule(r):
            u, v2 = str(r["start"]), str(r["end"])
            iv = col_idx[v2]
            phi_end = sv[:, iv].astype(np.float64, copy=False)
            xs = Xs[u].to_numpy(dtype=np.float64, copy=False)
            if float(np.std(xs)) < 1e-12 or float(np.std(phi_end)) < 1e-12:
                score = 0.5
                rr = float("nan")
            else:
                rr = _pearson_r_xy(xs, phi_end)
                if not np.isfinite(rr):
                    score = 0.5
                else:
                    score = float(np.clip(0.5 * (1.0 + sgn * rr), 0.0, 1.0))
            key = f"{u}×{v2}"
            per_ir[key] = float(rr) if np.isfinite(rr) else float("nan")
            per_isc[key] = score
            num += w * score
            den += w
            continue
        if str(r["end"]) != t:
            continue
        u = str(r["start"])
        if u not in col_idx:
            continue
        ki = col_idx[u]
        phi_u = sv[:, ki].astype(np.float64, copy=False)
        x_u = Xs[u].to_numpy(dtype=np.float64, copy=False)
        if float(np.std(phi_u)) < 1e-12 or float(np.std(x_u)) < 1e-12:
            score = 0.5
            rr = float("nan")
        else:
            rr = _pearson_r_xy(phi_u, x_u)
            if not np.isfinite(rr):
                score = 0.5
            else:
                score = float(np.clip(0.5 * (1.0 + sgn * rr), 0.0, 1.0))
        per_r[u] = float(rr) if np.isfinite(rr) else float("nan")
        per_sc[u] = score
        num += w * score
        den += w
    overall = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.5
    w_main = {
        str(r["start"]): abs(float(r["edge"]))
        for r in scm_rules
        if str(r["end"]) == t and str(r["start"]) in col_idx
    }
    w_ix = {
        f'{str(r["start"])}×{str(r["end"])}': abs(float(r["edge"]))
        for r in scm_rules
        if _feature_interaction_rule(r)
    }
    diag: dict[str, Any] = {
        "per_feature_pearson_r": per_r,
        "per_feature_score_0_1": per_sc,
        "per_interaction_pearson_r": per_ir,
        "per_interaction_score_0_1": per_isc,
        "weights_abs_edge": {**w_main, **w_ix},
    }
    return overall, diag


def _mass_and_count_aligned(vals: np.ndarray, sgn: float) -> tuple[float, float]:
    """(mass_aligned_frac, count_aligned_frac) in [0,1], each 0.5 if undefined."""
    v = np.asarray(vals, dtype=float).ravel()
    a = np.abs(v)
    n = int(v.size)
    tot = float(np.sum(a))
    sgn = 1.0 if sgn > 0 else -1.0
    if not np.isfinite(tot) or tot <= 1e-18 or n < 1:
        return 0.5, 0.5
    mass = float(np.sum(a[(sgn * v) > 0.0])) / tot
    med = float(np.median(a))
    tol = max(1e-18, 1e-6 * (med + 1e-18))
    mask = a > tol
    if int(mask.sum()) > 0:
        count = float(np.mean((sgn * v[mask]) > 0.0))
    else:
        count = 0.5
    return float(np.clip(mass, 0.0, 1.0)), float(np.clip(count, 0.0, 1.0))


def _rule_score(vals: np.ndarray, sgn: float) -> float:
    """Bee-swarm consistency: geometric mean of mass-aligned and row-sign agreement (both essential)."""
    mass, count = _mass_and_count_aligned(vals, sgn)
    return float(np.sqrt(max(1e-12, mass) * max(1e-12, count)))


def _rule_diag(vals: np.ndarray, sgn: float) -> dict[str, float]:
    v = np.asarray(vals, dtype=float).ravel()
    a = np.abs(v)
    n = int(v.size)
    med = float(np.median(a)) if n else 0.0
    tol = max(1e-18, 1e-6 * (med + 1e-18))
    mass_aligned, count_aligned = _mass_and_count_aligned(vals, sgn)
    score = _rule_score(vals, sgn)
    return {
        "n": n,
        "n_nonzero": int(np.count_nonzero(a > tol)),
        "mean": float(np.mean(v)) if n else 0.0,
        "mean_abs": float(np.mean(a)) if n else 0.0,
        "mass_aligned_frac": mass_aligned,
        "count_aligned_frac": count_aligned,
        "score_geom_mass_count": score,
    }


def _iter_rule_scores(
    pc: dict[str, Any] | None,
    iv: Any,
    cols: list[str],
    rules: list[dict[str, Any]],
    target_name: str,
    *,
    phi: np.ndarray | None = None,
):
    """Yield (rule, sgn, weight, score, source, diag) for each usable rule."""
    nf = (iv.shape[1] - 1) if iv is not None and getattr(iv, "ndim", 0) == 3 else 0
    for r in rules:
        u, v = str(r["start"]), str(r["end"])
        e = float(max(-1.0, min(1.0, float(r["edge"]))))
        w = abs(e)
        if w < 1e-12:
            continue
        sgn = 1.0 if e > 0 else -1.0
        source = "none"
        diag: dict[str, float] | None = None
        score = 0.5
        if v == target_name:
            if u not in cols:
                continue
            if phi is not None and getattr(phi, "ndim", 0) == 2 and phi.shape[1] == len(cols):
                vals = phi[:, cols.index(u)]
                score = _rule_score(vals, sgn)
                diag = _rule_diag(vals, sgn)
                source = "phi"
            else:
                mac = (pc or {}).get("mean_abs_contrib") or {}
                mc = (pc or {}).get("mean_contrib") or {}
                wu = float(mac.get(u, 0.0))
                if wu <= 1e-12:
                    score = 0.5
                else:
                    rr = max(-1.0, min(1.0, float(mc.get(u, 0.0)) / wu))
                    score = float(np.clip(0.5 + 0.5 * sgn * rr, 0.0, 1.0))
                diag = {
                    "n": 0,
                    "n_nonzero": 0,
                    "mean": float(mc.get(u, 0.0)),
                    "mean_abs": wu,
                    "mass_aligned_frac": score,
                    "count_aligned_frac": float("nan"),
                }
                source = "pc"
        # Feature–feature interaction rules (SHAP interaction tensor) disabled.
        # else:
        #     if u not in cols or v not in cols:
        #         continue
        #     ia, ib = cols.index(u), cols.index(v)
        #     if iv is None or ia >= nf or ib >= nf:
        #         continue
        #     vals = iv[:, ia, ib]
        #     score = _rule_score(vals, sgn)
        #     diag = _rule_diag(vals, sgn)
        #     source = "iv"
        else:
            continue
        yield r, sgn, w, score, source, diag


def feasibility_unified(
    pc: dict[str, Any] | None,
    iv: Any,
    cols: list[str],
    rules: list[dict[str, Any]],
    target_name: str,
    *,
    phi: np.ndarray | None = None,
) -> float:
    """Weighted mean of per-rule mass-aligned consistency scores (higher = bee-swarm more on the right side)."""
    num, den = 0.0, 0.0
    for _r, _sgn, w, score, _src, _diag in _iter_rule_scores(pc, iv, cols, rules, target_name, phi=phi):
        num += w * score
        den += w
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.5


def feasibility_report(
    pc: dict[str, Any] | None,
    iv: Any,
    cols: list[str],
    rules: list[dict[str, Any]],
    target_name: str,
    *,
    phi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-rule SHAP diagnostics (score, sign, alignment stats) + overall weighted score."""
    items: list[dict[str, Any]] = []
    num, den = 0.0, 0.0
    for r, sgn, w, score, source, diag in _iter_rule_scores(pc, iv, cols, rules, target_name, phi=phi):
        items.append({
            "start": str(r["start"]),
            "end": str(r["end"]),
            "edge": float(r["edge"]),
            "sign": int(sgn),
            "weight": float(w),
            "score": float(score),
            "source": source,
            "diag": diag or {},
        })
        num += w * score
        den += w
    overall = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.5
    return {"overall": overall, "rules": items}
