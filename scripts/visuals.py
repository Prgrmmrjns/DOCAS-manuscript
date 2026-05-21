from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import shap

from lib import (
    _confounder_feature_label,
    monotonic_aligned_fraction,
    monotonic_step_violations,
    plotted_scm_rules,
    rule_do_curve,
)

FEATURE_RING_RADIUS = 1.35
_RELATIONSHIP_STYLE: dict[str, tuple[str, str]] = {
    "SigmoidRelationship": ("#c0392b", "sig"),
    "MonotonicRelationship": ("#e67e22", "mono"),
    "LinearRelationship": ("#2980b9", "lin"),
    "NullRelationship": ("#7f8c8d", "null"),
    "OscillatingRelationship": ("#8e44ad", "osc"),
}
def _pearson(x: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n or x.size != y.size:
        return float("nan")
    x1, y1 = x[m], y[m]
    if np.std(x1) <= 1e-12 or np.std(y1) <= 1e-12:
        return float("nan")
    r = np.corrcoef(x1, y1)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def centered_radial_positions(feature_order: tuple[str, ...], target_name: str) -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {target_name: (0.0, 0.0)}
    n = len(feature_order)
    for k, name in enumerate(feature_order):
        ang = 2.0 * np.pi * k / max(n, 1) - np.pi / 2.0
        pos[name] = (float(FEATURE_RING_RADIUS * np.cos(ang)), float(FEATURE_RING_RADIUS * np.sin(ang)))
    return pos


def scm_layered_positions(
    *,
    target_name: str,
    core_cols: tuple[str, ...],
    observed_confounder_cols: tuple[str, ...],
    latent_confounder_cols: tuple[str, ...],
    min_node_separation: float = 2.6,
) -> dict[str, tuple[float, float]]:
    ring = tuple(core_cols) + tuple(observed_confounder_cols) + tuple(latent_confounder_cols)
    pos: dict[str, tuple[float, float]] = {target_name: (0.0, 0.0)}
    n = len(ring)
    radius = max(FEATURE_RING_RADIUS, n * min_node_separation / (2.0 * np.pi))
    for k, name in enumerate(ring):
        ang = 2.0 * np.pi * k / max(n, 1) - np.pi / 2.0
        pos[name] = (float(radius * np.cos(ang)), float(radius * np.sin(ang)))
    return pos


def _scm_short_label(name: str) -> str:
    aliases = {
        "glucose_context_m30": "glucose\ncontext_m30", "glucose_roc_30m": "glucose\nroc_30m",
        "glucose_roc_5m": "glucose\nroc_5m", "glucose_accel": "glucose\naccel",
        "glucose_std_1h": "glucose\nstd_1h",
        "pa_steps": "pa\nsteps", "pa_hr": "pa\nhr", "pa_accel": "pa\naccel", "pa_exercise": "pa\nexercise",
        "future glucose": "future\nglucose",
    }
    return aliases.get(name, name.replace("_", "\n") if len(name) > 12 else name)


def _norm_edge_width(w: float, w_min: float, w_max: float, lo: float = 0.5, hi: float = 6.0) -> float:
    if w_max <= w_min + 1e-18:
        return (lo + hi) / 2
    return float(lo + (w - w_min) / (w_max - w_min) * (hi - lo))


def _tree_shap(model: Any, X_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ex = shap.TreeExplainer(model)
    phi = ex.shap_values(X_np)
    phi = phi[-1] if isinstance(phi, list) else phi
    sv = np.asarray(phi, dtype=np.float64)
    if sv.ndim == 1:
        sv = sv.reshape(X_np.shape[0], -1)
    inter = ex.shap_interaction_values(X_np)
    inter = inter[-1] if isinstance(inter, list) else inter
    iv = np.asarray(inter, dtype=np.float64)
    return sv, iv[np.newaxis, ...] if iv.ndim == 2 else iv


def _joint_feature_product(Xn: np.ndarray, i: int, j: int) -> np.ndarray:
    prod = Xn[:, i].astype(np.float64) * Xn[:, j].astype(np.float64)
    prod[(~np.isfinite(Xn[:, i])) | (~np.isfinite(Xn[:, j]))] = np.nan
    return prod


def _bolt_latent_edges(
    G: nx.Graph, cols: list[str], target_name: str, sv: np.ndarray, iv: np.ndarray,
    Xn: np.ndarray, X_np: np.ndarray, model: Any, latent_names: tuple[str, ...],
    latent_fallback_corr: Mapping[str, float] | None,
) -> None:
    floor = max(0.035, max((float(d["w"]) for _, _, d in G.edges(data=True)), default=0.0) * 0.22) or 0.12
    preds = np.asarray(model.predict(X_np), dtype=np.float64).ravel()
    for lat in latent_names:
        if lat not in cols or lat not in G or G.degree[lat]:
            continue
        li = cols.index(lat)
        r_lt = next((v for v in (_pearson(sv[:, li], Xn[:, li]), _pearson(sv[:, li], preds)) if np.isfinite(v)), float("nan"))
        if np.isfinite(r_lt):
            G.add_edge(lat, target_name, sign=1.0 if r_lt >= 0 else -1.0, w=max(abs(r_lt), floor * 0.5), r=float(r_lt))
            continue
        scm_s = (latent_fallback_corr or {}).get(lat)
        if isinstance(scm_s, (int, float)) and scm_s:
            r_vis = 0.42 * np.sign(float(scm_s))
            G.add_edge(lat, target_name, sign=1.0 if r_vis >= 0 else -1.0, w=floor, r=float(r_vis))
        if G.degree[lat]:
            continue
        best_j, best_mag, best_r = -1, -1.0, float("nan")
        for j in range(len(cols)):
            if j == li:
                continue
            r_ij = _pearson(iv[:, li, j], _joint_feature_product(Xn, li, j))
            if np.isfinite(r_ij) and abs(r_ij) > best_mag:
                best_j, best_mag, best_r = j, abs(float(r_ij)), float(r_ij)
        if best_j >= 0:
            sgn = 1.0 if best_r >= 0 else -1.0
            G.add_edge(lat, cols[best_j], sign=sgn, w=max(best_mag, floor * 0.85), r=best_r)


def _shap_interaction_graph(
    model: Any, X_df: pd.DataFrame, cols: list[str], target_name: str, seed: int,
    max_shap_rows: int = 400, bolt_latents: tuple[str, ...] = (),
    latent_fallback_corr: Mapping[str, float] | None = None,
) -> tuple[nx.Graph, float, float]:
    n = min(max_shap_rows, len(X_df))
    idx = np.random.default_rng(seed).choice(len(X_df), n, replace=False) if len(X_df) > n else np.arange(len(X_df))
    Xs = X_df.iloc[idx]
    X_np = Xs.to_numpy(dtype=np.float64, copy=False)
    sv, iv = _tree_shap(model, X_np)
    Xn = X_np.copy()
    G = nx.Graph()
    for c in cols:
        G.add_node(c)
    G.add_node(target_name)
    weights: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s = _pearson(iv[:, i, j], _joint_feature_product(Xn, i, j))
            if np.isfinite(s):
                G.add_edge(cols[i], cols[j], sign=1.0 if s >= 0 else -1.0, w=abs(s), r=float(s))
                weights.append(abs(s))
        s = _pearson(sv[:, i], Xn[:, i])
        if np.isfinite(s):
            G.add_edge(cols[i], target_name, sign=1.0 if s >= 0 else -1.0, w=abs(s), r=float(s))
            weights.append(abs(s))
    if bolt_latents:
        _bolt_latent_edges(G, cols, target_name, sv, iv, Xn, X_np, model, bolt_latents, latent_fallback_corr)
        weights = [float(d["w"]) for _, _, d in G.edges(data=True)]
    w_min, w_max = (min(weights), max(weights)) if weights else (0.0, 1.0)
    return G, w_min, w_max


def _draw_shap_network_ax(
    ax: Any, G: nx.Graph, cols: list[str], target_name: str, pos: dict[str, tuple[float, float]],
    w_min: float, w_max: float, plot_title: str, show_legend: bool,
) -> None:
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal")
    nodes = list(cols) + [target_name]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=nodes,
        node_color=["#2ecc71" if n != target_name else "#9b59b6" for n in nodes],
        node_size=1100, alpha=0.95, edgecolors="#333",
    )
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in nodes}, font_size=8, font_weight="bold")
    for u, v, data in G.edges(data=True):
        w, sgn, r = float(data["w"]), float(data["sign"]), float(data.get("r", data["sign"] * data["w"]))
        color = "#c0392b" if sgn >= 0 else "#2980b9"
        x1, y1, x2, y2 = *pos[u], *pos[v]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=_norm_edge_width(w, w_min, w_max), alpha=0.85, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{r:+.2f}", fontsize=6, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="#fafafa", ec="none", alpha=0.9), zorder=3)
    ax.set_title(plot_title, fontsize=11)
    ax.axis("off")
    if show_legend:
        ax.legend(handles=[
            Line2D([0], [0], color="#c0392b", lw=4, label="positive correlation"),
            Line2D([0], [0], color="#2980b9", lw=4, label="negative correlation"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="feature"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6", markersize=10, label="target"),
        ], loc="upper left", fontsize=8)
    pad = 0.55
    ax.set_xlim(-FEATURE_RING_RADIUS - pad, FEATURE_RING_RADIUS + pad)
    ax.set_ylim(-FEATURE_RING_RADIUS - pad, FEATURE_RING_RADIUS + pad)


def save_shap_interaction_network_before_after_graph(
    *,
    cols: list[str],
    target_name: str,
    X_df_before: pd.DataFrame,
    model_before: Any,
    X_df_after: pd.DataFrame,
    model_after: Any,
    out_path: str | Path,
    seed: int,
    pos: dict[str, tuple[float, float]],
    cols_before: list[str] | None = None,
    pos_before: dict[str, tuple[float, float]] | None = None,
    bolt_latents_after: tuple[str, ...] = (),
    latent_fallback_corr_after: Mapping[str, float] | None = None,
) -> None:
    out_path = Path(out_path)
    cb = list(cols_before or cols)
    pb = pos_before or pos
    Gb, wb_lo, wb_hi = _shap_interaction_graph(model_before, X_df_before[cb], cb, target_name, seed=seed)
    Ga, wa_lo, wa_hi = _shap_interaction_graph(
        model_after, X_df_after, cols, target_name, seed=seed,
        bolt_latents=bolt_latents_after, latent_fallback_corr=latent_fallback_corr_after,
    )
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    _draw_shap_network_ax(axes[0], Gb, cb, target_name, pb, wb_lo, wb_hi, "Before augmentation", True)
    _draw_shap_network_ax(axes[1], Ga, cols, target_name, pos, wa_lo, wa_hi, "After augmentation", False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _edge_style(kind: str, fn: Any) -> tuple[str, str]:
    color, base = _RELATIONSHIP_STYLE.get(str(kind), ("#34495e", "edge"))
    inc = getattr(fn, "increasing", None)
    sign = getattr(fn, "sign", None)
    if inc is True:
        return color, f"{base}+"
    if inc is False:
        return color, f"{base}-"
    if isinstance(sign, (int, float)):
        return color, f"{base}{'+' if sign >= 0 else '-'}"
    return color, base


def _scm_node_appearance(n: str, target: str, latent: set[str], conf: set[str], dyn: set[str]) -> tuple[str, str, str]:
    if n == target:
        return "#9b59b6", "#333", "solid"
    if n in latent:
        return "#ecf0f1", "#7f8c8d", "dashed"
    if n in conf:
        return "#1abc9c", "#117a65", "solid"
    if n in dyn:
        return "#aed6f1", "#5dade2", "solid"
    return "#3498db", "#333", "solid"


def save_scm_graph_from_rules(
    *,
    rules: list[dict],
    feature_cols: tuple[str, ...],
    target_name: str,
    out_path: str | Path,
    pos: dict[str, tuple[float, float]],
    latent_cols: tuple[str, ...] = (),
    observed_confounder_cols: tuple[str, ...] = (),
    hidden_edges: tuple[tuple[str, str], ...] = (),
    flexible_edges: tuple[tuple[str, str], ...] = (),
    glucose_dynamics_cols: tuple[str, ...] = (),
    title: str | None = None,
    figsize: tuple[float, float] = (13, 11),
    short_labels: bool = False,
) -> Path:
    out_path = Path(out_path)
    latent_set = {str(c) for c in latent_cols}
    conf_set = {str(c) for c in observed_confounder_cols}
    dyn_set = {str(c) for c in glucose_dynamics_cols}
    G = nx.DiGraph()
    for n in feature_cols:
        G.add_node(n)
    G.add_node(target_name)
    for r in rules:
        G.add_edge(str(r["start"]), str(r["end"]))
    for u, v in flexible_edges:
        G.add_edge(str(u), str(v))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    ax.set_aspect("equal")
    nodes = list(feature_cols) + [target_name]
    styles = [_scm_node_appearance(n, target_name, latent_set, conf_set, dyn_set) for n in nodes]
    node_size = float(max(850, min(1400, 5200 / max(len(nodes), 1))))
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=nodes,
        node_color=[s[0] for s in styles], node_size=node_size, alpha=0.95,
        edgecolors=[s[1] for s in styles], linewidths=2.0,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: (_scm_short_label(n) if short_labels else n) for n in nodes},
        font_size=8 if short_labels else 9, font_weight="bold",
    )

    weights = [float(r.get("weight", 1.0)) for r in rules] or [1.0]
    w_lo, w_hi = min(weights), max(weights)
    seen_kinds: set[str] = set()
    tgt = str(target_name)
    for r in rules:
        u, v = str(r["start"]), str(r["end"])
        kind = str(r.get("kind", type(r.get("relationship_fn")).__name__ if r.get("relationship_fn") else ""))
        seen_kinds.add(kind)
        color, tag = _edge_style(kind, r.get("relationship_fn"))
        lw = _norm_edge_width(float(r.get("weight", 1.0)), w_lo, w_hi, lo=1.2, hi=5.5)
        ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, shrinkA=12, shrinkB=12, mutation_scale=14,
            connectionstyle="arc3,rad=0.0" if v == tgt else "arc3,rad=0.1", alpha=0.92,
        ), zorder=1)
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ox, oy = (mx * 0.22, my * 0.22) if v == tgt else (
            -((pos[v][1] - pos[u][1]) / (np.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]) or 1)) * 0.28,
            ((pos[v][0] - pos[u][0]) / (np.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]) or 1)) * 0.28,
        )
        ax.text(mx + ox, my + oy, f"{tag} w={float(r.get('weight', 1.0)):.1f}", fontsize=6 if short_labels else 7,
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="#fafafa", ec="none", alpha=0.88), zorder=4)

    for pairs, arrowstyle, linestyle in (
        (hidden_edges, "<|-|>", (0, (4, 3))),
        (flexible_edges, "-|>", (0, (5, 4))),
    ):
        for u, v in pairs:
            if u in pos and v in pos:
                ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(
                    arrowstyle=arrowstyle, color="#95a5a6",
                    lw=1.8 if arrowstyle == "<|-|>" else 1.6,
                    linestyle=linestyle, shrinkA=24 if arrowstyle == "<|-|>" else 12,
                    shrinkB=24 if arrowstyle == "<|-|>" else 12, mutation_scale=14,
                    connectionstyle="arc3,rad=0.0" if v == tgt else "arc3,rad=0.08",
                    alpha=0.9 if arrowstyle == "<|-|>" else 0.85,
                ), zorder=0)

    ax.set_title(title or "Structural causal model (Pearl-style, weighted edges)", fontsize=13)
    ax.axis("off")
    legend = [Line2D([0], [0], color=_edge_style(k, None)[0], lw=4, label=f"{k} ({_edge_style(k, None)[1]})") for k in sorted(seen_kinds)]
    if hidden_edges:
        legend.append(Line2D([0], [0], color="#95a5a6", lw=2, linestyle=(0, (4, 3)), label="hidden common cause"))
    if flexible_edges:
        legend.append(Line2D([0], [0], color="#95a5a6", lw=2, linestyle=(0, (5, 4)), label="flexible (glucose dynamics, not constrained)"))
    legend += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=12, label="core feature"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aed6f1", markeredgecolor="#5dade2", markeredgewidth=1.5, markersize=12, label="glucose dynamics"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1abc9c", markeredgecolor="#117a65", markeredgewidth=1.5, markersize=12, label="observable confounder"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ecf0f1", markeredgecolor="#7f8c8d", markeredgewidth=2, markersize=12, label="latent confounder"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6", markersize=12, label="target"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5, framealpha=0.92)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    pad = max(1.2, 0.22 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid_axes(n_rules: int) -> tuple[Any, int, int]:
    n_cols = min(2, n_rules)
    n_rows = int(np.ceil(n_rules / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 4.4 * n_rows), constrained_layout=True, squeeze=False)
    fig.patch.set_facecolor("#fafafa")
    return fig, axes, n_cols


_SNAPSHOT_COLORS = {"before": "#7f8c8d", "after": "#c0392b"}
_VIOLATION_COLOR = "#e67e22"
_MONO_TOL = 1e-9


def _sigmoid_template(
    g: np.ndarray, lo: float, hi: float, k: float, g0: float, *, increasing: bool,
) -> np.ndarray:
    inc = lo + (hi - lo) / (1.0 + np.exp(-k * (g - g0)))
    return inc if increasing else (hi + lo - inc)


def _anchored_decreasing_sigmoid(
    g: np.ndarray, y_lo: float, y_hi: float, k: float, g_min: float, g_max: float,
) -> np.ndarray:
    """Decreasing sigmoid: ``y_hi`` at low insulin, ``y_lo`` at high insulin."""
    t = (g - g_min) / max(g_max - g_min, 1e-12)
    return y_lo + (y_hi - y_lo) / (1.0 + np.exp(k * (t - 0.5) * 2.0))


def _fit_sigmoid_template(
    grid: np.ndarray, avg: np.ndarray, *, increasing: bool = False,
) -> tuple[np.ndarray, float]:
    g = np.asarray(grid, dtype=np.float64)
    y = np.asarray(avg, dtype=np.float64)
    mask = np.isfinite(g) & np.isfinite(y)
    g, y = g[mask], y[mask]
    if g.size < 3:
        return y, float("nan")
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    if y_hi <= y_lo + 1e-12:
        return np.full_like(g, y_lo), 1.0
    g_min, g_max = float(np.min(g)), float(np.max(g))
    best_r, best_t = -np.inf, y.copy()
    if increasing:
        for k in np.linspace(0.05, 3.0, 35):
            for g0 in np.linspace(g_min, g_max, 25):
                t = _sigmoid_template(g, y_lo, y_hi, float(k), float(g0), increasing=True)
                r = float(np.corrcoef(y, t)[0, 1])
                if np.isfinite(r) and r > best_r:
                    best_r, best_t = r, t
    else:
        for k in np.linspace(2.0, 24.0, 45):
            t = _anchored_decreasing_sigmoid(g, y_lo, y_hi, float(k), g_min, g_max)
            r = float(np.corrcoef(y, t)[0, 1])
            if np.isfinite(r) and r > best_r:
                best_r, best_t = r, t
    return best_t, best_r


def _overlay_monotonic_violations(
    ax: Any, grid: np.ndarray, avg: np.ndarray, *, increasing: bool,
) -> None:
    """Same violation overlay as ``save_single_rule_do_curve_figure``."""
    adj_segs = _adjacent_violation_segments(grid, avg, increasing=increasing)
    if not adj_segs:
        return
    viol_labeled = False
    for i, j in adj_segs:
        ax.plot(
            [grid[i], grid[j]], [avg[i], avg[j]], color=_VIOLATION_COLOR,
            linewidth=4.0, solid_capstyle="round", zorder=4,
            label="local monotonicity break" if not viol_labeled else None,
        )
        viol_labeled = True
    ends = sorted({idx for seg in adj_segs for idx in seg})
    ax.scatter(
        grid[ends], avg[ends], s=36, facecolors="white", edgecolors=_VIOLATION_COLOR,
        linewidths=2.0, zorder=5,
    )


def _first_violating_pair(
    grid: np.ndarray, avg: np.ndarray, *, increasing: bool,
) -> tuple[int, int] | None:
    order = np.argsort(grid, kind="mergesort")
    gs, ys = np.asarray(grid)[order], np.asarray(avg)[order]
    for i in range(int(gs.size)):
        for j in range(i + 1, int(gs.size)):
            if gs[i] >= gs[j] - _MONO_TOL:
                continue
            if (not increasing and ys[j] > ys[i] + _MONO_TOL) or (increasing and ys[j] < ys[i] - _MONO_TOL):
                return int(order[i]), int(order[j])
    return None


def _draw_monotonic_evaluation_example(
    ax: Any,
    grid: np.ndarray,
    avg: np.ndarray,
    *,
    patient_id: str,
    increasing: bool = False,
    feature: str = "insulin",
) -> None:
    plot_kw: dict[str, Any] = {"marker": "o", "markersize": 3} if grid.size <= 40 else {}
    nv, ns = monotonic_step_violations(avg, increasing=increasing, grid=grid)
    note = f" ({nv}/{ns} pair viol.)" if ns else ""

    ax.set_facecolor("#fafafa")
    ax.plot(
        grid, avg, color="#c0392b", linewidth=2.0, alpha=0.92,
        label=f"Before{note}", **plot_kw,
    )
    _overlay_monotonic_violations(ax, grid, avg, increasing=increasing)
    ax.set_ylabel(r"$E[\Delta y \mid \mathrm{do}(\mathrm{insulin})]$", fontsize=9)
    ax.set_title(
        rf"{feature} $\rightarrow$ target (MonotonicRelationship)",
        fontsize=9.5, fontweight="bold",
    )
    ax.grid(True, color="#ddd", linewidth=0.6)
    ax.legend(loc="best", fontsize=7)
    ax.tick_params(labelsize=8)


def _draw_sigmoid_pearson_example(
    ax: Any,
    grid: np.ndarray,
    avg: np.ndarray,
    *,
    patient_id: str,
    increasing: bool = False,
) -> None:
    y_lo, y_hi = float(np.min(avg)), float(np.max(avg))
    g_min, g_max = float(np.min(grid)), float(np.max(grid))
    template, r = _fit_sigmoid_template(grid, avg, increasing=increasing)

    ax.set_facecolor("#fafafa")
    plot_kw: dict[str, Any] = {"marker": "o", "markersize": 3} if grid.size <= 40 else {}
    ax.plot(grid, avg, color="#c0392b", linewidth=2.0, alpha=0.92, label=r"$\bar{f}(g)$", **plot_kw)
    ax.plot(
        grid, template, color="#2980b9", linewidth=2.0, ls="--", zorder=2,
        label=rf"Expected sigmoid ($\searrow$, {y_hi:.1f}$\rightarrow${y_lo:.1f})",
    )
    ax.set_xlabel(r"Insulin intervention $g$", fontsize=9)
    ax.set_ylabel(r"$E[\Delta y \mid \mathrm{do}(\mathrm{insulin})]$", fontsize=9)
    title_r = (
        rf"Sigmoid alignment, p.{patient_id}: Pearson $r={r:.2f}$"
        if np.isfinite(r) else rf"Sigmoid alignment, p.{patient_id}"
    )
    ax.set_title(title_r, fontsize=9.5, fontweight="bold")
    ax.grid(True, color="#ddd", linewidth=0.6)
    ax.legend(loc="best", fontsize=7)
    ax.tick_params(labelsize=8)


def save_irc_alignment_examples_figure(
    out_path: str | Path,
    *,
    grid: np.ndarray,
    avg: np.ndarray,
    patient_id: str = "588",
    increasing: bool = False,
    feature: str = "insulin",
) -> Path:
    """Stacked monotonic + sigmoid Pearson examples on a real insulin IRC."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.0), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    _draw_monotonic_evaluation_example(
        axes[0], grid, avg, patient_id=patient_id, increasing=increasing, feature=feature,
    )
    _draw_sigmoid_pearson_example(
        axes[1], grid, avg, patient_id=patient_id, increasing=increasing,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_irc_six_relationship_types_figure(
    out_path: str | Path,
    *,
    figsize: tuple[float, float] = (10.5, 6.8),
) -> Path:
    """Schematic IRC panels: three direct outcome curves + three interaction contribution curves."""
    g = np.linspace(0, 10, 80)
    curves: list[tuple[str, str, np.ndarray, str]] = [
        (
            "Direct — monotonic",
            r"Median $E[\,y \mid \mathrm{do}(u{=}g)\,]$",
            28.0 - 2.4 * g,
            "#c0392b",
        ),
        (
            "Direct — sigmoid",
            r"Median $E[\,y \mid \mathrm{do}(u{=}g)\,]$",
            6.0 + 18.0 / (1.0 + np.exp(-1.1 * (g - 5.0))),
            "#8e44ad",
        ),
        (
            "Direct — causal independence",
            r"Median $E[\,y \mid \mathrm{do}(u{=}g)\,]$",
            np.full_like(g, 12.0),
            "#7f8c8d",
        ),
        (
            "Interaction — monotonic",
            r"Median contrib.\ $\phi_v$ given $\mathrm{do}(u{=}g)$",
            -4.0 + 0.85 * g,
            "#2980b9",
        ),
        (
            "Interaction — sigmoid",
            r"Median contrib.\ $\phi_v$ given $\mathrm{do}(u{=}g)$",
            2.0 + 9.0 / (1.0 + np.exp(-1.3 * (g - 4.5))),
            "#16a085",
        ),
        (
            "Interaction — independence",
            r"Median contrib.\ $\phi_v$ given $\mathrm{do}(u{=}g)$",
            np.full_like(g, 3.5),
            "#95a5a6",
        ),
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor("#fafafa")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08, hspace=0.42, wspace=0.32)

    for ax, (title, ylab, y, color) in zip(axes.ravel(), curves):
        ax.set_facecolor("#fafafa")
        ax.plot(g, y, color=color, linewidth=2.6, marker="o", markersize=3, markevery=10)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlabel(r"Intervention level $g$ on feature $u$", fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=4)
        ax.grid(True, color="#ddd", linewidth=0.55)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def _adjacent_violation_segments(
    grid: np.ndarray, avg: np.ndarray, *, increasing: bool,
) -> list[tuple[int, int]]:
    """Consecutive grid steps that break monotonicity — for clean plot highlights."""
    g = np.asarray(grid, dtype=np.float64).ravel()
    y = np.asarray(avg, dtype=np.float64).ravel()
    order = np.argsort(g, kind="mergesort")
    gs, ys = g[order], y[order]
    out: list[tuple[int, int]] = []
    for i in range(int(gs.size) - 1):
        if gs[i + 1] <= gs[i] + _MONO_TOL:
            continue
        bad = (ys[i + 1] < ys[i] - _MONO_TOL) if increasing else (ys[i + 1] > ys[i] + _MONO_TOL)
        if bad:
            out.append((int(order[i]), int(order[i + 1])))
    return out


def _violating_sorted_pairs(
    grid: np.ndarray, avg: np.ndarray, *, increasing: bool,
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=np.float64).ravel()
    y = np.asarray(avg, dtype=np.float64).ravel()
    order = np.argsort(g, kind="mergesort")
    g, y = g[order], y[order]
    n = int(g.size)
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if g[i] >= g[j] - _MONO_TOL:
                continue
            if increasing and y[j] < y[i] - _MONO_TOL:
                out.append((int(order[i]), int(order[j])))
            elif not increasing and y[j] > y[i] + _MONO_TOL:
                out.append((int(order[i]), int(order[j])))
    return out


def save_single_rule_do_curve_figure(
    *,
    rule: dict[str, Any],
    model: Any,
    X_eval_df: pd.DataFrame,
    cols: list[str],
    target_name: str,
    out_path: str | Path,
    label: str = "Before",
    line_color: str | None = None,
    grid_size: int = 31,
    latent_columns: tuple[str, ...] = (),
    figsize: tuple[float, float] = (7.0, 4.4),
) -> Path | None:
    """Single do-curve panel — same axes/ styling as save_do_curve_progress_figure."""
    u = str(rule["start"])
    fn = rule.get("relationship_fn")
    fn_name = getattr(fn, "__name__", "") if callable(fn) else ""
    increasing = bool(getattr(fn, "increasing", True)) if callable(fn) else True
    X_np = X_eval_df[cols].to_numpy(dtype=np.float64, copy=False)
    grid, avg = rule_do_curve(
        model, X_np, cols, rule, target_name, latent_columns=latent_columns, grid_size=grid_size,
    )
    if not grid.size:
        return None
    y_label = (
        f"E[ {target_name} | do({u}=g) ]"
        if str(rule["end"]) == target_name
        else f"mean contrib.({rule['end']}) | do({u}=g)"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    plot_kw: dict[str, Any] = {"marker": "o", "markersize": 3} if grid.size <= 40 else {}
    frac = monotonic_aligned_fraction(avg, increasing=increasing, grid=grid)
    note = ""
    if np.isfinite(frac):
        nv, ns = monotonic_step_violations(avg, increasing=increasing, grid=grid)
        note = f" ({nv}/{ns} pair viol.)"
    color = line_color or _SNAPSHOT_COLORS.get(str(label).lower(), _SNAPSHOT_COLORS["after"])
    ax.plot(grid, avg, color=color, linewidth=2.0, alpha=0.92, label=f"{label}{note}", **plot_kw)
    _overlay_monotonic_violations(ax, grid, avg, increasing=increasing)
    ax.set_xlabel(u)
    ax.set_ylabel(y_label)
    ax.set_title(f"{u} -> {rule['end']}   ({fn_name})", fontsize=11)
    ax.grid(True, color="#ddd", linewidth=0.6)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_do_curve_progress_figure(
    *,
    cols: list[str],
    target_name: str,
    rules: list[dict],
    round_snapshots: list[tuple[str, Any]],
    X_eval_df: pd.DataFrame,
    out_path: str | Path,
    out_data_path: str | Path | None = None,
    latent_columns: tuple[str, ...] = (),
    X_eval_before_df: pd.DataFrame | None = None,
    cols_before: list[str] | None = None,
    grid_size: int = 31,
) -> Path | None:
    plotted = plotted_scm_rules(rules, cols, target_name)
    snapshots = list(round_snapshots)
    if not plotted or not snapshots:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cb = list(cols_before or cols)
    X_after = X_eval_df[cols].to_numpy(dtype=np.float64, copy=False)
    X_before = (
        X_eval_before_df[cb].to_numpy(dtype=np.float64, copy=False)
        if X_eval_before_df is not None
        else X_after
    )
    latent_set = {str(c) for c in latent_columns}
    fig, axes, n_cols = _grid_axes(len(plotted))
    n_rounds = len(snapshots)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.95, max(n_rounds, 1)))
    csv_rows: list[dict[str, Any]] = []

    for k, r in enumerate(plotted):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor("#fafafa")
        u = str(r["start"])
        fn = r.get("relationship_fn")
        fn_name = getattr(fn, "__name__", "") if callable(fn) else ""
        increasing = bool(getattr(fn, "increasing", True)) if callable(fn) else True
        plot_kw: dict[str, Any] = {}
        for ri, (label, mod) in enumerate(snapshots):
            if u in latent_set:
                continue
            is_before = str(label).lower() == "before"
            X_np = X_before if is_before else X_after
            cols_use = cb if is_before else cols
            lat = () if is_before else latent_columns
            grid_r, avg_r = rule_do_curve(
                mod, X_np, cols_use, r, target_name, latent_columns=lat, grid_size=grid_size,
            )
            y_label = (
                f"E[ {target_name} | do({u}=g) ]"
                if str(r["end"]) == target_name
                else f"mean contrib.({r['end']}) | do({u}=g)"
            )
            if not grid_r.size:
                continue
            if not plot_kw and grid_r.size <= 40:
                plot_kw = {"marker": "o", "markersize": 3}
            frac = monotonic_aligned_fraction(avg_r, increasing=increasing, grid=grid_r)
            note = ""
            if np.isfinite(frac):
                nv, ns = monotonic_step_violations(avg_r, increasing=increasing, grid=grid_r)
                note = f" ({nv}/{ns} pair viol.)"
            color = _SNAPSHOT_COLORS.get(str(label).lower(), cmap[ri])
            ax.plot(grid_r, avg_r, color=color, linewidth=2.0 if n_rounds <= 2 else 1.8,
                    alpha=0.92, label=f"{label}{note}", **plot_kw)
            csv_rows.extend({"rule": f"{u}->{r['end']}", "round": label, "grid_value": float(grid_r[i]), "avg_pred": float(avg_r[i])}
                          for i in range(min(len(grid_r), len(avg_r))))
        ax.set_xlabel(u)
        ax.set_ylabel(y_label)
        ax.set_title(f"{u} -> {r['end']}   ({fn_name})", fontsize=11)
        ax.grid(True, color="#ddd", linewidth=0.6)
        ax.legend(loc="best", fontsize=7)

    for k in range(len(plotted), axes.shape[0] * axes.shape[1]):
        axes[k // n_cols][k % n_cols].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if out_data_path and csv_rows:
        csv_path = Path(out_data_path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return out_path


def save_head_to_head_insulin_irc_figure(
    *,
    curves_by_approach: dict[str, list[np.ndarray]],
    approach_labels: dict[str, str],
    approach_colors: dict[str, str],
    out_path: str | Path,
    curves_by_patient: dict[str, dict[str, np.ndarray]] | None = None,
    figsize: tuple[float, float] = (10.0, 5.5),
) -> Path:
    """Single-panel insulin IRCs: faint per-patient curves + bold mean per approach."""
    del curves_by_patient
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, 25)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    for key, curves in curves_by_approach.items():
        if not curves:
            continue
        arr = np.vstack(curves)
        color = approach_colors.get(key, "#333333")
        label = approach_labels.get(key, key)
        for row in arr:
            ax.plot(t, row, color=color, alpha=0.14, linewidth=1.0, zorder=1)
        ax.plot(
            t, np.nanmean(arr, axis=0), color=color, linewidth=3.0, alpha=0.95,
            label=f"{label} (mean, n={len(curves)})", zorder=3,
        )
    ax.set_xlabel(r"Normalized insulin intervention ($g_{\min}\!\rightarrow\! g_{\max}$ per patient)")
    ax.set_ylabel(r"$E[\Delta y \mid \mathrm{do}(\mathrm{insulin})]$")
    ax.set_title("Insulin IRC — all Ohio patients", fontsize=11, fontweight="bold")
    ax.grid(True, color="#ddd", linewidth=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_shap_beeswarm_before_after_figure(
    *,
    model_before: Any,
    model_after: Any,
    X_val_df: pd.DataFrame,
    out_path: str | Path,
    X_val_before_df: pd.DataFrame | None = None,
    random_state: int = 42,
    max_samples: int = 200,
    title_before: str = "Before augmentation",
    title_after: str = "After augmentation",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(random_state))
    n = min(int(max_samples), len(X_val_df))
    idx = rng.choice(len(X_val_df), n, replace=False) if len(X_val_df) > n else np.arange(len(X_val_df))
    Xs_after = X_val_df.iloc[idx].reset_index(drop=True)
    Xs_before = (X_val_before_df.iloc[idx] if X_val_before_df is not None else Xs_after).reset_index(drop=True)

    def _drop_all_nan_cols(Xs: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in Xs.columns if Xs[c].notna().any()]
        return Xs[keep] if keep else Xs

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2))
    for ax, m, Xs, title in (
        (axes[0], model_before, _drop_all_nan_cols(Xs_before), title_before),
        (axes[1], model_after, _drop_all_nan_cols(Xs_after), title_after),
    ):
        phi = shap.TreeExplainer(m).shap_values(Xs.to_numpy(dtype=np.float64, copy=False))
        phi = phi[-1] if isinstance(phi, list) else phi
        plt.sca(ax)
        shap.summary_plot(np.asarray(phi, dtype=np.float64), features=Xs, plot_type="dot", max_display=len(Xs.columns), show=False)
        ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _latent_scatter_trend_ax(
    ax,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    yt_train: np.ndarray,
    yt_val: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    ax.set_facecolor("#fdfdfd")
    ax.scatter(np.asarray(x_train).ravel(), np.asarray(yt_train).ravel(), color="#bdc3c7", alpha=0.15, s=8, label="Train (noisy)")
    ax.scatter(np.asarray(x_val).ravel(), np.asarray(yt_val).ravel(), color="#8e44ad", alpha=0.4, s=16, label="Validation")
    x_all = np.concatenate([np.asarray(x_train).ravel(), np.asarray(x_val).ravel()])
    y_all = np.concatenate([np.asarray(yt_train).ravel(), np.asarray(yt_val).ravel()])
    idx = np.isfinite(x_all) & np.isfinite(y_all)
    if idx.sum() > 2:
        m, b = np.polyfit(x_all[idx], y_all[idx], 1)
        x_line = np.array([x_all[idx].min(), x_all[idx].max()])
        ax.plot(x_line, m * x_line + b, color="#e74c3c", lw=3, label=f"Trend (slope: {m:.2f})")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)


def _latent_confounder_row(
    ax_bar,
    ax_hist,
    ax_scatter_true,
    ax_scatter_hat,
    *,
    latent_name: str,
    target_name: str,
    params: dict[str, Any],
    z_train: np.ndarray,
    z_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_hat_train: np.ndarray,
    y_hat_val: np.ndarray,
) -> None:
    names = [_confounder_feature_label(n, target_name=target_name) for n in params["feature_names"]]
    coef = np.asarray(params["coef"], dtype=np.float64)
    order = np.argsort(-np.abs(coef))
    top = order[: min(12, len(order))]
    top = top[np.abs(coef[top]) > 1e-4]
    if top.size == 0:
        top = order[: min(6, len(order))]
    z_train = np.asarray(z_train, dtype=np.float64).ravel()
    z_val = np.asarray(z_val, dtype=np.float64).ravel()
    y_train_u = np.asarray(y_train, dtype=np.float64).ravel()
    y_val_u = np.asarray(y_val, dtype=np.float64).ravel()
    y_hat_train_u = np.asarray(y_hat_train, dtype=np.float64).ravel()
    y_hat_val_u = np.asarray(y_hat_val, dtype=np.float64).ravel()

    ax_bar.set_facecolor("#fdfdfd")
    labels = [names[i] for i in top]
    vals = coef[top]
    colors_bar = ["#e74c3c" if v >= 0 else "#3498db" for v in vals]
    y_pos = np.arange(len(labels))
    ax_bar.barh(y_pos, vals, color=colors_bar, edgecolor="#2c3e50", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.axvline(0.0, color="#2c3e50", lw=1.5)
    ax_bar.set_xlabel("Impact on Z", fontsize=10)
    ax_bar.set_title(f"{latent_name}: drivers", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax_bar.invert_yaxis()

    ax_hist.set_facecolor("#fdfdfd")
    bins = np.linspace(min(z_train.min(), z_val.min()), max(z_train.max(), z_val.max()), 30)
    ax_hist.hist(z_train, bins=bins, alpha=0.3, color="#bdc3c7", label="Train", density=True, edgecolor="white")
    ax_hist.hist(z_val, bins=bins, alpha=0.6, color="#8e44ad", label="Val", density=True, edgecolor="white")
    ax_hist.set_xlabel(latent_name, fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title(f"{latent_name}: distribution", fontsize=11, fontweight="bold")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, linestyle="--", alpha=0.5)

    _latent_scatter_trend_ax(
        ax_scatter_true,
        x_train=z_train,
        x_val=z_val,
        yt_train=y_train_u,
        yt_val=y_val_u,
        xlabel=latent_name,
        ylabel=f"True {target_name}",
        title=f"{latent_name}: vs true",
    )
    _latent_scatter_trend_ax(
        ax_scatter_hat,
        x_train=z_train,
        x_val=z_val,
        yt_train=y_hat_train_u,
        yt_val=y_hat_val_u,
        xlabel=latent_name,
        ylabel=f"Pred. {target_name}",
        title=f"{latent_name}: vs predicted",
    )


def save_latent_confounder_figure(
    *,
    latent_name: str,
    target_name: str,
    params: dict[str, Any],
    z_train: np.ndarray,
    z_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_hat_train: np.ndarray,
    y_hat_val: np.ndarray,
    out_path: str | Path,
) -> Path:
    """One latent: 2×2 panels (drivers, histogram, scatter vs target, scatter vs predictions)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Learned latent: '{latent_name}'", fontsize=16, fontweight="bold")
    _latent_confounder_row(
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1],
        latent_name=latent_name,
        target_name=target_name,
        params=params,
        z_train=z_train,
        z_val=z_val,
        y_train=y_train,
        y_val=y_val,
        y_hat_train=y_hat_train,
        y_hat_val=y_hat_val,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_joint_latent_confounders_figure(
    *,
    target_name: str,
    latent_names: list[str],
    cols: list[str],
    params_by_col: dict[str, dict[str, Any]],
    X_train_imp: np.ndarray,
    X_val_imp: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_hat_train: np.ndarray,
    y_hat_val: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Multiple latents: one PNG, one row per confounder × four columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(latent_names) == 1:
        c = latent_names[0]
        li = cols.index(c)
        return save_latent_confounder_figure(
            latent_name=c,
            target_name=target_name,
            params=params_by_col[c],
            z_train=X_train_imp[:, li],
            z_val=X_val_imp[:, li],
            y_train=y_train,
            y_val=y_val,
            y_hat_train=y_hat_train,
            y_hat_val=y_hat_val,
            out_path=out_path,
        )
    n = len(latent_names)
    fig, axes = plt.subplots(n, 4, figsize=(14, max(8.0, 3.1 * n)), constrained_layout=True, squeeze=False)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Learned latent confounders", fontsize=16, fontweight="bold")
    for i, c in enumerate(latent_names):
        li = cols.index(c)
        _latent_confounder_row(
            axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3],
            latent_name=c,
            target_name=target_name,
            params=params_by_col[c],
            z_train=X_train_imp[:, li],
            z_val=X_val_imp[:, li],
            y_train=y_train,
            y_val=y_val,
            y_hat_train=y_hat_train,
            y_hat_val=y_hat_val,
        )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
