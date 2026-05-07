from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import shap

FEATURE_RING_RADIUS = 1.35
INTERACTION_EDGE_MIN_ABS = 1e-8


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
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


def centered_radial_positions(feature_order: tuple[str, ...], target_name: str) -> dict[str, tuple[float, float]]:
    """Target at origin; features evenly on a circle (same order → same angle every plot)."""
    pos: dict[str, tuple[float, float]] = {target_name: (0.0, 0.0)}
    n = len(feature_order)
    if n == 0:
        return pos
    for k, name in enumerate(feature_order):
        ang = 2.0 * np.pi * k / n - np.pi / 2.0
        pos[name] = (
            float(FEATURE_RING_RADIUS * np.cos(ang)),
            float(FEATURE_RING_RADIUS * np.sin(ang)),
        )
    return pos


def _norm_edge_width(w: float, w_min: float, w_max: float, lo: float = 0.5, hi: float = 6.0) -> float:
    if w_max <= w_min + 1e-18:
        return (lo + hi) / 2
    t = (w - w_min) / (w_max - w_min)
    return float(lo + t * (hi - lo))


def _shap_interaction_graph_and_weights(
    *,
    model: Any,
    X_df: pd.DataFrame,
    cols: list[str],
    target_name: str,
    seed: int,
    max_shap_rows: int = 400,
    interaction_edge_min_abs: float = INTERACTION_EDGE_MIN_ABS,
) -> tuple[nx.Graph, float, float]:
    """Correlation network from SHAP terms and corresponding feature signals."""
    n = min(max_shap_rows, len(X_df))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_df), size=n, replace=False) if len(X_df) > n else np.arange(len(X_df))
    Xs = X_df.iloc[idx].copy()

    explainer = shap.TreeExplainer(model)
    iv = explainer.shap_interaction_values(Xs)
    if isinstance(iv, list):
        iv = np.asarray(iv[0])
    iv = np.asarray(iv, dtype=np.float64)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = np.asarray(sv[0])
    sv = np.asarray(sv, dtype=np.float64)

    G = nx.Graph()
    for c in cols:
        G.add_node(c)
    G.add_node(target_name)

    edge_weights: list[float] = []
    Xn = Xs.to_numpy(dtype=np.float64, copy=False)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            # Correlate pairwise interaction contribution with joint feature activity.
            s = _pearson_safe(iv[:, i, j], Xn[:, i] * Xn[:, j])
            if not np.isfinite(s):
                continue
            if abs(s) < interaction_edge_min_abs:
                continue
            G.add_edge(cols[i], cols[j], sign=np.sign(s), w=abs(s))
            edge_weights.append(abs(s))
    for i, c in enumerate(cols):
        # Correlate feature SHAP effect with feature value (same metric as feasibility logic).
        s = _pearson_safe(sv[:, i], Xs[c].to_numpy(dtype=np.float64, copy=False))
        if not np.isfinite(s):
            continue
        if abs(s) < interaction_edge_min_abs:
            continue
        G.add_edge(c, target_name, sign=np.sign(s), w=abs(s))
        edge_weights.append(abs(s))

    if not edge_weights:
        w_min, w_max = 0.0, 1.0
    else:
        w_min, w_max = min(edge_weights), max(edge_weights)
    return G, w_min, w_max


def _draw_shap_interaction_network_ax(
    ax: Any,
    G: nx.Graph,
    *,
    cols: list[str],
    target_name: str,
    pos: dict[str, tuple[float, float]],
    w_min: float,
    w_max: float,
    plot_title: str,
    show_legend: bool,
) -> None:
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal")

    node_list = list(cols) + [target_name]
    node_colors = ["#2ecc71" if n != target_name else "#9b59b6" for n in node_list]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=node_list, node_color=node_colors, node_size=1100, alpha=0.95, edgecolors="#333"
    )
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in node_list}, font_size=8, font_weight="bold")

    for u, v, data in G.edges(data=True):
        w = float(data["w"])
        sgn = float(data["sign"])
        lw = _norm_edge_width(w, w_min, w_max)
        color = "#c0392b" if sgn >= 0 else "#2980b9"
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.85, solid_capstyle="round", zorder=1)

    ax.set_title(plot_title, fontsize=11)
    ax.axis("off")

    if show_legend:
        legend_elems = [
            Line2D([0], [0], color="#c0392b", lw=4, label="positive correlation"),
            Line2D([0], [0], color="#2980b9", lw=4, label="negative correlation"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="feature"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6", markersize=10, label="target"),
        ]
        ax.legend(handles=legend_elems, loc="upper left", fontsize=8)

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
    max_shap_rows: int = 400,
    min_abs_pearson: float = 0.2,
) -> None:
    """Side-by-side SHAP interaction networks (same subsample seed; separate width scales per panel)."""
    out_path = Path(out_path)
    Gb, wb_lo, wb_hi = _shap_interaction_graph_and_weights(
        model=model_before, X_df=X_df_before, cols=cols, target_name=target_name, seed=seed,
        max_shap_rows=max_shap_rows,
        interaction_edge_min_abs=float(min_abs_pearson),
    )
    Ga, wa_lo, wa_hi = _shap_interaction_graph_and_weights(
        model=model_after, X_df=X_df_after, cols=cols, target_name=target_name, seed=seed,
        max_shap_rows=max_shap_rows,
        interaction_edge_min_abs=float(min_abs_pearson),
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    _draw_shap_interaction_network_ax(
        axes[0], Gb,
        cols=cols, target_name=target_name, pos=pos, w_min=wb_lo, w_max=wb_hi,
        plot_title="Before augmentation",
        show_legend=True,
    )
    _draw_shap_interaction_network_ax(
        axes[1], Ga,
        cols=cols, target_name=target_name, pos=pos, w_min=wa_lo, w_max=wa_hi,
        plot_title="After augmentation",
        show_legend=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scm_graph_from_rules(
    *,
    rules: list[dict],
    feature_cols: tuple[str, ...],
    target_name: str,
    out_path: str | Path,
    pos: dict[str, tuple[float, float]],
) -> None:
    """Directed SCM from dataset rules only; edge width ~ |edge|, red positive / blue negative."""
    out_path = Path(out_path)
    G = nx.DiGraph()
    for n in feature_cols:
        G.add_node(n)
    G.add_node(target_name)

    weights: list[float] = []
    for r in rules:
        u, v = str(r["start"]), str(r["end"])
        ed = float(r["edge"])
        w = abs(ed)
        weights.append(w)
        G.add_edge(u, v, sign=np.sign(ed), w=w)

    if not weights:
        w_min, w_max = 0.0, 1.0
    else:
        w_min, w_max = min(weights), max(weights)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    ax.set_aspect("equal")

    node_list = list(feature_cols) + [target_name]
    colors = ["#9b59b6" if n == target_name else "#3498db" for n in node_list]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list, node_color=colors, node_size=1400, alpha=0.95, edgecolors="#333")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in node_list}, font_size=9, font_weight="bold")

    for u, v, data in G.edges(data=True):
        w = float(data["w"])
        sgn = float(data["sign"])
        lw = _norm_edge_width(w, w_min, w_max)
        color = "#c0392b" if sgn >= 0 else "#2980b9"
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                shrinkA=22,
                shrinkB=22,
                mutation_scale=18,
                alpha=0.9,
            ),
            zorder=1,
        )

    ax.set_title("Structural causal model (dataset rules)", fontsize=13)
    ax.axis("off")

    legend_elems = [
        Line2D([0], [0], color="#c0392b", lw=4, label="positive edge"),
        Line2D([0], [0], color="#2980b9", lw=4, label="negative edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=12, label="feature"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6", markersize=12, label="target"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=10)
    pad = 0.55
    ax.set_xlim(-FEATURE_RING_RADIUS - pad, FEATURE_RING_RADIUS + pad)
    ax.set_ylim(-FEATURE_RING_RADIUS - pad, FEATURE_RING_RADIUS + pad)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _shap_explanation_for_plot(model: Any, X_df: pd.DataFrame) -> shap.Explanation:
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
    ev = np.asarray(ex.expected_value).ravel()
    ev0 = float(ev[0]) if ev.size else 0.0
    base = np.full(len(X_df), ev0, dtype=float)
    return shap.Explanation(
        values=sv,
        base_values=base,
        data=X_df.values.astype(float),
        feature_names=[str(c) for c in X_df.columns],
    )


def save_shap_beeswarm_before_after_figure(
    *,
    model_before: Any,
    model_after: Any,
    X_val_df: pd.DataFrame,
    out_path: str | Path,
    random_state: int = 42,
    max_samples: int = 200,
    title_before: str = "SHAP bee swarm — before",
    title_after: str = "SHAP bee swarm — after",
) -> Path:
    """Side-by-side SHAP beeswarm (before / after) on the same validation subsample."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(random_state))
    n = min(int(max_samples), len(X_val_df))
    idx = rng.choice(len(X_val_df), size=n, replace=False) if len(X_val_df) > n else np.arange(len(X_val_df))
    Xs = X_val_df.iloc[idx].copy().reset_index(drop=True)
    names = [str(c) for c in Xs.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2), constrained_layout=True)
    exp_b = _shap_explanation_for_plot(model_before, Xs)
    exp_a = _shap_explanation_for_plot(model_after, Xs)
    shap.plots.beeswarm(
        exp_b, max_display=len(names), show=False, ax=axes[0], plot_size=None, color_bar_label="Feature value",
    )
    axes[0].set_title(title_before)
    shap.plots.beeswarm(
        exp_a, max_display=len(names), show=False, ax=axes[1], plot_size=None, color_bar_label="Feature value",
    )
    axes[1].set_title(title_after)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


