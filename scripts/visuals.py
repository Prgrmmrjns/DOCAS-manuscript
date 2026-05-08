from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import shap

from scoring import _monotonic_signed_score

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


def _do_curve_for_target_rule(
    model: Any,
    X_np: np.ndarray,
    feature_idx: int,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (grid, mean prediction) for E[Y | do(X_u = g)] across the column range."""
    u_col = X_np[:, feature_idx]
    u_min = float(np.nanmin(u_col))
    u_max = float(np.nanmax(u_col))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min + 1e-12:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    grid = np.linspace(u_min, u_max, int(max(2, grid_size)), dtype=np.float64)
    avg = np.empty(grid.size, dtype=np.float64)
    for k, g in enumerate(grid):
        Xint = X_np.copy()
        Xint[:, feature_idx] = g
        preds = np.asarray(model.predict(Xint), dtype=np.float64)
        avg[k] = float(np.mean(preds)) if preds.size else float("nan")
    return grid, avg


def save_do_curve_before_after_figure(
    *,
    cols: list[str],
    target_name: str,
    rules: list[dict],
    model_before: Any,
    model_after: Any,
    X_eval_df: pd.DataFrame,
    out_path: str | Path,
    out_data_path: str | Path | None = None,
    grid_size: int = 11,
    max_samples: int = 200,
    random_state: int = 42,
) -> Path | None:
    """Side-by-side do-effect curves for every rule that points to the target.

    For each rule (u -> target, edge), plot E[Y | do(X_u = g)] across u's range
    for both models on the same axes. Slope direction directly reflects whether
    the model's interventional behavior matches the rule sign.
    """
    out_path = Path(out_path)
    target_rules = [r for r in rules if str(r.get("end")) == str(target_name) and str(r.get("start")) in cols]
    if not target_rules:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(random_state))
    n = min(int(max_samples), len(X_eval_df))
    idx = rng.choice(len(X_eval_df), size=n, replace=False) if len(X_eval_df) > n else np.arange(len(X_eval_df))
    Xs = X_eval_df.iloc[idx][cols].copy().reset_index(drop=True)
    X_np = Xs.to_numpy(dtype=np.float64, copy=False)
    col_idx = {c: i for i, c in enumerate(cols)}

    n_rules = len(target_rules)
    n_cols_grid = min(2, n_rules)
    n_rows_grid = int(np.ceil(n_rules / n_cols_grid))
    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid,
        figsize=(7.0 * n_cols_grid, 4.4 * n_rows_grid),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("#fafafa")
    curve_payload: list[dict[str, Any]] = []

    for k, r in enumerate(target_rules):
        ax = axes[k // n_cols_grid][k % n_cols_grid]
        ax.set_facecolor("#fafafa")
        u = str(r["start"])
        edge = float(r.get("edge", 0.0))
        i_u = col_idx[u]

        grid_b, avg_b = _do_curve_for_target_rule(model_before, X_np, i_u, grid_size=grid_size)
        grid_a, avg_a = _do_curve_for_target_rule(model_after, X_np, i_u, grid_size=grid_size)

        ideal_dir = "+" if edge > 0 else ("-" if edge < 0 else "0")
        if grid_b.size:
            ax.plot(grid_b, avg_b, color="#7f8c8d", linewidth=2.2, marker="o", markersize=4, label="before")
        if grid_a.size:
            ax.plot(grid_a, avg_a, color="#c0392b", linewidth=2.2, marker="o", markersize=4, label="after")

        ax.set_xlabel(u)
        ax.set_ylabel(f"E[ {target_name} | do({u}=g) ]")
        ax.set_title(f"{u} -> {target_name}   (rule edge {ideal_dir})", fontsize=11)
        ax.grid(True, color="#ddd", linewidth=0.6)
        ax.legend(loc="best", fontsize=9)

        steps_ok_before = (
            float(np.mean(np.diff(avg_b) * (1.0 if edge >= 0 else -1.0) >= -1e-12)) if avg_b.size > 1 else float("nan")
        )
        steps_ok_after = (
            float(np.mean(np.diff(avg_a) * (1.0 if edge >= 0 else -1.0) >= -1e-12)) if avg_a.size > 1 else float("nan")
        )
        ms_before = float(_monotonic_signed_score(avg_b)) if avg_b.size > 1 else float("nan")
        ms_after = float(_monotonic_signed_score(avg_a)) if avg_a.size > 1 else float("nan")
        grid_serial = grid_b.tolist() if grid_b.size else grid_a.tolist()
        curve_payload.append(
            {
                "rule": f"{u}->{target_name}",
                "edge": edge,
                "grid": grid_serial,
                "before_avg_pred": avg_b.tolist(),
                "after_avg_pred": avg_a.tolist(),
                "before_do_corr": None if not grid_b.size else float(_pearson_safe(grid_b, avg_b)),
                "after_do_corr": None if not grid_a.size else float(_pearson_safe(grid_a, avg_a)),
                "before_steps_aligned_fraction": None if not np.isfinite(steps_ok_before) else steps_ok_before,
                "after_steps_aligned_fraction": None if not np.isfinite(steps_ok_after) else steps_ok_after,
                "before_monotonic_signed_score": None if not np.isfinite(ms_before) else ms_before,
                "after_monotonic_signed_score": None if not np.isfinite(ms_after) else ms_after,
            }
        )

    for k in range(n_rules, n_rows_grid * n_cols_grid):
        ax = axes[k // n_cols_grid][k % n_cols_grid]
        ax.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    payload_obj: dict[str, Any] = {"target": target_name, "rules": curve_payload}
    if out_data_path is not None:
        out_data_path = Path(out_data_path)
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_data_path, "w") as fp:
            json.dump(payload_obj, fp, indent=2)
        rows: list[dict[str, Any]] = []
        for p in curve_payload:
            g = p["grid"]
            b = p["before_avg_pred"]
            a = p["after_avg_pred"]
            for i in range(min(len(g), len(b), len(a))):
                rows.append(
                    {
                        "rule": p["rule"],
                        "grid_value": g[i],
                        "before_avg_pred": b[i],
                        "after_avg_pred": a[i],
                    }
                )
        if rows:
            pd.DataFrame(rows).to_csv(out_data_path.with_suffix(".csv"), index=False)
    else:
        default_json = out_path.with_suffix(".json")
        with open(default_json, "w") as fp:
            json.dump(payload_obj, fp, indent=2)
    return out_path


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


