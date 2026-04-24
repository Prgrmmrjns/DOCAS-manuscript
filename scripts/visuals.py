from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from lib import (
    _pearson_r_xy,
    project_root,
    scm_parent_effect_sign,
    scm_structural_parents,
    scm_topological_order,
    tree_shap_df_sv_ev,
    tree_shap_values_from_df,
    val_subsample_for_shap_plot,
)


def _shap_explanation_for_plot(model: Any, X_df: pd.DataFrame) -> shap.Explanation:
    sv, ev0 = tree_shap_df_sv_ev(model, X_df)
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
) -> Path:
    """Side-by-side SHAP beeswarm (before / after) on the same validation subsample."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Xs = val_subsample_for_shap_plot(X_val_df, max_samples, random_state)
    names = [str(c) for c in Xs.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), constrained_layout=True)
    exp_b = _shap_explanation_for_plot(model_before, Xs)
    exp_a = _shap_explanation_for_plot(model_after, Xs)
    shap.plots.beeswarm(
        exp_b, max_display=len(names), show=False, ax=axes[0], plot_size=None, color_bar_label="Feature value",
    )
    axes[0].set_title("SHAP bee swarm — before")
    shap.plots.beeswarm(
        exp_a, max_display=len(names), show=False, ax=axes[1], plot_size=None, color_bar_label="Feature value",
    )
    axes[1].set_title("SHAP bee swarm — after")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _structural_interaction_rules(
    rules: Sequence[Mapping[str, Any]], target: str, cols: list[str],
) -> list[tuple[str, str, float]]:
    t, cs = str(target), set(cols)
    out: list[tuple[str, str, float]] = []
    for r in rules:
        u, v2 = str(r["start"]), str(r["end"])
        if v2 == t or u not in cs or v2 not in cs:
            continue
        out.append((u, v2, float(r["edge"])))
    return out


def save_interaction_cf_scatter_before_after_figure(
    *,
    model_before: Any,
    model_after: Any,
    X_val_df: pd.DataFrame,
    cols: list[str],
    rules: Sequence[Mapping[str, Any]],
    target: str,
    out_path: str | Path,
    random_state: int = 42,
    max_samples: int = 200,
) -> Path | None:
    """Observed ``start`` vs TreeSHAP(``end``); first feature→feature rule only. None if no such rule."""
    pairs = _structural_interaction_rules(rules, target, cols)
    if not pairs:
        return None
    start, end, _edge = pairs[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _one_ax(ax: plt.Axes, model: Any) -> None:
        Xs = val_subsample_for_shap_plot(X_val_df, max_samples, random_state)
        sv = np.asarray(tree_shap_values_from_df(model, Xs), dtype=np.float64)
        phi_end = sv[:, cols.index(end)].ravel()
        xs = Xs[start].to_numpy(dtype=np.float64, copy=False)
        rr = _pearson_r_xy(xs, phi_end)
        ax.scatter(xs, phi_end, alpha=0.35, s=14, c="#2c5282")
        ax.set_xlabel(f"{start} (observed)")
        ax.set_ylabel(f"SHAP({end})")
        ax.axhline(0.0, color="gray", lw=0.6, ls=":")
        rtxt = f"r = {rr:.3f}" if np.isfinite(rr) else "r = —"
        ax.text(0.03, 0.97, rtxt, transform=ax.transAxes, va="top", ha="left", fontsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _one_ax(axes[0], model_before)
    _one_ax(axes[1], model_after)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_interaction_cf_scatter_cli() -> None:
    """CLI: observed ``start`` vs TreeSHAP(``end``) from ``main.py`` pickle."""
    ap = argparse.ArgumentParser(description="Feature→feature rule scatter (uses main.py pickle).")
    ap.add_argument("--dataset", type=str, default="d1namo")
    ap.add_argument("--pickle", type=str, default="", help="Path to {dataset}_explain_models.pkl")
    ap.add_argument("--out", type=str, default="", help="Output PNG path")
    args = ap.parse_args()
    root = project_root(__file__)
    pkl = args.pickle or os.path.join(root, "results", f"{args.dataset}_explain_models.pkl")
    outp = args.out or os.path.join(root, "manuscript", "images", f"{args.dataset}_interaction_cf_scatter.png")
    if not os.path.isfile(pkl):
        raise SystemExit(f"missing {pkl} — run scripts/main.py for dataset {args.dataset} first")
    with open(pkl, "rb") as fp:
        b = pickle.load(fp)
    path = save_interaction_cf_scatter_before_after_figure(
        model_before=b["model_before"],
        model_after=b["model_after"],
        X_val_df=b["X_val_df"],
        cols=list(b["cols"]),
        rules=b["RULES"],
        target=str(b["TARGET"]),
        out_path=outp,
        random_state=int(b.get("random_state", 42)),
        max_samples=int(b.get("max_samples", 200)),
    )
    if path is None:
        raise SystemExit("no feature→feature rules in RULES for this bundle")
    print("wrote", path)


def load_scm_for_plot(
    rules: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], dict[str, Any]]:
    """Parents/order from structural rules via `lib` (feature→feature edges only)."""
    rl = list(rules)
    par = scm_structural_parents(rl, target=target)
    ord_ = scm_topological_order(feature_columns, rl, target=target)
    return par, ord_, {"rules": rl}


def export_scm_dag_dot(
    out_dot: str | Path,
    *,
    rules: Sequence[Mapping[str, Any]],
    target: str,
    feature_columns: tuple[str, ...],
    digraph_id: str = "scm_graph",
) -> Path:
    """Structural Graphviz DOT from feature→feature rules (see `lib.scm_*`)."""
    rl = list(rules)
    order: list[str] = [str(x) for x in scm_topological_order(feature_columns, rl, target=target)]
    parents_dict = scm_structural_parents(rl, target=target)
    parents: dict[str, list[str]] = {str(k): [str(x) for x in v] for k, v in parents_dict.items()}
    signs: dict[str, dict[str, int]] = scm_parent_effect_sign(rl, target=target)

    lines = [
        f"digraph {digraph_id} {{",
        "  graph [rankdir=TB, fontname=Helvetica, fontsize=10];",
        "  node [shape=ellipse, fontname=Helvetica, fontsize=9];",
        "  edge [fontname=Helvetica, fontsize=8];",
    ]
    seen: set[tuple[str, str]] = set()
    for child in order:
        for par in parents.get(child, []) or []:
            key = (par, child)
            if key in seen:
                continue
            seen.add(key)
            sgn = signs.get(child, {}).get(par, 0)
            if sgn > 0:
                lab, col = "+", "#1a7f1a"
            elif sgn < 0:
                lab, col = "\u2212", "#c41e3a"
            else:
                lab, col = "\u00b1", "#666666"
            lines.append(f'  "{par}" -> "{child}" [label="{lab}", fontcolor="{col}", color="{col}"];')
    lines.append("}")
    out = Path(out_dot)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    run_interaction_cf_scatter_cli()
