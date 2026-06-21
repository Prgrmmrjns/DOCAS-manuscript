from __future__ import annotations

import os
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str, *, dpi: int = 150) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pdp_curves(plot: dict, *, target: str, out_dir: str) -> str:
    pid, tlabel = plot["pid"], plot.get("target_name", target)
    nodes = np.asarray(plot["nodes"], float) * 100.0
    targets = np.asarray(plot["target_nodes"], float)
    base_nodes = np.asarray(plot["base_nodes"], float)
    after_nodes = np.asarray(plot["after_nodes"], float)
    rms_b = float(np.sqrt(np.mean((base_nodes - targets) ** 2)))
    rms_a = float(np.sqrt(np.mean((after_nodes - targets) ** 2)))

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.plot(nodes, base_nodes, "o-", color="#3498db", lw=2, ms=5, label=f"before (rms {rms_b:.2f})")
    ax.plot(nodes, after_nodes, "o-", color="#27ae60", lw=2, ms=5, label=f"after iSHAP (rms {rms_a:.2f})")
    ax.plot(nodes, targets, "o--", color="#7f8c8d", lw=1.5, ms=5, alpha=0.8, label="target sigmoid")
    ax.set(
        xlabel="active insulin (%)",
        ylabel=f"predicted {tlabel} (mg/dL)",
        title=f"insulin PDP · node rms {rms_b:.2f}→{rms_a:.2f} mg/dL",
    )
    ax.grid(True, color="#ddd", lw=0.6)
    ax.legend(fontsize=8)
    fig.suptitle(f"Patient {pid} · insulin marginal PDP", fontsize=11, y=1.02)
    return _save(fig, os.path.join(out_dir, "pdp_curves.png"))


def save_patient_figures(root: str, name: str, res: dict[str, Any], *, target: str) -> str:
    plot = res["plot"]
    img_dir = os.path.join(root, "images", name)
    os.makedirs(img_dir, exist_ok=True)
    return save_pdp_curves(plot, target=target, out_dir=img_dir)


def save_replay_figure(
    *,
    times,
    glucose: np.ndarray,
    sim_baseline_dss: np.ndarray,
    sim_ishap_dss: np.ndarray,
    meal_bolus_u: float,
    bolus_baseline: list[tuple],
    bolus_ishap: list[tuple],
    out_path: str,
    title: str = "",
) -> str:
    """ReplayBG-style DSS figure (LightGBM baseline vs iSHAP)."""
    from cib import TIR_HI, TIR_LO

    t = pd.to_datetime(times)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.8), sharex=True, height_ratios=[2.2, 1.2])

    ax = axes[0]
    ax.axhspan(TIR_LO, TIR_HI, color="#e8f4ea", alpha=0.7, zorder=0)
    ax.axhline(TIR_LO, color="#888", ls="--", lw=0.8)
    ax.axhline(TIR_HI, color="#888", ls="--", lw=0.8)
    ax.plot(t, glucose, "k:", lw=1.8, label="CGM (observed)")
    ax.plot(t, sim_baseline_dss, color="#c0392b", lw=2, ls="--", label="LightGBM baseline DSS")
    ax.plot(t, sim_ishap_dss, color="#27ae60", lw=2, ls="--", label="iSHAP DSS")
    y_top = float(np.nanmax(np.r_[glucose, sim_baseline_dss, sim_ishap_dss, [TIR_HI]]))
    ax.set_ylim(0, y_top * 1.08)

    def _mark(boluses: list[tuple], color: str, *, y_frac: float) -> None:
        for ts, dose in boluses:
            ts = pd.Timestamp(ts)
            ax.axvline(ts, color=color, ls=":", lw=1.0, alpha=0.75, zorder=1)
            ax.annotate(
                f"{int(round(dose))} U",
                xy=(ts, y_top * y_frac),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            )

    _mark(bolus_baseline, "#c0392b", y_frac=0.92)
    _mark(bolus_ishap, "#27ae60", y_frac=0.98)
    ax.set_ylabel("glucose (mg/dL)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(title or "ReplayBG · corrective insulin boluses")

    ax2 = axes[1]
    doses = [float(meal_bolus_u)] + [float(d) for _, d in bolus_baseline] + [float(d) for _, d in bolus_ishap]
    ymax = max(10.0, max(doses) * 1.15) if doses else 10.0
    meal_t = t.iloc[0]
    ax2.bar([meal_t], [meal_bolus_u], width=0.003, color="k", label="meal bolus", zorder=2)
    base_labeled = ishap_labeled = False
    for ts, dose in bolus_baseline:
        lbl = "baseline CIB" if not base_labeled else ""
        ax2.bar([pd.Timestamp(ts)], [dose], width=0.003, color="#c0392b", label=lbl, zorder=2)
        base_labeled = True
    for ts, dose in bolus_ishap:
        lbl = "iSHAP CIB" if not ishap_labeled else ""
        ax2.bar([pd.Timestamp(ts)], [dose], width=0.003, color="#27ae60", label=lbl, zorder=2)
        ishap_labeled = True
    ax2.set_ylabel("insulin bolus (U)")
    ax2.set_ylim(0, ymax)
    if bolus_baseline or bolus_ishap or meal_bolus_u:
        ax2.legend(loc="upper right", fontsize=7.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=25)
    axes[-1].set_xlabel("time")
    return _save(fig, out_path)
