"""End-to-end iSHAP run for a CSV dataset (classification or regression)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter

# Silence LibreSSL warning emitted by urllib3 on macOS.
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL.*")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ishap import ISHAPExplainer
from model import LightGBMClassifierModel, train_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SHAPIQ_FAST_CFG = {
    "sample_n_graph": 6,
    "sample_n_edges": 4,
    "sample_n_causal": 8,
    "max_features": 0,
}
BENCHMARK_FAST_CFG = {
    "sample_n": 80,
    "bg_n": 100,
    "run_full_benchmark": False,
}
PIPELINE_FAST_CFG = {
    "cv_splits": 3,
    "reuse_llm_cache": True,
    "reuse_shapiq_cache": True,
    "run_shapiq": False,
    "render_shapiq_graph": False,
}


def compute_cv(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run stratified CV for LightGBM and return fold-level and aggregate AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    for tr_i, te_i in cv.split(x, y):
        m = LightGBMClassifierModel()
        m.clf.set_params(verbose=-1)
        m.fit(x.iloc[tr_i], y.iloc[tr_i])
        p = m.predict_proba(x.iloc[te_i])[:, 1]
        aucs.append(float(roc_auc_score(y.iloc[te_i], p)))
    return {
        "n_splits": int(n_splits),
        "fold_auc": aucs,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs, ddof=1)),
    }


def main() -> None:
    load_dotenv()
    root = Path(__file__).resolve().parents[1]
    out_d = (root / "results").resolve()
    fig_d = (root / "manuscript").resolve()
    tgt = "mortality_flag"
    out_d.mkdir(parents=True, exist_ok=True)
    fig_d.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    def _timed(name: str, fn):
        t0 = perf_counter()
        out = fn()
        dt = float(perf_counter() - t0)
        timings[name] = dt
        print(f"[timing] {name}: {dt:.2f}s")
        return out

    exp = ISHAPExplainer(project_root=root)
    x, y = exp.load_dataset(
        csv_path=root / "mimic-iv_processed.csv",
        target_column=tgt,
    )

    cv_auc = _timed(
        "cv_auc",
        lambda: compute_cv(x, y, n_splits=PIPELINE_FAST_CFG["cv_splits"], random_state=42),
    )
    m, x_tr, x_te, y_tr, y_te, met = _timed(
        "train_model",
        lambda: train_model(
            x,
            y,
            classification=True,
            n_splits=PIPELINE_FAST_CFG["cv_splits"],
            verbose=-1,
            random_state=42,
        ),
    )
    exp.model = m
    exp.task_kind = "classification"

    g = _timed(
        "build_global_summary_bundle",
        lambda: exp.build_global_summary_bundle(X_tr=x_tr, X_te=x_te, sample_n=500, top_k=20),
    )
    g["performance"].update(met)
    g["performance"]["cv_auc"] = cv_auc

    td = exp.build_task_descriptor(
        x_tr,
        y_tr,
        task_title="ICU in-hospital mortality (binary)",
        task_description=(
            "Static demographics, comorbidity flags, treatment cohort indicators, and per-stay aggregated "
            "laboratory and vital-sign summaries from ICU tables. Target is in-hospital mortality."
        ),
        outcome_name="mortality",
    )
    td["hold_out_metrics_fold1"] = met
    td["cv_auc"] = cv_auc

    pa = _timed(
        "build_prediction_audit_bundle",
        lambda: exp.build_prediction_audit_bundle(
            x_tr, x_te, y_te, max_eval_rows=600, samples_per_stratum=2
        ),
    )

    tr_s = exp.build_training_samples_for_llm(x_tr, y_tr, n=8)
    pr_path = out_d / "alignment_and_rules.json"
    if PIPELINE_FAST_CFG["reuse_llm_cache"] and pr_path.exists():
        pr = json.loads(pr_path.read_text(encoding="utf-8"))
    else:
        pr = _timed(
            "generate_alignment_and_rules",
            lambda: exp.generate_alignment_and_rules(
                task_descriptor=td,
                global_shap_bundle=g,
                prediction_audit=pa,
                training_samples=tr_s,
            ),
        )
        pr_path.write_text(json.dumps(pr, indent=2, default=str), encoding="utf-8")

    syn = exp.generate_synthetic_rows_from_profiles(x_tr, pr, random_state=42)
    syn.to_csv(out_d / "synthetic_profile_samples.csv", index=False)

    bee_p = fig_d / "mimic_shap_vs_ishap_beeswarm.png"
    # Left: SHAP on held-out rows. Right: SHAP on LLM profile samples.
    _timed(
        "plot_shap_comparison_beeswarm",
        lambda: exp.plot_shap_comparison_beeswarm(
            x_tr,
            x_te,
            bee_p,
            profile_samples=syn,
            sample_n=160,
            max_display=18,
        ),
    )
    shap_eval = _timed(
        "evaluate_shap_objective_benchmark_comparison",
        lambda: exp.evaluate_shap_objective_benchmark_comparison(
            X_tr=x_tr,
            X_te=x_te,
            y_te=y_te,
            sample_n=BENCHMARK_FAST_CFG["sample_n"],
            bg_n=BENCHMARK_FAST_CFG["bg_n"],
            random_state=42,
            run_full_benchmark=BENCHMARK_FAST_CFG["run_full_benchmark"],
        ),
    )
    (out_d / "shap_objective_benchmark_comparison.json").write_text(
        json.dumps(shap_eval, indent=2, default=str),
        encoding="utf-8",
    )

    si_p = fig_d / "mimic_shapiq_interactions_graph.png"
    si_path = out_d / "shapiq_cohort_summary.json"
    ie_path = out_d / "interaction_empirics.json"
    if PIPELINE_FAST_CFG["run_shapiq"]:
        if PIPELINE_FAST_CFG["render_shapiq_graph"]:
            _timed(
                "plot_shapiq_interaction_graph_comparison",
                lambda: exp.plot_shapiq_interaction_graph_comparison(
                    X_eval_left=x_te,
                    X_eval_right=syn,
                    out_png=si_p,
                    sample_n=SHAPIQ_FAST_CFG["sample_n_graph"],
                    align_from_training=x_tr,
                    max_features=SHAPIQ_FAST_CFG["max_features"],
                ),
            )

        if PIPELINE_FAST_CFG["reuse_shapiq_cache"] and si_path.exists():
            si_s = _timed(
                "export_shapiq_interaction_edges_for_cohorts (cached)",
                lambda: json.loads(si_path.read_text(encoding="utf-8")),
            )
        else:
            si_s = _timed(
                "export_shapiq_interaction_edges_for_cohorts",
                lambda: exp.export_shapiq_interaction_edges_for_cohorts(
                    X_default=x_te,
                    X_profiles=syn,
                    sample_n=SHAPIQ_FAST_CFG["sample_n_edges"],
                    align_from_training=x_tr,
                    max_features=SHAPIQ_FAST_CFG["max_features"],
                ),
            )
            si_path.write_text(json.dumps(si_s, indent=2, default=str), encoding="utf-8")

        if PIPELINE_FAST_CFG["reuse_shapiq_cache"] and ie_path.exists():
            ie = _timed(
                "build_interaction_empirical_bundle (cached)",
                lambda: json.loads(ie_path.read_text(encoding="utf-8")),
            )
        else:
            ie = _timed(
                "build_interaction_empirical_bundle",
                lambda: exp.build_interaction_empirical_bundle(
                    x_tr,
                    syn,
                    si_s,
                    max_edges=15,
                    align_from_training=x_tr,
                ),
            )
            ie_path.write_text(json.dumps(ie, indent=2, default=str), encoding="utf-8")
    else:
        si_s = {}
        ie = {}
    p_dig = [
        {"profile_id": p.get("profile_id"), "title": p.get("title")}
        for p in (pr.get("targeted_profiles") or [])[:8]
        if isinstance(p, dict)
    ]
    ci_path = out_d / "causal_interaction_inference.json"
    if PIPELINE_FAST_CFG["reuse_llm_cache"] and ci_path.exists():
        ci = json.loads(ci_path.read_text(encoding="utf-8"))
    else:
        ci = _timed(
            "infer_causal_interaction_hypotheses",
            lambda: exp.infer_causal_interaction_hypotheses(
                task_descriptor=td,
                shapiq_cohort_summary=si_s,
                empirical_interactions=ie,
                global_shap_top=g.get("top_contributions"),
                targeted_profiles_digest=p_dig,
            ),
        )
    ci_path.write_text(
        json.dumps(ci, indent=2, default=str),
        encoding="utf-8",
    )

    x_syn = exp.align_profile_samples_to_training(x_tr, syn)
    pm_path = out_d / "pathophysiological_model.json"
    if PIPELINE_FAST_CFG["reuse_llm_cache"] and pm_path.exists():
        pm = json.loads(pm_path.read_text(encoding="utf-8"))
    else:
        pm = _timed(
            "elicit_pathophysiological_model",
            lambda: exp.elicit_pathophysiological_model(
                task_descriptor=td,
                global_shap_top=g.get("top_contributions") or [],
                shapiq_top_edges=si_s.get("top_edges_profiles") or [],
            ),
        )
    pm_path.write_text(
        json.dumps(pm, indent=2, default=str), encoding="utf-8"
    )
    pm_dot = pm.get("graphviz_dot") if isinstance(pm, dict) else None
    if isinstance(pm_dot, str) and pm_dot.strip():
        (out_d / "pathophysiological_model.dot").write_text(pm_dot, encoding="utf-8")
        (fig_d / "ishap_causal_context_explanation.dot").write_text(pm_dot, encoding="utf-8")

    run = {
        "target_column": tgt,
        "metrics": met,
        "timings_seconds": timings,
        "bottleneck_step": (max(timings, key=timings.get) if timings else None),
        "cv_auc": cv_auc,
        "shap_objective_benchmark_comparison": shap_eval,
        "training_samples": tr_s,
        "perturbation_rules": pr,
        "shapiq_cohort_summary": si_s,
        "interaction_empirics": ie,
        "causal_interaction_hypotheses": ci,
        "pathophysiological_model": pm,
    }
    (out_d / "ishap_run.json").write_text(
        json.dumps(run, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
