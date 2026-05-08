from __future__ import annotations

import importlib
import os
import warnings

from lib import iSHAP
from visuals import (
    centered_radial_positions,
    save_do_curve_before_after_figure,
    save_shap_beeswarm_before_after_figure,
    save_shap_interaction_network_before_after_graph,
    save_scm_graph_from_rules,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS: list[str] = ["d1namo"]

# Augmentation controls (forwarded to iSHAP.run_pipeline).
RUN_PIPELINE_KWARGS: dict[str, float | int] = {
    "random_state": 42,
    "test_size": 0.25,
    "synth_points_per_round": 20,
    "n_trials": 1000,
    "objective_metric_weight": 0.5,
}
MIN_ABS_PEARSON_FOR_INTERACTION_GRAPH = 0.4


def project_root(relative_to_file: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(relative_to_file)), ".."))


def run_dataset(dm: object) -> None:
    root = project_root(__file__)
    img_dir = os.path.join(root, "manuscript", "images", str(dm.NAME))
    results_dir = os.path.join(root, "results", str(dm.NAME))

    X, y = dm.load(root)
    result = iSHAP.run_pipeline(
        X, y,
        rules=list(dm.SCM_RULES),
        target=str(dm.TARGET),
        root=root,
        name=str(dm.NAME),
        task=str(getattr(dm, "TASK", "regression")),
        **RUN_PIPELINE_KWARGS,
    )

    cols       = result["cols"]
    target     = result["target"]
    feat_order = tuple(getattr(dm, "FEATURE_COLUMNS", tuple(cols)))
    pos        = centered_radial_positions(feat_order, target)

    save_shap_interaction_network_before_after_graph(
        cols=cols,
        target_name=target,
        X_df_before=result["X_eval_before"],
        model_before=result["model_before"],
        X_df_after=result["X_eval_after"],
        model_after=result["model_after"],
        out_path=os.path.join(img_dir, "shap_interaction_network_before_after_synthetic.png"),
        seed=42,
        pos=pos,
        min_abs_pearson=MIN_ABS_PEARSON_FOR_INTERACTION_GRAPH,
    )
    save_shap_beeswarm_before_after_figure(
        model_before=result["model_before"],
        model_after=result["model_after"],
        X_val_df=result["X_val"],
        out_path=os.path.join(img_dir, "shap_beeswarm_before_after_synthetic.png"),
        random_state=42,
        max_samples=200,
        title_before="Before augmentation",
        title_after="After augmentation",
    )
    save_scm_graph_from_rules(
        rules=list(dm.SCM_RULES),
        feature_cols=feat_order,
        target_name=target,
        out_path=os.path.join(img_dir, "scm_graph.png"),
        pos=pos,
    )
    save_do_curve_before_after_figure(
        cols=cols,
        target_name=target,
        rules=list(dm.SCM_RULES),
        model_before=result["model_before"],
        model_after=result["model_after"],
        X_eval_df=result["X_val"],
        out_path=os.path.join(img_dir, "do_curve_before_after.png"),
        out_data_path=os.path.join(results_dir, "do_curve_before_after.json"),
        grid_size=11,
        max_samples=200,
        random_state=42,
    )

for name in DATASETS:
    run_dataset(importlib.import_module(name))
