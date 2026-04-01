"""iSHAP: domain-informed SHAP + shapiq for tabular models (LLM-guided profile cohorts)."""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shapiq
import shap.benchmark as shap_benchmark

# pandas>=2.2: shapiq LightGBM conversion uses `.replace` with deprecated downcasting.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Downcasting behavior in `replace` is deprecated",
)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from shapiq.interaction_values import aggregate_interaction_values
from shapiq.plot import network_plot

from llm_expert import GRAPHVIZ_EXPLANATORY_MODEL_PROMPT, call_json
from model import LightGBMClassifierModel, LightGBMRegressorModel, train_model as fit_lightgbm


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba())
        return np.asarray(buf[:, :, :3]).copy()
    if hasattr(canvas, "tostring_argb"):
        r = canvas.get_renderer()
        w, h = int(r.width), int(r.height)
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return argb[:, :, [1, 2, 3]].copy()
    if hasattr(canvas, "tostring_rgb"):
        r = canvas.get_renderer()
        w, h = int(r.width), int(r.height)
        return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3)).copy()
    raise RuntimeError("Cannot extract raster from matplotlib canvas.")


class ISHAPExplainer:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self.model: LightGBMClassifierModel | LightGBMRegressorModel | None = None
        self.task_kind: str | None = None  # "classification" | "regression"
        self.safe_cols: list[str] = []
        self.backmap: dict[str, str] = {}

    def _is_classification(self) -> bool:
        return bool(self.task_kind == "classification")

    def _predict_target(self, X: pd.DataFrame) -> np.ndarray:
        if self._is_classification() and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    @staticmethod
    def _coerce_shap_array(shap_values: object) -> np.ndarray:
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
        return arr

    def _booster(self):
        if self.model is None:
            raise ValueError("Train model first.")
        return self.model.clf if hasattr(self.model, "clf") else self.model.reg

    @staticmethod
    def _sample_eval_rows(X: pd.DataFrame, sample_n: int, random_state: int) -> pd.DataFrame:
        return X.sample(min(len(X), sample_n), random_state=random_state)

    def load_dataset(
        self,
        *,
        csv_path: str | Path,
        target_column: str,
        drop_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        y_num = pd.to_numeric(df[target_column], errors="coerce")
        valid = y_num.notna()
        df = df.loc[valid].reset_index(drop=True)
        y_num = y_num.loc[valid].reset_index(drop=True)
        uniq = set(pd.unique(y_num.dropna()))
        is_binary = len(uniq) <= 2 and uniq.issubset({0, 1})
        y = (y_num.astype(int) if is_binary else y_num.astype(float)).reset_index(drop=True)
        self.task_kind = "classification" if is_binary else "regression"
        drop = set(drop_columns or [])
        drop.add(target_column)
        X = df[[c for c in df.columns if c not in drop]].copy()
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Keep NaN for continuous features (LightGBM can handle missing values).
        # For binary indicator columns, treat missing as 0 to preserve semantics.
        for c in X.columns:
            col = X[c]
            u = set(pd.unique(col.dropna()))
            if u.issubset({0, 1}):
                X[c] = col.fillna(0)

        new_cols: list[str] = []
        used: set[str] = set()
        mapping: dict[str, str] = {}
        for i, col in enumerate(X.columns):
            clean = re.sub(r"[^A-Za-z0-9_]", "_", str(col))
            if not clean:
                clean = f"feature_{i}"
            if clean[0].isdigit():
                clean = f"f_{clean}"
            base = clean
            j = 1
            while clean in used:
                clean = f"{base}_{j}"
                j += 1
            used.add(clean)
            mapping[clean] = str(col)
            new_cols.append(clean)
        self.safe_cols = new_cols
        self.backmap = mapping
        X.columns = self.safe_cols
        return X.reset_index(drop=True), y

    def train_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
        kind = self.task_kind or "classification"
        self.model, X_tr, X_te, y_tr, y_te, metrics = fit_lightgbm(
            X,
            y,
            classification=(kind == "classification"),
            n_splits=5,
            verbose=-1,
        )
        return X_tr, X_te, y_tr, y_te, metrics

    def align_profile_samples_to_training(self, X_tr: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Drop profile metadata / target / excluded scores; match ``X_tr`` columns; median-impute NaNs."""
        drop = [c for c in ("profile_id", "profile_title", "mortality_flag", "sofa") if c in df.columns]
        x = df.drop(columns=drop, errors="ignore")
        return x.reindex(columns=X_tr.columns).fillna(X_tr.median(numeric_only=True)).fillna(0.0)

    def plot_shap_comparison_beeswarm(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        out_png: str | Path,
        profile_samples: pd.DataFrame | None = None,
        sample_n: int = 300,
        max_display: int = 18,
        bg_n: int = 200,
        random_state: int = 42,
    ) -> None:
        """Left: interventional TreeSHAP on held-out rows. Right: tree-path SHAP on profile cohort."""
        n = min(sample_n, len(X_te))
        x_eval = X_te.sample(n, random_state=random_state)
        bg = X_tr.sample(min(bg_n, len(X_tr)), random_state=random_state)
        display_names = [self.backmap[c] for c in X_te.columns]

        booster = self._booster()
        explainer_std = shap.TreeExplainer(booster, data=bg, feature_perturbation="interventional")
        exp_std = explainer_std(x_eval, check_additivity=False)
        exp_std.feature_names = display_names

        x_profiles = None
        if profile_samples is not None and len(profile_samples):
            x_profiles = self.align_profile_samples_to_training(X_tr, profile_samples)
            if len(x_profiles):
                x_profiles = x_profiles.sample(min(len(x_profiles), sample_n), random_state=random_state + 1)
        if x_profiles is None or len(x_profiles) == 0:
            x_profiles = x_eval

        # For iSHAP profiles, compute SHAP directly via tree-path-dependent semantics to avoid
        # adding a second synthetic integration layer on top of synthetic profile cohorts.
        explainer_profiles = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
        exp_profiles = explainer_profiles(x_profiles, check_additivity=False)
        exp_profiles.feature_names = display_names

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        shap.plots.beeswarm(exp_std, max_display=max_display, show=False, ax=axes[0], plot_size=None)
        axes[0].set_title(f"Default SHAP (held-out rows, n={len(x_eval)})", fontsize=12, pad=12)
        shap.plots.beeswarm(exp_profiles, max_display=max_display, show=False, ax=axes[1], plot_size=None)
        axes[1].set_title(f"iSHAP (profiles, n={len(x_profiles)})", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _benchmark_result_to_dict(res: object) -> dict:
        """Convert a SHAP benchmark result object into JSON-safe fields."""
        out: dict[str, object] = {}
        for k in ("metric", "method", "value", "value_sign"):
            if hasattr(res, k):
                v = getattr(res, k)
                if isinstance(v, np.generic):
                    v = v.item()
                out[k] = v
        for k in ("curve_x", "curve_y", "curve_y_std"):
            if hasattr(res, k):
                v = getattr(res, k)
                if v is not None:
                    out[k] = np.asarray(v, dtype=float).tolist()
        return out

    def evaluate_shap_objective_benchmark_comparison(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        y_te: pd.Series | None = None,
        *,
        sample_n: int = 250,
        bg_n: int = 200,
        random_state: int = 42,
        run_full_benchmark: bool = False,
    ) -> dict:
        """
        Compare SHAP methods with objective benchmark metrics (as in SHAP docs).

        This includes explanation error, compute time, and sequential masking scores.
        """
        n = min(sample_n, len(X_te))
        x_eval = X_te.sample(n, random_state=random_state)
        bg = X_tr.sample(min(bg_n, len(X_tr)), random_state=random_state)
        booster = self._booster()
        x_eval_np = x_eval.to_numpy()

        exp_default = shap.TreeExplainer(
            booster,
            data=bg,
            feature_perturbation="interventional",
        )(x_eval, check_additivity=False)
        exp_ishap = shap.TreeExplainer(
            booster,
            feature_perturbation="tree_path_dependent",
        )(x_eval, check_additivity=False)

        methods = [("default_shap", exp_default), ("ishap", exp_ishap)]

        if self._is_classification() and hasattr(self.model, "clf"):
            predict_fn = lambda X: np.asarray(self.model.clf.predict(X, raw_score=True), dtype=float)
            y_eval = y_te.loc[x_eval.index].to_numpy() if y_te is not None else None

            def loss_fn(X, y):
                p = np.asarray(self.model.clf.predict_proba(X), dtype=float)
                y_arr = np.asarray(y, dtype=int)
                p_y = np.clip(p[np.arange(len(y_arr)), y_arr], 1e-12, 1.0)
                return (-np.log(p_y)).tolist()

            pred_scale = "raw_score"
        else:
            predict_fn = lambda X: np.asarray(self.model.predict(X), dtype=float)
            y_eval = y_te.loc[x_eval.index].to_numpy() if y_te is not None else None

            def loss_fn(X, y):
                pred = np.asarray(self.model.predict(X), dtype=float)
                y_arr = np.asarray(y, dtype=float)
                return ((y_arr - pred) ** 2).tolist()

            pred_scale = "prediction"

        eval_cols = list(x_eval.columns)

        def _as_model_frame(X_in: pd.DataFrame | np.ndarray) -> pd.DataFrame:
            if isinstance(X_in, pd.DataFrame):
                return X_in.reindex(columns=eval_cols)
            arr = np.asarray(X_in)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return pd.DataFrame(arr, columns=eval_cols)

        masker = shap.maskers.Independent(bg.to_numpy())
        metrics: dict[str, dict[str, object]] = {}
        timing_seconds: dict[str, float] = {}

        t0 = time.perf_counter()
        ee = shap_benchmark.ExplanationError(
            masker, lambda X: predict_fn(_as_model_frame(X)), x_eval_np
        )
        metrics["explanation_error"] = {
            name: self._benchmark_result_to_dict(ee(exp, name=name))
            for name, exp in methods
        }
        timing_seconds["explanation_error"] = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ct = shap_benchmark.ComputeTime()
        metrics["compute_time"] = {
            name: self._benchmark_result_to_dict(ct(exp, name=name))
            for name, exp in methods
        }
        timing_seconds["compute_time"] = float(time.perf_counter() - t0)

        if run_full_benchmark:
            for mask_type, ordering in [
                ("keep", "positive"),
                ("remove", "positive"),
                ("keep", "negative"),
                ("remove", "negative"),
            ]:
                t0 = time.perf_counter()
                sm = shap_benchmark.SequentialMasker(
                    mask_type,
                    ordering,
                    masker,
                    lambda X: predict_fn(_as_model_frame(X)),
                    x_eval_np,
                )
                key = f"{mask_type}_{ordering}"
                metrics[key] = {
                    name: self._benchmark_result_to_dict(sm(exp, name=name))
                    for name, exp in methods
                }
                timing_seconds[key] = float(time.perf_counter() - t0)

            if y_eval is not None:
                cmasker = shap.maskers.Composite(masker, shap.maskers.Fixed())
                for mask_type, ordering in [("keep", "absolute"), ("remove", "absolute")]:
                    t0 = time.perf_counter()
                    sm = shap_benchmark.SequentialMasker(
                        mask_type,
                        ordering,
                        cmasker,
                        lambda X, y: loss_fn(_as_model_frame(X), y),
                        x_eval_np,
                        y_eval,
                    )
                    key = f"{mask_type}_{ordering}_loss"
                    metrics[key] = {
                        name: self._benchmark_result_to_dict(sm(exp, name=name))
                        for name, exp in methods
                    }
                    timing_seconds[key] = float(time.perf_counter() - t0)

        summary = {
            "explanation_error": {
                "default_shap": float(metrics["explanation_error"]["default_shap"]["value"]),
                "ishap": float(metrics["explanation_error"]["ishap"]["value"]),
            },
            "compute_time_seconds": {
                "default_shap": float(metrics["compute_time"]["default_shap"]["value"]),
                "ishap": float(metrics["compute_time"]["ishap"]["value"]),
            },
        }
        summary["lower_explanation_error_method"] = (
            "default_shap"
            if summary["explanation_error"]["default_shap"] <= summary["explanation_error"]["ishap"]
            else "ishap"
        )
        if timing_seconds:
            summary["timing_seconds"] = timing_seconds
            summary["bottleneck_metric"] = max(timing_seconds, key=timing_seconds.get)

        return {
            "prediction_scale": pred_scale,
            "n_samples": int(len(x_eval)),
            "run_full_benchmark": bool(run_full_benchmark),
            "metrics": metrics,
            "summary": summary,
        }

    def build_global_summary_bundle(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        *,
        sample_n: int = 500,
        top_k: int = 20,
    ) -> dict:
        """Global SHAP summary for LLM (sample of held-out rows)."""
        n = min(sample_n, len(X_te))
        x_eval = X_te.sample(n, random_state=42)
        booster = self._booster()
        explainer = shap.TreeExplainer(booster, data=X_tr.sample(min(300, len(X_tr)), random_state=42))
        shap_arr = self._coerce_shap_array(explainer.shap_values(x_eval, check_additivity=False))
        mean_abs = np.abs(shap_arr).mean(axis=0)
        signed_mean = shap_arr.mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        cols = X_te.columns
        top_feats = [
            {
                "feature": self.backmap[cols[i]],
                "mean_abs_shap": float(mean_abs[i]),
                "mean_signed_shap": float(signed_mean[i]),
            }
            for i in top_idx
        ]

        pred = self._predict_target(X_te)
        bundle = {
            "scope": "global",
            "n_samples_evaluated": int(n),
            "performance": {
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
            },
            "top_contributions": top_feats,
        }
        return bundle

    def _shapiq_mean_abs_interactions(
        self,
        X_eval: pd.DataFrame,
        *,
        sample_n: int = 120,
        random_state: int = 42,
        max_features: int | None = 25,
    ) -> tuple[np.ndarray, list[str]]:
        """Mean |order-2 shapiq interactions| over a row sample."""
        x = self._sample_eval_rows(X_eval, sample_n, random_state)
        x = self._select_shapiq_feature_subset(x, max_features=max_features)
        expl = shapiq.TreeExplainer(self._booster())
        mats = [
            np.asarray(expl.explain(x.iloc[i].to_numpy()).get_n_order_values(2)) for i in range(len(x))
        ]
        mean_abs = np.nanmean(np.abs(np.stack(mats, axis=0)), axis=0)
        np.fill_diagonal(mean_abs, 0.0)
        return mean_abs, [self.backmap.get(c, c) for c in x.columns]

    def _select_shapiq_feature_subset(
        self,
        X_eval: pd.DataFrame,
        *,
        max_features: int | None = 25,
    ) -> pd.DataFrame:
        """Return shapiq input in model feature order with names preserved.

        We intentionally keep all model features. shapiq's tree explainer stores
        relevant-feature indices in the original model index space; hard column
        subsetting can trigger index-out-of-bounds errors.
        """
        model_cols: list[str] = []
        try:
            model_cols = list(self._booster().feature_name())
        except Exception:
            model_cols = []
        if model_cols and all(c in X_eval.columns for c in model_cols):
            return X_eval.reindex(columns=model_cols).copy()
        return X_eval.copy()

    @staticmethod
    def _edges_from_interaction_matrix(mat: np.ndarray, labels: list[str], cohort: str) -> pd.DataFrame:
        d = mat.shape[0]
        ti, tj = np.triu_indices(d, k=1)
        return (
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "feature_a": [labels[i] for i in ti],
                    "feature_b": [labels[j] for j in tj],
                    "mean_abs_interaction": mat[ti, tj].astype(float),
                }
            )
            .sort_values("mean_abs_interaction", ascending=False)
            .reset_index(drop=True)
        )

    def _shapiq_global_interaction_values(
        self,
        X_eval: pd.DataFrame,
        *,
        sample_n: int = 120,
        random_state: int = 42,
        max_features: int | None = 25,
    ) -> tuple[shapiq.InteractionValues, list[str]]:
        """Mean aggregated shapiq InteractionValues (order 2, k-SII)."""
        x = self._sample_eval_rows(X_eval, sample_n, random_state)
        x = self._select_shapiq_feature_subset(x, max_features=max_features)
        expl = shapiq.TreeExplainer(model=self._booster(), max_order=2, index="k-SII")
        ivs = [expl.explain(x=x.iloc[i].to_numpy()) for i in range(len(x))]
        return aggregate_interaction_values(ivs, aggregation="mean"), [
            self.backmap.get(c, c) for c in x.columns
        ]

    @staticmethod
    def _keep_top_k_interactions(
        iv: shapiq.InteractionValues, k: int
    ) -> shapiq.InteractionValues:
        """Return a deep copy of iv with all but the top-k order-2 interactions zeroed out."""
        
        filtered = copy.deepcopy(iv)
        lookup = filtered.interaction_lookup
        order2 = [(tup, lookup[tup]) for tup in lookup if len(tup) == 2]
        if len(order2) > k:
            order2.sort(key=lambda x: -abs(float(filtered.values[x[1]])))
            for _, idx in order2[k:]:
                filtered.values[idx] = 0.0
        return filtered

    def plot_shapiq_interaction_graph_comparison(
        self,
        X_eval_left: pd.DataFrame,
        X_eval_right: pd.DataFrame,
        out_png: str | Path,
        *,
        sample_n: int = 120,
        random_state: int = 42,
        align_from_training: pd.DataFrame | None = None,
        top_k_interactions: int = 15,
        max_features: int | None = 25,
    ) -> None:
        """Side-by-side shapiq interaction network plots for two cohorts."""
        x_right = (
            self.align_profile_samples_to_training(align_from_training, X_eval_right)
            if align_from_training is not None
            else X_eval_right
        )
        left_iv, labels = self._shapiq_global_interaction_values(
            X_eval_left, sample_n=sample_n, random_state=random_state, max_features=max_features
        )
        right_iv, _ = self._shapiq_global_interaction_values(
            x_right, sample_n=sample_n, random_state=random_state + 1, max_features=max_features
        )

        def _raster(iv, title: str) -> np.ndarray:
            filtered = self._keep_top_k_interactions(iv, top_k_interactions)
            fig, _ = network_plot(filtered, feature_names=labels, show=False)
            fig.suptitle(title, fontsize=12, y=0.98)
            arr = _fig_to_rgb_array(fig)
            plt.close(fig)
            return arr

        left_img = _raster(left_iv, f"Held-out cohort (n={min(sample_n, len(X_eval_left))})")
        right_img = _raster(right_iv, f"Profile cohort (n={min(sample_n, len(x_right))})")

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        axes[0].imshow(left_img)
        axes[0].axis("off")
        axes[1].imshow(right_img)
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def build_task_descriptor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        task_title: str,
        task_description: str,
        outcome_name: str,
    ) -> dict:
        """Structured summary of the prediction task and feature inventory for the LLM."""
        desc = [
            {
                "feature": self.backmap[s],
                "dtype": "numeric",
                **{k: float(v) for k, v in X[s].agg(["mean", "std", "min", "max"]).items()},
            }
            for s in self.safe_cols
        ]
        return {
            "task_title": task_title,
            "task_description": task_description,
            "outcome_name": outcome_name,
            "task_kind": self.task_kind,
            "n_samples": int(len(X)),
            "outcome_mean": float(y.mean()),
            "outcome_std": float(y.std()),
            "prevalence_positive": (float(y.mean()) if self._is_classification() else None),
            "n_features": len(self.safe_cols),
            "features": desc,
        }

    def compute_shap_matrix(
        self,
        X_tr: pd.DataFrame,
        x_eval: pd.DataFrame,
        bg_n: int = 300,
        use_background: bool = True,
    ) -> np.ndarray:
        """SHAP values for each row of x_eval.

        use_background=True -> interventional TreeSHAP with sampled background.
        use_background=False -> tree_path_dependent TreeSHAP (no background integration).
        """
        booster = self._booster()
        if use_background:
            bg = X_tr.sample(min(bg_n, len(X_tr)), random_state=42)
            explainer = shap.TreeExplainer(booster, data=bg, feature_perturbation="interventional")
        else:
            explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(x_eval, check_additivity=False)
        return self._coerce_shap_array(sv)

    def export_shapiq_interaction_edges_for_cohorts(
        self,
        X_default: pd.DataFrame,
        X_profiles: pd.DataFrame,
        *,
        out_default_edges_csv: str | Path | None = None,
        out_profiles_edges_csv: str | Path | None = None,
        out_compare_json: str | Path | None = None,
        write_edge_csvs: bool = False,
        sample_n: int = 40,
        random_state: int = 42,
        align_from_training: pd.DataFrame | None = None,
        max_features: int | None = 25,
    ) -> dict:
        """Pairwise shapiq interactions; optional full edge CSVs and/or a standalone summary JSON."""
        x_profiles = (
            self.align_profile_samples_to_training(align_from_training, X_profiles)
            if align_from_training is not None
            else X_profiles
        )
        mat_def, labels = self._shapiq_mean_abs_interactions(
            X_default, sample_n=sample_n, random_state=random_state, max_features=max_features
        )
        mat_pro, _ = self._shapiq_mean_abs_interactions(
            x_profiles, sample_n=sample_n, random_state=random_state + 1, max_features=max_features
        )

        edges_def = self._edges_from_interaction_matrix(mat_def, labels, "default")
        edges_pro = self._edges_from_interaction_matrix(mat_pro, labels, "profiles")
        if write_edge_csvs:
            if out_default_edges_csv is None or out_profiles_edges_csv is None:
                raise ValueError("write_edge_csvs requires out_default_edges_csv and out_profiles_edges_csv.")
            for path, df_edges in (
                (Path(out_default_edges_csv), edges_def),
                (Path(out_profiles_edges_csv), edges_pro),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                df_edges.to_csv(path, index=False)

        mass_def = mat_def.sum(axis=0)
        mass_pro = mat_pro.sum(axis=0)
        diff = mass_pro - mass_def
        top_up = np.argsort(diff)[::-1][:8]
        top_down = np.argsort(diff)[:8]
        comp = {
            "sample_n_per_cohort": int(min(sample_n, len(X_default), len(x_profiles))),
            "mean_abs_interaction_total_default": float(np.nansum(mat_def)),
            "mean_abs_interaction_total_profiles": float(np.nansum(mat_pro)),
            "top_features_more_interactive_in_profiles": [
                {"feature": labels[i], "delta_interaction_mass": float(diff[i])} for i in top_up
            ],
            "top_features_less_interactive_in_profiles": [
                {"feature": labels[i], "delta_interaction_mass": float(diff[i])} for i in top_down
            ],
            "top_edges_default": edges_def.head(10).to_dict(orient="records"),
            "top_edges_profiles": edges_pro.head(10).to_dict(orient="records"),
        }
        if out_compare_json is not None:
            out_p = Path(out_compare_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        return comp

    def build_interaction_empirical_bundle(
        self,
        X_tr: pd.DataFrame,
        syn_df: pd.DataFrame,
        shapiq_cohort_summary: dict,
        *,
        max_edges: int = 15,
        align_from_training: pd.DataFrame | None = None,
    ) -> dict:
        """Model-based interaction summaries on the synthetic cohort (median splits; diff-in-diff on preds).

        These statistics describe **associations under the fitted model** on profile data, not randomized
        causal identification. They ground the LLM causal discussion in quantitative contrasts.
        """
        ref = align_from_training if align_from_training is not None else X_tr
        x = self.align_profile_samples_to_training(ref, syn_df)
        if len(x) < 8:
            return {"n_rows": len(x), "edges": [], "note": "Too few synthetic rows for stable quadrants."}

        p = np.asarray(self._predict_target(x), dtype=float)
        inv_display = {v: k for k, v in self.backmap.items()}
        raw_edges = shapiq_cohort_summary.get("top_edges_default") or []
        seen: set[tuple[str, str]] = set()
        rows_out: list[dict] = []

        for edge in raw_edges[: max_edges * 2]:
            if not isinstance(edge, dict):
                continue
            fa = str(edge.get("feature_a") or "")
            fb = str(edge.get("feature_b") or "")
            key = tuple(sorted((fa, fb)))
            if not fa or not fb or key in seen:
                continue
            seen.add(key)

            sa, sb = inv_display.get(fa), inv_display.get(fb)
            if sa is None or sb is None or sa not in x.columns or sb not in x.columns:
                continue

            a = pd.to_numeric(x[sa], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(x[sb], errors="coerce").to_numpy(dtype=float)
            ma = float(np.nanmedian(a))
            mb = float(np.nanmedian(b))
            A_hi = a >= ma
            B_hi = b >= mb
            valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(p)

            def cell(mask: np.ndarray) -> float:
                m = mask & valid
                return float(np.mean(p[m])) if np.any(m) else float("nan")

            m_ll = cell(~A_hi & ~B_hi)
            m_lh = cell(~A_hi & B_hi)
            m_hl = cell(A_hi & ~B_hi)
            m_hh = cell(A_hi & B_hi)
            dod = (m_hh - m_hl) - (m_lh - m_ll)
            rows_out.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "model_evidence_strength_model_only": float(edge.get("mean_abs_interaction") or 0.0),
                    "n": int(len(x)),
                    "quadrant_mean_prediction": {
                        "a_low_b_low": m_ll,
                        "a_low_b_high": m_lh,
                        "a_high_b_low": m_hl,
                        "a_high_b_high": m_hh,
                    },
                    "diff_in_diff_prediction": float(dod),
                }
            )
            if len(rows_out) >= max_edges:
                break

        return {
            "n_rows": int(len(x)),
            "task_kind": self.task_kind,
            "edges": rows_out,
            "disclaimer": (
                "Median splits on synthetic profile rows; diff-in-diff is a summary of model predictions, "
                "not an RCT. Confounding and feedback remain possible."
            ),
        }

    def infer_causal_interaction_hypotheses(
        self,
        *,
        task_descriptor: dict,
        shapiq_cohort_summary: dict,
        empirical_interactions: dict,
        global_shap_top: list[dict] | None = None,
        targeted_profiles_digest: list[dict] | None = None,
        model: str = "mistral-small-2603",
    ) -> dict:
        """Use an LLM to translate shapiq edges + synthetic-cohort contrasts into *hypotheses* about causal structure.

        Output is explicitly non-definitive: the model proposes directional narratives, effect modifiers, and
        falsifiers.
        """
        schema = (
            "Return one JSON object with keys:\n"
            "- summary (string): 2-4 sentences on what interaction evidence does and does NOT imply.\n"
            "- causal_hypotheses (array, max 12), each:\n"
            "  - hypothesis_id (string)\n"
            "  - feature_a (string, display names as in evidence)\n"
            "  - feature_b (string)\n"
            "  - proposed_relation (string: e.g. 'effect_modification', 'synergy_on_risk', "
            "'buffering', 'common_cause_pattern', 'unknown')\n"
            "  - narrative (string): mechanistic/clinical voice, hedged; no certainty claims.\n"
            "  - direction_on_outcome (string: 'increases'|'decreases'|'ambiguous'|'not_applicable')\n"
            "  - tying_evidence (array of strings): cite shapiq edge rank, empirical diff-in-diff sign, SHAP rank.\n"
            "  - confounding_risks (array of strings)\n"
            "  - suggested_falsifiers (array of strings): experiments or data that could refute the hypothesis.\n"
            "  - confidence_llm (string: 'low'|'medium'|'high') — epistemic, not statistical p-value.\n"
        )
        payload = {
            "task_descriptor": task_descriptor,
            "shapiq_top_edges_default": shapiq_cohort_summary.get("top_edges_default"),
            "shapiq_top_edges_profiles": shapiq_cohort_summary.get("top_edges_profiles"),
            "shapiq_feature_deltas": {
                "more_interactive_in_profiles": shapiq_cohort_summary.get(
                    "top_features_more_interactive_in_profiles"
                ),
                "less_interactive_in_profiles": shapiq_cohort_summary.get(
                    "top_features_less_interactive_in_profiles"
                ),
            },
            "empirical_model_interactions_on_synthetic_rows": empirical_interactions,
            "global_shap_top": global_shap_top or [],
            "targeted_profiles_digest": targeted_profiles_digest or [],
        }
        user = schema + "\nEvidence JSON:\n" + json.dumps(payload, indent=2, default=str)
        return call_json(
            system=(
                "You are a careful clinical epidemiologist + ML interpreter. Tree/shapiq interactions and "
                "model prediction contrasts are **correlational under a learned predictor** and may reflect "
                "confounding, colliders, or label leakage. Never claim proven causality. Propose concise, "
                "testable hypotheses and name key biases. "
                "Return valid JSON only."
            ),
            user=user,
            model=model,
        )

    def elicit_pathophysiological_model(
        self,
        *,
        task_descriptor: dict,
        global_shap_top: list[dict],
        shapiq_top_edges: list[dict],
        model: str = "mistral-small-2603",
    ) -> dict:
        """Ask the LLM to synthesize SHAP and shapiq into a pathophysiological organ system model.
        
        This agentic step maps raw machine learning artifacts (features and statistical interactions) 
        into a domain-grounded representation of human organ systems and their clinical couplings.
        """
        schema = (
            "Return ONLY one JSON object representing a pathophysiological graph with the following keys:\n"
            "- organ_systems (array), each:\n"
            "  - system_name (string: e.g., 'Cardiovascular System', 'Renal System')\n"
            "  - mechanism (string: e.g., 'Hemodynamic Instability / Shock')\n"
            "  - features (array of strings: exact feature names from the inputs mapped to this system)\n"
            "- cross_system_interactions (array), each:\n"
            "  - source_system (string: must match a system_name)\n"
            "  - target_system (string: must match a system_name)\n"
            "  - clinical_coupling (string: e.g., 'Vasopressor-Device Interaction', 'Sepsis-Shock Coupling')\n"
            "  - underlying_shapiq_features (array of strings: the specific features driving this interaction)\n"
            "- outcome_node (string: e.g., 'ICU Mortality Risk (Survival vs. Death)')\n"
            "- graphviz_dot (string): complete Graphviz DOT for a directed graph named pathophysiology.\n"
            "  Requirements: valid `digraph pathophysiology { ... }`, no markdown fences, no comments outside DOT,\n"
            "  include all organ systems as nodes, include cross-system edges with labels, and include one outcome node.\n"
        )
        payload = {
            "task_descriptor": task_descriptor,
            "top_shap_features": global_shap_top[:15],
            "top_shapiq_interactions": shapiq_top_edges[:10],
        }
        user = schema + "\nContext Data:\n" + json.dumps(payload, indent=2, default=str)
        out = call_json(
            system=(
                "You are an expert critical care physician and AI agent. "
                "Your task is to contextualize machine learning artifacts (SHAP, shapiq) "
                "into a clinical pathophysiological model. Group features into human organ systems and "
                "describe their cross-system interactions based on the provided data. "
                "Also output a valid Graphviz DOT graph for this model. "
                + GRAPHVIZ_EXPLANATORY_MODEL_PROMPT + " "
                "Use only feature names present in the context data. Return valid JSON only."
            ),
            user=user,
            model=model,
        )
        # Ensure downstream code always has a DOT artifact, even if the LLM omits it.
        if not isinstance(out, dict):
            out = {}
        dot = out.get("graphviz_dot")
        if not isinstance(dot, str) or "digraph" not in dot:
            out["graphviz_dot"] = self._build_pathophysiology_dot(out)
        return out

    @staticmethod
    def _dot_safe(s: str) -> str:
        return re.sub(r'[^A-Za-z0-9_]+', "_", s.strip()).strip("_") or "node"

    def _build_pathophysiology_dot(self, pm: dict) -> str:
        """Deterministic DOT fallback when LLM output lacks graphviz_dot."""
        orgs = pm.get("organ_systems") if isinstance(pm, dict) else None
        ints = pm.get("cross_system_interactions") if isinstance(pm, dict) else None
        outcome = pm.get("outcome_node") if isinstance(pm, dict) else None
        orgs = orgs if isinstance(orgs, list) else []
        ints = ints if isinstance(ints, list) else []
        out_lbl = str(outcome or "Outcome")

        lines = [
            "digraph pathophysiology {",
            "  rankdir=LR;",
            '  graph [fontname="Helvetica"];',
            '  node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#ffe5cc", color="#c05621"];',
            '  edge [fontname="Helvetica", color="#7c3aed"];',
        ]

        name_to_id: dict[str, str] = {}
        for i, o in enumerate(orgs):
            if not isinstance(o, dict):
                continue
            n = str(o.get("system_name") or f"System {i+1}")
            mech = str(o.get("mechanism") or "")
            feats = o.get("features") or []
            feats_txt = ", ".join(str(f) for f in feats[:4]) if isinstance(feats, list) else ""
            label_parts = [n]
            if mech:
                label_parts.append(mech)
            if feats_txt:
                label_parts.append(f"features: {feats_txt}")
            node_id = f"sys_{self._dot_safe(n)}_{i}"
            name_to_id[n] = node_id
            lines.append(f'  {node_id} [label="{ " | ".join(label_parts).replace(chr(34), "") }"];')

        out_id = "outcome_node"
        lines.append(f'  {out_id} [label="{out_lbl.replace(chr(34), "")}", fillcolor="#e6ffed", color="#2f855a"];')

        for e in ints:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source_system") or "")
            tgt = str(e.get("target_system") or "")
            lbl = str(e.get("clinical_coupling") or "")
            src_id = name_to_id.get(src)
            tgt_id = name_to_id.get(tgt)
            if src_id and tgt_id:
                if lbl:
                    lines.append(f'  {src_id} -> {tgt_id} [label="{lbl.replace(chr(34), "")}"];')
                else:
                    lines.append(f"  {src_id} -> {tgt_id};")

        for n_id in name_to_id.values():
            lines.append(f"  {n_id} -> {out_id} [color=\"#15803d\"];")
        lines.append("}")
        return "\n".join(lines)

    def build_prediction_audit_bundle(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        y_te: pd.Series,
        *,
        max_eval_rows: int = 600,
        samples_per_stratum: int = 2,
    ) -> dict:
        """Stratified held-out rows: classification (correct/incorrect) or regression (error strata)."""
        n = min(max_eval_rows, len(X_te))
        x_eval = X_te.iloc[:n].copy()
        y_sub = y_te.iloc[:n].to_numpy(dtype=float)
        shap_mat = self.compute_shap_matrix(X_tr, x_eval)
        is_cls = hasattr(self.model, "predict_proba")
        if is_cls:
            p = self.model.predict_proba(x_eval)[:, 1]
            pred_c = (p >= 0.5).astype(int)
            correct = pred_c == y_sub.astype(int)
            conf = np.maximum(p, 1.0 - p)

            def pick_mask(mask: np.ndarray, k: int) -> list[int]:
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return []
                sub = idx[np.argsort(conf[idx])[::-1][:k]]
                return [int(i) for i in sub]

            strata = {
                "correct_high_risk": pick_mask(correct & (y_sub == 1), samples_per_stratum),
                "correct_low_risk": pick_mask(correct & (y_sub == 0), samples_per_stratum),
                "false_positive": pick_mask((~correct) & (y_sub == 0), samples_per_stratum),
                "false_negative": pick_mask((~correct) & (y_sub == 1), samples_per_stratum),
                "uncertain_correct": pick_mask(correct & (np.abs(p - 0.5) < 0.15), samples_per_stratum),
            }
        else:
            p = self.model.predict(x_eval)
            correct = None
            err = np.abs(p - y_sub)
            q1 = float(np.quantile(err, 0.25))
            q3 = float(np.quantile(err, 0.75))
            idx = np.arange(len(err))
            hi = idx[err >= q3]
            lo = idx[err <= q1]
            strata = {
                "low_error": [int(i) for i in lo[:samples_per_stratum]],
                "high_error": [int(i) for i in hi[:samples_per_stratum]],
            }

        audits = []
        seen = set()
        for bucket, indices in strata.items():
            for i in indices:
                if i in seen:
                    continue
                seen.add(i)
                row_shap = shap_mat[i]
                top_idx = np.argsort(np.abs(row_shap))[::-1][:8]
                feats = [
                    {
                        "feature": self.backmap[x_eval.columns[j]],
                        "value": float(x_eval.iloc[i, j]),
                        "shap": float(row_shap[j]),
                    }
                    for j in top_idx
                ]
                pred_i = float(p[i])
                audits.append(
                    {
                        "stratum": bucket,
                        "row_index_in_eval_slice": i,
                        "y_true": float(y_sub[i]),
                        "prediction": pred_i,
                        "error_abs": float(abs(pred_i - y_sub[i])),
                        "correct": (bool(correct[i]) if correct is not None else None),
                        "top_local_shap": feats,
                    }
                )

        return {
            "n_eval_rows": n,
            "accuracy_slice": (float(correct.mean()) if correct is not None else None),
            "stratified_audits": audits,
        }

    def generate_alignment_and_rules(
        self,
        *,
        task_descriptor: dict,
        global_shap_bundle: dict,
        prediction_audit: dict,
        training_samples: list[dict] | None = None,
        model: str = "mistral-small-2603",
    ) -> dict:
        """Return targeted synthetic profiles as JSON (bias/function based)."""
        schema_hint = (
            "Return a single JSON object with keys:\n"
            "- targeted_profiles (array, length 10), each:\n"
            "  - profile_id (string)\n"
            "  - title (string)\n"
            "  - clinical_story (string)\n"
            "  - why_this_profile (string)\n"
            "  - bias_functions (array): each is an object with keys:\n"
            "    - feature (string)\n"
            "    - kind (string, one of: 'shift', 'scale', 'affine', 'set_prevalence')\n"
            "    - params (object):\n"
            "      - for shift: {delta}\n"
            "      - for scale: {mult}\n"
            "      - for affine: {mult, delta}\n"
            "      - for set_prevalence (binary features only): {target_rate}\n"
            "    - notes (string)\n"
            "  - hard_constraints (array of strings)\n"
            "  - sampling_notes (string)\n"
            "  - n_samples (int, must be 100)\n"
        )
        payload = {
            "task_descriptor": task_descriptor,
            "global_shap_top": global_shap_bundle.get("top_contributions"),
            "prediction_stratified_audit": prediction_audit,
            "training_samples": training_samples or [],
        }
        user = schema_hint + "\nEvidence JSON:\n" + json.dumps(payload, indent=2)
        return call_json(
            system=(
                "You are generating realistic ICU patient profiles by applying small, interpretable bias functions "
                "to real training rows. You MUST keep values within plausible ranges implied by the training samples "
                "and dataset description. Avoid extreme shifts. Prefer small affine transforms (e.g., age +10, BMI *1.1). "
                "For binary features, use set_prevalence to modestly increase/decrease rates. Return a valid JSON object."
            ),
            user=user,
            model=model,
        )

    def build_training_samples_for_llm(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        *,
        n: int = 8,
        random_state: int = 42,
    ) -> list[dict]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_tr), size=min(n, len(X_tr)), replace=False)
        out: list[dict] = []
        for i in idx:
            row = X_tr.iloc[int(i)]
            out.append(
                {
                    "row_index": int(i),
                    "y": int(y_tr.iloc[int(i)]),
                    "features": {self.backmap.get(c, c): float(row[c]) for c in X_tr.columns},
                }
            )
        return out

    def generate_synthetic_rows_from_profiles(
        self,
        X_tr: pd.DataFrame,
        profiles_json: dict,
        *,
        random_state: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(random_state)
        profiles = list(profiles_json.get("targeted_profiles") or [])
        disp_to_safe = {self.backmap.get(c, c): c for c in X_tr.columns}
        binary_safe = {
            c for c in X_tr.columns if set(pd.unique(X_tr[c].dropna())).issubset({0, 1})
        }
        frames: list[pd.DataFrame] = []
        for prof in profiles:
            if not isinstance(prof, dict):
                continue
            pid = str(prof.get("profile_id") or "")
            title = str(prof.get("title") or "")
            n_samples = int(prof.get("n_samples") or 100)

            base_idx = rng.choice(len(X_tr), size=n_samples, replace=True)
            base = X_tr.iloc[base_idx].copy()

            bias_fns = prof.get("bias_functions") or []
            if isinstance(bias_fns, list):
                for fn in bias_fns:
                    if not isinstance(fn, dict):
                        continue
                    disp = fn.get("feature")
                    safe = disp_to_safe.get(str(disp))
                    if safe is None:
                        continue
                    kind = str(fn.get("kind") or "")
                    params = fn.get("params") or {}
                    if not isinstance(params, dict):
                        params = {}
                    if kind == "shift":
                        delta = float(params.get("delta") or 0.0)
                        base[safe] = base[safe].astype(float) + delta
                    elif kind == "scale":
                        mult = float(params.get("mult") or 1.0)
                        base[safe] = base[safe].astype(float) * mult
                    elif kind == "affine":
                        mult = float(params.get("mult") or 1.0)
                        delta = float(params.get("delta") or 0.0)
                        base[safe] = base[safe].astype(float) * mult + delta
                    elif kind == "set_prevalence":
                        if safe not in binary_safe:
                            continue
                        target = float(params.get("target_rate") or 0.5)
                        target = max(0.0, min(1.0, target))
                        base[safe] = (rng.rand(n_samples) < target).astype(float)

            base.insert(0, "profile_id", pid)
            base.insert(1, "profile_title", title)
            frames.append(base)

        return pd.concat(frames, axis=0, ignore_index=True)
