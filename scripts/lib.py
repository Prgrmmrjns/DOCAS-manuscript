from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from model import make_model
from scoring import CombinedScorer
from synthesis import OptunaTPESyntheticOptimizer, uniform_bounds_from_training

class SCMRule(TypedDict):
    start: str
    end: str
    edge: float


@dataclass(frozen=True)
class AugContext:
    X_train: pd.DataFrame
    y_train: pd.Series
    cols: list[str]
    rules: list[dict[str, Any]]
    target: str
    rng: np.random.Generator


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64, order="C").ravel()
    y = np.asarray(y, dtype=np.float64, order="C").ravel()
    if x.size < 2 or x.size != y.size:
        return float("nan")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


class iSHAP:
    @staticmethod
    def optimize_synthetic_points(
        ctx: AugContext,
        fitness_fn: Callable[[pd.DataFrame, pd.Series], tuple[float, float]],
        *,
        n_trials: int = 100,
        n_points: int = 10,
        objective_metric_weight: float = 0.5,
        metric_scale: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Optimize synthetic points via pluggable optimizer backend."""
        space = uniform_bounds_from_training(ctx.X_train, ctx.y_train, ctx.cols)
        scorer = CombinedScorer(metric_weight=objective_metric_weight, metric_anchor=metric_scale)
        optimizer = OptunaTPESyntheticOptimizer()
        X_best, y_best = optimizer.optimize(
            space=space,
            n_points=n_points,
            n_trials=n_trials,
            rng=ctx.rng,
            scorer=scorer,
            fitness_fn=fitness_fn,
        )
        return X_best, y_best.rename(ctx.y_train.name)

    @staticmethod
    def compact_rule_deltas(before_rules: list[dict[str, Any]], after_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        B = {str(r.get("rule")): r for r in before_rules}
        A = {str(r.get("rule")): r for r in after_rules}
        out: list[dict[str, Any]] = []
        for key in sorted(set(B) | set(A)):
            b, a = B.get(key, {}), A.get(key, {})
            br, ar_ = b.get("pearson_r"), a.get("pearson_r")
            out.append(
                {
                    "rule": key,
                    "weight": a.get("weight", b.get("weight")),
                    "ideal_pearson": a.get("ideal_pearson", b.get("ideal_pearson")),
                    "pearson_r_before": br,
                    "pearson_r_after": ar_,
                    "pearson_r_delta_after_minus_before": None
                    if br is None or ar_ is None
                    else round(float(ar_) - float(br), 4),
                }
            )
        return out

    @staticmethod
    def model_report(
        model: Any,
        X_df: pd.DataFrame,
        cols: list[str],
        rules: list[dict[str, Any]],
        target: str,
        *,
        random_state: int = 42,
        max_samples: int = 200,
    ) -> dict[str, Any]:
        t = str(target)
        col_idx = {c: i for i, c in enumerate(cols)}
        n = min(int(max_samples), len(X_df))
        if len(X_df) > n:
            rng = np.random.default_rng(int(random_state))
            idx = rng.choice(len(X_df), size=n, replace=False)
            Xs = X_df.iloc[idx].copy().reset_index(drop=True)
        else:
            Xs = X_df.reset_index(drop=True)
        ex = shap.TreeExplainer(model)
        try:
            sv = ex.shap_values(Xs, check_additivity=False)
        except TypeError:
            sv = ex.shap_values(Xs)
        if isinstance(sv, list):
            sv = np.asarray(sv[0])
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 1:
            sv = sv.reshape(len(Xs), -1)
        sv = np.asarray(sv, dtype=np.float64)
        items: list[dict[str, Any]] = []
        w_sum = dev_sum = 0.0
        for r in rules:
            edge = float(r["edge"])
            u, v = str(r["start"]), str(r["end"])
            edge_mag = abs(edge)
            w = (edge_mag if edge_mag > 1e-12 else 1.0) * (0.55 if v != t else 1.0)
            ideal = 0.0 if edge_mag <= 1e-12 else (1.0 if edge > 0 else -1.0)
            if v == t:
                if u not in col_idx:
                    continue
                rr = pearson_r(sv[:, col_idx[u]], Xs[u].to_numpy(np.float64, copy=False))
                key = f"{u}->{t}"
            else:
                if u not in col_idx or v not in col_idx:
                    continue
                rr = pearson_r(Xs[u].to_numpy(np.float64, copy=False), sv[:, col_idx[v]])
                key = f"{u}->{v}"
            if not np.isfinite(rr):
                dev, ro, diff = 2.0, None, None
            else:
                ro = float(rr)
                dev = abs(ro - ideal)
                diff = round(ro - ideal, 4)
            w_sum += w
            dev_sum += w * dev
            items.append(
                {
                    "rule": key,
                    "weight": round(w, 4),
                    "pearson_r": None if ro is None else round(ro, 4),
                    "ideal_pearson": ideal,
                    "pearson_r_diff": diff,
                }
            )
        feasibility = 0.5 if w_sum < 1e-18 else float(np.clip(1.0 - dev_sum / w_sum / 2.0, 0.0, 1.0))
        return {"feasibility_score_0_1": round(feasibility, 4), "rules": items}

    @staticmethod
    def regression_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))
        
    @staticmethod
    def run_pipeline(
        X: pd.DataFrame,
        y: pd.Series,
        rules: list[dict[str, Any]],
        target: str,
        root: str,
        name: str,
        *,
        task: str = "regression",
        random_state: int = 42,
        test_size: float = 0.25,
        synth_points_per_round: int = 10,
        n_trials: int = 1000,
        objective_metric_weight: float = 0.5,
    ) -> dict[str, Any]:
        cols = list(X.columns)
        t = str(target)

        X_all = X[cols].to_numpy(dtype=np.float64)
        y_all = y.to_numpy(dtype=np.float64)
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=test_size, random_state=random_state, shuffle=True,
        )
        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        X_val_df = pd.DataFrame(X_val, columns=cols)
        is_classification = str(task).lower() == "classification"
        classes = np.unique(y_train) if is_classification else np.asarray([], dtype=np.float64)

        def _metric_and_loss(m: Any) -> tuple[float, float]:
            pred = np.asarray(m.predict(X_val), dtype=np.float64)
            if is_classification:
                d = np.abs(pred[:, None] - classes[None, :])
                pred_cls = classes[np.argmin(d, axis=1)]
                acc = float(accuracy_score(y_val, pred_cls))
                return acc, 1.0 - acc
            error = iSHAP.regression_error(y_val, pred)
            return error, error

        feas_max_samples = 200
        feas_n = min(feas_max_samples, len(X_val_df))
        if len(X_val_df) > feas_n:
            _rng_feas = np.random.default_rng(int(random_state))
            _idx_feas = _rng_feas.choice(len(X_val_df), size=feas_n, replace=False)
            X_feas_df = X_val_df.iloc[_idx_feas].copy().reset_index(drop=True)
        else:
            X_feas_df = X_val_df.reset_index(drop=True)

        def _report(m: Any) -> tuple[dict[str, Any], float]:
            rep = iSHAP.model_report(
                m,
                X_feas_df,
                cols,
                rules,
                t,
                random_state=random_state,
                max_samples=feas_max_samples,
            )
            return rep, float(rep.get("feasibility_score_0_1") or 0.0)

        model = make_model(random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        feas_before, score_before = _report(model)
        metric_before, loss_before = _metric_and_loss(model)

        scorer = CombinedScorer(metric_weight=objective_metric_weight, metric_anchor=loss_before)
        def _combined(loss_v: float, feas_v: float) -> float:
            return scorer.score(loss_v, feas_v)
        metric_name = "accuracy" if is_classification else "error"
        print(
            f"initial {metric_name}={metric_before:.6f} feasibility={score_before:.4f} "
            f"combined={_combined(loss_before, score_before):.6f}"
        )

        rng = np.random.default_rng(random_state)
        X_aug = X_train.copy()
        y_aug = y_train.copy()
        feasibility_per_added_datapoint: list[dict[str, float | int]] = []
        n_added = 0
        best_metric = float(metric_before)
        best_loss = float(loss_before)
        best_score = float(score_before)
        best_combined = _combined(best_loss, best_score)

        while True:
            def _fitness_round(X_s: pd.DataFrame, y_s: pd.Series) -> tuple[float, float]:
                X_one = X_s[cols].to_numpy(dtype=np.float64)
                y_one = y_s.to_numpy(dtype=np.float64)
                X_fit = np.vstack([X_aug, X_one])
                y_fit = np.concatenate([y_aug, y_one])
                m = make_model(random_state=random_state, n_jobs=1)
                m.fit(X_fit, y_fit)
                _, feas = _report(m)
                _, loss = _metric_and_loss(m)
                return loss, feas

            ctx = AugContext(
                X_train=pd.DataFrame(X_aug, columns=cols),
                y_train=pd.Series(y_aug, name=y.name),
                cols=cols,
                rules=rules,
                target=t,
                rng=rng,
            )
            X_one_df, y_one_s = iSHAP.optimize_synthetic_points(
                ctx,
                _fitness_round,
                n_trials=n_trials,
                n_points=synth_points_per_round,
                objective_metric_weight=objective_metric_weight,
                metric_scale=scorer.metric_anchor,
            )
            X_one = X_one_df[cols].to_numpy(dtype=np.float64)
            y_one = y_one_s.to_numpy(dtype=np.float64)

            X_try = np.vstack([X_aug, X_one])
            y_try = np.concatenate([y_aug, y_one])

            m_step = make_model(random_state=random_state, n_jobs=-1)
            m_step.fit(X_try, y_try)
            _, step_score = _report(m_step)
            step_metric, step_loss = _metric_and_loss(m_step)
            step_combined = _combined(step_loss, step_score)

            if step_combined > best_combined + 1e-9:
                X_aug = X_try
                y_aug = y_try
                n_added += len(X_one)
                best_metric = float(step_metric)
                best_loss = float(step_loss)
                best_score = float(step_score)
                best_combined = float(step_combined)
                row = {
                    "n_added_synth_rows": int(n_added),
                    "metric_holdout": round(step_metric, 6),
                    "feasibility_score_0_1": round(step_score, 4),
                    "combined_score": round(step_combined, 6),
                }
                feasibility_per_added_datapoint.append(row)
                print(
                    f"added={row['n_added_synth_rows']} {metric_name}={row['metric_holdout']:.6f} "
                    f"feasibility={row['feasibility_score_0_1']:.4f} "
                    f"combined={row['combined_score']:.6f}"
                )
            else:
                print(
                    f"reject {metric_name}={step_metric:.6f} feasibility={step_score:.4f} "
                    f"combined={step_combined:.6f} best_combined={best_combined:.6f}"
                )
                print(
                    f"stop best_{metric_name}={best_metric:.6f} best_feasibility={best_score:.4f} "
                    f"best_combined={best_combined:.6f}"
                )
                break

        model_synth = make_model(random_state=random_state, n_jobs=-1)
        model_synth.fit(X_aug, y_aug)
        X_eval_synth = np.vstack([X_aug, X_val])
        feas_after, score_after = _report(model_synth)
        metric_after, _ = _metric_and_loss(model_synth)

        X_eval_df = pd.DataFrame(np.vstack([X_train, X_val]), columns=cols)
        X_eval_synth_df = pd.DataFrame(X_eval_synth, columns=cols)

        results_dir = os.path.join(root, "results", name)
        os.makedirs(results_dir, exist_ok=True)

        payload = {
            "feasibility": {
                "before": round(score_before, 4),
                "after": round(score_after, 4),
                "delta_after_minus_before": round(score_after - score_before, 4),
            },
            "feasibility_per_added_datapoint": feasibility_per_added_datapoint,
            "rule_components": iSHAP.compact_rule_deltas(
                list(feas_before.get("rules", [])),
                list(feas_after.get("rules", [])),
            ),
            "holdout_metric": {
                "metric_name": metric_name,
                "n_synthetic_rows_added": int(n_added),
                "metric_holdout_before_synthetic": round(metric_before, 6),
                "metric_holdout_after_synthetic": round(metric_after, 6),
                "delta_metric_holdout": round(metric_after - metric_before, 6),
            },
        }
        with open(os.path.join(results_dir, f"{name}_feasibility.json"), "w") as fp:
            json.dump(payload, fp, indent=2)

        return {
            "cols": cols,
            "target": t,
            "model_before": model,
            "model_after": model_synth,
            "X_eval_before": X_eval_df,
            "X_eval_after": X_eval_synth_df,
            "X_val": X_val_df,
            "results_dir": results_dir,
        }
