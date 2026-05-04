from __future__ import annotations

import importlib
import json
import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lib import (
    domain_target_pearson_objective,
    project_root,
    scm_structural_parents,
    scm_topological_order,
    tree_shap_mean_abs_weights,
    tree_shap_values_from_df,
)
from visuals import (
    save_interaction_cf_scatter_before_after_figure,
    save_shap_beeswarm_before_after_figure,
)

DATASETS: list[str] = ["d1namo"]

random_state, test_size, val_frac = 42, 0.25, 0.25
max_outer_iters = 10
n_trials = 300
show_progress_bar = True
objective_pearson_weight = 0.5
LGB_PARAMS = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.3, "verbosity": -1, "n_jobs": 1}
EARLY_STOPPING_ROUNDS = 10
SYNTH_BATCH_SIZE = 64


@dataclass
class SynthParams:
    shap_struct_weight_power: float = 0.62
    shap_leaf_noise_base: float = 0.09
    shap_y_diversify: float = 0.24
    shap_synth_guide_samples: int = 512
    y_label_head_anchor: float = 0.08
    steer_rule_feature: str | None = None
    steer_feature_high_prob: float = 0.0
    steer_feature_x_low: float = 0.86
    steer_y_pull: float = 0.48
    steer_insulin_z_scale: float = 1.65


def default_synth_params(dm: Any) -> SynthParams:
    # d1namo: tuned by scripts/grid_synth_d1namo.py (grid_score on insulin SHAP tail + Pearson − RMSE pen)
    if getattr(dm, "NAME", "") == "d1namo":
        return SynthParams(
            steer_rule_feature="insulin",
            steer_feature_high_prob=0.32,
            steer_feature_x_low=0.87,
            steer_y_pull=0.45,
            steer_insulin_z_scale=1.65,
        )
    return SynthParams()


def composite_score(rmse_v: float, pearson_01: float, *, rmse_ref: float) -> float:
    w = min(1.0, max(0.0, float(objective_pearson_weight)))
    d = max(float(rmse_ref), 1e-9)
    rmse_q = float(np.clip(0.5 + 0.5 * (d - float(rmse_v)) / d, 0.0, 1.0))
    p_q = float(np.clip(pearson_01, 0.0, 1.0))
    return float(np.clip((1.0 - w) * rmse_q + w * p_q, 0.0, 1.0))


def run_dataset(
    dm: Any,
    synth_params: SynthParams | None = None,
    *,
    max_outer_iters_override: int | None = None,
    n_trials_override: int | None = None,
    skip_outputs: bool = False,
    uniform_shap_weights: bool = False,
    random_state_override: int | None = None,
) -> dict[str, Any]:
    sp = synth_params if synth_params is not None else default_synth_params(dm)
    rs = int(random_state_override) if random_state_override is not None else random_state
    max_iters = int(max_outer_iters if max_outer_iters_override is None else max_outer_iters_override)
    n_trials_local = int(n_trials if n_trials_override is None else n_trials_override)
    RULES = list(dm.SCM_RULES)
    X, y = dm.load(project_root(__file__))
    X_train_pool, X_test_df, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs,
        stratify=y if getattr(dm, "TASK", "regression") == "classification" else None,
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=val_frac, random_state=rs, shuffle=True,
        stratify=y_train_pool if getattr(dm, "TASK", "regression") == "classification" else None,
    )

    cols = list(X_train_df.columns)
    d_features = len(cols)
    sc_x = MinMaxScaler((0.0, 1.0)).fit(X_train_df)
    sc_y = MinMaxScaler((0.0, 1.0)).fit(y_train.to_numpy(float).reshape(-1, 1))
    X_train = sc_x.transform(X_train_df).astype(np.float32, copy=False)
    X_val = sc_x.transform(X_val_df).astype(np.float32, copy=False)
    X_test = sc_x.transform(X_test_df).astype(np.float32, copy=False)
    y_train_s = sc_y.transform(y_train.to_numpy(float).reshape(-1, 1)).ravel().astype(np.float32, copy=False)
    y_val_s = sc_y.transform(y_val.to_numpy(float).reshape(-1, 1)).ravel().astype(np.float32, copy=False)
    y_val_orig, y_test_orig = y_val.to_numpy(np.float64, copy=False), y_test.to_numpy(np.float64, copy=False)
    X_val_df = pd.DataFrame(X_val, columns=cols)
    SHAP_PEARSON_MAX_SAMPLES = min(200, len(X_val_df))

    scm_order = scm_topological_order(tuple(cols), RULES, target=dm.TARGET)
    scm_parents = scm_structural_parents(RULES, target=dm.TARGET)

    def _domain_pearson(model: Any, Xdf: pd.DataFrame, max_samples: int | None) -> tuple[float, dict[str, Any]]:
        cap = SHAP_PEARSON_MAX_SAMPLES if max_samples is None else min(int(max_samples), len(Xdf))
        return domain_target_pearson_objective(
            model, Xdf, cols, RULES, dm.TARGET, max_samples=cap, random_state=rs,
        )

    def shap_pearson_report(model: Any, Xdf: pd.DataFrame) -> dict[str, Any]:
        s01, diag = _domain_pearson(model, Xdf, None)
        return {"pearson_objective_0_1": s01, **diag}

    def fit_scm_ols(X_base: np.ndarray, col_list: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        col_idx = {c: i for i, c in enumerate(col_list)}
        out: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for name in scm_order:
            if name not in col_idx:
                continue
            parents = tuple(p for p in scm_parents.get(name, ()) if p in col_idx)
            if not parents:
                continue
            pi = np.array([col_idx[p] for p in parents], dtype=int)
            yc, xp = X_base[:, col_idx[name]], X_base[:, pi]
            lr = LinearRegression().fit(xp, yc)
            out[name] = (pi.astype(np.int64), lr.coef_.astype(np.float32), float(lr.intercept_))
        return out

    def scm_struct_residual_scales(
        X_base: np.ndarray, col_list: list[str], scm_ols: dict[str, tuple[np.ndarray, np.ndarray, float]],
    ) -> dict[str, float]:
        col_idx = {c: i for i, c in enumerate(col_list)}
        out: dict[str, float] = {}
        for name, (pi, coef, icept) in scm_ols.items():
            j = col_idx.get(name)
            if j is None or len(X_base) < 3:
                continue
            pred, yc = X_base[:, pi] @ coef + icept, X_base[:, j]
            out[name] = max(float(np.std(yc - pred)), 0.02)
        return out

    def fit_target_head(X_base: np.ndarray, y_scaled: np.ndarray) -> tuple[LinearRegression, float]:
        lr = LinearRegression().fit(X_base, y_scaled)
        resid = y_scaled.astype(np.float64, copy=False) - lr.predict(X_base)
        std = float(np.std(resid)) if len(resid) > 1 else 0.1
        return lr, float(np.clip(2.5 * std, 0.04, 0.38))

    def rule_push_y(row: np.ndarray, col_list: list[str]) -> float:
        col_idx = {c: i for i, c in enumerate(col_list)}
        s = 0.0
        for r in RULES:
            if str(r["end"]) != dm.TARGET:
                continue
            u = str(r["start"])
            if u not in col_idx:
                continue
            s += float(r["edge"]) * (float(row[col_idx[u]]) - 0.5)
        return s

    def _scm_struct_z(trial_id: int, j: int, name: str, salt: int = 0) -> float:
        key = (
            trial_id * 1_000_003 + int(salt) * 50_021 + j * 524_287 + sum((i + 1) * ord(c) for i, c in enumerate(name))
        ) % (2**31 - 1)
        return float(np.random.default_rng(int(key)).standard_normal())

    def _scm_fill_row(
        X_base: np.ndarray,
        col_idx: dict[str, int],
        scm_ols: dict[str, tuple[np.ndarray, np.ndarray, float]],
        struct_scales: dict[str, float],
        tid: int,
        row_salt: int,
        noise: float,
        shap_w: np.ndarray | None,
    ) -> np.ndarray:
        n = len(X_base)
        row = np.zeros(d_features, dtype=np.float32)
        for name in scm_order:
            if name not in col_idx:
                continue
            j = col_idx[name]
            parents = scm_parents.get(name, ())
            ix = (tid + j * 31 + row_salt) % n
            wj = float(shap_w[j]) ** sp.shap_struct_weight_power if shap_w is not None else 1.0
            z = _scm_struct_z(tid, j, name, salt=int(row_salt))
            if sp.steer_rule_feature and name == sp.steer_rule_feature:
                z *= float(sp.steer_insulin_z_scale)
            if not parents or not all(p in col_idx for p in parents) or name not in scm_ols:
                base = float(X_base[ix, j])
                if shap_w is None:
                    row[j] = np.float32(base)
                else:
                    leaf_n = noise * sp.shap_leaf_noise_base * float(np.sqrt(float(shap_w[j]))) * z
                    row[j] = np.float32(np.clip(base + leaf_n, 0.0, 1.0))
                continue
            pi, coef, icept = scm_ols[name]
            pred = float(np.dot(coef, row[pi]) + icept)
            sig = float(struct_scales.get(name, 0.05))
            row[j] = np.float32(np.clip(pred + noise * sig * z * wj, 0.0, 1.0))
        return row

    def _y_shap_diversify(row: np.ndarray, tid: int, row_salt: int, shap_w: np.ndarray | None) -> float:
        if shap_w is None or shap_w.shape[0] != d_features:
            return 0.0
        rng_y = np.random.default_rng(int(tid) * 7919 + int(row_salt) * 13 + 1)
        zy = rng_y.standard_normal(d_features)
        center = row.astype(np.float64, copy=False) - 0.5
        return float(sp.shap_y_diversify * float(np.dot(shap_w.astype(np.float64), zy * center)))

    def steer_row_if_needed(row_1d: np.ndarray, col_idx: dict[str, int], tid: int, rsalt: int) -> tuple[np.ndarray, bool]:
        if not sp.steer_rule_feature or sp.steer_feature_high_prob <= 0.0:
            return row_1d, False
        j = col_idx.get(sp.steer_rule_feature)
        if j is None:
            return row_1d, False
        rng_s = np.random.default_rng(int(tid) * 500_009 + int(rsalt) * 17 + 3)
        if float(rng_s.random()) >= float(sp.steer_feature_high_prob):
            return row_1d, False
        out = row_1d.astype(np.float32, copy=True)
        x_hi = float(rng_s.uniform(float(sp.steer_feature_x_low), 0.998))
        out[j] = np.float32(max(float(out[j]), x_hi))
        return out, True

    def scm_sample_row(
        X_base: np.ndarray,
        col_list: list[str],
        scm_ols: dict[str, tuple[np.ndarray, np.ndarray, float]],
        struct_scales: dict[str, float],
        y_head: LinearRegression,
        y_delta_hi: float,
        trial: optuna.Trial,
        shap_w: np.ndarray | None,
    ) -> tuple[np.ndarray, np.float32]:
        col_idx = {c: i for i, c in enumerate(col_list)}
        tid = int(trial.number)
        noise = trial.suggest_float("scm_noise", 0.0, 0.14)
        row = _scm_fill_row(X_base, col_idx, scm_ols, struct_scales, tid, 0, noise, shap_w).ravel()
        row, steered = steer_row_if_needed(row, col_idx, tid, 0)
        y_hat = float(y_head.predict(row.reshape(1, -1))[0])
        y_delta = trial.suggest_float("y_delta", -y_delta_hi, y_delta_hi)
        rule_push = trial.suggest_float("rule_push", 0.12, 1.0)
        y_push = rule_push * rule_push_y(row, col_list)
        y_raw = y_hat + y_delta + y_push + _y_shap_diversify(row, tid, 0, shap_w)
        if steered and sp.steer_rule_feature:
            ji = col_idx.get(sp.steer_rule_feature)
            if ji is not None:
                y_raw -= float(sp.steer_y_pull) * max(0.0, float(row[ji]) - 0.5)
        return row.reshape(1, -1), np.float32(np.clip(y_raw, 0.0, 1.0))

    def scm_from_bp(
        X_base: np.ndarray,
        col_list: list[str],
        scm_ols: dict[str, tuple[np.ndarray, np.ndarray, float]],
        struct_scales: dict[str, float],
        y_head: LinearRegression,
        y_delta_hi: float,
        bp: dict[str, Any],
        trial_number: int,
        *,
        row_salt: int = 0,
        noise_scale: float = 1.0,
        y_delta_jitter: float = 0.0,
        rule_push_scale: float = 1.0,
        y_head_anchor: float = 0.0,
        shap_w: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.float32]:
        tid = int(trial_number)
        col_idx = {c: i for i, c in enumerate(col_list)}
        noise = float(np.clip(float(bp["scm_noise"]) * float(noise_scale), 0.0, 0.22))
        y_delta = float(np.clip(float(bp["y_delta"]) + float(y_delta_jitter), -y_delta_hi, y_delta_hi))
        row = _scm_fill_row(X_base, col_idx, scm_ols, struct_scales, tid, int(row_salt), noise, shap_w).ravel()
        row, steered = steer_row_if_needed(row, col_idx, tid, int(row_salt))
        y_hat = float(y_head.predict(row.reshape(1, -1))[0])
        rule_push = float(np.clip(float(bp.get("rule_push", 0.0)) * float(rule_push_scale), 0.12, 1.0))
        y_raw = y_hat + y_delta + rule_push * rule_push_y(row, col_list) + _y_shap_diversify(row, tid, int(row_salt), shap_w)
        if steered and sp.steer_rule_feature:
            ji = col_idx.get(sp.steer_rule_feature)
            if ji is not None:
                y_raw -= float(sp.steer_y_pull) * max(0.0, float(row[ji]) - 0.5)
        a = float(np.clip(y_head_anchor, 0.0, 0.45))
        y_mid = (1.0 - a) * y_raw + a * y_hat
        return row.reshape(1, -1), np.float32(np.clip(y_mid, 0.0, 1.0))

    def scm_batch_from_bp(
        X_base: np.ndarray,
        col_list: list[str],
        scm_ols: dict[str, tuple[np.ndarray, np.ndarray, float]],
        struct_scales: dict[str, float],
        y_head: LinearRegression,
        y_delta_hi: float,
        bp: dict[str, Any],
        trial_number: int,
        batch_size: int,
        rng: np.random.Generator,
        shap_w: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        b = int(batch_size)
        out_x, out_y = np.empty((b, d_features), dtype=np.float32), np.empty(b, dtype=np.float32)
        for k in range(b):
            salt = k * 1_009 + 17 * (k % 7)
            x_row, y_s = scm_from_bp(
                X_base, col_list, scm_ols, struct_scales, y_head, y_delta_hi, bp, trial_number,
                row_salt=salt,
                noise_scale=float(rng.uniform(0.42, 1.08)),
                y_delta_jitter=float(rng.normal(0.0, 0.22 * y_delta_hi)),
                rule_push_scale=float(rng.uniform(0.62, 1.22)),
                y_head_anchor=sp.y_label_head_anchor,
                shap_w=shap_w,
            )
            out_x[k], out_y[k] = x_row.ravel(), float(y_s)
        return out_x, out_y

    def _train_lgb(X_fit: np.ndarray, y_fit: np.ndarray) -> Any:
        m = LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X_fit, y_fit, eval_set=[(X_val, y_val_s)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return m

    def rmse_original_y(
        X_train_fit: np.ndarray, y_train_fit_scaled: np.ndarray, X_eval: np.ndarray, y_eval_original: np.ndarray,
    ) -> float:
        pred_o = sc_y.inverse_transform(_train_lgb(X_train_fit, y_train_fit_scaled).predict(X_eval).reshape(-1, 1)).ravel()
        return sqrt(mean_squared_error(y_eval_original, pred_o))

    def fit_rmse_and_feas(X_train_fit: np.ndarray, y_train_fit_scaled: np.ndarray) -> tuple[float, float, Any]:
        m = _train_lgb(X_train_fit, y_train_fit_scaled)
        pred_o = sc_y.inverse_transform(m.predict(X_val).reshape(-1, 1)).ravel()
        rmse_v = sqrt(mean_squared_error(y_val_orig, pred_o))
        pearson_01, _ = _domain_pearson(m, X_val_df, None)
        return rmse_v, pearson_01, m

    baseline_rmse, pearson0, model_before = fit_rmse_and_feas(X_train, y_train_s)
    base_c = prev_score = composite_score(baseline_rmse, pearson0, rmse_ref=baseline_rmse)
    test0 = rmse_original_y(X_train, y_train_s, X_test, y_test_orig)
    X_train_aug, y_train_aug = X_train.copy(), y_train_s.copy()
    synth_rows_added = 0

    for it in range(max_iters):
        scm_ols = fit_scm_ols(X_train_aug, cols)
        struct_scales = scm_struct_residual_scales(X_train_aug, cols, scm_ols)
        y_head, y_delta_hi = fit_target_head(X_train_aug, y_train_aug)
        ref_m = _train_lgb(X_train_aug, y_train_aug)
        try:
            X_df_aug = pd.DataFrame(X_train_aug, columns=cols)
            m_guide = min(int(sp.shap_synth_guide_samples), len(X_df_aug))
            shap_w = tree_shap_mean_abs_weights(
                ref_m, X_df_aug, max_samples=m_guide, random_state=rs + 1_003 * it,
            )
            if int(shap_w.shape[0]) != d_features:
                shap_w = np.ones(d_features, dtype=np.float32)
        except Exception:
            shap_w = np.ones(d_features, dtype=np.float32)
        if uniform_shap_weights:
            shap_w = np.ones(d_features, dtype=np.float32)

        def objective(trial: optuna.Trial) -> float:
            scm_sample_row(X_train_aug, cols, scm_ols, struct_scales, y_head, y_delta_hi, trial, shap_w)
            bp_t = dict(trial.params)
            rng_t = np.random.default_rng(int(rs) + 1_000_003 * int(it) + int(trial.number))
            x_b, y_b = scm_batch_from_bp(
                X_train_aug, cols, scm_ols, struct_scales, y_head, y_delta_hi, bp_t, trial.number, SYNTH_BATCH_SIZE, rng_t,
                shap_w,
            )
            rmse_v, p01, _ = fit_rmse_and_feas(np.vstack((X_train_aug, x_b)), np.concatenate((y_train_aug, y_b)))
            trial.set_user_attr("val_rmse", float(rmse_v))
            trial.set_user_attr("val_pearson01", float(p01))
            return composite_score(rmse_v, p01, rmse_ref=baseline_rmse)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(constant_liar=True, multivariate=True))
        study.optimize(
            objective, n_trials=n_trials_local, show_progress_bar=show_progress_bar and not skip_outputs,
        )

        top = sorted(
            (t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None),
            key=lambda t: float(t.value),
            reverse=True,
        )[:3]
        if not top:
            break

        scored: list[tuple[float, float, float, Any]] = []
        for t in top:
            rng_k = np.random.default_rng(int(rs) + 7_001_003 * int(it) + int(t.number))
            x_k, y_k = scm_batch_from_bp(
                X_train_aug, cols, scm_ols, struct_scales, y_head, y_delta_hi, dict(t.params), t.number,
                SYNTH_BATCH_SIZE, rng_k, shap_w,
            )
            rmse_v, p01, _ = fit_rmse_and_feas(np.vstack((X_train_aug, x_k)), np.concatenate((y_train_aug, y_k)))
            scored.append((composite_score(rmse_v, p01, rmse_ref=baseline_rmse), rmse_v, p01, t))
        cand_score, cand_rmse, cand_pearson, best_trial = max(scored, key=lambda z: z[0])

        if cand_score <= prev_score:
            break

        rng_b = np.random.default_rng(int(rs) + 9_000_011 * int(it) + int(best_trial.number))
        x_batch, y_batch = scm_batch_from_bp(
            X_train_aug, cols, scm_ols, struct_scales, y_head, y_delta_hi, dict(best_trial.params), best_trial.number,
            SYNTH_BATCH_SIZE, rng_b, shap_w,
        )
        X_train_aug = np.vstack((X_train_aug, x_batch))
        y_train_aug = np.concatenate((y_train_aug, y_batch))
        d_comp, prev_score = cand_score - prev_score, cand_score
        synth_rows_added += int(x_batch.shape[0])
        if not skip_outputs:
            print(f"composite={cand_score:.4f} (↑{d_comp:+.4f})  val_rmse={cand_rmse:.4f} pearson01={cand_pearson:.3f} +{x_batch.shape[0]} synth")

    rmse_end, pearson_end, model_after = fit_rmse_and_feas(X_train_aug, y_train_aug)
    test1 = rmse_original_y(X_train_aug, y_train_aug, X_test, y_test_orig)
    end_c = composite_score(rmse_end, pearson_end, rmse_ref=baseline_rmse)

    def insulin_val_shap_stats(m: Any) -> dict[str, float]:
        if "insulin" not in cols:
            return {}
        keys = (
            "insulin_shap_q10",
            "insulin_shap_q25",
            "insulin_shap_mean",
            "frac_insulin_shap_lt_m03",
            "frac_insulin_shap_lt_m08",
        )
        try:
            sv = np.asarray(tree_shap_values_from_df(m, X_val_df), dtype=np.float64)
            ki = cols.index("insulin")
            phi = sv[:, ki].ravel()
            return {
                "insulin_shap_q10": float(np.percentile(phi, 10)),
                "insulin_shap_q25": float(np.percentile(phi, 25)),
                "insulin_shap_mean": float(np.mean(phi)),
                "frac_insulin_shap_lt_m03": float(np.mean(phi < -0.03)),
                "frac_insulin_shap_lt_m08": float(np.mean(phi < -0.08)),
            }
        except Exception:
            return {k: float("nan") for k in keys}

    insulin_stats = insulin_val_shap_stats(model_after)
    shap_report = {
        "dataset": dm.NAME,
        "target": dm.TARGET,
        "rules": [dict(r) for r in RULES],
        "n_val": len(X_val_df),
        "synth_batch_size": SYNTH_BATCH_SIZE,
        "synth_rows_added": synth_rows_added,
        "synth_params": asdict(sp),
        "baseline": shap_pearson_report(model_before, X_val_df),
        "augmented": shap_pearson_report(model_after, X_val_df),
        "insulin_val_tail": insulin_stats,
    }
    summary: dict[str, Any] = {
        "dataset": dm.NAME,
        "synth_params": asdict(sp),
        "baseline_rmse": float(baseline_rmse),
        "val_rmse_end": float(rmse_end),
        "pearson_end": float(pearson_end),
        "composite_end": float(end_c),
        "test_rmse_baseline": float(test0),
        "test_rmse_end": float(test1),
        "synth_rows_added": int(synth_rows_added),
        **insulin_stats,
    }
    if not skip_outputs:
        results_dir = os.path.join(project_root(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{dm.NAME}_shap_pearson.json"), "w") as fp:
            json.dump(shap_report, fp, indent=2, default=float)
        try:
            sv_b = np.asarray(tree_shap_values_from_df(model_before, X_val_df), dtype=np.float64)
            sv_a = np.asarray(tree_shap_values_from_df(model_after, X_val_df), dtype=np.float64)
            parts: list[pd.DataFrame | pd.Series] = [
                X_val_df.reset_index(drop=True).add_prefix("x_"),
                pd.DataFrame(sv_b, columns=[f"shap_before_{c}" for c in cols]),
                pd.DataFrame(sv_a, columns=[f"shap_after_{c}" for c in cols]),
                pd.Series(y_val_s, name="y_scaled"),
                pd.Series(y_val_orig, name="y_original"),
            ]
            shap_csv = pd.concat(parts, axis=1)
            shap_csv.insert(0, "row_idx", np.arange(len(shap_csv)))
            shap_csv.to_csv(os.path.join(results_dir, f"{dm.NAME}_shap_values_val.csv"), index=False)
        except Exception:
            pass
        try:
            with open(os.path.join(results_dir, f"{dm.NAME}_explain_models.pkl"), "wb") as fp:
                pickle.dump(
                    {
                        "model_before": model_before,
                        "model_after": model_after,
                        "cols": cols,
                        "X_val_df": X_val_df,
                        "TARGET": dm.TARGET,
                        "RULES": [dict(r) for r in RULES],
                        "random_state": rs,
                        "max_samples": SHAP_PEARSON_MAX_SAMPLES,
                    },
                    fp,
                )
        except Exception:
            pass
        _img = os.path.join(project_root(__file__), "manuscript", "images")
        save_shap_beeswarm_before_after_figure(
            model_before=model_before, model_after=model_after, X_val_df=X_val_df,
            out_path=os.path.join(_img, f"{dm.NAME}_shap_beeswarm_before_after.png"), random_state=rs,
            max_samples=SHAP_PEARSON_MAX_SAMPLES,
        )
        try:
            icf = save_interaction_cf_scatter_before_after_figure(
                model_before=model_before,
                model_after=model_after,
                X_val_df=X_val_df,
                cols=cols,
                rules=RULES,
                target=dm.TARGET,
                out_path=os.path.join(_img, f"{dm.NAME}_interaction_cf_scatter.png"),
                random_state=rs,
                max_samples=SHAP_PEARSON_MAX_SAMPLES,
            )
            if icf is not None:
                print(f"[{dm.NAME}] feature-rule SHAP scatter: {icf}")
        except Exception:
            pass
        print(
            f"[{dm.NAME}] n_features={d_features} n_train={len(X_train)} n_val={len(X_val_df)} n_test={len(X_test_df)} synth_rows_added={synth_rows_added}\n"
            f"[baseline] val_rmse={baseline_rmse:.4f} pearson01={pearson0:.3f} composite={base_c:.4f} test_rmse={test0:.4f}\n"
            f"[+synth]   val_rmse={rmse_end:.4f} pearson01={pearson_end:.3f} composite={end_c:.4f} test_rmse={test1:.4f}"
        )
        if insulin_stats:
            print(f"[insulin SHAP on val] q10={insulin_stats.get('insulin_shap_q10', float('nan')):.4f} "
                  f"mean={insulin_stats.get('insulin_shap_mean', float('nan')):.4f} "
                  f"frac<-0.08={insulin_stats.get('frac_insulin_shap_lt_m08', float('nan')):.3f}")
    return summary


if __name__ == "__main__":
    for _dataset_mod in DATASETS:
        run_dataset(importlib.import_module(_dataset_mod))
