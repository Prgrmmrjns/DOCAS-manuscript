"""LightGBM helpers used by iSHAP scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class ClfCfg:
    objective: str = "binary"
    n_estimators: int = 450
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 30
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class RegCfg:
    objective: str = "regression"
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


class LightGBMClassifierModel:
    """Classifier wrapper exposing `.clf` for SHAP and SHAPIQ."""

    def __init__(self, cfg: ClfCfg | None = None, params: dict | None = None) -> None:
        base = asdict(cfg or ClfCfg())
        if params:
            base.update(params)
        self.clf = LGBMClassifier(**base)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LightGBMClassifierModel:
        self.clf.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(x)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.clf.predict(x)


class LightGBMRegressorModel:
    """Regressor wrapper exposing `.reg` for SHAP and SHAPIQ."""

    def __init__(self, cfg: RegCfg | None = None, params: dict | None = None) -> None:
        base = asdict(cfg or RegCfg())
        if params:
            base.update(params)
        self.reg = LGBMRegressor(**base)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LightGBMRegressorModel:
        self.reg.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(x)


def _split_idx(
    x: pd.DataFrame, y: pd.Series, *, cls: bool, k: int, rs: int
) -> tuple[np.ndarray, np.ndarray]:
    if cls:
        return next(StratifiedKFold(n_splits=k, shuffle=True, random_state=rs).split(x, y))
    return next(KFold(n_splits=k, shuffle=True, random_state=rs).split(x, y))


def _clf_m(y: pd.Series, p: np.ndarray) -> dict:
    o = {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }
    return o


def _reg_m(y: pd.Series, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p))
    return {"rmse": rmse, "r2": r2}


def train_model(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    classification: bool = True,
    n_splits: int = 5,
    params: dict | None = None,
    verbose: int = -1,
    random_state: int = 42,
) -> tuple[
    LightGBMClassifierModel | LightGBMRegressorModel,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict,
]:
    """Train one split-selected model and return train/test frames plus metrics."""
    tr_i, te_i = _split_idx(x, y, cls=classification, k=n_splits, rs=random_state)
    x_tr, x_te = x.iloc[tr_i], x.iloc[te_i]
    y_tr, y_te = y.iloc[tr_i], y.iloc[te_i]

    if classification:
        m = LightGBMClassifierModel(params=params)
        m.clf.set_params(verbose=verbose)
        m.fit(x_tr, y_tr)
        p = m.predict_proba(x_te)[:, 1]
        met = _clf_m(y_te, p)
        return m, x_tr, x_te, y_tr, y_te, met

    m = LightGBMRegressorModel(params=params)
    m.reg.set_params(verbose=verbose)
    m.fit(x_tr, y_tr)
    p = m.predict(x_te)
    met = _reg_m(y_te, p)
    return m, x_tr, x_te, y_tr, y_te, met
