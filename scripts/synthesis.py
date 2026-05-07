from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from scoring import CombinedScorer


@dataclass(frozen=True)
class SynthSearchSpace:
    cols: list[str]
    feat_min: np.ndarray
    feat_max: np.ndarray
    y_min: float
    y_max: float


def uniform_bounds_from_training(
    X_train_df: pd.DataFrame,
    y_train_s: pd.Series,
    cols: list[str],
) -> SynthSearchSpace:
    Xc = X_train_df[cols]
    return SynthSearchSpace(
        cols=list(cols),
        feat_min=Xc.min().to_numpy(dtype=np.float64),
        feat_max=Xc.max().to_numpy(dtype=np.float64),
        y_min=float(y_train_s.min()),
        y_max=float(y_train_s.max()),
    )


class OptunaTPESyntheticOptimizer:
    def optimize(
        self,
        *,
        space: SynthSearchSpace,
        n_points: int,
        n_trials: int,
        rng: np.random.Generator,
        scorer: CombinedScorer,
        fitness_fn: Callable[[pd.DataFrame, pd.Series], tuple[float, float]],
    ) -> tuple[pd.DataFrame, pd.Series]:
        cols = space.cols
        seed = int(rng.integers(0, 2**31 - 1))

        def objective(trial: optuna.Trial) -> float:
            x_rows = [
                [
                    trial.suggest_float(f"x_{k}_{c}", float(space.feat_min[i]), float(space.feat_max[i]))
                    for i, c in enumerate(cols)
                ]
                for k in range(int(n_points))
            ]
            y_vals = [trial.suggest_float(f"y_{k}", space.y_min, space.y_max) for k in range(int(n_points))]
            X_s = pd.DataFrame(x_rows, columns=cols)
            y_s = pd.Series(y_vals)
            metric_loss, feasibility = fitness_fn(X_s, y_s)
            return scorer.score(metric_loss, feasibility)

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

        best = study.best_trial
        X_best = pd.DataFrame(
            [[best.params[f"x_{k}_{c}"] for c in cols] for k in range(int(n_points))],
            columns=cols,
        )
        y_best = pd.Series([best.params[f"y_{k}"] for k in range(int(n_points))])
        return X_best, y_best
