from __future__ import annotations

from typing import Any

import lightgbm as lgb

MODEL_KW: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "verbose": -1,
}


def make_model(*, random_state: int, n_jobs: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**MODEL_KW, random_state=int(random_state), n_jobs=int(n_jobs))
