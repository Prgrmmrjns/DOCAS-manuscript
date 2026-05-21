from __future__ import annotations

from typing import Any

import lightgbm as lgb

MODEL_KW: dict[str, Any] = dict(
    max_depth=3,
    n_estimators=50,
    learning_rate=0.3,
    verbose=-1,
)


def monotonic_constraints_for(
    cols: list[str],
    rules: list[dict[str, Any]],
    target: str,
) -> list[int]:
    """Map SCM monotonic rules to LightGBM constraint codes (+1 inc, -1 dec, 0 free)."""
    cmap: dict[str, int] = {}
    t = str(target)
    for r in rules:
        if str(r.get("end", "")) != t:
            continue
        fn = r.get("relationship_fn")
        inc = bool(getattr(fn, "increasing", True))
        cmap[str(r["start"])] = 1 if inc else -1
    return [int(cmap.get(c, 0)) for c in cols]


def make_model(
    *,
    random_state: int | None = None,
    n_jobs: int = -1,
    monotonic_constraints: list[int] | None = None,
    model_kw: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    kw: dict[str, Any] = {**MODEL_KW, **(model_kw or {}), "n_jobs": n_jobs}
    if random_state is not None:
        kw["random_state"] = random_state
    if monotonic_constraints is not None:
        kw["monotonic_constraints"] = monotonic_constraints
    return lgb.LGBMRegressor(**kw)
