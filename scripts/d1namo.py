from __future__ import annotations

import os

import pandas as pd

from lib import SCMRule

NAME = "d1namo"
TASK = "regression"
TARGET = "future glucose"
CSV = "d1namo_combined.csv"

# Column order for synthetic-row traversal and topological tie-breaking.
FEATURE_COLUMNS: tuple[str, ...] = (
    "time",
    "glucose",
    "glucose_change",
    "dietary_fibers",
    "fats",
    "proteins",
    "insulin",
    "simple_sugars",
    "complex_sugars",
)


def load(root: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(root, CSV)
    df = pd.read_csv(path)
    X = df[list(FEATURE_COLUMNS)].astype(float, copy=False)
    return X.reset_index(drop=True), df[TARGET].reset_index(drop=True)


# Domain rules: target edges → sign of Pearson(SHAP(feature), x_feature).
# Feature→feature edges → sign of Pearson(x_start, SHAP(end)): negative edge means higher ``start``
# should associate with *lower* SHAP on ``end`` (e.g.\ more fiber → less simple-sugars attribution).
SCM_RULES: list[SCMRule] = [
    #{"start": "insulin", "end": TARGET, "edge": -1.0},
    #{"start": "simple_sugars", "end": TARGET, "edge": 0.95},
    #{"start": "complex_sugars", "end": TARGET, "edge": 0.45},
    #{"start": "insulin", "end": "simple_sugars", "edge": -0.88},
    {"start": "dietary_fibers", "end": "simple_sugars", "edge": -1.0},
]
