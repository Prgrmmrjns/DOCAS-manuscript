from __future__ import annotations

import os

import pandas as pd

from lib import SCMRule

NAME = "heart_disease"
TASK = "classification"
TARGET = "disease"
CSV = "heart_disease.csv"

FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

# Exercise capacity (higher max HR) is protective; ischemic ST depression and vessel score raise risk.
SCM_RULES: list[SCMRule] = [
    {"start": "thalach", "end": TARGET, "edge": -0.92},
    {"start": "oldpeak", "end": TARGET, "edge": 0.88},
    {"start": "ca", "end": TARGET, "edge": 0.78},
]


def load(root: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(root, CSV)
    df = pd.read_csv(path)
    X = df[list(FEATURE_COLUMNS)].astype(float, copy=False)
    return X.reset_index(drop=True), df[TARGET].reset_index(drop=True)
