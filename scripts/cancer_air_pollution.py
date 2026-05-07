from __future__ import annotations

import os

import pandas as pd

from lib import SCMRule

NAME = "cancer_air_pollution"
TASK = "classification"
TARGET = "Level"
CSV = "cancer patient data sets.csv"


def load(root: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(root, CSV)
    df = pd.read_csv(path)

    y = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .map({"Low": 0.0, "Medium": 1.0, "High": 2.0})
        .astype(float)
    )

    X = df.drop(columns=[TARGET, "index", "Patient Id"], errors="ignore")
    return X.reset_index(drop=True), y.reset_index(drop=True)


SCM_RULES: list[SCMRule] = [
    {"start": "Air Pollution", "end": TARGET, "edge": 1.0},
]
