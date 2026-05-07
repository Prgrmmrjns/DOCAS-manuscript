from __future__ import annotations

import os

import pandas as pd

from lib import SCMRule

NAME = "d1namo"
TASK = "regression"
TARGET = "future glucose"
CSV = "d1namo_combined.csv"



def load(root: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(root, CSV)
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    return X.reset_index(drop=True), df[TARGET].reset_index(drop=True)


# Coarse glucose-metabolism SCM:
# upstream nutrients/hormone effects + direct parents of future glucose.
SCM_RULES: list[SCMRule] = [
    # Feature -> feature relations
    #{"start": "fats", "end": "simple_sugars", "edge": 0.2},
    #{"start": "fats", "end": "complex_sugars", "edge": 0.2},
    #{"start": "dietary_fibers", "end": "simple_sugars", "edge": -0.5},
    #{"start": "dietary_fibers", "end": "complex_sugars", "edge": -0.5},
    # Direct effects on target
    #{"start": "simple_sugars", "end": TARGET, "edge": 1.0},
    #{"start": "complex_sugars", "end": TARGET, "edge": 0.6},
    #{"start": "proteins", "end": TARGET, "edge": 0.2},
    {"start": "insulin", "end": TARGET, "edge": -1.0},
    #{"start": "fats", "end": TARGET, "edge": 0.0},
    #{"start": "dietary_fibers", "end": TARGET, "edge": 0.0},
]