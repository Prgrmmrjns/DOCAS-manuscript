from __future__ import annotations

import os

import pandas as pd

from lib import SCMRule

NAME = "cancer_risk"
TASK = "regression"
TARGET = "Overall_Risk_Score"
CSV = "cancer_risk_factors.csv"

# Numeric risk factors only (exclude ID, cancer label, ordinal risk band).
FEATURE_COLUMNS: tuple[str, ...] = (
    "Age",
    "Gender",
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "Air_Pollution",
    "Occupational_Hazards",
    "BRCA_Mutation",
    "H_Pylori_Infection",
    "Calcium_Intake",
    "BMI",
    "Physical_Activity_Level",
)

# Tobacco and processed meat raise composite risk; fruit/vegetable intake is protective.
SCM_RULES: list[SCMRule] = [
    {"start": "Smoking", "end": TARGET, "edge": 0.92},
    {"start": "Fruit_Veg_Intake", "end": TARGET, "edge": -0.72},
    {"start": "Diet_Red_Meat", "end": TARGET, "edge": 0.58},
]


def load(root: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(root, CSV)
    df = pd.read_csv(path)
    X = df[list(FEATURE_COLUMNS)].astype(float, copy=False)
    return X.reset_index(drop=True), df[TARGET].reset_index(drop=True)
