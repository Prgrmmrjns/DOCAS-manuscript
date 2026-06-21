"""Corrective insulin bolus (CIB) helpers for DSS replay figures."""

from __future__ import annotations

import numpy as np

CIB_GRID_U = np.linspace(0, 10, 21)
CIB_TARGET_MGDL = 120.0
CIB_TRIGGER_MGDL = 180.0
CIB_DELAY_MIN = 120.0
CIB_MIN_GAP_MIN = 60.0
TIR_LO, TIR_HI = 70.0, 180.0


def predict_abs_glucose(model, row: np.ndarray, g_prev: np.ndarray) -> np.ndarray:
    delta = np.asarray(model.predict(row), float).ravel()
    return np.asarray(g_prev, float).ravel() + delta


def cib_dose(
    model, row: np.ndarray, j_ins: int, g_now: float, iob_u: float = 0.0,
    correction_factor: float | None = None,
) -> int:
    """Grid search (Prendin et al. 2023, Eq. 11): dose that brings prediction closest to target."""
    if g_now <= CIB_TRIGGER_MGDL:
        return 0
    best_u, best_err = 0, abs(g_now - CIB_TARGET_MGDL)
    for u in CIB_GRID_U[1:]:
        x = row.copy()
        x[j_ins] = x[j_ins] + float(iob_u) + float(u)
        g_pred = float(predict_abs_glucose(model, x.reshape(1, -1), np.array([g_now]))[0])
        err = abs(g_pred - CIB_TARGET_MGDL)
        if err < best_err:
            best_err, best_u = err, int(round(u))
    return best_u
