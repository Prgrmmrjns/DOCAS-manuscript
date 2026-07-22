"""Ablation: five DOCAS synthetic targets on OhioT1DM patient 588 (PH30).
Regimes: hardcoded hyperbolic, independence (flat), CGM interaction, CHO surface,
full glycaemic range. Uncomment the ablation block in main.py; tables/figures via manuscript.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docas import DOCAS, F0_REF, future
from forecasters import train_baseline
from ohio_t1dm_eval import DATA_ROOT, _alignment, _rmse_mgdl, load_baseline_model, round_metric
from ohio_t1dm_preprocessing import horizon_steps, load_patient

OUT_JSON = DATA_ROOT / "results" / "ablation" / "patient588_ph30.json"
PATIENT_ID, HORIZON_MIN, SEED, HYPER_K = "588", 30, 42, 2.5
NOMINAL_DROP, FULL_RANGE_F0_LO = 0.15, 70.0
_MODEL_CACHE = {}


def hyperbolic(u, k):
    u = np.maximum(np.atleast_1d(np.asarray(u, float)), 0.0)
    return (u / (k + u)) / max(1.0 / (float(k) + 1.0), 1e-12)


def _delta_response(amp, u, k=HYPER_K):
    """ΔBG at F0_REF for fractional amplitude ``amp`` (matches DOCAS.alignment)."""
    return -F0_REF * float(amp) * hyperbolic(u, k)


def _fit(X, y, **kw):
    return train_baseline(X, y, model_kwargs={"n_jobs": 1, **kw})


def _band(X, baseline, ji, gi, ins_s, cgm_s, tgt_s, f0_lo):
    f0 = future(baseline, X, ji, gi, (0.0,), DOCAS.SPAN, ins_s, tgt_s, cgm_s, False).ravel()
    m = f0 > float(f0_lo)
    return X[m], f0[m]


def _rows_1d(A, f0_a, u, ji, gi, ins_s, cgm_s, tgt_s, *, amp, k):
    shape, n, g = hyperbolic(u, k), len(A), len(u)
    cgm = np.repeat(cgm_s.inverse_transform(A[:, gi : gi + 1]).ravel(), g)
    future = np.repeat(f0_a, g) * (1.0 - amp * np.tile(shape, n))
    y = tgt_s.transform((future - cgm).reshape(-1, 1)).ravel()
    xd = np.repeat(A, g, 0)
    xd[:, ji] = ins_s.transform((np.tile(u, n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    return xd, y


def _aug_1d(A, f0_a, ji, gi, ins_s, cgm_s, tgt_s, rng, *, amp, n_u, n_anchors):
    idx = rng.integers(0, len(A), n_anchors)
    return _rows_1d(A[idx], f0_a[idx], np.linspace(0.0, 1.0, n_u), ji, gi, ins_s, cgm_s, tgt_s, amp=amp, k=HYPER_K)


def _aug_cgm(A, ji, gi, ins_s, cgm_s, tgt_s, rng, *, amp, k, span_mgdl=60.0):
    u, d = np.linspace(0.0, 1.0, 21), np.linspace(-span_mgdl, span_mgdl, 21)
    anchors = A[rng.integers(0, len(A), 400)]
    cgm0 = cgm_s.inverse_transform(anchors[:, gi : gi + 1]).ravel()
    Ug, Dg = (g.ravel() for g in np.meshgrid(u, d, indexing="ij"))
    slope, n, g = 1.0 - amp * hyperbolic(Ug, k), len(anchors), len(Ug)
    cgm_new = np.repeat(cgm0, g) + np.tile(Dg, n)
    future = cgm_new * np.tile(slope, n)
    y = tgt_s.transform((future - cgm_new).reshape(-1, 1)).ravel()
    xd = np.repeat(anchors, g, 0)
    xd[:, gi] = cgm_s.transform(cgm_new.reshape(-1, 1)).ravel()
    xd[:, ji] = ins_s.transform((np.tile(Ug, n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    return xd, y


def _aug_cho(A, f0_a, cgm0, ji, ci, ins_s, cho_s, tgt_s, rng, *, delta, k, cho_max=80.0):
    u, c = np.linspace(0.0, 1.0, 21), np.linspace(0.0, cho_max, 7)
    idx = rng.integers(0, len(A), 700)
    anchors, f0s, cgm0s = A[idx], f0_a[idx], cgm0[idx]
    Ug, Cg = (g.ravel() for g in np.meshgrid(u, c, indexing="ij"))
    r, c_norm, n, g = hyperbolic(Ug, k), Cg / cho_max, len(anchors), len(Ug)
    future = np.repeat(f0s, g) * (1.0 + delta * (np.tile(c_norm, n) - np.tile(r, n)))
    y = tgt_s.transform((future - np.repeat(cgm0s, g)).reshape(-1, 1)).ravel()
    xd = np.repeat(anchors, g, 0)
    xd[:, ji] = ins_s.transform((np.tile(Ug, n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    xd[:, ci] = cho_s.transform(np.tile(Cg, n).reshape(-1, 1)).ravel()
    return xd, y


def _cgm_slopes(model, anchors, ji, gi, ins_s, cgm_s, tgt_s, u_probe, delta_mgdl=5.0):
    cgm0 = cgm_s.inverse_transform(anchors[:, gi : gi + 1]).ravel()
    out = []
    for u in u_probe:
        rows = []
        for sign in (-1.0, 1.0):
            x = anchors.copy()
            x[:, gi] = cgm_s.transform((cgm0 + sign * delta_mgdl).reshape(-1, 1)).ravel()
            x[:, ji] = ins_s.transform(np.full((len(anchors), 1), u * DOCAS.SPAN)).ravel()
            rows.append(tgt_s.inverse_transform(model.predict(x).reshape(-1, 1)).ravel())
        out.append(1.0 + float(np.mean((rows[1] - rows[0]) / (2 * delta_mgdl))))
    return out


def _surface_align(model, anchors, f0s, cgm0, ji, ci, ins_s, cho_s, tgt_s, *, delta, k, cho_max=80.0):
    u, c = np.linspace(0.0, 1.0, 21), np.linspace(0.0, cho_max, 7)
    Ug, Cg = (g.ravel() for g in np.meshgrid(u, c, indexing="ij"))
    r, c_norm = hyperbolic(Ug, k), Cg / cho_max
    target = np.repeat(f0s, len(Ug)) * (1.0 + delta * (np.tile(c_norm, len(anchors)) - np.tile(r, len(anchors))))
    n, g = len(anchors), len(Ug)
    xd = np.repeat(anchors, g, 0)
    xd[:, ji] = ins_s.transform((np.tile(Ug, n) * DOCAS.SPAN).reshape(-1, 1)).ravel()
    xd[:, ci] = cho_s.transform(np.tile(Cg, n).reshape(-1, 1)).ravel()
    pred_future = np.repeat(cgm0, g) + tgt_s.inverse_transform(model.predict(xd).reshape(-1, 1)).ravel()
    return float(np.sqrt(np.mean((pred_future - target) ** 2)))


def _train_aug(X_tr, y_tr, xd, y):
    return _fit(np.vstack([X_tr, xd]), np.concatenate([y_tr, y]))


def _rmse_delta(model, X_te, y_te, tgt_s, base_rmse):
    return _rmse_mgdl(model, X_te, y_te, tgt_s) - base_rmse


def run() -> dict:
    rng = np.random.default_rng(SEED)
    data = load_patient(str(DATA_ROOT), PATIENT_ID, horizon_steps=horizon_steps(HORIZON_MIN))
    baseline = load_baseline_model(HORIZON_MIN, PATIENT_ID)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    ci = data.feature_names.index("CHO")
    ins_s, cgm_s, cho_s, tgt_s = data.train_scalers[ji], data.train_scalers[gi], data.train_scalers[ci], data.cgm_scaler
    align_kw = dict(ji=ji, yi=gi, u=np.linspace(0, 1, 41), span=DOCAS.SPAN,
                    a_s=ins_s, dy_s=tgt_s, y_s=cgm_s, y0_lo=DOCAS.Y0_LO,
                    y0_hi=DOCAS.Y0_HI, additive=False)
    align_full = {**align_kw, "y0_lo": FULL_RANGE_F0_LO}
    band = lambda X, lo: _band(X, baseline, ji, gi, ins_s, cgm_s, tgt_s, lo)
    A_tr, f0_tr = band(data.X_train, DOCAS.Y0_LO)
    A_te, f0_te = band(data.X_test, DOCAS.Y0_LO)
    A_tr_full, f0_tr_full = band(data.X_train, FULL_RANGE_F0_LO)
    A_te_full, f0_te_full = band(data.X_test, FULL_RANGE_F0_LO)
    base_rmse = _rmse_mgdl(baseline, data.X_test, data.y_test, tgt_s)
    X_tr, y_tr, X_te, y_te = data.X_train, data.y_train, data.X_test, data.y_test
    results = {}
    tau = lambda drop: _delta_response(drop, align_kw["u"])
    xd, y = _aug_1d(A_tr, f0_tr, ji, gi, ins_s, cgm_s, tgt_s, rng, amp=NOMINAL_DROP, n_u=61, n_anchors=8000)
    m1 = _train_aug(X_tr, y_tr, xd, y)
    results["hardcoded"] = dict(
        alignment_error=round_metric(_alignment(m1, baseline, A_te, response=tau(NOMINAL_DROP), **align_kw)),
        rmse_delta_mgdl=_rmse_delta(m1, X_te, y_te, tgt_s, base_rmse),
    )
    xd, y = _aug_1d(A_tr, f0_tr, ji, gi, ins_s, cgm_s, tgt_s, rng, amp=0.0, n_u=41, n_anchors=6000)
    m2 = _train_aug(X_tr, y_tr, xd, y)
    results["independence"] = dict(
        alignment_error=round_metric(_alignment(m2, baseline, A_te, response=tau(0.0), **align_kw)),
        rmse_delta_mgdl=_rmse_delta(m2, X_te, y_te, tgt_s, base_rmse),
    )
    xd, y = _aug_cgm(A_tr, ji, gi, ins_s, cgm_s, tgt_s, rng, amp=NOMINAL_DROP, k=HYPER_K)
    m3 = _train_aug(X_tr, y_tr, xd, y)
    slopes = _cgm_slopes(m3, A_te, ji, gi, ins_s, cgm_s, tgt_s, (0.0, 1.0))
    tgt_sl = [1.0, 1.0 - NOMINAL_DROP]
    results["cgm_interaction"] = dict(
        slope_u0=slopes[0], slope_u1=slopes[1], target_slope_u0=tgt_sl[0], target_slope_u1=tgt_sl[1],
        interaction_error=round_metric(abs(slopes[0] - tgt_sl[0]) + abs(slopes[1] - tgt_sl[1])),
        rmse_delta_mgdl=_rmse_delta(m3, X_te, y_te, tgt_s, base_rmse),
    )
    cgm0_tr = cgm_s.inverse_transform(A_tr[:, gi : gi + 1]).ravel()
    xd, y = _aug_cho(A_tr, f0_tr, cgm0_tr, ji, ci, ins_s, cho_s, tgt_s, rng, delta=0.15, k=HYPER_K)
    m4 = _train_aug(X_tr, y_tr, xd, y)
    cgm0_te = cgm_s.inverse_transform(A_te[:, gi : gi + 1]).ravel()
    results["cho_surface"] = dict(
        alignment_error=round_metric(_surface_align(m4, A_te, f0_te, cgm0_te, ji, ci, ins_s, cho_s, tgt_s, delta=0.15, k=HYPER_K)),
        rmse_delta_mgdl=_rmse_delta(m4, X_te, y_te, tgt_s, base_rmse),
    )
    xd, y = _aug_1d(A_tr_full, f0_tr_full, ji, gi, ins_s, cgm_s, tgt_s, rng, amp=NOMINAL_DROP, n_u=61, n_anchors=12000)
    m5 = _train_aug(X_tr, y_tr, xd, y)
    aln_full = lambda m: round_metric(_alignment(m, baseline, A_te_full, response=tau(NOMINAL_DROP), **align_full))
    results["full_range"] = dict(
        f0_lo=FULL_RANGE_F0_LO, n_train_anchors=int(len(A_tr_full)), n_hyper_train_anchors=int(len(A_tr)),
        alignment_error=round_metric(_alignment(m5, baseline, A_te, response=tau(NOMINAL_DROP), **align_kw)),
        alignment_error_full=aln_full(m5), hardcoded_alignment_error_full=aln_full(m1),
        rmse_delta_mgdl=_rmse_delta(m5, X_te, y_te, tgt_s, base_rmse),
    )
    payload = dict(patient_id=PATIENT_ID, horizon_min=HORIZON_MIN, baseline_test_rmse_mgdl=base_rmse, **results)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    _MODEL_CACHE.update(dict(
        data=data, baseline=baseline, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5,
        gi=gi, ji=ji, ci=ci, ins_s=ins_s, cgm_s=cgm_s, cho_s=cho_s, tgt_s=tgt_s,
        A_te=A_te, f0_te=f0_te, A_te_full=A_te_full, f0_te_full=f0_te_full,
    ))
    return payload
