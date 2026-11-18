"""DOCAS core: counterfactual sampling, alignment, absolute-τ target."""
from __future__ import annotations

import numpy as np

from forecasters import train_baseline

_inv = lambda s, a: s.inverse_transform(np.asarray(a, float).reshape(-1, 1)).reshape(np.asarray(a).shape)
_shape = lambda t, u: np.asarray(t(np.asarray(u, float)), float).ravel()


def _context_anchors(low, meal, n, p_meal, p_low, rng):
    """Draw anchors from the meal × low-glucose cells with the given probabilities."""
    p_meal = float(np.clip(p_meal, 0.0, 1.0))
    p_low = float(np.clip(p_low, 0.0, 1.0))
    cells = (
        (meal & low, p_meal * p_low),
        (meal & ~low, p_meal * (1.0 - p_low)),
        (~meal & low, (1.0 - p_meal) * p_low),
        (~meal & ~low, (1.0 - p_meal) * (1.0 - p_low)),
    )
    groups, w = [], []
    for mask, p in cells:
        idx = np.flatnonzero(mask)
        if len(idx) and p > 0.0:
            groups.append(idx)
            w.append(p)
    if not groups:
        return rng.choice(len(low), n, replace=True)
    w = np.asarray(w, float)
    w /= w.sum()
    counts = rng.multinomial(int(n), w)
    parts = [rng.choice(g, int(c), replace=True) for g, c in zip(groups, counts) if c]
    return rng.permutation(np.concatenate(parts))


def _binned_dose(n, weights, emp_frac, u_pool, rng):
    """Mixture of the empirical dose pool and a piecewise-uniform dose density."""
    w = np.clip(np.asarray(weights, float).ravel(), 1e-6, None)
    w = w / w.sum()
    edges = np.linspace(0.0, 1.0, len(w) + 1)
    emp = rng.random(n) < float(np.clip(emp_frac, 0.0, 1.0))
    u = np.empty(n, float)
    n_emp = int(emp.sum())
    if n_emp:
        u[emp] = rng.choice(u_pool, n_emp)
    n_bin = int(n - n_emp)
    if n_bin:
        b = rng.choice(len(w), n_bin, p=w)
        u[~emp] = rng.uniform(edges[b], edges[b + 1])
    return u


def _stratified_anchors(f0, n, k, rng):
    """Equal-count anchor draw across k quantile bins of the ambient level f0."""
    k = int(k)
    if k < 2:
        return rng.choice(len(f0), n, replace=True)
    edges = np.quantile(f0, np.linspace(0.0, 1.0, k + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], f0, side="right"), 0, k - 1)
    groups = [np.flatnonzero(bins == b) for b in range(k)]
    groups = [g for g in groups if len(g)]
    per = int(np.ceil(n / len(groups)))
    return rng.permutation(np.concatenate(
        [rng.choice(g, per, replace=True) for g in groups]))[:n]


def _ctx_codes(cgm, cho):
    cho = np.maximum(np.asarray(cho, float), 0.0)
    c = cho / (cho + 40.0)
    gz = np.clip((np.asarray(cgm, float) - 140.0) / 80.0, -2.0, 2.0)
    return c, gz


def _place_indices(cgm, cho, n, spec, rng):
    """Anchor draw whose log-weight has a carbohydrate × glucose term."""
    c, gz = _ctx_codes(cgm, cho)
    logw = (float(spec["a_c"]) * c + float(spec["a_g"]) * gz
            + float(spec["a_int"]) * c * gz)
    logw = np.clip(logw - logw.max(), -20.0, 0.0)
    w = np.exp(logw)
    w /= w.sum()
    return rng.choice(len(c), int(n), replace=True, p=w)


def _place_dose(cgm, cho, u_pool, spec, rng):
    """Beta dose whose mean depends on carbohydrate, glucose, and their product."""
    c, gz = _ctx_codes(cgm, cho)
    n = len(c)
    logit = (float(spec["b0"]) + float(spec["b_c"]) * c + float(spec["b_g"]) * gz
             + float(spec["b_int"]) * c * gz)
    mean = np.clip(1.0 / (1.0 + np.exp(-np.clip(logit, -12.0, 12.0))), 0.02, 0.98)
    conc = max(float(spec["conc"]), 0.5)
    u = rng.beta(mean * conc, (1.0 - mean) * conc)
    emp = rng.random(n) < float(np.clip(spec["emp"], 0.0, 1.0))
    k = int(emp.sum())
    if k:
        u[emp] = rng.choice(u_pool, k)
    return np.clip(u, 0.0, 1.0)


def sample_synthetic(
    X, y, *, target, ji, yi, span, a_s, dy_s, y_s, rng,
    tau_drop, synth_ratio, u_ref,
    ctx_jitter, cho_idx, cho_s, emp_frac=0.5, tau_mode="absolute",
    fit_space="delta", level_strata=0, exact_frac=True, level_coupled=True,
    level_ref="current", tau_exact=None, anchors=None, relabel=None,
    dose_weights=None, dose_weights_meal=None, dose_weights_fast=None,
    anchor_p_meal=None, anchor_p_low=None, point_spec=None,
    context_remap=False, cgm_share=0.0, curve_scale=1.0, cho_floor=None, **_,
):
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    sp = max(float(span), 1e-12)
    amb = _inv(a_s, X[:, ji])
    u_pool = np.clip(amb / sp, 0.0, 1.0)
    frac = str(tau_mode).lower().startswith("frac")
    if anchors is not None:
        i = np.asarray(anchors, int).ravel()
        n = int(len(i))
    else:
        n = max(1, int(float(synth_ratio) * len(y)))
        if point_spec is not None:
            cgm_all = _inv(y_s, X[:, yi])
            cho_all = np.maximum(_inv(cho_s, X[:, cho_idx]), 0.0)
            i = _place_indices(cgm_all, cho_all, n, point_spec, rng)
            u = _place_dose(cgm_all[i], cho_all[i], u_pool, point_spec, rng)
        elif anchor_p_meal is not None:
            cgm_all = _inv(y_s, X[:, yi])
            cho_all = np.maximum(_inv(cho_s, X[:, cho_idx]), 0.0)
            p_low = (cgm_all < 140.0).mean() if anchor_p_low is None else anchor_p_low
            i = _context_anchors(
                cgm_all < 140.0, cho_all >= 40.0, n, anchor_p_meal, p_low, rng)
        else:
            f0_all = _inv(y_s, X[:, yi]) + _inv(dy_s, y)
            if frac and int(level_strata) > 1:
                i = _stratified_anchors(f0_all, n, level_strata, rng)
            else:
                i = rng.choice(len(X), n, replace=True)
    if point_spec is not None:
        pass
    elif dose_weights_meal is not None and dose_weights_fast is not None:
        cho_i = np.maximum(_inv(cho_s, X[i, cho_idx]), 0.0)
        meal_i = cho_i >= 40.0
        u = np.empty(n, float)
        for mask, w in ((meal_i, dose_weights_meal), (~meal_i, dose_weights_fast)):
            k = int(mask.sum())
            if k:
                u[mask] = _binned_dose(k, w, 0.0, u_pool, rng)
    elif dose_weights is not None:
        u = _binned_dose(n, dose_weights, emp_frac, u_pool, rng)
    else:
        emp = rng.random(n) < float(np.clip(emp_frac, 0.0, 1.0))
        u = np.where(emp, rng.choice(u_pool, n), rng.random(n))

    r_ref = float(_shape(target, [u_ref])[0]) or 1.0
    ru = _shape(target, u) / r_ref
    ro = _shape(target, np.clip(amb[i] / sp, 0.0, 1.0)) / r_ref
    dy_obs = _inv(dy_s, y[i])
    cgm0 = _inv(y_s, X[i, yi])

    # Context noise is drawn first so the ambient-scaled label can stay consistent
    # with the CGM and CHO the synthetic row actually shows.
    cho0 = _inv(cho_s, X[i, cho_idx])
    if context_remap:
        # Remap CGM and CHO from another observed training row (not independent jitter).
        j = rng.choice(len(X), n, replace=True)
        cgm = np.maximum(_inv(y_s, X[j, yi]), 40.0)
        cho = np.maximum(_inv(cho_s, X[j, cho_idx]), 0.0)
    elif point_spec is None:
        cgm = np.maximum(cgm0 + rng.normal(0.0, float(ctx_jitter), n), 40.0)
        cho = np.maximum(cho0 + rng.normal(0.0, float(ctx_jitter), n), 0.0)
    else:
        c_i, gz_i = _ctx_codes(cgm0, cho0)
        sig_g = float(point_spec["sig_g"]) * np.exp(
            np.clip(float(point_spec["k_g"]) * c_i, -2.0, 2.0))
        sig_c = float(point_spec["sig_c"]) * np.exp(
            np.clip(float(point_spec["k_c"]) * gz_i, -2.0, 2.0))
        cgm = np.maximum(cgm0 + float(point_spec["bias_g"]) + rng.normal(0.0, sig_g), 40.0)
        cho = np.maximum(cho0 + float(point_spec["bias_c"]) + rng.normal(0.0, sig_c), 0.0)
    level_fit = frac and str(fit_space).lower().startswith("lev")
    if relabel is not None:
        lab = np.asarray(relabel(u, cgm, cho, i), float).ravel()
        level_fit = False
    elif frac:
        base = (cgm if level_coupled else cgm0) + dy_obs
        if str(level_ref).startswith("cur"):
            if tau_exact is not None:
                # τ re-simulated at the row's own ambient (and optional PK state)
                u_obs = np.clip(amb[i] / sp, 0.0, 1.0)
                try:
                    du = tau_exact(u, cgm, i) - tau_exact(u_obs, cgm, i)
                except TypeError:
                    du = tau_exact(u, cgm) - tau_exact(u_obs, cgm)
                lvl = base + du
            else:
                # τ scales with the ambient CGM at dosing time: additive, exactly invertible
                lvl = base + cgm * float(tau_drop) * (ru - ro)
        elif exact_frac:
            # untreated forecast implied by the factual dose, then re-dose at u
            g_un = base / np.maximum(1.0 + float(tau_drop) * ro, 0.1)
            lvl = g_un * (1.0 + float(tau_drop) * ru)
        else:
            lvl = base * (1.0 + float(tau_drop) * (ru - ro))
        lab = lvl if level_fit else lvl - cgm
    else:
        lab = dy_obs + float(tau_drop) * (ru - ro)
    share = float(cgm_share)
    scale = float(curve_scale)
    if not level_fit and (share or scale != 1.0 or cho_floor is not None):
        dphi = lab - dy_obs
        if scale != 1.0:
            dphi = dphi * scale
        if cho_floor is not None:
            c = np.maximum(cho, 0.0)
            mix = c ** 2 / (c ** 2 + 30.0 ** 2)
            dphi = dphi * (1.0 - (1.0 - float(cho_floor)) * mix)
        if share:
            moved = np.clip(cgm + share * dphi, 40.0, None)
            lab = dy_obs + dphi - (moved - cgm)
            cgm = moved
        else:
            lab = dy_obs + dphi
    y_out_s = y_s if level_fit else dy_s

    Xs = X[i].copy()
    Xs[:, yi] = y_s.transform(cgm.reshape(-1, 1)).ravel()
    Xs[:, cho_idx] = cho_s.transform(cho.reshape(-1, 1)).ravel()
    Xs[:, ji] = a_s.transform((u * sp).reshape(-1, 1)).ravel()
    return Xs, y_out_s.transform(lab[:, None]).ravel(), dict(
        n_anchors=int(len(np.unique(i))), n_synth=len(lab),
        tau_mode=("fractional" if frac else "absolute"), predicts_level=level_fit,
        u=np.asarray(u, float), anchor_idx=np.asarray(i, int))


class Cfg:
    N_GRID_POINTS = 100
    SEED = 42
    LOAD = False
    SI_MODE = 2.3 * 5e-4
    BODY_WEIGHT_KG = 75.0
    G0_TAU = 220.0
    DATASET = "ohio_t1dm"
    SENSOR = {}
    RBG = {}
    AE_WEIGHT = 1.0


def ape(rmse, ae, *, ae_weight=None) -> float:
    """Alignment-adjusted prediction error, APE_λ = RMSE + λ·AE (mg/dL).

    λ is how many mg/dL of forecast error one mg/dL of dose-response error is
    worth. λ=1 treats them alike; λ=0.1 is an accuracy-first user who still
    wants the audit curve to count.
    """
    w = Cfg.AE_WEIGHT if ae_weight is None else ae_weight
    return float(rmse) + float(w) * float(ae)


def select_by_ape(candidates, *, ae_weight=None):
    """Pick the candidate minimising APE_λ.

    `candidates` maps a setting to an (rmse, ae) pair; ties go to the smaller
    setting so the cheapest adequate option wins.
    """
    w = Cfg.AE_WEIGHT if ae_weight is None else ae_weight
    return min(candidates, key=lambda k: (ape(*candidates[k], ae_weight=w), k))


def ape_selection_map(candidates, *, tol=1e-9):
    """Winning candidate as a function of λ, for every λ ≥ 0.

    APE_λ is linear in λ for each candidate, so the argmin is piecewise
    constant and switches at the λ where two candidates tie. Returns
    ``[(lo, hi, key), ...]`` covering ``[0, inf)``: any λ a user names falls in
    one interval, with no grid and no re-running of the sweep.
    """
    pts = {k: (float(r), float(a)) for k, (r, a) in candidates.items()}
    cur = min(pts, key=lambda k: (pts[k][0], pts[k][1], k))  # λ=0: pure accuracy
    out, lo = [], 0.0
    while True:
        r0, a0 = pts[cur]
        nxt, sw = None, float("inf")
        for k, (r, a) in pts.items():
            if a >= a0 - tol:
                continue
            lam = (r - r0) / (a0 - a)  # tie point: r0 + λ a0 == r + λ a
            if lam < sw - tol or (abs(lam - sw) <= tol and a < pts[nxt][1]):
                nxt, sw = k, lam
        if nxt is None:
            out.append((lo, float("inf"), cur))
            return [(a, b, k) for a, b, k in out if b - a > tol]
        out.append((lo, sw, cur))
        lo, cur = sw, nxt


def sync_cfg(**kw) -> None:
    for k, v in kw.items():
        setattr(Cfg, k, v)
    DOCAS.N_GRID_POINTS = Cfg.N_GRID_POINTS
    DOCAS.SEED = Cfg.SEED
    DOCAS.LOAD = bool(Cfg.LOAD)
    DOCAS.SPAN = None
    DOCAS.U_GRID = np.linspace(0.0, 1.0, int(Cfg.N_GRID_POINTS))


def future(model, X, ji, yi, u, span, a_s, dy_s, y_s, *, level=False):
    X, u = np.asarray(X, float), np.asarray(u, float).ravel()
    if hasattr(model, "predict_intervention"):
        if level:
            return _inv(y_s, model.predict_intervention(X, u, span))
        return _inv(y_s, X[:, yi])[:, None] + _inv(dy_s, model.predict_intervention(X, u, span))
    rows = np.repeat(X, len(u), 0)
    rows[:, ji] = a_s.transform(np.tile(u * float(span), len(X)).reshape(-1, 1)).ravel()
    pred = model.predict(rows).reshape(len(X), len(u))
    if level:
        return _inv(y_s, pred)
    return _inv(y_s, X[:, yi])[:, None] + _inv(dy_s, pred)


class DOCAS:
    SPAN = None
    N_GRID_POINTS = Cfg.N_GRID_POINTS
    U_GRID = np.linspace(0.0, 1.0, Cfg.N_GRID_POINTS)
    SEED = Cfg.SEED
    LOAD = Cfg.LOAD
    train_fn = None

    def __init__(
        self, target, *, intervention_idx, outcome_idx, action_scaler, level_scaler, delta_scaler,
        span, tau_drop, u_ref=1.0, synth_ratio=1.0, ctx_jitter=0.0,
        cho_idx=0, cho_scaler=None, emp_frac=0.5, tau_mode="absolute",
        fit_space="delta", level_strata=0, predicts_level=None,
        exact_frac=True, level_coupled=True, level_ref="current",
        seed=Cfg.SEED, model_kwargs=(), tau_exact=None, relabel=None,
        dose_weights=None, dose_weights_meal=None, dose_weights_fast=None,
        anchor_p_meal=None, anchor_p_low=None, point_spec=None,
        context_remap=False, cgm_share=0.0, curve_scale=1.0, cho_floor=None, **_,
    ):
        self.tau_exact = tau_exact
        self.relabel = relabel
        self.context_remap = bool(context_remap)
        self.point_spec = None if point_spec is None else dict(point_spec)
        self.dose_weights = None if dose_weights is None else [float(v) for v in dose_weights]
        self.dose_weights_meal = None if dose_weights_meal is None else [float(v) for v in dose_weights_meal]
        self.dose_weights_fast = None if dose_weights_fast is None else [float(v) for v in dose_weights_fast]
        self.anchor_p_meal = None if anchor_p_meal is None else float(anchor_p_meal)
        self.anchor_p_low = None if anchor_p_low is None else float(anchor_p_low)
        self.target = lambda u, _t=target: np.asarray(_t(u), float)
        self.ji, self.yi = int(intervention_idx), int(outcome_idx)
        self.a_s, self.y_s, self.dy_s = action_scaler, level_scaler, delta_scaler
        self.span = float(span)
        self.train_fn = type(self).train_fn
        self.u_grid = np.asarray(type(self).U_GRID, float)
        self.seed = int(seed)
        self.u_ref = float(u_ref)
        self.tau_drop = float(tau_drop)
        self.tau_mode = str(tau_mode).lower()
        self.fit_space = str(fit_space).lower()
        self.level_strata = int(level_strata)
        self.exact_frac = bool(exact_frac)
        self.level_coupled = bool(level_coupled)
        self.level_ref = str(level_ref).lower()
        self.predicts_level = bool(
            self.tau_mode.startswith("frac") and self.fit_space.startswith("lev")
            if predicts_level is None else predicts_level)
        self.synth_ratio = float(synth_ratio)
        self.emp_frac = float(emp_frac)
        self.ctx_jitter = float(ctx_jitter)
        self.cho_idx, self.cho_s = int(cho_idx), cho_scaler
        self.cgm_share = float(cgm_share)
        self.curve_scale = float(curve_scale)
        self.cho_floor = None if cho_floor is None else float(cho_floor)
        self.model_kwargs = dict(model_kwargs)
        self.model_ = self.baseline_ = self.meta_ = None

    def alignment(self, model, X, *, u_grid=None):
        """RMSE (mg/dL) of the predicted dose-response vs τ, on the glucose level scale."""
        X = np.asarray(X, float)
        grid = self.u_grid if u_grid is None else np.asarray(u_grid, float).ravel()
        u0 = np.concatenate([[0.0], grid])
        fut = future(model, X, self.ji, self.yi, u0, self.span, self.a_s, self.dy_s, self.y_s,
                     level=self.predicts_level)
        f0m = fut[:, 0]
        g = _inv(self.y_s, X[:, self.yi])[:, None]
        if self.tau_exact is not None:
            idx = np.arange(len(X))
            try:
                tgt = f0m[:, None] + self.tau_exact(u0[1:][None, :], g, idx)
            except TypeError:
                tgt = f0m[:, None] + self.tau_exact(u0[1:][None, :], g)
            return float(np.sqrt(np.mean((fut[:, 1:] - tgt) ** 2)))
        r_ref = float(_shape(self.target, [self.u_ref])[0]) or 1.0
        Ru = _shape(self.target, u0[1:]) / r_ref
        if not self.tau_mode.startswith("frac"):
            gain = np.full(len(X), self.tau_drop)
        elif self.level_ref.startswith("cur"):
            gain = _inv(self.y_s, X[:, self.yi]) * self.tau_drop
        else:
            gain = f0m * self.tau_drop
        tgt = f0m[:, None] + gain[:, None] * Ru[None, :]
        return float(np.sqrt(np.mean((fut[:, 1:] - tgt) ** 2)))

    def alignment_absolute(self, model, X, *, max_dose_u=5.0, n=None):
        """Alignment error on a common absolute dose grid (0–max_dose_u), span-free.

        Probing a fixed 0–5 U window makes AE comparable across patients whose
        normalised axis spans very different absolute doses; for small-span
        patients the upper doses extrapolate beyond the training range.
        """
        n = int(n or len(self.u_grid))
        return self.alignment(
            model, X, u_grid=np.linspace(0.0, float(max_dose_u) / self.span, n)[1:])

    def fit(self, X_train, y_train, *, baseline_model, synth_weight=None):
        X, y = np.asarray(X_train, float), np.asarray(y_train, float).ravel()
        ji, yi, span = self.ji, self.yi, float(self.span)
        kw = dict(self.model_kwargs)
        rng = np.random.default_rng(self.seed)
        if self.predicts_level:
            # train on future CGM levels (scaled with level scaler)
            y_fit = self.y_s.transform(
                (_inv(self.y_s, X[:, yi]) + _inv(self.dy_s, y)).reshape(-1, 1)).ravel()
        else:
            y_fit = y
        Xs, ys, meta = sample_synthetic(
            X, y, target=self.target, ji=ji, yi=yi, span=span,
            a_s=self.a_s, dy_s=self.dy_s, y_s=self.y_s, rng=rng,
            tau_drop=self.tau_drop, synth_ratio=self.synth_ratio,
            u_ref=self.u_ref, tau_mode=self.tau_mode,
            fit_space=self.fit_space, level_strata=self.level_strata,
            exact_frac=self.exact_frac, level_coupled=self.level_coupled,
            level_ref=self.level_ref, ctx_jitter=self.ctx_jitter,
            cho_idx=self.cho_idx, cho_s=self.cho_s, emp_frac=self.emp_frac,
            tau_exact=self.tau_exact, relabel=self.relabel,
            dose_weights=self.dose_weights,
            dose_weights_meal=self.dose_weights_meal,
            dose_weights_fast=self.dose_weights_fast,
            anchor_p_meal=self.anchor_p_meal, anchor_p_low=self.anchor_p_low,
            point_spec=self.point_spec, context_remap=self.context_remap,
            cgm_share=self.cgm_share, curve_scale=self.curve_scale,
            cho_floor=self.cho_floor)
        X_fit = np.vstack([X, Xs])
        y_all = np.concatenate([y_fit, ys])
        sw = None
        if synth_weight is not None:
            sw = np.concatenate([np.ones(len(X)), np.full(len(Xs), float(synth_weight))])
        model = self.train_fn(
            X_fit, y_all, model_kwargs=kw, n_factual=len(X),
            **({} if sw is None else {"sample_weight": sw}))
        self.model_, self.baseline_, self.meta_ = model, baseline_model, dict(
            model=model, baseline_model=baseline_model, n_anchors=meta["n_anchors"], n_synth=meta["n_synth"],
            grid_n=len(self.u_grid), span=span, tau_drop=self.tau_drop, synth_ratio=self.synth_ratio,
            tau_mode=self.tau_mode, fit_space=self.fit_space, level_strata=self.level_strata,
            predicts_level=self.predicts_level)
        return self.meta_


DOCAS.train_fn = lambda X, y, model_kwargs=None, n_factual=None, sample_weight=None, **_: train_baseline(
    X, y, model_kwargs=model_kwargs, n_factual=n_factual, sample_weight=sample_weight)


def _default_span(span):
    return float(span or DOCAS.SPAN or 1.0)


def _rbg_params(*, SI=None, tau=None, alpha=None, body_weight_kg=None):
    prof = dict(Cfg.SENSOR.get(str(Cfg.DATASET).lower().replace("-", "_"), {}))
    si = float(SI if SI is not None else Cfg.SI_MODE)
    bw = float(body_weight_kg if body_weight_kg is not None else Cfg.BODY_WEIGHT_KG)
    p = {
        **Cfg.RBG,
        "SI": si,
        "tau": float(tau if tau is not None else prof["tau"]),
        "alpha": float(alpha if alpha is not None else prof["alpha"]),
    }
    return p, bw, float(Cfg.G0_TAU)


def _rbg_init(n, *, p, bw, g0):
    basal = 1000.0 / (60.0 * bw)
    return dict(
        basal=basal,
        G=np.full(n, g0),
        IG=np.full(n, g0),
        X=np.zeros(n),
        Isc1=np.full(n, basal / p["kd"]),
        Isc2=np.full(n, basal / p["ka2"]),
        Ip=np.full(n, basal / p["ke"]),
    )


def _rbg_step(state, doses_u, t, *, p, bw):
    basal = state["basal"]
    bolus = doses_u * 1000.0 / bw if t == int(round(p["tau"])) + 1 else 0.0
    insulin = basal + bolus
    state["Isc1"] = (state["Isc1"] + insulin) / (1.0 + p["kd"])
    state["Isc2"] = (state["Isc2"] + p["kd"] * state["Isc1"]) / (1.0 + p["ka2"])
    state["Ip"] = (state["Ip"] + p["ka2"] * state["Isc2"]) / (1.0 + p["ke"])
    state["X"] = (
        state["X"] + p["p2"] * (p["SI"] / p["VI"]) * (state["Ip"] - basal / p["ke"])
    ) / (1.0 + p["p2"])
    G = state["G"]
    risk = np.where(
        G < p["Gb"],
        1.0 + 10.0 * p["r1"] * (
            np.log(np.maximum(G, 60.0)) ** p["r2"] - np.log(p["Gb"]) ** p["r2"]
        ) ** 2,
        1.0,
    )
    state["G"] = (G + p["SG"] * p["Gb"]) / (1.0 + p["SG"] + risk * state["X"])
    state["IG"] = (p["alpha"] * state["IG"] + state["G"]) / (1.0 + p["alpha"])
    return state


def replaybg_simulate(
    doses_u, horizon_min, *, SI=None, tau=None, alpha=None, body_weight_kg=None, g0=None,
):
    p, bw, g0_cfg = _rbg_params(SI=SI, tau=tau, alpha=alpha, body_weight_kg=body_weight_kg)
    g0 = float(g0 if g0 is not None else g0_cfg)
    doses = np.asarray(doses_u, float).ravel()
    n, H = len(doses), int(horizon_min)
    state = _rbg_init(n, p=p, bw=bw, g0=g0)
    t_min = np.arange(H + 1, dtype=float)
    IG = np.empty((H + 1, n), float)
    IG[0] = state["IG"]
    for t in range(1, H + 1):
        _rbg_step(state, doses, t, p=p, bw=bw)
        IG[t] = state["IG"]
    return t_min, IG


def replaybg_drop_trace(
    dose_u=1.0, horizon_min=270, *, SI=None, tau=None, alpha=None, body_weight_kg=None, g0=None,
):
    t_min, IG = replaybg_simulate(
        [float(dose_u), 0.0], int(horizon_min),
        SI=SI, tau=tau, alpha=alpha, body_weight_kg=body_weight_kg, g0=g0)
    return t_min, IG[:, 0] - IG[:, 1]


def replaybg_response(
    u, horizon_min, *, SI=Cfg.SI_MODE, tau=None, alpha=None,
    body_weight_kg=Cfg.BODY_WEIGHT_KG, span=None, drop_mode="endpoint",
):
    """Fractional CGM lowering R(u) = drop / G0.

    drop_mode:
      - "endpoint" (default): IG_ctl(H) − IG_u(H), matching the ΔCGM target
      - "peak": max lowering over [0, H] (legacy)
    """
    sp = _default_span(span)
    doses = np.r_[np.asarray(u, float).ravel() * sp, 0.0]
    _, IG = replaybg_simulate(
        doses, int(horizon_min), SI=SI, tau=tau, alpha=alpha, body_weight_kg=body_weight_kg)
    ctl, treated = IG[:, -1], IG[:, :-1]
    if str(drop_mode).lower().startswith("peak"):
        drop = np.max(ctl[:, None] - treated, axis=0)
    else:
        drop = ctl[-1, None] - treated[-1]
    g0 = float(Cfg.G0_TAU)
    return (drop / g0).reshape(np.asarray(u).shape)


def calibrated_target(
    horizon_min, *, body_weight_kg=Cfg.BODY_WEIGHT_KG, SI=Cfg.SI_MODE,
    tau=None, alpha=None, span=None, drop_mode="endpoint",
):
    h = int(horizon_min)
    return lambda u, _h=h, _bw=body_weight_kg, _si=SI, _tau=tau, _a=alpha, _sp=span, _dm=drop_mode: replaybg_response(
        u, _h, SI=_si, body_weight_kg=_bw, tau=_tau, alpha=_a, span=_sp, drop_mode=_dm)


def ambient_exact_tau(horizon_min: int, span, *, si=Cfg.SI_MODE,
                      body_weight_kg=Cfg.BODY_WEIGHT_KG, g_lo=40.0, g_hi=400.0,
                      g_step=5.0, n_u=129):
    """τ(u; g) re-simulated at each ambient g, instead of scaled from the G0 response.

    The study target measures the bolus response once at G0 and extrapolates it
    proportionally to ambient glucose. The plant is not proportional in glucose,
    so this builds a lookup over (g, u) and interpolates, which keeps the target
    cheap enough to use inside synthetic-label construction.
    """
    sp = _default_span(span)
    gs = np.arange(float(g_lo), float(g_hi) + float(g_step), float(g_step))
    us = np.linspace(0.0, 1.0, int(n_u))
    tab = np.empty((len(gs), len(us)), float)
    for a, g in enumerate(gs):
        _, IG = replaybg_simulate(np.r_[us * sp, 0.0], int(horizon_min),
                                  SI=si, body_weight_kg=body_weight_kg, g0=float(g))
        tab[a] = IG[-1, -1] - IG[-1, :-1]

    def tau(u, g, idx=None, _t=tab, _gs=gs, _nu=len(us)):
        del idx  # ambient ODE depends only on (u, g)
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        g = np.clip(np.asarray(g, float), _gs[0], _gs[-1])
        u, g = np.broadcast_arrays(u, g)
        fa = (g - _gs[0]) / (_gs[1] - _gs[0])
        a0 = np.clip(np.floor(fa).astype(int), 0, len(_gs) - 2)
        wa = fa - a0
        fb = u * (_nu - 1)
        b0 = np.clip(np.floor(fb).astype(int), 0, _nu - 2)
        wb = fb - b0
        lo = _t[a0, b0] * (1.0 - wb) + _t[a0, b0 + 1] * wb
        hi = _t[a0 + 1, b0] * (1.0 - wb) + _t[a0 + 1, b0 + 1] * wb
        return -(lo * (1.0 - wa) + hi * wa)

    return tau


def circadian_si_mult(hour, *, alpha=0.20, trough_hour=7.0):
    """Diurnal SI multiplier: trough near Dawn (07:00), peak 12 h later."""
    h = np.asarray(hour, float)
    return 1.0 - float(alpha) * np.cos(2.0 * np.pi * (h - float(trough_hour)) / 24.0)


def causal_pk_states(
    grid_ns: np.ndarray, bolus_times_ns: np.ndarray, bolus_doses_u: np.ndarray,
    *, SI=None, body_weight_kg=None, g0=None,
) -> dict[str, np.ndarray]:
    """Forward 3-compartment PK + active insulin along a 5-min grid (causal).

    Boluses with time ≤ grid[t] contribute; no future leakage. Glucose is held at
    ``g0`` only for optional IG bookkeeping; the returned compartments are linear
    in the insulin path and independent of glucose.
    """
    p, bw, g0_cfg = _rbg_params(SI=SI, body_weight_kg=body_weight_kg)
    g0 = float(g0 if g0 is not None else g0_cfg)
    grid = np.asarray(grid_ns, np.int64).ravel()
    n = int(grid.size)
    bt = np.asarray(bolus_times_ns, np.int64).ravel()
    bd = np.asarray(bolus_doses_u, float).ravel()
    step_ns = int(grid[1] - grid[0]) if n > 1 else int(5 * 60 * 1e9)
    step_min = max(1, int(round(step_ns / 1e9 / 60.0)))
    doses = np.zeros(n, float)
    if bt.size:
        t0 = int(grid[0])
        for t_ns, d in zip(bt, bd):
            j = int(round((int(t_ns) - t0) / step_ns))
            if 0 <= j < n:
                doses[j] += float(d)
    state = _rbg_init(1, p=p, bw=bw, g0=g0)
    out = {k: np.empty(n, float) for k in ("Isc1", "Isc2", "Ip", "X")}
    lag = int(round(p["tau"]))
    # Sub-minute loop with bolus applied at lag after the 5-min bin that logged it.
    pending = []  # (minute_index_from_start, dose_u)
    minute = 0
    for j in range(n):
        if doses[j] > 0:
            pending.append((minute + lag + 1, float(doses[j])))
        for _ in range(step_min):
            minute += 1
            bolus = 0.0
            keep = []
            for tm, du in pending:
                if tm == minute:
                    bolus += du
                elif tm > minute:
                    keep.append((tm, du))
            pending = keep
            # One-sample step with an explicit bolus (bypass t==tau gating).
            basal = state["basal"]
            insulin = basal + bolus * 1000.0 / bw
            state["Isc1"] = (state["Isc1"] + insulin) / (1.0 + p["kd"])
            state["Isc2"] = (state["Isc2"] + p["kd"] * state["Isc1"]) / (1.0 + p["ka2"])
            state["Ip"] = (state["Ip"] + p["ka2"] * state["Isc2"]) / (1.0 + p["ke"])
            state["X"] = (
                state["X"] + p["p2"] * (p["SI"] / p["VI"]) * (state["Ip"] - basal / p["ke"])
            ) / (1.0 + p["p2"])
            G = state["G"]
            risk = np.where(
                G < p["Gb"],
                1.0 + 10.0 * p["r1"] * (
                    np.log(np.maximum(G, 60.0)) ** p["r2"] - np.log(p["Gb"]) ** p["r2"]
                ) ** 2,
                1.0,
            )
            state["G"] = (G + p["SG"] * p["Gb"]) / (1.0 + p["SG"] + risk * state["X"])
            state["IG"] = (p["alpha"] * state["IG"] + state["G"]) / (1.0 + p["alpha"])
        for k in out:
            out[k][j] = float(np.asarray(state[k]).ravel()[0])
    out["basal"] = float(state["basal"])
    return out


def _state_ode_ig_batch(
    u_norm, *, Isc1, Isc2, Ip, X, g0, u_obs_norm, span, horizon_min, SI, body_weight_kg,
):
    """Endpoint IG under probe doses from causal PK states (vectorised).

    Accepts either a single state with many ``u`` values, or many states with one
    ``u`` (or matching shapes broadcast to a common batch length).
    """
    p, bw, _ = _rbg_params(
        SI=float(np.mean(np.asarray(SI, float))) if np.ndim(SI) else SI,
        body_weight_kg=body_weight_kg)
    u = np.atleast_1d(np.asarray(u_norm, float)).astype(float)
    u_obs = np.atleast_1d(np.asarray(u_obs_norm, float)).astype(float)
    g0 = np.atleast_1d(np.asarray(g0, float)).astype(float)
    Isc1 = np.atleast_1d(np.asarray(Isc1, float)).astype(float)
    Isc2 = np.atleast_1d(np.asarray(Isc2, float)).astype(float)
    Ip = np.atleast_1d(np.asarray(Ip, float)).astype(float)
    X = np.atleast_1d(np.asarray(X, float)).astype(float)
    si = np.atleast_1d(np.asarray(SI, float)).astype(float)
    n = max(u.size, u_obs.size, g0.size, Isc1.size)
    def _br(a):
        a = np.atleast_1d(a)
        return np.full(n, float(a[0])) if a.size == 1 else a.ravel()[:n]
    u, u_obs, g0 = _br(u), np.clip(_br(u_obs), 0.0, 1.0), _br(g0)
    Isc1, Isc2, Ip, X, si = _br(Isc1), _br(Isc2), _br(Ip), _br(X), _br(si)
    eps = 1e-9
    ratio = np.ones(n, float)
    new_bolus = np.zeros(n, float)
    no_obs = u_obs <= eps
    ratio = np.where(no_obs, 0.0, ratio)
    new_bolus = np.where(no_obs, np.clip(u, 0.0, 1.0) * float(span), new_bolus)
    low = (~no_obs) & (u < u_obs - eps)
    high = (~no_obs) & (u > u_obs + eps)
    ratio = np.where(low, np.clip(u / np.maximum(u_obs, eps), 0.0, 1.0), ratio)
    ratio = np.where(u <= eps, 0.0, ratio)
    new_bolus = np.where(high, (u - u_obs) * float(span), new_bolus)
    Isc1, Isc2, Ip, X = Isc1 * ratio, Isc2 * ratio, Ip * ratio, X * ratio
    basal = 1000.0 / (60.0 * bw)
    G = g0.copy()
    IG = g0.copy()
    lag = int(round(p["tau"])) + 1
    H = int(horizon_min)
    for t in range(1, H + 1):
        bolus = np.where(t == lag, new_bolus * 1000.0 / bw, 0.0)
        insulin = basal + bolus
        Isc1 = (Isc1 + insulin) / (1.0 + p["kd"])
        Isc2 = (Isc2 + p["kd"] * Isc1) / (1.0 + p["ka2"])
        Ip = (Ip + p["ka2"] * Isc2) / (1.0 + p["ke"])
        X = (X + p["p2"] * (si / p["VI"]) * (Ip - basal / p["ke"])) / (1.0 + p["p2"])
        risk = np.where(
            G < p["Gb"],
            1.0 + 10.0 * p["r1"] * (
                np.log(np.maximum(G, 60.0)) ** p["r2"] - np.log(p["Gb"]) ** p["r2"]
            ) ** 2,
            1.0,
        )
        G = (G + p["SG"] * p["Gb"]) / (1.0 + p["SG"] + risk * X)
        IG = (p["alpha"] * IG + G) / (1.0 + p["alpha"])
    return IG


def _state_ode_ig(
    u_norm, *, Isc1, Isc2, Ip, X, g0, u_obs_norm, span, horizon_min, SI, body_weight_kg,
):
    """Endpoint interstitial glucose under probe ``u_norm`` from causal PK state."""
    return _state_ode_ig_batch(
        u_norm, Isc1=Isc1, Isc2=Isc2, Ip=Ip, X=X, g0=g0, u_obs_norm=u_obs_norm,
        span=span, horizon_min=horizon_min, SI=SI, body_weight_kg=body_weight_kg)


def make_state_tau(
    *, Isc1, Isc2, Ip, X, g_obs, u_obs_u, hours, span, horizon_min,
    si_base=None, si_row=None, circadian=False, circadian_alpha=0.20,
    shuffle=None, flip_circadian=False, body_weight_kg=None, n_u=33,
):
    """Build ``tau_exact(u, g, idx)`` → ΔCGM from null dose using causal PK state.

    Precomputes a per-row dose table at each row's observed glucose so training and
    alignment stay cheap. ``shuffle`` permutes only the PK age state across rows.
    """
    Isc1 = np.asarray(Isc1, float).ravel()
    Isc2 = np.asarray(Isc2, float).ravel()
    Ip = np.asarray(Ip, float).ravel()
    X = np.asarray(X, float).ravel()
    g_obs = np.asarray(g_obs, float).ravel()
    u_obs_u = np.asarray(u_obs_u, float).ravel()
    hours = np.asarray(hours, float).ravel()
    n = len(Isc1)
    sp = max(float(span), 1e-12)
    u_obs_n = np.clip(u_obs_u / sp, 0.0, 1.0)
    si_base = float(si_base if si_base is not None else Cfg.SI_MODE)
    if si_row is None:
        si_row = np.full(n, si_base)
    else:
        si_row = np.asarray(si_row, float).ravel()
    if circadian:
        mult = circadian_si_mult(hours, alpha=circadian_alpha)
        if flip_circadian:
            mult = circadian_si_mult(hours + 12.0, alpha=circadian_alpha)
        mult = mult / max(float(mult.mean()), 1e-12)
        si_row = si_row * mult
    if shuffle is not None:
        perm = np.asarray(shuffle, int).ravel()
        Isc1, Isc2, Ip, X = Isc1[perm], Isc2[perm], Ip[perm], X[perm]
    bw = body_weight_kg
    u_tab = np.linspace(0.0, 1.0, int(n_u))
    # Batch over rows for each probe on the normalised dose grid.
    ig0 = _state_ode_ig_batch(
        np.zeros(n), Isc1=Isc1, Isc2=Isc2, Ip=Ip, X=X, g0=g_obs, u_obs_norm=u_obs_n,
        span=sp, horizon_min=horizon_min, SI=si_row, body_weight_kg=bw)
    table = np.empty((n, len(u_tab)), float)
    table[:, 0] = 0.0
    for k, uk in enumerate(u_tab[1:], start=1):
        ig = _state_ode_ig_batch(
            np.full(n, uk), Isc1=Isc1, Isc2=Isc2, Ip=Ip, X=X, g0=g_obs,
            u_obs_norm=u_obs_n, span=sp, horizon_min=horizon_min, SI=si_row,
            body_weight_kg=bw)
        table[:, k] = ig - ig0

    def tau(u, g, idx=None):
        u = np.asarray(u, float)
        if idx is None:
            idx = np.arange(n)
        idx = np.clip(np.asarray(idx, int).ravel(), 0, n - 1)
        # Scale the stored (g_obs) response by g / g_obs so jittered ambient still
        # moves the label, while the shape stays the precomputed state-ODE curve.
        g = np.asarray(g, float).ravel()
        scale = np.ones(len(idx), float)
        if g.size == len(idx):
            den = np.maximum(g_obs[idx], 40.0)
            scale = np.clip(g / den, 0.25, 4.0)
        elif g.size == 1:
            den = np.maximum(g_obs[idx], 40.0)
            scale = np.clip(float(g[0]) / den, 0.25, 4.0)
        if u.ndim == 1 and u.size == len(idx):
            out = np.empty(len(idx), float)
            for k, j in enumerate(idx):
                out[k] = float(np.interp(float(u[k]), u_tab, table[j])) * scale[k]
            return out
        u_grid = u.ravel()
        out = np.empty((len(idx), len(u_grid)), float)
        for k, j in enumerate(idx):
            out[k] = np.interp(u_grid, u_tab, table[j]) * scale[k]
        return out

    return tau


def multivar_gain(g, cho, *, kappa=0.0, cob_beta=0.0, cob_gamma=0.0,
                  cob50=40.0, g_ref=180.0, **_):
    """Context gain on the DRC: (g/g_ref)^κ / (1 + β·c) · (1 + γ·c·(g/g_ref − 1)).

    ``c = CHO / (CHO + cob50)``. Only forecaster features (CGM, CHO) enter, so the
    learner can express the gain; the dose shape stays the ODE's.
    """
    r = np.maximum(np.asarray(g, float), 40.0) / max(float(g_ref), 40.0)
    cho = np.maximum(np.asarray(cho, float), 0.0)
    c = cho / (cho + max(float(cob50), 1e-6))
    m = r ** float(kappa) / (1.0 + float(cob_beta) * c)
    m = m * (1.0 + float(cob_gamma) * c * (r - 1.0))
    return np.clip(m, 0.05, 8.0)


def make_multivar_tau(
    base_tau, *, cho, hours=None, si_pop, si_pat,
    si_scale=1.0, patient_blend=0.0, circ_alpha=0.0, norm=1.0, **gain_kw,
):
    """Wrap a base ``tau_exact`` with patient SI, circadian, and context gain.

    ``τ ← τ_base · s_I · circ(h) · gain(g, CHO) / norm``. Passing ``norm`` as the
    mean training gain keeps the cohort-average slope of the base DRC, so the
    search cannot lower RMSE by flattening the prior.
    """
    cho = np.asarray(cho, float).ravel()
    n = len(cho)
    hours = np.zeros(n) if hours is None else np.asarray(hours, float).ravel()
    si_pop = float(si_pop)
    blend = float(np.clip(patient_blend, 0.0, 1.0))
    si_eff = (1.0 - blend) * si_pop + blend * float(si_pat)
    s0 = float(si_scale) * si_eff / max(si_pop, 1e-18) / max(float(norm), 1e-9)

    def tau(u, g, idx=None, _base=base_tau):
        if idx is None:
            idx = np.arange(n)
        idx = np.clip(np.asarray(idx, int).ravel(), 0, n - 1)
        du = np.asarray(_base(u, g, idx), float)
        g_arr = np.asarray(g, float).ravel()
        if g_arr.size != len(idx):
            g_arr = np.resize(g_arr, len(idx))
        mult = s0 * multivar_gain(g_arr, cho[idx], **gain_kw)
        if float(circ_alpha) > 1e-12:
            circ = circadian_si_mult(hours[idx], alpha=float(circ_alpha))
            mult = mult * circ / max(float(circ.mean()), 1e-12)
        mult = np.asarray(mult, float).ravel()
        return du * mult[:, None] if du.ndim == 2 else du * mult

    return tau


# --- τ from pump settings only (no glucose-insulin ODE model) -----------------
# Walsh--Roberts exponential insulin-action curve, the form programmed in insulin
# pumps and open-source AID: peak activity time and duration of insulin action.
INSULIN_PEAK_MIN, INSULIN_DIA_MIN = 75.0, 300.0


def insulin_action_fraction(t_min, *, t_peak=INSULIN_PEAK_MIN, t_dia=INSULIN_DIA_MIN):
    """Fraction of a bolus's total glucose-lowering action realised by t_min."""
    t = np.asarray(t_min, float)
    tp, td = float(t_peak), float(t_dia)
    tau = tp * (1.0 - tp / td) / (1.0 - 2.0 * tp / td)
    a = 2.0 * tau / td
    s = 1.0 / (1.0 - a + (1.0 + a) * np.exp(-td / tau))
    iob = 1.0 - s * (1.0 - a) * (
        (t ** 2 / (tau * td * (1.0 - a)) - t / tau - 1.0) * np.exp(-t / tau) + 1.0)
    iob = np.where(t <= 0.0, 1.0, np.where(t >= td, 0.0, iob))
    return np.clip(1.0 - iob, 0.0, 1.0)


def pump_response(u, horizon_min, *, cf_mgdl_per_u, span=None,
                  t_peak=INSULIN_PEAK_MIN, t_dia=INSULIN_DIA_MIN):
    """Fractional lowering from the correction factor times insulin action.

    A bolus of ``d`` units is expected to lower glucose by ``cf * d`` in total
    (the clinician's correction factor), delivered over time by the insulin-action
    curve, so the drop by ``horizon_min`` is ``cf * d * A(horizon_min)``. Returned
    as a fraction of ``G0_TAU`` like the ReplayBG response, so fractional τ mode
    scales it to ambient glucose. Nothing here comes from a physiological
    simulator.
    """
    sp = _default_span(span)
    d = np.clip(np.asarray(u, float), 0.0, 1.0) * sp
    frac = float(insulin_action_fraction(horizon_min, t_peak=t_peak, t_dia=t_dia))
    return (float(cf_mgdl_per_u) * d * frac / float(Cfg.G0_TAU)).reshape(
        np.asarray(u).shape)


def pump_target(horizon_min, *, cf_mgdl_per_u, span=None,
                t_peak=INSULIN_PEAK_MIN, t_dia=INSULIN_DIA_MIN):
    h = int(horizon_min)
    return lambda u, _h=h, _cf=cf_mgdl_per_u, _sp=span, _tp=t_peak, _td=t_dia: pump_response(
        u, _h, cf_mgdl_per_u=_cf, span=_sp, t_peak=_tp, t_dia=_td)


def pump_frac_tau_drop(horizon_min: int, span, *, cf_mgdl_per_u,
                       t_peak=INSULIN_PEAK_MIN, t_dia=INSULIN_DIA_MIN):
    r = float(np.ravel(pump_response(
        [1.0], int(horizon_min), cf_mgdl_per_u=cf_mgdl_per_u, span=span,
        t_peak=t_peak, t_dia=t_dia))[0]) or 1.0
    return -r, 1.0


def tau_from_cfg(horizon_min: int, span, cfg: dict):
    """(target, tau_drop, u_ref, tau_mode) for the τ family named by ``cfg``.

    ``cfg["tau_kind"] == "pump"`` builds τ from the correction factor and the
    insulin-action curve; anything else uses the ReplayBG dose response.
    """
    mode = str(cfg.get("tau_mode", "fractional")).lower()
    if str(cfg.get("tau_kind", "replaybg")).lower() == "pump":
        cf = float(cfg["cf_mgdl_per_u"])
        tp = float(cfg.get("insulin_peak_min", INSULIN_PEAK_MIN))
        td = float(cfg.get("insulin_dia_min", INSULIN_DIA_MIN))
        drop, u_ref = pump_frac_tau_drop(
            horizon_min, span, cf_mgdl_per_u=cf, t_peak=tp, t_dia=td)
        target = pump_target(
            horizon_min, cf_mgdl_per_u=cf, span=span, t_peak=tp, t_dia=td)
        return target, drop, u_ref, "fractional"
    si, bw = float(cfg["SI"]), float(cfg["body_weight_kg"])
    drop_mode = str(cfg.get("drop_mode", "endpoint"))
    drop, u_ref = (
        frac_tau_drop(horizon_min, span, si=si, body_weight_kg=bw, drop_mode=drop_mode)
        if mode.startswith("frac") else
        abs_tau_drop(horizon_min, span, si=si, body_weight_kg=bw, drop_mode=drop_mode))
    target = calibrated_target(
        horizon_min, body_weight_kg=bw, SI=si, span=span, drop_mode=drop_mode)
    return target, drop, u_ref, mode


def abs_tau_drop(horizon_min: int, span, *, si=Cfg.SI_MODE, body_weight_kg=Cfg.BODY_WEIGHT_KG,
                 drop_mode="endpoint"):
    sp = _default_span(span)
    u_ref = 1.0
    r = float(replaybg_response(
        [u_ref], int(horizon_min), SI=si, body_weight_kg=body_weight_kg, span=sp,
        drop_mode=drop_mode)[0]) or 1.0
    return -float(Cfg.G0_TAU) * r, u_ref


def frac_tau_drop(horizon_min: int, span, *, si=Cfg.SI_MODE, body_weight_kg=Cfg.BODY_WEIGHT_KG,
                  drop_mode="endpoint"):
    """Edge fractional lowering −R(u_ref); scale by ambient ĝ_b at train/align time."""
    sp = _default_span(span)
    u_ref = 1.0
    r = float(replaybg_response(
        [u_ref], int(horizon_min), SI=si, body_weight_kg=body_weight_kg, span=sp,
        drop_mode=drop_mode)[0]) or 1.0
    return -float(r), u_ref


def docas_fit_kw(data, span, horizon_min, *, cfg, model_kwargs=(), **over):
    cfg = {**cfg, **over}
    si, bw = float(cfg["SI"]), float(cfg["body_weight_kg"])
    target, drop, u_ref, mode = tau_from_cfg(int(horizon_min), span, cfg)
    gi, ji = data.feature_names.index("CGM"), data.insulin_idx
    cho_idx = data.feature_names.index("CHO")
    return dict(
        target=target,
        intervention_idx=ji, outcome_idx=gi,
        action_scaler=data.train_scalers[ji], level_scaler=data.train_scalers[gi],
        delta_scaler=data.cgm_scaler, span=float(span),
        model_kwargs=dict(model_kwargs),
        tau_drop=drop, u_ref=u_ref, tau_mode=mode,
        fit_space=str(cfg.get("fit_space", "delta")),
        level_strata=int(cfg.get("level_strata", 0)),
        exact_frac=bool(cfg.get("exact_frac", True)),
        level_coupled=bool(cfg.get("level_coupled", True)),
        level_ref=str(cfg.get("level_ref", "current")),
        synth_ratio=float(cfg["synth_ratio"]),
        ctx_jitter=float(cfg["ctx_jitter"]),
        emp_frac=float(cfg.get("emp_frac", 0.5)),
        cho_idx=cho_idx, cho_scaler=data.train_scalers[cho_idx],
    ), drop, u_ref, si, bw
