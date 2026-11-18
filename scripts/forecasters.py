"""Research forecasters (CatBoost, MLP, persistence, insulin-removed)."""
from __future__ import annotations

import warnings
import numpy as np

SEED = 42
VAL_FRAC, EARLY_STOP_ROUNDS = 0.2, 50
# CatBoost library defaults (depth=6, iterations=1000, auto learning rate);
# only pin seed / silence for reproducibility.
CAT_KWARGS = dict(random_seed=SEED, verbose=False, allow_writing_files=False)
MLP_HIDDEN, MLP_EPOCHS, MLP_BATCH, MLP_LR = (256,), 100, 512, 0.03
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message=".*remainder.*ColumnTransformer.*", category=FutureWarning)


def _val_split(n, val_frac, n_factual, seed):
    """Early-stopping split: chronological tail of the factual block (CGM rows are
    autocorrelated, so a random split leaks), random tail of the synthetic block."""
    n_val = int(round(float(val_frac) * n)) if val_frac > 0 else 0
    if n_val < 50:
        return None, None
    nf = int(n if n_factual is None else n_factual)
    k = max(1, int(round(float(val_frac) * nf)))
    va = [np.arange(nf - k, nf)]
    tr = [np.arange(0, nf - k)]
    if nf < n:
        rest = np.random.default_rng(seed).permutation(np.arange(nf, n))
        m = max(1, int(round(float(val_frac) * (n - nf))))
        va.append(rest[:m])
        tr.append(rest[m:])
    return np.concatenate(tr), np.concatenate(va)


class CatBoostForecaster:
    """CatBoost trunk at library defaults; boosting length is chosen per arm by
    early stopping.

    Each arm validates on a held-out slice of its own training distribution, so
    the DOCAS trunk (factual + counterfactual rows) is not capped at the number
    of iterations that suffices for the smaller factual-only baseline, and the
    baseline is not forced to overfit a budget sized for the augmented problem.
    """

    def __init__(self, **model_kwargs):
        self._kw, self._model = {**CAT_KWARGS, **model_kwargs}, None
        self._val = float(self._kw.pop("val_frac", VAL_FRAC))
        self._patience = int(self._kw.pop("early_stopping_rounds", EARLY_STOP_ROUNDS))

    def fit(self, X, y, *, n_factual=None, sample_weight=None, **_):
        import catboost as cb
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        w = None if sample_weight is None else np.asarray(sample_weight, float).ravel()
        self._model = cb.CatBoostRegressor(**self._kw)
        tr, va = _val_split(len(y), self._val, n_factual,
                            int(self._kw.get("random_seed", SEED)))
        if va is None:
            if w is None:
                self._model.fit(X, y)
            else:
                self._model.fit(cb.Pool(X, y, weight=w))
            return self
        if w is None:
            self._model.fit(X[tr], y[tr], eval_set=(X[va], y[va]),
                            early_stopping_rounds=self._patience, use_best_model=True)
        else:
            self._model.fit(
                cb.Pool(X[tr], y[tr], weight=w[tr]),
                eval_set=cb.Pool(X[va], y[va], weight=w[va]),
                early_stopping_rounds=self._patience, use_best_model=True)
        return self

    @property
    def n_trees(self):
        return int(self._model.tree_count_)

    def predict(self, X):
        return self._model.predict(X)


class PersistenceForecaster:
    """Zero-change predictor: ΔCGM = 0 mg/dL.

    Targets are min–max scaled, so a raw 0 decodes to the most negative ΔCGM
    seen in training, not to no change; the caller passes the scaled image of
    0 mg/dL.
    """

    def __init__(self, zero_scaled=0.0):
        self.zero_scaled = float(zero_scaled)

    def fit(self, X, y, **_):
        return self

    def predict(self, X):
        return np.full(len(np.atleast_2d(np.asarray(X, float))), self.zero_scaled, float)


class InsulinRemovedForecaster:
    """CatBoost trunk with insulin column(s) dropped."""

    def __init__(self, insulin_idx, **model_kwargs):
        self._drop = (
            [int(insulin_idx)] if np.isscalar(insulin_idx)
            else [int(i) for i in insulin_idx])
        self._kw = dict(model_kwargs)
        self.base = None
        self.keep = None

    def fit(self, X, y, **_):
        X = np.asarray(X, float)
        self.keep = [c for c in range(X.shape[1]) if c not in self._drop]
        self.base = CatBoostForecaster(**self._kw).fit(X[:, self.keep], y)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return self.base.predict(X[:, self.keep])


def train_baseline(X, y, *, model_kwargs=(), n_factual=None, sample_weight=None, **_):
    return CatBoostForecaster(**dict(model_kwargs)).fit(
        X, y, n_factual=n_factual, sample_weight=sample_weight)


def train_persistence(X, y, *, model_kwargs=(), delta_scaler=None, **_):
    zero = 0.0 if delta_scaler is None else float(delta_scaler.transform([[0.0]])[0, 0])
    return PersistenceForecaster(zero).fit(X, y)


def train_monotone(X, y, *, model_kwargs=(), insulin_idx=1, **_):
    """CatBoost with monotone decreasing constraint on insulin → ΔCGM."""
    n = np.asarray(X, float).shape[1]
    constraints = [0] * n
    constraints[int(insulin_idx)] = -1
    kw = {**dict(model_kwargs), "monotone_constraints": constraints}
    return CatBoostForecaster(**kw).fit(X, y)


def train_insulin_removed(X, y, *, model_kwargs=(), insulin_idx=1, **_):
    return InsulinRemovedForecaster(insulin_idx, **dict(model_kwargs)).fit(X, y)


class MLPForecaster:
    def __init__(self, hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, lr=MLP_LR, seed=SEED):
        self.hidden = tuple(int(h) for h in hidden)
        self.epochs, self.batch_size, self.lr, self.seed = int(epochs), int(batch_size), float(lr), int(seed)
        self._model = None

    def _build(self, d_in):
        import torch.nn as nn
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        layers, prev = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        self._model = self._build(X.shape[1])
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        n, bs = len(X), min(len(X), self.batch_size)
        self._model.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(n).split(bs):
                pred = self._model(xt[idx]).squeeze(-1)
                loss = ((pred - yt[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    def predict(self, X):
        import torch
        with torch.inference_mode():
            return self._model(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


def train_mlp(X, y, *, model_kwargs=(), **_):
    return MLPForecaster(**dict(model_kwargs)).fit(X, y)


PENALTY_LAMBDA = 0.25
N_PENALTY_CTX = 32
N_PENALTY_GRID = 11


class PenaltyMLPForecaster(MLPForecaster):
    """MLP with a soft penalty that pulls each context's insulin curve toward the target."""

    def __init__(self, *, aligner, penalty_lambda=PENALTY_LAMBDA, **kw):
        super().__init__(**kw)
        self.aligner = aligner
        self.penalty_lambda = float(penalty_lambda)

    def fit(self, X, y, *, n_factual=None, **_):
        import torch
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        nf = int(n_factual if n_factual is not None else len(X))
        X_fact = X[:nf]
        self._model = self._build(X.shape[1])
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        n, bs = len(X), min(len(X), self.batch_size)
        a = self.aligner
        u_grid = np.linspace(0.0, 1.0, N_PENALTY_GRID)
        ji, yi, span = a.ji, a.yi, float(a.span)
        a_s, dy_s, y_s = a.a_s, a.dy_s, a.y_s
        g_scale = float(y_s.scale_[0])
        g_min = float(y_s.min_[0])
        dy_scale = float(dy_s.scale_[0])
        dy_min = float(dy_s.min_[0])
        r_ref = float(np.asarray(a.target([a.u_ref]), float).ravel()[0]) or 1.0
        R = np.asarray(a.target(u_grid), float).ravel() / r_ref
        self._model.train()
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(n).split(bs):
                pred = self._model(xt[idx]).squeeze(-1)
                loss = ((pred - yt[idx]) ** 2).mean()
                ctx_idx = rng.choice(nf, size=min(N_PENALTY_CTX, nf), replace=False)
                ctx = X_fact[ctx_idx]
                g = (ctx[:, yi] - g_min) / g_scale
                probes = np.tile(u_grid * span, len(ctx)).reshape(-1, 1)
                rows = np.repeat(ctx, len(u_grid), 0)
                rows[:, ji] = a_s.transform(probes).ravel()
                pred_s = self._model(torch.as_tensor(rows, dtype=torch.float32)).squeeze(-1)
                pred_delta = (pred_s - dy_min) / dy_scale
                pred_delta = pred_delta.reshape(len(ctx), len(u_grid))
                target_delta = torch.as_tensor(
                    (g * a.tau_drop)[:, None] * R[None, :], dtype=torch.float32)
                ice = pred_delta - pred_delta[:, :1]
                tgt = target_delta - target_delta[:, :1]
                loss = loss + self.penalty_lambda * ((ice - tgt) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self._model.eval()
        return self
