"""Research forecasters (LightGBM, Prendin-style LSTM). Not part of the PyPI package."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

LGBM_KWARGS = dict(n_estimators=100, max_depth=3, learning_rate=.1, random_state=42, verbose=-1, n_jobs=1)
LSTM_SEED = 42
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


class LightGBMForecaster:
    def __init__(self, **model_kwargs):
        self.model_kwargs, self._model = dict(model_kwargs), None

    def fit(self, X, y, sample_weight=None):
        import lightgbm as lgb
        self._model = lgb.LGBMRegressor(**{**LGBM_KWARGS, **self.model_kwargs})
        kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight, float)}
        self._model.fit(np.asarray(X, float), np.asarray(y, float).ravel(), **kw)
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, float))


class MonotonicLightGBMForecaster(LightGBMForecaster):
    def __init__(self, *, insulin_idx=1, n_features=3, **model_kwargs):
        super().__init__(**model_kwargs)
        c = [0] * n_features
        c[insulin_idx] = -1
        self.model_kwargs["monotone_constraints"] = c


class LSTMForecaster:
    """Prendin-style LSTM(64)+Dense (interpretable-DL4BGP)."""

    def __init__(self, *, units=64, epochs=100, batch_size=256, dense_activation=None):
        self.units, self.epochs, self.batch_size = units, epochs, batch_size
        self.dense_activation = dense_activation
        self._model = None

    @staticmethod
    def _iob_kernel(*, pad=18, ts_min=5.0):
        k1, k2, k3 = 0.0173, 0.0116, 6.75
        curve = np.zeros(360, float)
        for t in range(360):
            curve[t] = 1.0 - 0.75 * (
                (-k3 / (k2 * (k1 - k2)) * (np.exp(-k2 * t / 0.75) - 1.0)
                 + k3 / (k1 * (k1 - k2)) * (np.exp(-k1 * t / 0.75) - 1.0)) / 2.4947e4)
        curve = np.concatenate([np.zeros(int(pad), float), curve])
        k = curve[::int(ts_min)][::-1]
        return k / max(float(k.sum()), 1e-12)

    @staticmethod
    def _cob_kernel(*, ts_min=5.0, dynamic="slow"):
        mat = Path(__file__).resolve().parents[1] / "interpretable-DL4BGP" / "Sources" / "addsOn" / "COB.mat"
        if mat.is_file():
            import scipy.io
            cob = np.asarray(scipy.io.loadmat(mat)["COB"], float)
            col = 1 if dynamic == "slow" else 0
            k = cob[:, col][::int(ts_min)][::-1]
            return k / max(float(k.sum()), 1e-12)
        t = np.arange(0.0, 361.0, float(ts_min))
        k = t * np.exp(-t / 40.0)
        return k / max(float(k.sum()), 1e-12)

    @classmethod
    def physiological_smooth(cls, X, *, ji, ci, pad=18):
        X = np.asarray(X, float).copy()
        iob_k, cob_k = cls._iob_kernel(pad=pad), cls._cob_kernel()
        if len(X) > max(len(iob_k), len(cob_k)):
            X[:, ji] = np.convolve(X[:, ji], iob_k, mode="same")
            X[:, ci] = np.convolve(X[:, ci], cob_k, mode="same")
        return X

    def _build(self, n_features):
        import tensorflow as tf
        np.random.seed(LSTM_SEED)
        tf.random.set_seed(LSTM_SEED)
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, n_features)),
            tf.keras.layers.LSTM(self.units),
            tf.keras.layers.Dense(1, activation=self.dense_activation),
        ])
        m.compile(loss="mean_squared_error", optimizer="adam")
        return m

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        self._model = self._build(X.shape[1])
        kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight, float)}
        self._model.fit(X.reshape(len(X), 1, X.shape[1]), y, epochs=self.epochs,
                        batch_size=self.batch_size, verbose=0, **kw)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self._model.predict(X.reshape(len(X), 1, X.shape[1]), verbose=0).ravel()


class NPLSTMForecaster(LSTMForecaster):
    """Prendin np-LSTM: plain LSTM(64)+Dense(1)."""


class PLSTMForecaster(LSTMForecaster):
    """Prendin p-LSTM: offline IOB/COB smoothing + Dense(tanh)."""

    def __init__(self, *, ji, ci, pad=18, **kwargs):
        kwargs.setdefault("dense_activation", "tanh")
        super().__init__(**kwargs)
        self.ji, self.ci, self.pad = ji, ci, pad

    def smooth(self, X):
        return self.physiological_smooth(X, ji=self.ji, ci=self.ci, pad=self.pad)

    def fit(self, X, y, sample_weight=None):
        return super().fit(self.smooth(X), y, sample_weight=sample_weight)


def train_baseline(X, y, *, model_kwargs=None, sample_weight=None):
    return LightGBMForecaster(**dict(model_kwargs or {})).fit(X, y, sample_weight=sample_weight)


def train_monotonic(X, y, *, insulin_idx=1, n_features=3, model_kwargs=None, sample_weight=None):
    return MonotonicLightGBMForecaster(
        insulin_idx=insulin_idx, n_features=n_features, **dict(model_kwargs or {}),
    ).fit(X, y, sample_weight=sample_weight)
