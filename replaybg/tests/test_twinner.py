"""Tests for ``twinner.twinner`` (the MAP estimator).

Two tiers:

* Deterministic unit tests for the pieces that make up the objective — the
  Gaussian-copula structure builder ``_build_correlation_structure`` and the
  ``Twinner._log_prior`` / ``_log_likelihood`` / ``_log_posterior`` methods.
  These use lightweight fakes so the expected values can be computed by hand,
  independently of any physiological model.
* One end-to-end tier that drives the real ``SingleMealT1DModel`` /
  ``SingleMealT1DData`` through ``Twinner.twin`` on noise-free synthetic data,
  checking the returned contract (shape, rounding, clipping, determinism) and
  that a known parameter is recovered.

The end-to-end tests run sequentially with a small number of starts on one real
day of data (``example/data_day_1.parquet`` with ``example/patient_info.parquet``)
so the ~22-hour window makes kinetic parameters like ``SI`` identifiable. They
pay a one-time Numba JIT compilation cost on the first model instantiation.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from distributions import Gamma, LogNormal, Normal, Uniform
from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from utils.numba_dicts import to_typed_f64_dict
from twinner.twinner import Twinner, _build_correlation_structure


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"


@pytest.fixture
def tw():
    """A quiet, sequential twinner (config irrelevant for the unit-method tests)."""
    return Twinner(parallelize=False, n_starts=4, verbose=False)


# --------------------------------------------------------------------------- #
# _build_correlation_structure
# --------------------------------------------------------------------------- #
def test_correlation_structure_none_when_empty():
    prior = {"a": {}, "b": {}}
    assert _build_correlation_structure(prior, None) is None
    assert _build_correlation_structure(prior, {}) is None


def test_correlation_structure_matrix_terms():
    prior = {"a": {}, "b": {}}
    rho = 0.5
    cs = _build_correlation_structure(prior, {("a", "b"): rho})

    R = np.array([[1.0, rho], [rho, 1.0]])
    expected_inv_minus_i = np.linalg.inv(R) - np.eye(2)
    _, expected_logdet = np.linalg.slogdet(R)

    np.testing.assert_allclose(cs["R_inv_minus_I"], expected_inv_minus_i)
    assert cs["logdet_R"] == pytest.approx(expected_logdet)


def test_correlation_structure_rejects_non_pd_matrix():
    # rho = 1.1 makes R indefinite (not positive definite).
    prior = {"a": {}, "b": {}}
    with pytest.raises(ValueError, match="positive definite"):
        _build_correlation_structure(prior, {("a", "b"): 1.1})


# --------------------------------------------------------------------------- #
# _log_prior
# --------------------------------------------------------------------------- #
def _independent_prior():
    return {
        "SI": {"prior": Gamma(3.3, 1 / 5e-4), "min": 0.0, "max": 0.1},
        "Gb": {"prior": Normal(119.13, 7.11), "min": 70.0, "max": 150.0},
    }


def test_log_prior_matches_truncated_marginal_sum(tw):
    prior = _independent_prior()
    model = SimpleNamespace(SI=1e-3, Gb=120.0)

    expected = 0.0
    for name, v in prior.items():
        x = getattr(model, name)
        density = v["prior"].evaluate(x)
        z = v["prior"].cdf(v["max"]) - v["prior"].cdf(v["min"])
        expected += np.log(density) - np.log(z)

    assert tw._log_prior(model, prior) == pytest.approx(expected)


def test_log_prior_out_of_bounds_is_neg_inf(tw):
    prior = _independent_prior()
    # SI above its max -> -inf regardless of the other (valid) parameter.
    model = SimpleNamespace(SI=0.2, Gb=120.0)
    assert tw._log_prior(model, prior) == -np.inf


def test_log_prior_zero_correlation_matches_independent(tw):
    # rho = 0 gives R = I, so the copula correction term is exactly zero.
    prior = _independent_prior()
    model = SimpleNamespace(SI=1e-3, Gb=120.0)
    cs = _build_correlation_structure(prior, {("SI", "Gb"): 0.0})
    assert tw._log_prior(model, prior, cs) == pytest.approx(tw._log_prior(model, prior))


def test_log_prior_copula_correction_matches_formula(tw):
    prior = _independent_prior()
    model = SimpleNamespace(SI=1e-3, Gb=120.0)
    rho = 0.5
    cs = _build_correlation_structure(prior, {("SI", "Gb"): rho})

    # Reconstruct the Gaussian-copula correction independently.
    z = []
    for name in cs["names"]:
        v = prior[name]
        x = getattr(model, name)
        fa = v["prior"].cdf(v["min"])
        fb = v["prior"].cdf(v["max"])
        fx = v["prior"].cdf(x)
        u = np.clip((fx - fa) / (fb - fa), 1e-12, 1.0 - 1e-12)
        z.append(ndtri(u))
    z = np.array(z)
    R = np.array([[1.0, rho], [rho, 1.0]])
    _, logdet = np.linalg.slogdet(R)
    correction = -0.5 * logdet - 0.5 * z @ (np.linalg.inv(R) - np.eye(2)) @ z

    got = tw._log_prior(model, prior, cs) - tw._log_prior(model, prior)
    assert got == pytest.approx(correction)


# --------------------------------------------------------------------------- #
# _log_likelihood
# --------------------------------------------------------------------------- #
class _ConstModel:
    """A stand-in model whose output is a fixed glucose value at every step."""

    def __init__(self, g):
        self.g = g

    def output(self, t):
        return self.g

    def step(self, u, t):
        pass


def test_log_likelihood_gaussian_cv_model(tw):
    g = 100.0
    tsteps, yts = 10, 2
    y_samples = np.array([90.0, 110.0, 100.0, 105.0, 95.0])
    rbg = SimpleNamespace(
        tsteps=tsteps,
        yts=yts,
        u=np.zeros((tsteps, 5)),
        y=y_samples,
        y_idxs=np.arange(y_samples.shape[0]),
    )

    ll = tw._log_likelihood(_ConstModel(g), rbg)

    # The model is constant, so the subsampled prediction is g everywhere.
    out = np.full(y_samples.shape[0], g)
    residuals = out - y_samples
    sdn = 0.05 * np.abs(out)  # cv = 5%, hard-coded in the method
    n = len(residuals)
    expected = (
        -0.5 * np.sum((residuals / sdn) ** 2)
        - np.sum(np.log(sdn))
        - 0.5 * n * np.log(2 * np.pi)
    )
    assert ll == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# _log_posterior
# --------------------------------------------------------------------------- #
class _FakeModel(_ConstModel):
    """Constant-output model that also records parameters set via ``reset``."""

    def __init__(self, g):
        super().__init__(g)
        self.SI = 0.0
        self.beta = 0.0

    def reset(self, theta):
        for k in theta:
            setattr(self, k, float(theta[k]))


@pytest.fixture
def posterior_rbg():
    tsteps, yts = 10, 2
    y_samples = np.array([90.0, 110.0, 100.0, 105.0, 95.0])
    return SimpleNamespace(
        tsteps=tsteps,
        yts=yts,
        u=np.zeros((tsteps, 5)),
        y=y_samples,
        y_idxs=np.arange(y_samples.shape[0]),
    )


def test_log_posterior_rounds_integer_parameters(tw, posterior_rbg):
    prior = {"beta": {"prior": Uniform(0, 60), "min": 0, "max": 60, "integer": True}}
    model = _FakeModel(100.0)
    tw._log_posterior(np.array([3.7]), model, posterior_rbg, prior)
    # 3.7 must be rounded to 4 before being written into the model.
    assert model.beta == 4.0


def test_log_posterior_out_of_bounds_returns_neg_inf_triple(tw, posterior_rbg):
    prior = {"SI": {"prior": Gamma(3.3, 1 / 5e-4), "min": 0.0, "max": 0.1}}
    model = _FakeModel(100.0)
    lp, ll, post = tw._log_posterior(np.array([0.5]), model, posterior_rbg, prior)
    assert (lp, ll, post) == (-np.inf, -np.inf, -np.inf)


def test_log_posterior_is_sum_of_components(tw, posterior_rbg):
    prior = {"SI": {"prior": Gamma(3.3, 1 / 5e-4), "min": 0.0, "max": 0.1}}
    model = _FakeModel(100.0)
    lp, ll, post = tw._log_posterior(np.array([1e-3]), model, posterior_rbg, prior)
    assert np.isfinite(post)
    assert post == pytest.approx(lp + ll)


# --------------------------------------------------------------------------- #
# twin (end-to-end)
# --------------------------------------------------------------------------- #
def _simulate(model, rbg_data):
    """Forward-simulate ``model`` on ``rbg_data`` inputs; return subsampled output.

    Mirrors the simulation loop inside ``Twinner._log_likelihood`` so we can
    fabricate noise-free observations consistent with the model.
    """
    out = np.zeros(rbg_data.tsteps)
    out[0] = model.output(0)
    for k in range(1, rbg_data.tsteps):
        model.step(rbg_data.u[k], k)
        out[k] = model.output(k)
    return out[0 :: rbg_data.yts]


@pytest.fixture
def sm_setup(env):
    """A prepared single-meal data object plus a fresh model over it.

    Loads one real day of data (``example/data_day_1.parquet``) with the subject
    body weight from ``example/patient_info.parquet``. The ~22-hour window (264
    5-minute samples) is long enough for kinetic parameters to be identifiable.
    """
    df = pd.read_parquet(EXAMPLE_DIR / "data_day_1.parquet")
    df["t"] = pd.to_datetime(df["t"])
    patient_info = pd.read_parquet(EXAMPLE_DIR / "patient_info.parquet")
    body_weight = float(patient_info["bw"].iloc[0])

    rbg_data = SingleMealT1DData(
        data=df, body_weight=body_weight, environment=env
    )
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)
    return model, rbg_data


def test_twin_recovers_known_parameter(sm_setup):
    model, rbg_data = sm_setup

    # Generate noise-free observations from the model at a known SI, then make
    # them the data. With a flat (Uniform) prior the MAP estimate equals the
    # MLE, which for exact data is the generating SI. Over this full-day window
    # (with real meals and boluses driving insulin action) SI is identifiable.
    true_si = 1.2e-3
    model.reset(to_typed_f64_dict({"SI": true_si}))
    synthetic = _simulate(model, rbg_data)
    rbg_data.y = synthetic.astype(float)
    rbg_data.y_idxs = np.arange(synthetic.shape[0])

    prior = {"SI": {"prior": Uniform(a=0.0, b=0.1), "min": 0.0, "max": 0.1}}
    result = Twinner(parallelize=False, n_starts=8, verbose=False).twin(
        model, rbg_data, prior
    )

    assert set(result) == {"fun", "x"}
    assert result["x"].shape == (1,)
    assert np.isfinite(result["fun"])
    assert result["x"][0] == pytest.approx(true_si, rel=0.1)


def test_twin_rounds_and_clips_result(sm_setup):
    model, rbg_data = sm_setup
    prior = {
        "SI": {"prior": Uniform(a=0.0, b=0.1), "min": 0.0, "max": 0.1},
        "beta": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
    }
    result = Twinner(parallelize=False, n_starts=4, verbose=False).twin(
        model, rbg_data, prior
    )

    si, beta = result["x"]
    assert 0.0 <= si <= 0.1
    assert 0 <= beta <= 60
    assert beta == float(int(beta))  # integer parameter is rounded


def test_twin_is_deterministic(sm_setup):
    model, rbg_data = sm_setup
    prior = {"SI": {"prior": Uniform(a=0.0, b=0.1), "min": 0.0, "max": 0.1}}
    twinner = Twinner(parallelize=False, n_starts=4, verbose=False)

    r1 = twinner.twin(model, rbg_data, prior)
    r2 = twinner.twin(model, rbg_data, prior)

    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["fun"] == r2["fun"]
