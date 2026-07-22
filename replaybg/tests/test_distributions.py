"""Tests for the prior distributions in ``distributions``.

Each distribution is a Numba ``@jitclass`` exposing the same public API:
``__init__`` (stores parameters), ``evaluate`` (pdf), ``cdf`` and ``sample``
(bounded draw). The suite covers every method of every distribution: densities
and CDFs are checked against ``scipy.stats`` references (plus their support /
edge cases), and ``sample`` is checked for the bounded-draw contract,
seed-reproducibility and clipping.

Note: Numba ``@jitclass`` methods reject keyword arguments, so ``seed`` is
always passed positionally as ``sample(min_val, max_val, seed)``.
"""
import numpy as np
import pytest
from scipy import special, stats

from distributions import Gamma, LogNormal, Normal, Uniform


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def test_normal_constructor_stores_parameters():
    d = Normal(1.5, 2.0)
    assert d.mu == 1.5
    assert d.sigma == 2.0


def test_lognormal_constructor_stores_parameters():
    d = LogNormal(0.5, 0.75)
    assert d.mu == 0.5
    assert d.sigma == 0.75


def test_uniform_constructor_stores_parameters():
    d = Uniform(2.0, 6.0)
    assert d.a == 2.0
    assert d.b == 6.0


def test_gamma_constructor_caches_log_normalizer():
    alpha, beta = 2.0, 3.0
    d = Gamma(alpha, beta)
    assert d.alpha == alpha
    assert d.beta == beta
    # _log_normalizer = alpha*log(beta) - gammaln(alpha); check via scipy.
    expected = alpha * np.log(beta) - special.gammaln(alpha)
    assert d._log_normalizer == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# evaluate (pdf)
# --------------------------------------------------------------------------- #
def test_normal_pdf_matches_scipy():
    d = Normal(1.5, 2.0)
    for x in (-3.0, 0.0, 1.5, 4.2):
        assert d.evaluate(x) == pytest.approx(stats.norm.pdf(x, 1.5, 2.0))


def test_lognormal_pdf_matches_scipy():
    mu, sigma = 0.5, 0.75
    d = LogNormal(mu, sigma)
    for x in (0.1, 1.0, 2.5, 6.0):
        ref = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        assert d.evaluate(x) == pytest.approx(ref)


def test_gamma_pdf_matches_scipy():
    alpha, beta = 2.0, 3.0  # rate parameterization -> scale = 1/beta
    d = Gamma(alpha, beta)
    for x in (0.1, 0.5, 1.0, 3.0):
        ref = stats.gamma.pdf(x, a=alpha, scale=1.0 / beta)
        assert d.evaluate(x) == pytest.approx(ref, rel=1e-6)


def test_gamma_pdf_non_positive_is_zero():
    d = Gamma(2.0, 3.0)
    assert d.evaluate(0.0) == 0.0
    assert d.evaluate(-1.0) == 0.0


def test_uniform_pdf_and_support():
    d = Uniform(2.0, 6.0)
    assert d.evaluate(4.0) == pytest.approx(0.25)
    assert d.evaluate(2.0) == pytest.approx(0.25)  # inclusive lower bound
    assert d.evaluate(6.0) == pytest.approx(0.25)  # inclusive upper bound
    assert d.evaluate(1.9) == 0.0
    assert d.evaluate(6.1) == 0.0


# --------------------------------------------------------------------------- #
# cdf
# --------------------------------------------------------------------------- #
def test_normal_cdf_matches_scipy():
    d = Normal(1.5, 2.0)
    for x in (-3.0, 0.0, 1.5, 4.2):
        assert d.cdf(x) == pytest.approx(stats.norm.cdf(x, 1.5, 2.0))


def test_lognormal_cdf_matches_scipy():
    mu, sigma = 0.5, 0.75
    d = LogNormal(mu, sigma)
    for x in (0.1, 1.0, 2.5, 6.0):
        ref = stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
        assert d.cdf(x) == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_lognormal_cdf_non_positive_is_zero():
    d = LogNormal(0.0, 1.0)
    assert d.cdf(0.0) == 0.0
    assert d.cdf(-1.0) == 0.0


def test_gamma_cdf_matches_scipy():
    alpha, beta = 2.0, 3.0
    d = Gamma(alpha, beta)
    # Spans both branches of the incomplete-gamma routine: series (beta*x < a+1)
    # for the small x, continued fraction (beta*x >= a+1) for the large x.
    for x in (0.1, 0.5, 1.0, 3.0, 10.0):
        ref = stats.gamma.cdf(x, a=alpha, scale=1.0 / beta)
        assert d.cdf(x) == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_gamma_cdf_non_positive_is_zero():
    d = Gamma(2.0, 3.0)
    assert d.cdf(0.0) == 0.0
    assert d.cdf(-1.0) == 0.0


def test_uniform_cdf_clips_to_unit_interval():
    d = Uniform(2.0, 6.0)
    assert d.cdf(1.0) == 0.0
    assert d.cdf(2.0) == 0.0
    assert d.cdf(4.0) == pytest.approx(0.5)
    assert d.cdf(6.0) == 1.0
    assert d.cdf(7.0) == 1.0


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #
ALL_DISTS = [
    Normal(0.0, 1.0),
    LogNormal(0.0, 1.0),
    Gamma(2.0, 3.0),
    Uniform(-5.0, 5.0),
]


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_sample_respects_bounds(dist):
    lo, hi = 0.2, 0.8
    for _ in range(200):
        s = dist.sample(lo, hi)
        assert lo <= s <= hi


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_sample_is_reproducible_with_seed(dist):
    a = dist.sample(-10.0, 10.0, 123)
    b = dist.sample(-10.0, 10.0, 123)
    assert a == b


@pytest.mark.parametrize("dist", ALL_DISTS)
def test_sample_clips_to_degenerate_bounds(dist):
    # min(max(draw, c), c) == c for any draw, so an empty [c, c] interval pins
    # every sample to c regardless of the underlying distribution.
    for _ in range(50):
        assert dist.sample(5.0, 5.0) == 5.0
