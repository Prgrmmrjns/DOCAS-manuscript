import math

import numpy as np
from numba import float64

from environment import jitclass_, njit_

_SQRT2 = math.sqrt(2.0)

@jitclass_([
    ("mu", float64),
    ("sigma", float64),
])
class LogNormal(object):
    """Log-normal distribution helper.

    The class provides density evaluation and bounded random sampling for a
    log-normal distribution parameterized by the mean and standard deviation
    of the underlying normal distribution.

    ...
    Attributes
    ----------
    mu : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, mu, sigma):
        """Constructs all the necessary attributes for the LogNormal object.

        Parameters
        ----------
        mu : float
            Mean of the underlying normal distribution.
        sigma : float
            Standard deviation of the underlying normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """Evaluates the log-normal probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density. Must be positive.

        Returns
        -------
        float
            The probability density value at ``x``.
        """
        return 1 / (x * self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2)

    def cdf(self, x):
        """Evaluates the log-normal cumulative distribution function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability at ``x``. Returns ``0.0`` if ``x <= 0``.
        """
        if x <= 0:
            return 0.0
        return 0.5 * (1.0 + math.erf((np.log(x) - self.mu) / (self.sigma * _SQRT2)))

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.lognormal(self.mu, self.sigma), min_val), max_val)

@jitclass_([
    ("mu", float64),
    ("sigma", float64),
])
class Normal(object):
    """Normal distribution helper.

    The class provides density evaluation and bounded random sampling for a
    Gaussian distribution parameterized by its mean and standard deviation.

    ...
    Attributes
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, mu, sigma):
        """Constructs all the necessary attributes for the Normal object.

        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sigma : float
            Standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """Evaluates the normal probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density.

        Returns
        -------
        float
            The probability density value at ``x``.
        """
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - self.mu) / self.sigma) ** 2)

    def cdf(self, x):
        """Evaluates the normal cumulative distribution function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability at ``x``.
        """
        return 0.5 * (1.0 + math.erf((x - self.mu) / (self.sigma * _SQRT2)))

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.normal(self.mu, self.sigma), min_val), max_val)

@jitclass_([
    ("alpha", float64),
    ("beta", float64),
    ("_log_normalizer", float64),
])
class Gamma:
    """Gamma distribution helper.

    The class provides density evaluation and bounded random sampling for a
    Gamma distribution parameterized in rate form.

    ...
    Attributes
    ----------
    alpha : float
        Shape parameter.
    beta : float
        Rate parameter, where ``beta = 1 / scale``.
    _log_normalizer : float
        Cached log-normalization constant.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, alpha, beta):
        """Constructs all the necessary attributes for the Gamma object.

        Parameters
        ----------
        alpha : float
            Shape parameter.
        beta : float
            Rate parameter, where ``beta = 1 / scale``.
        """
        self.alpha = alpha
        self.beta = beta  # rate parameterization (beta = 1/scale)
        self._log_normalizer = alpha * np.log(beta) - _gammaln(alpha)

    def evaluate(self, x):
        """Evaluates the Gamma probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density. Must be positive.

        Returns
        -------
        float
            The probability density value at ``x``. Returns ``0.0`` if
            ``x <= 0``.
        """
        if x <= 0:
            return 0.0
        return np.exp(self._log_normalizer + (self.alpha - 1) * np.log(x) - self.beta * x)

    def cdf(self, x):
        """Evaluates the Gamma cumulative distribution function.

        Uses the regularized lower incomplete gamma function ``P(alpha, beta*x)``.

        Parameters
        ----------
        x : float
            Value at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability at ``x``. Returns ``0.0`` if ``x <= 0``.
        """
        if x <= 0:
            return 0.0
        return _gammainc_lower_reg(self.alpha, self.beta * x)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.gamma(self.alpha, 1.0 / self.beta), min_val), max_val)

@jitclass_([
    ("a", float64),
    ("b", float64),
])
class Uniform(object):
    """Uniform distribution helper.

    The class provides density evaluation and bounded random sampling for a
    uniform distribution over the interval ``[a, b]``.

    ...
    Attributes
    ----------
    a : float
        Lower bound of the distribution support.
    b : float
        Upper bound of the distribution support.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, a, b):
        """Constructs all the necessary attributes for the Uniform object.

        Parameters
        ----------
        a : float
            Lower bound of the distribution support.
        b : float
            Upper bound of the distribution support.
        """
        self.a = a
        self.b = b

    def evaluate(self, x):
        """Evaluates the uniform probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density.

        Returns
        -------
        float
            The probability density value at ``x``. Returns ``0.0`` if ``x`` is
            outside ``[a, b]``.
        """
        if x < self.a or x > self.b:
            return 0.0
        return 1.0 / (self.b - self.a)

    def cdf(self, x):
        """Evaluates the uniform cumulative distribution function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability at ``x``, clipped to ``[0, 1]``.
        """
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.uniform(self.a, self.b), min_val), max_val)

@njit_
def _gammaln(x):
    """Approximates the natural logarithm of the gamma function.

    Uses the Lanczos approximation.

    Parameters
    ----------
    x : float
        Positive input value.

    Returns
    -------
    float
        An approximation of ``log(Gamma(x))``.
    """
    coeffs = np.array([
         76.18009172947146,
        -86.50532032941677,
         24.01409824083091,
         -1.231739572450155,
          0.1208650973866179e-2,
         -0.5395239384953e-5
    ])
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015
    for c in coeffs:
        y += 1.0
        ser += c / y
    return -tmp + np.log(2.5066282746310005 * ser / x)

@njit_
def _gammainc_lower_reg(a, x):
    """Regularized lower incomplete gamma function ``P(a, x)``.

    Numba does not provide ``scipy.special.gammainc``, so this implements the
    standard Numerical Recipes split: a series expansion for ``x < a + 1`` and a
    continued-fraction expansion (giving the upper part ``Q = 1 - P``) for
    ``x >= a + 1``. Reuses :func:`_gammaln` for the normalization.

    Parameters
    ----------
    a : float
        Shape parameter (must be positive).
    x : float
        Upper integration limit (must be non-negative).

    Returns
    -------
    float
        ``P(a, x)`` in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0

    gln = _gammaln(a)

    if x < a + 1.0:
        # Series expansion for P(a, x).
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * np.exp(-x + a * np.log(x) - gln)

    # Continued-fraction expansion for Q(a, x) = 1 - P(a, x) (Lentz's method).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = np.exp(-x + a * np.log(x) - gln) * h
    return 1.0 - q