r"""Probability distributions — a uniform interface over SciPy.

A single :func:`distribution` factory returns a :class:`Distribution` wrapper
exposing a consistent API (``pdf``/``pmf``, ``cdf``, ``ppf``, ``sample``,
``moments``) for both continuous and discrete laws, so downstream code does
not branch on the distribution family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _ss

from core.exceptions import ModelError

# Map friendly names -> (scipy distribution, parameter names).
_REGISTRY: dict[str, tuple] = {
    "normal": (_ss.norm, ("loc", "scale")),
    "uniform": (_ss.uniform, ("loc", "scale")),
    "exponential": (_ss.expon, ("loc", "scale")),
    "gamma": (_ss.gamma, ("a", "loc", "scale")),
    "beta": (_ss.beta, ("a", "b", "loc", "scale")),
    "student_t": (_ss.t, ("df", "loc", "scale")),
    "chi2": (_ss.chi2, ("df", "loc", "scale")),
    "lognormal": (_ss.lognorm, ("s", "loc", "scale")),
    "poisson": (_ss.poisson, ("mu",)),
    "binomial": (_ss.binom, ("n", "p")),
    "geometric": (_ss.geom, ("p",)),
    "bernoulli": (_ss.bernoulli, ("p",)),
}

_DISCRETE = {"poisson", "binomial", "geometric", "bernoulli"}


@dataclass(slots=True)
class Moments:
    mean: float
    variance: float
    skewness: float
    kurtosis: float  # excess kurtosis (0 for the normal)


class Distribution:
    """A frozen probability distribution with a uniform interface."""

    def __init__(self, name: str, frozen, discrete: bool) -> None:
        self.name = name
        self._d = frozen
        self.discrete = discrete

    def pdf(self, x):
        """Probability density (continuous) or mass (discrete) at ``x``."""
        return self._d.pmf(x) if self.discrete else self._d.pdf(x)

    # Alias for discrete users who expect ``pmf``.
    pmf = pdf

    def cdf(self, x):
        """Cumulative distribution function."""
        return self._d.cdf(x)

    def ppf(self, q):
        """Quantile / inverse-CDF at probability ``q``."""
        return self._d.ppf(q)

    def sample(self, size: int = 1, *, seed: int | None = None):
        """Draw ``size`` random variates."""
        return self._d.rvs(size=size, random_state=np.random.default_rng(seed))

    def moments(self) -> Moments:
        """Mean, variance, skewness and excess kurtosis."""
        m, v, s, k = self._d.stats(moments="mvsk")
        return Moments(float(m), float(v), float(s), float(k))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Distribution({self.name!r})"


def distribution(name: str, **params) -> Distribution:
    """Construct a :class:`Distribution` by name with keyword parameters.

    Examples
    --------
    >>> distribution("normal", loc=0, scale=1).cdf(0.0)
    0.5
    >>> distribution("poisson", mu=3).pmf(2)  # doctest: +ELLIPSIS
    0.224...
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ModelError(f"unknown distribution {name!r}; known: {sorted(_REGISTRY)}")
    dist, allowed = _REGISTRY[key]
    bad = set(params) - set(allowed)
    if bad:
        raise ModelError(f"{name}: unexpected parameters {sorted(bad)}; allowed: {allowed}")
    return Distribution(key, dist(**params), discrete=key in _DISCRETE)
