r"""Parameter estimation — MLE, confidence intervals, and the bootstrap.

Includes closed-form Gaussian MLE, a generic maximum-likelihood optimizer for
any SciPy distribution family, Student-t confidence intervals for a mean, and
a nonparametric percentile bootstrap for an arbitrary statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _ss

from core.exceptions import ConvergenceError, ModelError


@dataclass(slots=True)
class Estimate:
    """A point estimate with an interval."""

    value: float
    lower: float
    upper: float
    confidence: float

    @property
    def margin(self) -> float:
        return (self.upper - self.lower) / 2.0


def normal_mle(data) -> tuple[float, float]:
    """MLE of a normal distribution: ``(mu_hat, sigma_hat)``.

    Note ``sigma_hat`` uses the ``1/n`` (biased) normalization, as the
    likelihood prescribes — not the ``1/(n-1)`` sample variance.
    """
    x = np.asarray(data, dtype=float)
    if x.size < 1:
        raise ModelError("need at least one observation")
    mu = float(x.mean())
    sigma = float(np.sqrt(((x - mu) ** 2).mean()))
    return mu, sigma


def mle(data, dist_name: str, *, start: tuple[float, ...] | None = None) -> tuple[float, ...]:
    """Maximum-likelihood fit of a named SciPy distribution to ``data``.

    Returns the fitted shape/loc/scale parameters. Thin, robust wrapper over
    ``scipy.stats.<dist>.fit`` with a clear error on failure.
    """
    x = np.asarray(data, dtype=float)
    dist = getattr(_ss, dist_name, None)
    if dist is None:
        raise ModelError(f"unknown scipy distribution {dist_name!r}")
    try:
        return tuple(float(p) for p in dist.fit(x))
    except Exception as exc:  # pragma: no cover - scipy optimizer failure
        raise ConvergenceError(f"MLE fit failed for {dist_name!r}: {exc}") from exc


def confidence_interval_mean(data, confidence: float = 0.95) -> Estimate:
    """Student-t confidence interval for the population mean."""
    x = np.asarray(data, dtype=float)
    n = x.size
    if n < 2:
        raise ModelError("need at least two observations")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    tcrit = float(_ss.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return Estimate(mean, mean - tcrit * se, mean + tcrit * se, confidence)


def bootstrap(
    data,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> Estimate:
    """Nonparametric percentile bootstrap CI for an arbitrary ``statistic``."""
    x = np.asarray(data, dtype=float)
    n = x.size
    if n < 2:
        raise ModelError("need at least two observations")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    reps = np.array([statistic(x[i]) for i in idx])
    lo = (1.0 - confidence) / 2.0
    return Estimate(
        value=float(statistic(x)),
        lower=float(np.quantile(reps, lo)),
        upper=float(np.quantile(reps, 1.0 - lo)),
        confidence=confidence,
    )


def maximize_likelihood(
    neg_log_likelihood: Callable[[np.ndarray], float],
    x0,
    *,
    bounds=None,
) -> np.ndarray:
    """Generic MLE by minimizing a user-supplied negative log-likelihood."""
    res = _opt.minimize(neg_log_likelihood, np.asarray(x0, dtype=float),
                        method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise ConvergenceError(f"likelihood optimization failed: {res.message}")
    return res.x
