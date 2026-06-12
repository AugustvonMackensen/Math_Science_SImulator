r"""Stochastic calculus: Brownian motion, Itô SDEs, and Itô integration.

Scalar Itô stochastic differential equations of the form

.. math::

    dX_t = a(t, X_t)\,dt + b(t, X_t)\,dW_t

are integrated with the **Euler-Maruyama** and **Milstein** schemes,
vectorized over many sample paths so Monte-Carlo expectations are cheap.
The module also evaluates **Itô integrals** :math:`\int_0^T f(t, W_t)\,dW_t`
(left-endpoint sums, as the Itô convention demands) and ships the two
canonical processes — geometric Brownian motion and the Ornstein-Uhlenbeck
process — with built-in drift/diffusion.

All routines accept a ``seed`` for reproducibility and return plot-ready
arrays (``t`` of shape ``(n_steps+1,)`` and ``paths`` of shape
``(n_paths, n_steps+1)``).

Examples
--------
>>> import numpy as np
>>> from maths.stochastic import geometric_brownian_motion
>>> sol = geometric_brownian_motion(x0=1.0, mu=0.05, sigma=0.2,
...                                 t_span=(0.0, 1.0), n_steps=500,
...                                 n_paths=20000, seed=0)
>>> abs(sol.paths[:, -1].mean() - np.exp(0.05)) < 0.01  # E[X_T] = x0 e^{mu T}
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.exceptions import ModelError

# drift a(t, x) and diffusion b(t, x); x is an array over sample paths.
Coefficient = Callable[[float, np.ndarray], np.ndarray]


@dataclass(slots=True)
class SDESolution:
    """Result of integrating an SDE over many sample paths.

    Attributes
    ----------
    t
        Time grid, shape ``(n_steps + 1,)``.
    paths
        Sample paths, shape ``(n_paths, n_steps + 1)``; row ``k`` is one
        realization of the process.
    method
        Name of the integration scheme.
    """

    t: np.ndarray
    paths: np.ndarray
    method: str

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def terminal(self) -> np.ndarray:
        """Terminal values ``X_T`` across paths, shape ``(n_paths,)``."""
        return self.paths[:, -1]

    def mean(self) -> np.ndarray:
        """Ensemble mean ``E[X_t]`` at each time, shape ``(n_steps + 1,)``."""
        return self.paths.mean(axis=0)

    def std(self) -> np.ndarray:
        """Ensemble standard deviation at each time."""
        return self.paths.std(axis=0)

    def confidence_band(self, q: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper empirical quantile bands at level ``q`` over time."""
        lo = (1.0 - q) / 2.0
        return (
            np.quantile(self.paths, lo, axis=0),
            np.quantile(self.paths, 1.0 - lo, axis=0),
        )


def _validate(t_span: tuple[float, float], n_steps: int, n_paths: int) -> float:
    t0, t1 = t_span
    if t1 <= t0:
        raise ModelError(f"require t1 > t0, got t_span={t_span}")
    if n_steps < 1:
        raise ModelError("n_steps must be >= 1")
    if n_paths < 1:
        raise ModelError("n_paths must be >= 1")
    return (t1 - t0) / n_steps


def brownian_motion(
    t_span: tuple[float, float],
    *,
    n_steps: int = 500,
    n_paths: int = 1,
    x0: float = 0.0,
    seed: int | None = None,
) -> SDESolution:
    r"""Generate standard Brownian (Wiener) paths ``W_t`` with ``W_0 = x0``.

    Increments are :math:`\Delta W \sim \mathcal N(0, \Delta t)`.
    """
    dt = _validate(t_span, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    t0, t1 = t_span
    t = np.linspace(t0, t1, n_steps + 1)
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(dW, axis=1)
    return SDESolution(t=t, paths=paths, method="brownian-motion")


def euler_maruyama(
    drift: Coefficient,
    diffusion: Coefficient,
    x0: float | np.ndarray,
    t_span: tuple[float, float],
    *,
    n_steps: int = 500,
    n_paths: int = 1,
    seed: int | None = None,
) -> SDESolution:
    r"""Integrate ``dX = a(t,X) dt + b(t,X) dW`` with the Euler-Maruyama scheme.

    .. math:: X_{n+1} = X_n + a(t_n, X_n)\,\Delta t + b(t_n, X_n)\,\Delta W_n

    Strong order 0.5, weak order 1.0. ``drift`` and ``diffusion`` are called
    with ``(t, x)`` where ``x`` is the length-``n_paths`` array of current
    states, and must return an array broadcastable to that shape.
    """
    dt = _validate(t_span, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    t0, t1 = t_span
    t = np.linspace(t0, t1, n_steps + 1)
    sqdt = np.sqrt(dt)

    x = np.full(n_paths, float(x0)) if np.isscalar(x0) else np.array(x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    for n in range(n_steps):
        dW = rng.normal(0.0, sqdt, size=n_paths)
        x = x + drift(t[n], x) * dt + diffusion(t[n], x) * dW
        paths[:, n + 1] = x
    return SDESolution(t=t, paths=paths, method="euler-maruyama")


def milstein(
    drift: Coefficient,
    diffusion: Coefficient,
    x0: float | np.ndarray,
    t_span: tuple[float, float],
    *,
    diffusion_x: Coefficient | None = None,
    n_steps: int = 500,
    n_paths: int = 1,
    seed: int | None = None,
) -> SDESolution:
    r"""Integrate an Itô SDE with the Milstein scheme (strong order 1.0).

    .. math::

        X_{n+1} = X_n + a\,\Delta t + b\,\Delta W
        + \tfrac12 b\, b' \,\big((\Delta W)^2 - \Delta t\big)

    where ``b' = ∂b/∂x``. Supply it via ``diffusion_x`` for exactness;
    otherwise it is estimated by central finite differences.
    """
    dt = _validate(t_span, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    t0, t1 = t_span
    t = np.linspace(t0, t1, n_steps + 1)
    sqdt = np.sqrt(dt)

    if diffusion_x is None:
        def diffusion_x(tn: float, x: np.ndarray) -> np.ndarray:  # noqa: ANN001
            h = 1e-6 * (1.0 + np.abs(x))
            return (diffusion(tn, x + h) - diffusion(tn, x - h)) / (2.0 * h)

    x = np.full(n_paths, float(x0)) if np.isscalar(x0) else np.array(x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    for n in range(n_steps):
        dW = rng.normal(0.0, sqdt, size=n_paths)
        b = diffusion(t[n], x)
        x = (
            x
            + drift(t[n], x) * dt
            + b * dW
            + 0.5 * b * diffusion_x(t[n], x) * (dW**2 - dt)
        )
        paths[:, n + 1] = x
    return SDESolution(t=t, paths=paths, method="milstein")


def ito_integral(
    integrand: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    *,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    r"""Approximate the Itô integral :math:`\int_{t_0}^{t_1} f(t, W_t)\,dW_t`.

    Uses the **left-endpoint** (non-anticipating) Riemann-Stieltjes sum that
    defines the Itô integral:

    .. math:: \sum_n f(t_n, W_{t_n})\,(W_{t_{n+1}} - W_{t_n}).

    Returns one value per sample path, shape ``(n_paths,)``. For example,
    :math:`\int_0^T W\,dW = \tfrac12 (W_T^2 - T)` (note the ``-T``: the
    hallmark of Itô vs. ordinary calculus).
    """
    dt = _validate(t_span, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    t0, t1 = t_span
    sqdt = np.sqrt(dt)

    W = np.zeros(n_paths)
    total = np.zeros(n_paths)
    tn = t0
    for _ in range(n_steps):
        dW = rng.normal(0.0, sqdt, size=n_paths)
        total += integrand(tn, W) * dW  # left endpoint -> Itô
        W = W + dW
        tn += dt
    return total


# --- canonical processes ---------------------------------------------------

def geometric_brownian_motion(
    x0: float,
    mu: float,
    sigma: float,
    t_span: tuple[float, float],
    *,
    n_steps: int = 500,
    n_paths: int = 1,
    seed: int | None = None,
) -> SDESolution:
    r"""Geometric Brownian motion ``dX = mu X dt + sigma X dW`` (e.g. asset prices).

    Closed-form moments: :math:`E[X_t] = x_0 e^{\mu t}`.
    """
    if x0 <= 0:
        raise ModelError("geometric Brownian motion requires x0 > 0")
    sol = euler_maruyama(
        lambda t, x: mu * x,
        lambda t, x: sigma * x,
        x0,
        t_span,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    return SDESolution(t=sol.t, paths=sol.paths, method="geometric-brownian-motion")


def ornstein_uhlenbeck(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    t_span: tuple[float, float],
    *,
    n_steps: int = 500,
    n_paths: int = 1,
    seed: int | None = None,
) -> SDESolution:
    r"""Ornstein-Uhlenbeck process ``dX = theta (mu - X) dt + sigma dW``.

    Mean-reverting to ``mu`` at rate ``theta``; the canonical model of a
    noisy system relaxing to equilibrium. :math:`E[X_t] = mu + (x_0-mu)e^{-\theta t}`.
    """
    if theta < 0:
        raise ModelError("Ornstein-Uhlenbeck requires theta >= 0")
    sol = euler_maruyama(
        lambda t, x: theta * (mu - x),
        lambda t, x: sigma * np.ones_like(x),
        x0,
        t_span,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    return SDESolution(t=sol.t, paths=sol.paths, method="ornstein-uhlenbeck")


def monte_carlo_expectation(
    payoff: Callable[[np.ndarray], np.ndarray],
    solution: SDESolution,
) -> tuple[float, float]:
    r"""Monte-Carlo estimate of ``E[payoff(X_T)]`` from a solved SDE.

    Returns ``(estimate, standard_error)`` where the standard error is
    ``std / sqrt(n_paths)``.
    """
    values = np.asarray(payoff(solution.terminal), dtype=float)
    estimate = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(values.size))
    return estimate, stderr
