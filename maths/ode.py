"""Numerical integration of ordinary differential equations.

Two complementary engines:

* :func:`integrate` — adaptive, general-purpose IVP solving on top of
  :func:`scipy.integrate.solve_ivp` (RK45/DOP853/Radau/...), with dense
  output and event handling.
* :func:`velocity_verlet` — a fixed-step *symplectic* integrator for
  separable systems ``q'' = a(t, q)``. Unlike Runge-Kutta methods it nearly
  conserves energy over long horizons, which matters for Hamiltonian/
  Lagrangian mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from core.exceptions import ConvergenceError, ModelError

ArrayLike = Sequence[float] | np.ndarray
RHS = Callable[[float, np.ndarray], np.ndarray]
Accel = Callable[[float, np.ndarray], np.ndarray]


@dataclass(slots=True)
class ODEResult:
    """Result of an ODE integration.

    Attributes
    ----------
    t
        Times at which the solution was sampled, shape ``(n_t,)``.
    y
        Solution array of shape ``(n_dim, n_t)`` — row ``i`` is component ``i``.
    method
        Name of the integrator used.
    success
        Whether the integrator reported success.
    message
        Human-readable solver status message.
    """

    t: np.ndarray
    y: np.ndarray
    method: str
    success: bool
    message: str

    @property
    def n_dim(self) -> int:
        return self.y.shape[0]

    def component(self, i: int) -> np.ndarray:
        """Return the time series of state component ``i``."""
        return self.y[i]


def integrate(
    rhs: RHS,
    y0: ArrayLike,
    t_span: tuple[float, float],
    *,
    t_eval: ArrayLike | None = None,
    n_points: int = 500,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-12,
    events: Callable | Sequence[Callable] | None = None,
    max_step: float | None = None,
) -> ODEResult:
    """Integrate the first-order system ``y' = rhs(t, y)``.

    Parameters
    ----------
    rhs
        Right-hand side ``f(t, y) -> dy/dt``; ``y`` is a 1-D array.
    y0
        Initial state.
    t_span
        ``(t0, t1)`` integration interval.
    t_eval
        Explicit output times. If ``None``, ``n_points`` points are sampled
        uniformly across ``t_span``.
    n_points
        Number of uniform output points when ``t_eval`` is ``None``.
    method
        Any ``solve_ivp`` method. ``"DOP853"`` (high-order explicit) and
        ``"Radau"`` (implicit, for stiff problems) are good defaults.
    rtol, atol
        Relative/absolute tolerances.
    events
        Optional event function(s) for root-finding during integration.
    max_step
        Optional cap on the internal step size.

    Returns
    -------
    ODEResult

    Raises
    ------
    ModelError
        If ``t_span`` is degenerate or ``y0`` is empty.
    ConvergenceError
        If the solver fails to integrate the requested span.
    """
    y0 = np.asarray(y0, dtype=float)
    if y0.ndim != 1 or y0.size == 0:
        raise ModelError("y0 must be a non-empty 1-D array")
    t0, t1 = t_span
    if t1 == t0:
        raise ModelError(f"degenerate t_span: t0 == t1 == {t0}")

    if t_eval is None:
        t_eval = np.linspace(t0, t1, n_points)

    kwargs: dict = dict(rtol=rtol, atol=atol)
    if events is not None:
        kwargs["events"] = events
    if max_step is not None:
        kwargs["max_step"] = max_step

    sol = solve_ivp(rhs, (t0, t1), y0, method=method, t_eval=t_eval, **kwargs)
    if not sol.success:
        raise ConvergenceError(f"{method} failed: {sol.message}")

    return ODEResult(
        t=sol.t,
        y=sol.y,
        method=method,
        success=sol.success,
        message=sol.message,
    )


def velocity_verlet(
    accel: Accel,
    q0: ArrayLike,
    v0: ArrayLike,
    t_span: tuple[float, float],
    *,
    n_steps: int = 10_000,
) -> ODEResult:
    """Symplectic velocity-Verlet integration of ``q'' = accel(t, q)``.

    Suited to conservative mechanical systems where long-term energy
    behaviour matters. The returned state stacks positions on top of
    velocities: rows ``0..d-1`` are ``q``, rows ``d..2d-1`` are ``v``.

    Parameters
    ----------
    accel
        Acceleration field ``a(t, q) -> q''`` (force per unit mass).
    q0, v0
        Initial positions and velocities (same length ``d``).
    t_span
        ``(t0, t1)`` interval.
    n_steps
        Number of fixed steps; step size ``h = (t1 - t0) / n_steps``.

    Returns
    -------
    ODEResult
        With ``y`` of shape ``(2d, n_steps + 1)``.
    """
    q = np.asarray(q0, dtype=float)
    v = np.asarray(v0, dtype=float)
    if q.shape != v.shape or q.ndim != 1 or q.size == 0:
        raise ModelError("q0 and v0 must be non-empty 1-D arrays of equal length")
    t0, t1 = t_span
    if n_steps < 1:
        raise ModelError("n_steps must be >= 1")

    h = (t1 - t0) / n_steps
    d = q.size
    ts = np.linspace(t0, t1, n_steps + 1)
    out = np.empty((2 * d, n_steps + 1), dtype=float)
    out[:d, 0] = q
    out[d:, 0] = v

    a = np.asarray(accel(t0, q), dtype=float)
    for k in range(n_steps):
        t = ts[k]
        q = q + v * h + 0.5 * a * h * h
        a_next = np.asarray(accel(t + h, q), dtype=float)
        v = v + 0.5 * (a + a_next) * h
        a = a_next
        out[:d, k + 1] = q
        out[d:, k + 1] = v

    return ODEResult(
        t=ts,
        y=out,
        method="velocity-verlet",
        success=True,
        message="fixed-step symplectic integration completed",
    )
