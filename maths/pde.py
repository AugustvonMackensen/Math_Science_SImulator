r"""Partial differential equations — 1-D finite-difference solvers.

Two staples of mathematical physics on a line segment with Dirichlet
boundaries:

* :func:`heat_equation_1d` — the diffusion equation
  :math:`u_t = \alpha\,u_{xx}` via **Crank-Nicolson** (implicit, second-order,
  unconditionally stable).
* :func:`wave_equation_1d` — :math:`u_{tt} = c^2 u_{xx}` via the explicit
  central-difference scheme (stable when the Courant number
  :math:`c\,\Delta t/\Delta x \le 1`).

Both return a :class:`FieldSolution` carrying the space grid ``x``, time grid
``t`` and the field ``u`` of shape ``(n_t, n_x)`` — ready to plot or animate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.exceptions import ModelError


@dataclass(slots=True)
class FieldSolution:
    """A scalar field ``u(t, x)`` sampled on a space-time grid."""

    x: np.ndarray            # shape (n_x,)
    t: np.ndarray            # shape (n_t,)
    u: np.ndarray            # shape (n_t, n_x)
    method: str

    @property
    def final(self) -> np.ndarray:
        """Field at the last time level."""
        return self.u[-1]


def _grid(length: float, t_final: float, nx: int, nt: int):
    if length <= 0 or t_final <= 0:
        raise ModelError("length and t_final must be positive")
    if nx < 3 or nt < 2:
        raise ModelError("require nx >= 3 and nt >= 2")
    x = np.linspace(0.0, length, nx)
    t = np.linspace(0.0, t_final, nt)
    return x, t, x[1] - x[0], t[1] - t[0]


def _eval_ic(ic: Callable[[np.ndarray], np.ndarray] | np.ndarray, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(ic(x) if callable(ic) else ic, dtype=float)
    if arr.shape != x.shape:
        raise ModelError(f"initial condition must have shape {x.shape}, got {arr.shape}")
    return arr


def heat_equation_1d(
    u0: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    *,
    alpha: float,
    length: float,
    t_final: float,
    nx: int = 101,
    nt: int = 201,
    left: float = 0.0,
    right: float = 0.0,
) -> FieldSolution:
    r"""Solve ``u_t = alpha u_xx`` with Dirichlet BCs ``u(0)=left``, ``u(L)=right``.

    Crank-Nicolson: solve the tridiagonal system
    :math:`(I - \tfrac r2 D) u^{n+1} = (I + \tfrac r2 D) u^n`
    each step with :math:`r = \alpha\,\Delta t/\Delta x^2`.
    """
    if alpha <= 0:
        raise ModelError("alpha (diffusivity) must be positive")
    x, t, dx, dt = _grid(length, t_final, nx, nt)
    r = alpha * dt / dx**2

    u = np.empty((nt, nx))
    u[0] = _eval_ic(u0, x)
    u[0, 0], u[0, -1] = left, right

    n = nx - 2  # interior points
    main = (1.0 + r) * np.ones(n)
    off = (-r / 2.0) * np.ones(n - 1)
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    Bmain = (1.0 - r) * np.ones(n)
    Boff = (r / 2.0) * np.ones(n - 1)
    B = np.diag(Bmain) + np.diag(Boff, 1) + np.diag(Boff, -1)

    bc = np.zeros(n)
    bc[0] = r * left
    bc[-1] = r * right

    for k in range(nt - 1):
        rhs = B @ u[k, 1:-1] + bc
        u[k + 1, 1:-1] = np.linalg.solve(A, rhs)
        u[k + 1, 0], u[k + 1, -1] = left, right

    return FieldSolution(x=x, t=t, u=u, method="crank-nicolson")


def wave_equation_1d(
    u0: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    *,
    c: float,
    length: float,
    t_final: float,
    v0: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
    nx: int = 201,
    nt: int = 401,
) -> FieldSolution:
    r"""Solve ``u_tt = c^2 u_xx`` with fixed ends, initial shape ``u0`` and velocity ``v0``.

    Explicit leapfrog; raises if the Courant number
    :math:`\lambda = c\,\Delta t/\Delta x` exceeds 1 (CFL instability).
    """
    if c <= 0:
        raise ModelError("wave speed c must be positive")
    x, t, dx, dt = _grid(length, t_final, nx, nt)
    lam = c * dt / dx
    if lam > 1.0 + 1e-12:
        raise ModelError(
            f"CFL violated: Courant number {lam:.3f} > 1; refine dt or coarsen dx"
        )
    lam2 = lam**2

    u = np.zeros((nt, nx))
    u[0] = _eval_ic(u0, x)
    u[0, 0] = u[0, -1] = 0.0
    vel = np.zeros(nx) if v0 is None else _eval_ic(v0, x)

    # First step uses the initial velocity (Taylor expansion).
    u[1, 1:-1] = (
        u[0, 1:-1]
        + dt * vel[1:-1]
        + 0.5 * lam2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])
    )
    for k in range(1, nt - 1):
        u[k + 1, 1:-1] = (
            2 * u[k, 1:-1]
            - u[k - 1, 1:-1]
            + lam2 * (u[k, 2:] - 2 * u[k, 1:-1] + u[k, :-2])
        )
    return FieldSolution(x=x, t=t, u=u, method="leapfrog")
