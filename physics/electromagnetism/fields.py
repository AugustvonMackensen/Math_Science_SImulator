r"""Electromagnetism — electrostatics of point charges and Laplace's equation.

Two complementary tools:

* Point-charge **superposition**: the electric field and potential of a set of
  :class:`PointCharge` sources via Coulomb's law.
* A **boundary-value solver** for Laplace's equation
  :math:`\nabla^2 \varphi = 0` on a rectangular grid with Dirichlet edges,
  by finite differences — the workhorse for conductor/cavity problems.

SI units throughout; the Coulomb constant is taken from
:mod:`core.constants` (``1/(4 pi eps_0)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.constants import constant
from core.exceptions import ModelError

_K_E = 1.0 / (4.0 * np.pi * constant("eps_0").value)  # Coulomb constant, N m^2 C^-2
_SOFTENING = 1e-12  # avoid division by zero at a source location


@dataclass(slots=True)
class PointCharge:
    """A point charge ``q`` (coulombs) at ``position`` (metres, 2-D or 3-D)."""

    q: float
    position: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.ndim != 1 or self.position.size not in (2, 3):
            raise ModelError("position must be a 2-D or 3-D vector")


def _stack_points(points: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    P = np.asarray(points, dtype=float)
    single = P.ndim == 1
    if single:
        P = P[None, :]
    if P.shape[1] != dim:
        raise ModelError(f"points must have dimension {dim} to match the charges")
    return P, single


def electric_potential(charges: list[PointCharge], points) -> np.ndarray:
    r"""Electrostatic potential :math:`\varphi = k_e \sum_i q_i / r_i` (volts)."""
    if not charges:
        raise ModelError("need at least one charge")
    dim = charges[0].position.size
    P, single = _stack_points(points, dim)
    phi = np.zeros(len(P))
    for ch in charges:
        r = np.linalg.norm(P - ch.position, axis=1) + _SOFTENING
        phi += _K_E * ch.q / r
    return float(phi[0]) if single else phi


def electric_field(charges: list[PointCharge], points) -> np.ndarray:
    r"""Electric field :math:`\mathbf E = k_e \sum_i q_i \hat r_i / r_i^2` (V/m).

    Returns a vector per evaluation point (same shape as ``points``).
    """
    if not charges:
        raise ModelError("need at least one charge")
    dim = charges[0].position.size
    P, single = _stack_points(points, dim)
    E = np.zeros_like(P)
    for ch in charges:
        d = P - ch.position
        r = np.linalg.norm(d, axis=1) + _SOFTENING
        E += _K_E * ch.q * d / r[:, None] ** 3
    return E[0] if single else E


def coulomb_force(q1: PointCharge, q2: PointCharge) -> np.ndarray:
    r"""Force on ``q2`` due to ``q1`` (newtons), :math:`k_e q_1 q_2 \hat r / r^2`."""
    d = q2.position - q1.position
    r = np.linalg.norm(d) + _SOFTENING
    return _K_E * q1.q * q2.q * d / r**3


@dataclass(slots=True)
class PotentialField:
    """Solution of Laplace's equation on a rectangular grid."""

    x: np.ndarray         # shape (nx,)
    y: np.ndarray         # shape (ny,)
    phi: np.ndarray       # shape (ny, nx)

    def electric_field(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Field :math:`\mathbf E = -\nabla\varphi` as ``(Ex, Ey)`` arrays."""
        gy, gx = np.gradient(self.phi, self.y, self.x)
        return -gx, -gy


def solve_laplace_2d(
    *,
    nx: int,
    ny: int,
    width: float = 1.0,
    height: float = 1.0,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 20_000,
) -> PotentialField:
    r"""Solve :math:`\nabla^2\varphi = 0` with constant Dirichlet edge values.

    Uses Gauss-Seidel relaxation with successive over-relaxation (SOR) for
    fast convergence. Returns the potential and grid.
    """
    if nx < 3 or ny < 3:
        raise ModelError("require nx, ny >= 3")
    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    phi = np.zeros((ny, nx))
    phi[:, 0] = left
    phi[:, -1] = right
    phi[-1, :] = top
    phi[0, :] = bottom

    # Optimal SOR factor for a rectangular grid.
    omega = 2.0 / (1.0 + np.sin(np.pi / max(nx, ny)))
    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                updated = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1])
                new = (1 - omega) * phi[i, j] + omega * updated
                delta = max(delta, abs(new - phi[i, j]))
                phi[i, j] = new
        if delta < tol:
            break
    return PotentialField(x=x, y=y, phi=phi)
