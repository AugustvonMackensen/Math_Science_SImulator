r"""Quantum mechanics — the 1-D time-independent Schrödinger equation.

Solves the eigenvalue problem

.. math:: \left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)\right]\psi = E\psi

on a finite interval with Dirichlet (hard-wall) boundaries, by a
finite-difference discretization of the kinetic operator. The Hamiltonian is
symmetric tridiagonal, so eigenpairs come from the fast
``scipy.linalg.eigh_tridiagonal``. Wavefunctions are returned normalized to
:math:`\int|\psi|^2\,dx = 1`.

Helpers build the two textbook potentials — the infinite square well and the
harmonic oscillator — whose analytic spectra make excellent correctness checks
(:math:`E_n = \hbar\omega(n+\tfrac12)` for the oscillator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal

from core.exceptions import ModelError


@dataclass(slots=True)
class StationaryStates:
    """Bound-state spectrum of a 1-D Hamiltonian."""

    x: np.ndarray              # grid, shape (n_points,)
    energies: np.ndarray       # ascending eigenvalues, shape (n_states,)
    wavefunctions: np.ndarray  # shape (n_states, n_points); row k is psi_k(x)

    def probability_density(self, k: int) -> np.ndarray:
        r""":math:`|\psi_k(x)|^2` for state ``k``."""
        return self.wavefunctions[k] ** 2


def solve_schrodinger(
    potential: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    *,
    n_points: int = 1000,
    mass: float = 1.0,
    hbar: float = 1.0,
    n_states: int = 6,
) -> StationaryStates:
    """Compute the lowest ``n_states`` bound states for a potential ``V(x)``.

    Parameters
    ----------
    potential
        Callable ``V(x)`` evaluated on the grid (vectorized over NumPy arrays).
    x_min, x_max
        Interval endpoints; the wavefunction is pinned to zero just outside.
    n_points
        Number of interior grid points.
    mass, hbar
        Particle mass and reduced Planck constant (choose your unit system).
    n_states
        Number of lowest eigenstates to return.
    """
    if x_max <= x_min:
        raise ModelError("require x_max > x_min")
    if n_points < n_states + 2:
        raise ModelError("n_points must exceed n_states")

    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]
    V = np.asarray(potential(x), dtype=float)
    if V.shape != x.shape:
        raise ModelError("potential must return one value per grid point")

    coeff = hbar**2 / (2.0 * mass * dx**2)
    diag = 2.0 * coeff + V
    offdiag = -coeff * np.ones(n_points - 1)

    energies, vecs = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, n_states - 1))
    # Normalize columns so that the trapezoidal integral of |psi|^2 is 1.
    psi = vecs.T  # shape (n_states, n_points)
    norms = np.sqrt(np.trapezoid(psi**2, x, axis=1))
    psi = psi / norms[:, None]
    # Fix a sign convention: make the first significant lobe positive.
    for k in range(psi.shape[0]):
        if psi[k, np.argmax(np.abs(psi[k]))] < 0:
            psi[k] = -psi[k]
    return StationaryStates(x=x, energies=energies, wavefunctions=psi)


def infinite_square_well_energies(
    n_states: int, *, width: float = 1.0, mass: float = 1.0, hbar: float = 1.0
) -> np.ndarray:
    r"""Analytic levels :math:`E_n = n^2\pi^2\hbar^2/(2mL^2)`, ``n = 1, 2, ...``."""
    n = np.arange(1, n_states + 1)
    return (n**2 * np.pi**2 * hbar**2) / (2.0 * mass * width**2)


def harmonic_oscillator_energies(
    n_states: int, *, omega: float = 1.0, hbar: float = 1.0
) -> np.ndarray:
    r"""Analytic levels :math:`E_n = \hbar\omega(n+\tfrac12)`, ``n = 0, 1, ...``."""
    n = np.arange(n_states)
    return hbar * omega * (n + 0.5)


def harmonic_potential(mass: float = 1.0, omega: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    r"""Return the potential ``V(x) = \tfrac12 m\omega^2 x^2``."""
    return lambda x: 0.5 * mass * omega**2 * x**2
