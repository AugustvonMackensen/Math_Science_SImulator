"""Pure-mathematics engines."""

from __future__ import annotations

from . import calculus, geometry, linalg, pde
from .ode import ODEResult, integrate, velocity_verlet
from .stochastic import (
    SDESolution,
    brownian_motion,
    euler_maruyama,
    geometric_brownian_motion,
    ito_integral,
    milstein,
    monte_carlo_expectation,
    ornstein_uhlenbeck,
)

__all__ = [
    "ODEResult",
    "integrate",
    "velocity_verlet",
    "SDESolution",
    "brownian_motion",
    "euler_maruyama",
    "milstein",
    "ito_integral",
    "geometric_brownian_motion",
    "ornstein_uhlenbeck",
    "monte_carlo_expectation",
]
