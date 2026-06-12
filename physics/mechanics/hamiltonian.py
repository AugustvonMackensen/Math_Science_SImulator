r"""Hamiltonian mechanics — Hamilton's canonical equations from ``H(q, p, t)``.

Companion to :class:`~physics.mechanics.lagrangian.LagrangianSystem`. The user
types a Hamiltonian; the engine forms Hamilton's equations

.. math:: \dot q_i = \frac{\partial H}{\partial p_i}, \qquad
          \dot p_i = -\frac{\partial H}{\partial q_i},

lambdifies them, and integrates in phase space. For each coordinate named
``q`` the conjugate momentum symbol is ``p_q``.

Example — harmonic oscillator
-----------------------------
>>> from physics.mechanics import HamiltonianSystem
>>> sho = HamiltonianSystem(
...     coordinates=["x"],
...     parameters=["m", "k"],
...     hamiltonian="p_x**2/(2*m) + k*x**2/2",
... )
>>> res = sho.simulate({"x": (1.0, 0.0)}, (0.0, 10.0), {"m": 1.0, "k": 1.0})
>>> res.y.shape[0]  # [x, p_x]
2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from core.exceptions import ModelError
from maths.ode import ODEResult, integrate

_ALLOWED_FUNCS = {
    name: getattr(sp, name)
    for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "sign", "pi", "cosh", "sinh")
}


@dataclass(slots=True)
class _Compiled:
    t: sp.Symbol
    q_syms: list[sp.Symbol]
    p_syms: list[sp.Symbol]
    par_syms: list[sp.Symbol]
    qdot: list[sp.Expr]
    pdot: list[sp.Expr]
    H: sp.Expr


class HamiltonianSystem:
    """A mechanical system defined by a Hamiltonian ``H(q, p, t)``."""

    def __init__(
        self,
        coordinates: Sequence[str],
        hamiltonian: str,
        parameters: Sequence[str] | None = None,
    ) -> None:
        if not coordinates:
            raise ModelError("at least one coordinate is required")
        self.coordinates = list(coordinates)
        self.momenta = [f"p_{c}" for c in self.coordinates]
        self.parameters = list(parameters or [])
        self.hamiltonian_str = hamiltonian
        self._c = self._compile(hamiltonian)
        args = (self._c.t, *self._c.q_syms, *self._c.p_syms, *self._c.par_syms)
        self._qdot_fn = sp.lambdify(args, self._c.qdot, modules="numpy")
        self._pdot_fn = sp.lambdify(args, self._c.pdot, modules="numpy")
        self._H_fn = sp.lambdify(args, self._c.H, modules="numpy")

    def _compile(self, hamiltonian: str) -> _Compiled:
        t = sp.Symbol("t", real=True)
        q_syms = [sp.Symbol(c, real=True) for c in self.coordinates]
        p_syms = [sp.Symbol(m, real=True) for m in self.momenta]
        par_syms = [sp.Symbol(p, real=True) for p in self.parameters]

        local = {**_ALLOWED_FUNCS, "t": t}
        for s in (*q_syms, *p_syms, *par_syms):
            local[s.name] = s
        try:
            H = sp.sympify(hamiltonian, locals=local)
        except (sp.SympifyError, SyntaxError, TypeError) as exc:
            raise ModelError(f"could not parse Hamiltonian: {exc}") from exc

        qdot = [sp.simplify(sp.diff(H, p_syms[i])) for i in range(len(q_syms))]
        pdot = [sp.simplify(-sp.diff(H, q_syms[i])) for i in range(len(q_syms))]
        return _Compiled(t, q_syms, p_syms, par_syms, qdot, pdot, sp.simplify(H))

    @property
    def n_dof(self) -> int:
        return len(self.coordinates)

    def hamiltons_equations(self) -> dict[str, sp.Expr]:
        """Symbolic Hamilton equations keyed by ``<var>_dot``."""
        out: dict[str, sp.Expr] = {}
        for i, c in enumerate(self.coordinates):
            out[f"{c}_dot"] = self._c.qdot[i]
            out[f"p_{c}_dot"] = self._c.pdot[i]
        return out

    def _param_values(self, parameters: Mapping[str, float] | None) -> tuple[float, ...]:
        parameters = parameters or {}
        missing = [p for p in self.parameters if p not in parameters]
        if missing:
            raise ModelError(f"missing parameter values: {missing}")
        return tuple(float(parameters[p]) for p in self.parameters)

    def simulate(
        self,
        initial: Mapping[str, tuple[float, float]],
        t_span: tuple[float, float],
        parameters: Mapping[str, float] | None = None,
        *,
        n_points: int = 500,
        method: str = "DOP853",
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> ODEResult:
        """Integrate the canonical equations; state rows are ``[q..., p...]``."""
        missing = [c for c in self.coordinates if c not in initial]
        if missing:
            raise ModelError(f"missing initial conditions for: {missing}")
        p_vals = self._param_values(parameters)
        d = self.n_dof
        q0 = np.array([initial[c][0] for c in self.coordinates], dtype=float)
        p0 = np.array([initial[c][1] for c in self.coordinates], dtype=float)
        y0 = np.concatenate([q0, p0])

        qdot_fn, pdot_fn = self._qdot_fn, self._pdot_fn

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            q, p = y[:d], y[d:]
            qd = np.atleast_1d(np.asarray(qdot_fn(t, *q, *p, *p_vals), dtype=float)).ravel()
            pd = np.atleast_1d(np.asarray(pdot_fn(t, *q, *p, *p_vals), dtype=float)).ravel()
            return np.concatenate([qd, pd])

        return integrate(rhs, y0, t_span, n_points=n_points, method=method, rtol=rtol, atol=atol)

    def energy(self, state: np.ndarray, parameters: Mapping[str, float] | None = None):
        """Evaluate ``H`` over a ``(2d, n)`` state array (conserved if ``H`` is time-independent)."""
        p_vals = self._param_values(parameters)
        d = self.n_dof
        arr = np.asarray(state, dtype=float)
        if arr.shape[0] != 2 * d:
            raise ModelError(f"state must have {2 * d} rows")
        q = list(arr[:d])
        p = list(arr[d:])
        return np.asarray(self._H_fn(0.0, *q, *p, *p_vals), dtype=float)
