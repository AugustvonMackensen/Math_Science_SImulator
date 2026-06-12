r"""Analytical (Lagrangian) mechanics engine.

Given a Lagrangian :math:`L(q, \dot q, t)` typed as an expression string,
this module:

1. builds time-dependent generalized coordinates,
2. derives the Euler-Lagrange equations
   :math:`\frac{d}{dt}\frac{\partial L}{\partial \dot q_i}
   - \frac{\partial L}{\partial q_i} = 0` **symbolically**,
3. solves them for the accelerations :math:`\ddot q` via the mass matrix,
4. lambdifies the result into a fast numerical right-hand side, and
5. integrates the motion with the :mod:`maths.ode` engine.

It also exposes the conserved energy (Hamiltonian via a Legendre transform),
so energy drift can be used as a correctness check.

Example — simple pendulum
-------------------------
>>> from physics.mechanics import LagrangianSystem
>>> import numpy as np
>>> sys = LagrangianSystem(
...     coordinates=["theta"],
...     parameters=["m", "l", "g"],
...     lagrangian="m*l**2*theta_dot**2/2 + m*g*l*cos(theta)",
... )
>>> res = sys.simulate(
...     initial={"theta": (0.3, 0.0)},
...     t_span=(0.0, 10.0),
...     parameters={"m": 1.0, "l": 1.0, "g": 9.81},
... )
>>> res.y.shape[0]  # [theta, theta_dot]
2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from core.exceptions import ModelError
from maths.ode import ODEResult, integrate

# Functions a user may reference inside a Lagrangian string.
_ALLOWED_FUNCS = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "exp", "log", "sqrt", "Abs", "sign", "pi",
    )
}


@dataclass(slots=True)
class _SymbolicModel:
    """Internal symbolic representation built once at construction."""

    t: sp.Symbol
    q_syms: list[sp.Symbol]          # plain position symbols, e.g. theta
    v_syms: list[sp.Symbol]          # plain velocity symbols, e.g. theta_dot
    p_syms: list[sp.Symbol]          # parameter symbols
    accel_exprs: list[sp.Expr]       # q̈_i as functions of (t, q, v, p)
    energy_expr: sp.Expr             # Hamiltonian H(q, v, p)
    el_eqs: list[sp.Eq]              # readable Euler-Lagrange equations


class LagrangianSystem:
    r"""A mechanical system defined by a Lagrangian.

    Parameters
    ----------
    coordinates
        Names of the generalized coordinates, e.g. ``["theta"]`` or
        ``["x", "y"]``. For each name ``q`` the symbol ``q`` denotes the
        coordinate and ``q_dot`` its time derivative inside the Lagrangian
        string.
    lagrangian
        The Lagrangian as an expression string in terms of the coordinate
        symbols, their ``_dot`` velocities, any parameters, and ``t``.
    parameters
        Names of constant parameters appearing in the Lagrangian
        (e.g. masses, lengths, ``g``). Optional.

    Raises
    ------
    ModelError
        If no coordinates are given, the Lagrangian cannot be parsed, or the
        mass matrix is singular (an ill-posed / non-dynamical model).
    """

    def __init__(
        self,
        coordinates: Sequence[str],
        lagrangian: str,
        parameters: Sequence[str] | None = None,
    ) -> None:
        if not coordinates:
            raise ModelError("at least one generalized coordinate is required")
        self.coordinates = list(coordinates)
        self.parameters = list(parameters or [])
        self.lagrangian_str = lagrangian
        self._model = self._compile(lagrangian)
        # Cache a numeric accel function and energy function.
        args = (self._model.t, *self._model.q_syms, *self._model.v_syms, *self._model.p_syms)
        self._accel_fn = sp.lambdify(args, self._model.accel_exprs, modules="numpy")
        self._energy_fn = sp.lambdify(args, self._model.energy_expr, modules="numpy")

    # -- construction -------------------------------------------------------

    def _compile(self, lagrangian: str) -> _SymbolicModel:
        # Use the same time symbol that ``dynamicsymbols`` differentiates
        # against, so d/dt of the coordinates does not collapse to zero.
        t = dynamicsymbols._t

        # Plain symbols used for parsing and final lambdified expressions.
        q_syms = [sp.Symbol(name, real=True) for name in self.coordinates]
        v_syms = [sp.Symbol(f"{name}_dot", real=True) for name in self.coordinates]
        p_syms = [sp.Symbol(name, real=True) for name in self.parameters]

        local_dict = {**_ALLOWED_FUNCS, "t": t}
        for sym in (*q_syms, *v_syms, *p_syms):
            local_dict[sym.name] = sym

        try:
            L_plain = sp.sympify(lagrangian, locals=local_dict)
        except (sp.SympifyError, SyntaxError, TypeError) as exc:
            raise ModelError(f"could not parse Lagrangian: {exc}") from exc

        # Time-dependent versions of the coordinates for proper d/dt.
        q_funcs = [dynamicsymbols(name) for name in self.coordinates]
        qd_funcs = [f.diff(t) for f in q_funcs]
        qdd_funcs = [f.diff(t, 2) for f in q_funcs]

        # Lift the plain Lagrangian into the time-dependent picture.
        to_dynamic = {q_syms[i]: q_funcs[i] for i in range(len(q_syms))}
        to_dynamic.update({v_syms[i]: qd_funcs[i] for i in range(len(v_syms))})
        L_dyn = L_plain.subs(to_dynamic)

        # Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0.
        el = [
            sp.simplify(L_dyn.diff(qd_funcs[i]).diff(t) - L_dyn.diff(q_funcs[i]))
            for i in range(len(q_funcs))
        ]

        # These are linear in q̈. Solve the linear system M·q̈ = b.
        tmp_acc = [sp.Symbol(f"__acc_{i}") for i in range(len(q_funcs))]
        el_lin = [e.subs({qdd_funcs[i]: tmp_acc[i] for i in range(len(q_funcs))}) for e in el]
        try:
            M, b = sp.linear_eq_to_matrix(el_lin, tmp_acc)
        except Exception as exc:  # pragma: no cover - sympy internal failure
            raise ModelError(f"equations are not linear in accelerations: {exc}") from exc
        if M.det() == 0:
            raise ModelError("singular mass matrix: model is not dynamical / ill-posed")
        acc_dyn = M.LUsolve(b)  # symbolic q̈ in time-dependent variables

        # Map everything back to plain symbols for lambdification.
        to_plain = {q_funcs[i]: q_syms[i] for i in range(len(q_syms))}
        to_plain.update({qd_funcs[i]: v_syms[i] for i in range(len(v_syms))})
        accel_exprs = [sp.simplify(a.subs(to_plain)) for a in acc_dyn]

        # Conserved energy via Legendre transform: H = Σ q̇·∂L/∂q̇ − L.
        H_dyn = sum(qd_funcs[i] * L_dyn.diff(qd_funcs[i]) for i in range(len(q_funcs))) - L_dyn
        energy_expr = sp.simplify(H_dyn.subs(to_plain))

        # Readable Euler-Lagrange equations in plain symbols.
        el_eqs = [
            sp.Eq(sp.simplify(e.subs({qdd_funcs[i]: sp.Symbol(f"{self.coordinates[i]}_ddot")
                                      for i in range(len(q_funcs))}).subs(to_plain)), 0)
            for e in el
        ]

        return _SymbolicModel(
            t=t,
            q_syms=q_syms,
            v_syms=v_syms,
            p_syms=p_syms,
            accel_exprs=accel_exprs,
            energy_expr=energy_expr,
            el_eqs=el_eqs,
        )

    # -- introspection ------------------------------------------------------

    @property
    def n_dof(self) -> int:
        """Number of generalized coordinates (degrees of freedom)."""
        return len(self.coordinates)

    def euler_lagrange_equations(self) -> list[sp.Eq]:
        """Return the symbolic Euler-Lagrange equations (``= 0`` form)."""
        return list(self._model.el_eqs)

    def acceleration_expressions(self) -> dict[str, sp.Expr]:
        """Return symbolic accelerations ``q̈_i`` keyed by coordinate name."""
        return {f"{c}_ddot": e for c, e in zip(self.coordinates, self._model.accel_exprs)}

    def energy_expression(self) -> sp.Expr:
        """Return the symbolic conserved energy (Hamiltonian)."""
        return self._model.energy_expr

    # -- numerics -----------------------------------------------------------

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
        """Integrate the equations of motion.

        Parameters
        ----------
        initial
            Mapping ``coordinate -> (q0, qdot0)`` of initial position and
            velocity for every coordinate.
        t_span
            ``(t0, t1)`` time interval.
        parameters
            Numeric values for every declared parameter.
        n_points, method, rtol, atol
            Forwarded to :func:`maths.ode.integrate`.

        Returns
        -------
        ODEResult
            State rows are ordered ``[q_0..q_{d-1}, v_0..v_{d-1}]``.
        """
        missing = [c for c in self.coordinates if c not in initial]
        if missing:
            raise ModelError(f"missing initial conditions for: {missing}")
        p_vals = self._param_values(parameters)
        d = self.n_dof

        q0 = np.array([initial[c][0] for c in self.coordinates], dtype=float)
        v0 = np.array([initial[c][1] for c in self.coordinates], dtype=float)
        y0 = np.concatenate([q0, v0])

        accel_fn = self._accel_fn

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            q = y[:d]
            v = y[d:]
            acc = accel_fn(t, *q, *v, *p_vals)
            return np.concatenate([v, np.atleast_1d(np.asarray(acc, dtype=float)).ravel()])

        return integrate(
            rhs, y0, t_span, n_points=n_points, method=method, rtol=rtol, atol=atol
        )

    def energy(
        self,
        state: Mapping[str, tuple[float, float]] | np.ndarray,
        parameters: Mapping[str, float] | None = None,
    ) -> float | np.ndarray:
        """Evaluate the conserved energy at a state (or array of states).

        ``state`` may be a mapping ``coordinate -> (q, v)`` for a single point,
        or a ``(2d, n)`` array as produced by :meth:`simulate` for a series.
        """
        p_vals = self._param_values(parameters)
        d = self.n_dof
        if isinstance(state, Mapping):
            q = [state[c][0] for c in self.coordinates]
            v = [state[c][1] for c in self.coordinates]
            return float(self._energy_fn(0.0, *q, *v, *p_vals))
        arr = np.asarray(state, dtype=float)
        if arr.shape[0] != 2 * d:
            raise ModelError(f"state array must have {2 * d} rows, got {arr.shape[0]}")
        q = list(arr[:d])
        v = list(arr[d:])
        return np.asarray(self._energy_fn(0.0, *q, *v, *p_vals), dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LagrangianSystem(coordinates={self.coordinates}, "
            f"parameters={self.parameters})"
        )
