r"""Symbolic calculus — single-variable and vector calculus via SymPy.

Accepts expressions as strings (ideal for a web IDE / formula box) or as
SymPy objects. Covers differentiation, integration (definite & indefinite),
limits, Taylor series, and the vector-calculus operators grad / div / curl /
Laplacian, plus the Jacobian and Hessian.

Examples
--------
>>> from maths.calculus import derivative, integral, gradient
>>> str(derivative("sin(x)*exp(x)", "x"))
'exp(x)*sin(x) + exp(x)*cos(x)'
>>> integral("x**2", "x", (0, 1))
1/3
>>> gradient("x**2 + y**2", ["x", "y"])
[2*x, 2*y]
"""

from __future__ import annotations

from typing import Sequence

import sympy as sp

from core.exceptions import ModelError

_ALLOWED = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
        "tanh", "exp", "log", "sqrt", "Abs", "sign", "pi", "E", "gamma", "erf",
    )
}


def sym(name: str) -> sp.Symbol:
    """Create a real symbol (or several, space-separated)."""
    return sp.symbols(name, real=True)


def parse(expr: str | sp.Expr, variables: Sequence[str] = ()) -> sp.Expr:
    """Parse a string into a SymPy expression with a safe function namespace."""
    if isinstance(expr, sp.Expr):
        return expr
    local = dict(_ALLOWED)
    for v in variables:
        local[v] = sp.Symbol(v, real=True)
    try:
        return sp.sympify(expr, locals=local)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise ModelError(f"could not parse expression {expr!r}: {exc}") from exc


def derivative(expr: str | sp.Expr, var: str, order: int = 1) -> sp.Expr:
    """``d^order/dvar^order`` of ``expr``."""
    e = parse(expr, [var])
    return sp.diff(e, sp.Symbol(var, real=True), order)


def integral(
    expr: str | sp.Expr,
    var: str,
    limits: tuple[float | str, float | str] | None = None,
) -> sp.Expr:
    """Indefinite integral, or definite integral over ``limits = (a, b)``."""
    x = sp.Symbol(var, real=True)
    e = parse(expr, [var])
    if limits is None:
        return sp.integrate(e, x)
    a, b = (parse(str(lim)) for lim in limits)
    return sp.integrate(e, (x, a, b))


def limit(expr: str | sp.Expr, var: str, point: float | str, direction: str = "+") -> sp.Expr:
    """Limit of ``expr`` as ``var -> point`` from ``direction`` ('+', '-', or '+-')."""
    x = sp.Symbol(var, real=True)
    e = parse(expr, [var])
    return sp.limit(e, x, parse(str(point)), dir=direction)


def taylor_series(
    expr: str | sp.Expr, var: str, point: float | str = 0, order: int = 6
) -> sp.Expr:
    """Taylor/Maclaurin series of ``expr`` about ``point`` to ``O(var^order)``."""
    x = sp.Symbol(var, real=True)
    e = parse(expr, [var])
    return sp.series(e, x, parse(str(point)), order).removeO()


def jacobian(funcs: Sequence[str | sp.Expr], variables: Sequence[str]) -> sp.Matrix:
    """Jacobian matrix ``J[i, j] = ∂f_i/∂x_j``."""
    syms = [sp.Symbol(v, real=True) for v in variables]
    F = sp.Matrix([parse(f, variables) for f in funcs])
    return F.jacobian(syms)


def hessian(expr: str | sp.Expr, variables: Sequence[str]) -> sp.Matrix:
    """Hessian matrix ``H[i, j] = ∂^2 f/∂x_i ∂x_j``."""
    syms = [sp.Symbol(v, real=True) for v in variables]
    return sp.hessian(parse(expr, variables), syms)


def gradient(scalar: str | sp.Expr, variables: Sequence[str]) -> list[sp.Expr]:
    """Gradient ``∇f`` as a list of partials."""
    f = parse(scalar, variables)
    return [sp.diff(f, sp.Symbol(v, real=True)) for v in variables]


def divergence(field: Sequence[str | sp.Expr], variables: Sequence[str]) -> sp.Expr:
    """Divergence ``∇·F`` of a vector field (one component per variable)."""
    if len(field) != len(variables):
        raise ModelError("field and variables must have equal length")
    return sp.simplify(
        sum(sp.diff(parse(field[i], variables), sp.Symbol(variables[i], real=True))
            for i in range(len(variables)))
    )


def curl(field: Sequence[str | sp.Expr], variables: Sequence[str]) -> list[sp.Expr]:
    """Curl ``∇×F`` of a 3-D vector field over variables ``(x, y, z)``."""
    if len(field) != 3 or len(variables) != 3:
        raise ModelError("curl is defined for 3-component fields in 3 variables")
    x, y, z = (sp.Symbol(v, real=True) for v in variables)
    Fx, Fy, Fz = (parse(field[i], variables) for i in range(3))
    return [
        sp.simplify(sp.diff(Fz, y) - sp.diff(Fy, z)),
        sp.simplify(sp.diff(Fx, z) - sp.diff(Fz, x)),
        sp.simplify(sp.diff(Fy, x) - sp.diff(Fx, y)),
    ]


def laplacian(scalar: str | sp.Expr, variables: Sequence[str]) -> sp.Expr:
    """Laplacian ``∇²f = Σ ∂²f/∂x_i²``."""
    f = parse(scalar, variables)
    return sp.simplify(sum(sp.diff(f, sp.Symbol(v, real=True), 2) for v in variables))


def numeric(expr: str | sp.Expr, variables: Sequence[str]):
    """Compile a SymPy expression into a fast NumPy-callable function."""
    syms = [sp.Symbol(v, real=True) for v in variables]
    return sp.lambdify(syms, parse(expr, variables), modules="numpy")
