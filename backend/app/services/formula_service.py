"""Formula parsing, simplification, rendering, and calculus.

Accepts an expression as SymPy text (robust, always available) or LaTeX
(when SymPy's optional LaTeX parser is installed). Produces a LaTeX string
for display in the frontend formula box, a simplified form, and an optional
numeric value.
"""

from __future__ import annotations

import sympy as sp

from core.exceptions import ModelError
from maths import calculus

from ..schemas import (
    CalculusRequest,
    CalculusResponse,
    FormulaRequest,
    FormulaResponse,
)


def _parse_latex(text: str) -> sp.Expr:
    try:
        from sympy.parsing.latex import parse_latex
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise ModelError(
            "LaTeX parsing needs an optional dependency "
            "(pip install antlr4-python3-runtime); send input_format='text' instead."
        ) from exc
    return parse_latex(text)


def parse_expression(text: str, input_format: str) -> sp.Expr:
    """Parse ``text`` into a SymPy expression per the requested format."""
    text = text.strip()
    if not text:
        raise ModelError("expression is empty")
    if input_format == "latex":
        return _parse_latex(text)
    if input_format == "text":
        return calculus.parse(text, _names(text))
    # auto: a backslash strongly implies LaTeX; fall back to text on failure.
    if "\\" in text:
        try:
            return _parse_latex(text)
        except ModelError:
            pass
    return calculus.parse(text, _names(text))


def _names(text: str) -> list[str]:
    # Let SymPy discover symbols; we only need real-symbol hints for common ones.
    return [c for c in "abcdefghijklmnopqrstuvwxyz" if c in text]


def evaluate_formula(req: FormulaRequest) -> FormulaResponse:
    expr = parse_expression(req.expression, req.input_format)
    simplified = sp.simplify(expr)
    value: float | None = None
    if req.variables:
        subs = {sp.Symbol(k, real=True): v for k, v in req.variables.items()}
        try:
            value = float(expr.subs(subs).evalf())
        except (TypeError, ValueError):
            value = None  # still symbolic after substitution
    return FormulaResponse(
        input=req.expression,
        latex=sp.latex(expr),
        simplified=str(simplified),
        simplified_latex=sp.latex(simplified),
        value=value,
        free_symbols=sorted(s.name for s in expr.free_symbols),
    )


def run_calculus(req: CalculusRequest) -> CalculusResponse:
    expr = req.expression
    op = req.operation
    if op == "derivative":
        result = calculus.derivative(expr, req.variable, req.order)
    elif op == "integral":
        limits = None
        if req.lower is not None and req.upper is not None:
            limits = (req.lower, req.upper)
        result = calculus.integral(expr, req.variable, limits)
    elif op == "limit":
        if req.point is None:
            raise ModelError("limit requires a 'point'")
        result = calculus.limit(expr, req.variable, req.point)
    elif op == "series":
        point = req.point if req.point is not None else "0"
        result = calculus.taylor_series(expr, req.variable, point, req.order)
    else:  # pragma: no cover - guarded by schema Literal
        raise ModelError(f"unknown operation {op!r}")
    return CalculusResponse(operation=op, result=str(result), result_latex=sp.latex(result))
