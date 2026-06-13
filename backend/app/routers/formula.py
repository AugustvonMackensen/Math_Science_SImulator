"""Formula endpoints — evaluate/render expressions and run calculus."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.exceptions import MathSciError

from ..schemas import (
    CalculusRequest,
    CalculusResponse,
    FormulaRequest,
    FormulaResponse,
)
from ..services import formula_service

router = APIRouter(prefix="/api/formula", tags=["formula"])


@router.post("/evaluate", response_model=FormulaResponse)
def evaluate(req: FormulaRequest) -> FormulaResponse:
    """Parse an expression, return its LaTeX, simplified form, and optional value."""
    try:
        return formula_service.evaluate_formula(req)
    except MathSciError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/calculus", response_model=CalculusResponse)
def calculus(req: CalculusRequest) -> CalculusResponse:
    """Differentiate, integrate, take a limit, or expand a Taylor series."""
    try:
        return formula_service.run_calculus(req)
    except MathSciError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
