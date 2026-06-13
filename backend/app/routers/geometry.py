"""Geometry endpoint — build a drawable scene from shape specs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.exceptions import MathSciError

from ..schemas import GeometryRequest, GeometryResponse
from ..services import geometry_service

router = APIRouter(prefix="/api/geometry", tags=["geometry"])


@router.post("/scene", response_model=GeometryResponse)
def scene(req: GeometryRequest) -> GeometryResponse:
    """Validate shapes, derive constructs, compute metrics, and render an SVG."""
    try:
        return geometry_service.build_scene(req)
    except MathSciError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
