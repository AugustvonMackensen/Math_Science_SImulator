"""Health and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..sandbox import active_executor
from ..sandbox.runner import docker_available

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict:
    """Liveness probe plus a summary of how code will be executed."""
    return {
        "status": "ok",
        "executor": active_executor(),
        "docker_available": docker_available(),
        "configured_executor": settings.executor,
    }
