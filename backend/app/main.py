"""FastAPI application entry point.

Run locally with::

    uvicorn backend.app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import execute, formula, geometry, health

app = FastAPI(
    title="Math & Science Simulator API",
    version="0.1.0",
    description="Compute engine for postgraduate physics & mathematics, "
    "with code execution, formula rendering, and geometry drawing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(execute.router)
app.include_router(formula.router)
app.include_router(geometry.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"name": "Math & Science Simulator API", "docs": "/docs", "health": "/health"}
