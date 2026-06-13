"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- code execution --------------------------------------------------------

class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Python source to run in the sandbox.")
    timeout: float | None = Field(None, gt=0, le=60, description="Override timeout (seconds).")


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool
    executor: str
    images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded PNGs captured from matplotlib figures.",
    )


# --- formulas --------------------------------------------------------------

class FormulaRequest(BaseModel):
    expression: str = Field(..., description="Math expression (SymPy text or LaTeX).")
    input_format: Literal["auto", "text", "latex"] = "auto"
    variables: dict[str, float] | None = Field(
        None, description="Optional values to numerically evaluate the expression."
    )


class FormulaResponse(BaseModel):
    input: str
    latex: str
    simplified: str
    simplified_latex: str
    value: float | None = None
    free_symbols: list[str]


class CalculusRequest(BaseModel):
    expression: str
    variable: str = "x"
    operation: Literal["derivative", "integral", "limit", "series"]
    order: int = 1
    point: str | None = None
    lower: str | None = None
    upper: str | None = None


class CalculusResponse(BaseModel):
    operation: str
    result: str
    result_latex: str


# --- geometry --------------------------------------------------------------

class ShapeSpec(BaseModel):
    kind: Literal["point", "segment", "line", "circle", "triangle", "polygon"]
    # Coordinates depend on kind; validated in the service layer.
    points: list[list[float]] = Field(default_factory=list)
    center: list[float] | None = None
    radius: float | None = None
    label: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)


class GeometryRequest(BaseModel):
    shapes: list[ShapeSpec]
    # Auto-add derived constructs (e.g. a triangle's circumcircle).
    derive: bool = False
    render_svg: bool = True


class Drawable(BaseModel):
    kind: str
    data: dict[str, Any]
    label: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)


class GeometryResponse(BaseModel):
    drawables: list[Drawable]
    bounds: dict[str, float]
    metrics: dict[str, Any] = Field(default_factory=dict)
    svg: str | None = None
