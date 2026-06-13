"""Geometry scene building — turn shape specs into drawable primitives.

Validates each :class:`ShapeSpec`, normalizes it into a frontend-friendly
:class:`Drawable`, optionally derives classical constructs (a triangle's
circumcircle, incircle, and centroid), computes a few metrics, and renders an
SVG preview with matplotlib (Agg, headless).
"""

from __future__ import annotations

import io

import numpy as np

from core.exceptions import ModelError
from maths.geometry import euclidean as eu

from ..schemas import Drawable, GeometryRequest, GeometryResponse, ShapeSpec


def _require_points(spec: ShapeSpec, n: int) -> list[list[float]]:
    if len(spec.points) != n:
        raise ModelError(f"{spec.kind} needs {n} point(s), got {len(spec.points)}")
    for p in spec.points:
        if len(p) != 2:
            raise ModelError("each point must be [x, y]")
    return spec.points


def _drawable_for(spec: ShapeSpec) -> tuple[Drawable, list[list[float]]]:
    """Return (drawable, coordinates-for-bounds) for one spec."""
    k = spec.kind
    if k == "point":
        (x, y), = _require_points(spec, 1)
        return Drawable(kind="point", data={"x": x, "y": y}, label=spec.label,
                        style=spec.style), [[x, y]]
    if k == "segment":
        (a, b) = _require_points(spec, 2)
        return Drawable(kind="segment", data={"x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]},
                        label=spec.label, style=spec.style), [a, b]
    if k == "line":
        (a, b) = _require_points(spec, 2)
        return Drawable(kind="line", data={"points": [a, b]}, label=spec.label,
                        style=spec.style), [a, b]
    if k == "circle":
        if spec.center is None or spec.radius is None:
            raise ModelError("circle needs 'center' and 'radius'")
        cx, cy = spec.center
        r = spec.radius
        box = [[cx - r, cy - r], [cx + r, cy + r]]
        return Drawable(kind="circle", data={"cx": cx, "cy": cy, "r": r},
                        label=spec.label, style=spec.style), box
    if k == "triangle":
        pts = _require_points(spec, 3)
        return Drawable(kind="polygon", data={"points": pts, "closed": True},
                        label=spec.label, style=spec.style), pts
    if k == "polygon":
        if len(spec.points) < 3:
            raise ModelError("polygon needs >= 3 points")
        return Drawable(kind="polygon", data={"points": spec.points, "closed": True},
                        label=spec.label, style=spec.style), spec.points
    raise ModelError(f"unknown shape kind {k!r}")  # pragma: no cover


def _derive(spec: ShapeSpec) -> tuple[list[Drawable], dict]:
    """Derived constructs + metrics for a shape (currently triangles & circles)."""
    extra: list[Drawable] = []
    metrics: dict = {}
    if spec.kind == "triangle":
        tri = eu.Triangle(*spec.points)
        cc = tri.circumcircle
        ic = tri.incircle
        extra.append(Drawable(kind="circle", data={"cx": float(cc.center[0]),
                     "cy": float(cc.center[1]), "r": float(cc.radius)},
                     label="circumcircle", style={"dashed": True}))
        extra.append(Drawable(kind="circle", data={"cx": float(ic.center[0]),
                     "cy": float(ic.center[1]), "r": float(ic.radius)},
                     label="incircle", style={"dashed": True}))
        extra.append(Drawable(kind="point", data={"x": float(tri.centroid[0]),
                     "y": float(tri.centroid[1])}, label="centroid"))
        a, b, c = tri.angles()
        metrics["triangle"] = {
            "area": tri.area, "perimeter": tri.perimeter,
            "circumradius": float(cc.radius), "inradius": float(ic.radius),
            "angles_deg": [float(np.degrees(a)), float(np.degrees(b)), float(np.degrees(c))],
        }
    elif spec.kind == "circle" and spec.radius is not None:
        metrics["circle"] = {"area": float(np.pi * spec.radius**2),
                             "circumference": float(2 * np.pi * spec.radius)}
    elif spec.kind == "polygon" and len(spec.points) >= 3:
        metrics["polygon"] = {"area": abs(eu.polygon_area(spec.points))}
    return extra, metrics


def _bounds(coords: list[list[float]]) -> dict[str, float]:
    if not coords:
        return {"xmin": -1.0, "ymin": -1.0, "xmax": 1.0, "ymax": 1.0}
    arr = np.asarray(coords, dtype=float)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    pad = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)
    return {"xmin": float(xmin - pad), "ymin": float(ymin - pad),
            "xmax": float(xmax + pad), "ymax": float(ymax + pad)}


def _render_svg(drawables: list[Drawable], bounds: dict[str, float]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle

    fig, ax = plt.subplots(figsize=(5, 5))
    for d in drawables:
        dashed = d.style.get("dashed", False)
        ls = "--" if dashed else "-"
        if d.kind == "point":
            ax.plot(d.data["x"], d.data["y"], "o", color="crimson")
            if d.label:
                ax.annotate(d.label, (d.data["x"], d.data["y"]), fontsize=8)
        elif d.kind == "segment":
            ax.plot([d.data["x1"], d.data["x2"]], [d.data["y1"], d.data["y2"]], ls, color="steelblue")
        elif d.kind == "line":
            (x1, y1), (x2, y2) = d.data["points"]
            ax.axline((x1, y1), (x2, y2), color="steelblue", linestyle=ls)
        elif d.kind == "circle":
            ax.add_patch(MplCircle((d.data["cx"], d.data["cy"]), d.data["r"],
                         fill=False, linestyle=ls, edgecolor="seagreen"))
        elif d.kind == "polygon":
            pts = np.asarray(d.data["points"], dtype=float)
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], ls, color="darkorange")

    ax.set_xlim(bounds["xmin"], bounds["xmax"])
    ax.set_ylim(bounds["ymin"], bounds["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_scene(req: GeometryRequest) -> GeometryResponse:
    drawables: list[Drawable] = []
    coords: list[list[float]] = []
    metrics: dict = {}

    for spec in req.shapes:
        d, box = _drawable_for(spec)
        drawables.append(d)
        coords.extend(box)
        if req.derive:
            extra, m = _derive(spec)
            drawables.extend(extra)
            metrics.update(m)
            for e in extra:
                if e.kind == "circle":
                    coords.append([e.data["cx"] - e.data["r"], e.data["cy"] - e.data["r"]])
                    coords.append([e.data["cx"] + e.data["r"], e.data["cy"] + e.data["r"]])

    bounds = _bounds(coords)
    svg = _render_svg(drawables, bounds) if req.render_svg else None
    return GeometryResponse(drawables=drawables, bounds=bounds, metrics=metrics, svg=svg)
