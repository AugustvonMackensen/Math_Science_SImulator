r"""Euclidean & plane geometry — points, lines, circles, triangles, polygons.

Numerically robust constructions and predicates for 2-D plane geometry:
distances, line/segment/circle intersections, triangle centers
(centroid, circumcenter, incenter), polygon area, and the convex hull.
Points are plain length-2 NumPy arrays; helpers accept anything array-like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.exceptions import ModelError

Point = np.ndarray
_EPS = 1e-12


def _cross2(a, b) -> float:
    """Scalar 2-D cross product (z-component), avoiding deprecated np.cross."""
    return float(a[0] * b[1] - a[1] * b[0])


def _pt(p) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.shape != (2,):
        raise ModelError(f"expected a 2-D point, got shape {a.shape}")
    return a


def distance(p, q) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(_pt(p) - _pt(q)))


def collinear(a, b, c, *, tol: float = 1e-9) -> bool:
    """True if points ``a, b, c`` are collinear."""
    a, b, c = _pt(a), _pt(b), _pt(c)
    return abs(_cross2(b - a, c - a)) <= tol


def polygon_area(vertices) -> float:
    """Signed area of a simple polygon (shoelace formula); CCW is positive."""
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2 or len(V) < 3:
        raise ModelError("need >= 3 points of shape (n, 2)")
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass(slots=True)
class Line:
    """Infinite line through ``point`` with (unnormalized) ``direction``."""

    point: np.ndarray
    direction: np.ndarray

    @classmethod
    def through(cls, a, b) -> "Line":
        a, b = _pt(a), _pt(b)
        d = b - a
        if np.linalg.norm(d) <= _EPS:
            raise ModelError("cannot build a line through two identical points")
        return cls(point=a, direction=d)

    def distance_to(self, p) -> float:
        """Perpendicular distance from point ``p`` to the line."""
        p = _pt(p)
        d = self.direction / np.linalg.norm(self.direction)
        w = p - self.point
        return float(abs(_cross2(d, w)))

    def intersect(self, other: "Line"):
        """Intersection point with ``other``; ``None`` if parallel."""
        p, r = self.point, self.direction
        q, s = other.point, other.direction
        rxs = _cross2(r, s)
        if abs(rxs) <= _EPS:
            return None
        t = _cross2(q - p, s) / rxs
        return p + t * r


@dataclass(slots=True)
class Segment:
    """Line segment from ``a`` to ``b``."""

    a: np.ndarray
    b: np.ndarray

    def intersect(self, other: "Segment"):
        """Proper intersection point of two segments, or ``None``."""
        p, r = _pt(self.a), _pt(self.b) - _pt(self.a)
        q, s = _pt(other.a), _pt(other.b) - _pt(other.a)
        rxs = _cross2(r, s)
        qp = q - p
        if abs(rxs) <= _EPS:
            return None  # parallel or collinear
        t = _cross2(qp, s) / rxs
        u = _cross2(qp, r) / rxs
        if -_EPS <= t <= 1 + _EPS and -_EPS <= u <= 1 + _EPS:
            return p + t * r
        return None


@dataclass(slots=True)
class Circle:
    """Circle with ``center`` and ``radius``."""

    center: np.ndarray
    radius: float

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def circumference(self) -> float:
        return float(2 * np.pi * self.radius)

    def intersect_line(self, line: Line) -> list[np.ndarray]:
        """Intersection points with a line (0, 1, or 2 of them)."""
        d = line.direction / np.linalg.norm(line.direction)
        f = _pt(line.point) - _pt(self.center)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.radius**2
        disc = b * b - 4 * c
        if disc < -_EPS:
            return []
        disc = max(disc, 0.0)
        sq = np.sqrt(disc)
        ts = {(-b - sq) / 2, (-b + sq) / 2}
        return [line.point + t * d for t in sorted(ts)]

    def intersect_circle(self, other: "Circle") -> list[np.ndarray]:
        """Intersection points with another circle (0, 1, or 2 of them)."""
        p0, p1 = _pt(self.center), _pt(other.center)
        d = float(np.linalg.norm(p1 - p0))
        if d <= _EPS or d > self.radius + other.radius or d < abs(self.radius - other.radius):
            return []
        a = (self.radius**2 - other.radius**2 + d**2) / (2 * d)
        h2 = self.radius**2 - a**2
        mid = p0 + a * (p1 - p0) / d
        if h2 <= _EPS:
            return [mid]
        h = np.sqrt(h2)
        perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
        return [mid + h * perp, mid - h * perp]


@dataclass(slots=True)
class Triangle:
    """Triangle with vertices ``a, b, c``."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        self.a, self.b, self.c = _pt(self.a), _pt(self.b), _pt(self.c)
        if collinear(self.a, self.b, self.c):
            raise ModelError("degenerate triangle: vertices are collinear")

    @property
    def area(self) -> float:
        return abs(polygon_area([self.a, self.b, self.c]))

    @property
    def perimeter(self) -> float:
        return distance(self.a, self.b) + distance(self.b, self.c) + distance(self.c, self.a)

    @property
    def centroid(self) -> np.ndarray:
        return (self.a + self.b + self.c) / 3.0

    @property
    def circumcircle(self) -> Circle:
        """Circle passing through all three vertices."""
        ax, ay = self.a
        bx, by = self.b
        cx, cy = self.c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / d
        center = np.array([ux, uy])
        return Circle(center=center, radius=distance(center, self.a))

    @property
    def incircle(self) -> Circle:
        """Largest circle fitting inside the triangle."""
        la = distance(self.b, self.c)
        lb = distance(self.a, self.c)
        lc = distance(self.a, self.b)
        peri = la + lb + lc
        center = (la * self.a + lb * self.b + lc * self.c) / peri
        radius = 2 * self.area / peri
        return Circle(center=center, radius=radius)

    def angles(self) -> tuple[float, float, float]:
        """Interior angles (radians) at vertices ``a, b, c``."""
        def ang(p, q, r):
            u = _pt(q) - _pt(p)
            v = _pt(r) - _pt(p)
            cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return float(np.arccos(np.clip(cosv, -1.0, 1.0)))
        return ang(self.a, self.b, self.c), ang(self.b, self.a, self.c), ang(self.c, self.a, self.b)


def convex_hull(points) -> np.ndarray:
    """Convex hull (CCW) of a point set via Andrew's monotone chain."""
    P = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h: list[np.ndarray] = []
        for p in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(P) + half(P[::-1]))
