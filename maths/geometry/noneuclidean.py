r"""Non-Euclidean geometry — spherical and hyperbolic.

**Spherical** (positive curvature): great-circle distances on a sphere of
radius ``R`` and the angle-excess area law for spherical triangles.

**Hyperbolic** (negative curvature): the two standard conformal models —
the **Poincaré disk** and the **upper half-plane** — with their geodesic
distances, plus the angle-*defect* area law.

A unifying sanity check across all three: for a triangle the area is
``R^2 * |angle sum - pi|``, positive on the sphere (excess) and negative in
the hyperbolic plane (defect), and zero in the Euclidean limit.
"""

from __future__ import annotations

import numpy as np

from core.exceptions import ModelError

_EPS = 1e-12


# --- spherical -------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2, *, radius: float = 1.0) -> float:
    """Great-circle distance between two (lat, lon) points in radians-of-latitude.

    Inputs are in radians. Returns arc length on a sphere of given ``radius``.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def spherical_distance(u, v, *, radius: float = 1.0) -> float:
    """Great-circle distance between two points given as 3-D unit vectors."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(radius * np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def spherical_triangle_area(a: float, b: float, c: float, *, radius: float = 1.0) -> float:
    """Area of a spherical triangle from its three interior angles (radians).

    Girard's theorem: ``area = R^2 (a + b + c - pi)`` (the spherical excess).
    """
    excess = a + b + c - np.pi
    if excess <= -_EPS:
        raise ModelError("spherical triangle angles must sum to more than pi")
    return float(radius**2 * excess)


# --- hyperbolic: Poincaré disk ---------------------------------------------

def _in_unit_disk(z: complex) -> bool:
    return abs(z) < 1.0 - 1e-9


def poincare_disk_distance(z1: complex, z2: complex) -> float:
    r"""Hyperbolic distance in the Poincaré disk model (unit disk).

    .. math:: d(z_1, z_2) = \operatorname{arccosh}
        \left(1 + \frac{2|z_1 - z_2|^2}{(1-|z_1|^2)(1-|z_2|^2)}\right).
    """
    z1, z2 = complex(z1), complex(z2)
    if not (_in_unit_disk(z1) and _in_unit_disk(z2)):
        raise ModelError("points must lie strictly inside the unit disk")
    num = 2 * abs(z1 - z2) ** 2
    den = (1 - abs(z1) ** 2) * (1 - abs(z2) ** 2)
    return float(np.arccosh(1 + num / den))


# --- hyperbolic: upper half-plane ------------------------------------------

def upper_half_plane_distance(z1: complex, z2: complex) -> float:
    r"""Hyperbolic distance in the upper half-plane model (``Im z > 0``).

    .. math:: d(z_1, z_2) = \operatorname{arccosh}
        \left(1 + \frac{|z_1 - z_2|^2}{2\,\operatorname{Im}z_1\,\operatorname{Im}z_2}\right).
    """
    z1, z2 = complex(z1), complex(z2)
    if z1.imag <= 0 or z2.imag <= 0:
        raise ModelError("points must have positive imaginary part")
    num = abs(z1 - z2) ** 2
    den = 2 * z1.imag * z2.imag
    return float(np.arccosh(1 + num / den))


def hyperbolic_triangle_area(a: float, b: float, c: float, *, radius: float = 1.0) -> float:
    """Area of a hyperbolic triangle from its interior angles (radians).

    Gauss-Bonnet: ``area = R^2 (pi - a - b - c)`` (the angle defect).
    """
    defect = np.pi - (a + b + c)
    if defect <= -_EPS:
        raise ModelError("hyperbolic triangle angles must sum to less than pi")
    return float(radius**2 * defect)
