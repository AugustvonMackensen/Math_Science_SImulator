r"""Affine transformations of the plane (and 3-D rotations).

Plane transforms are represented as 3x3 homogeneous matrices so that
translations compose with linear maps by ordinary matrix multiplication.
Build primitives (:func:`translation`, :func:`rotation`, :func:`scaling`,
:func:`reflection`, :func:`shear`), :func:`compose` them right-to-left, and
:func:`apply` them to point sets.
"""

from __future__ import annotations

import numpy as np

from core.exceptions import ModelError


def identity() -> np.ndarray:
    """3x3 identity transform."""
    return np.eye(3)


def translation(dx: float, dy: float) -> np.ndarray:
    """Translate by ``(dx, dy)``."""
    T = np.eye(3)
    T[0, 2], T[1, 2] = dx, dy
    return T


def rotation(theta: float, about: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Rotate by ``theta`` radians about the point ``about`` (CCW)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if about == (0.0, 0.0):
        return R
    ax, ay = about
    return translation(ax, ay) @ R @ translation(-ax, -ay)


def scaling(sx: float, sy: float | None = None, about: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Scale by ``sx`` (and ``sy``, default equal) about ``about``."""
    if sy is None:
        sy = sx
    S = np.diag([sx, sy, 1.0])
    if about == (0.0, 0.0):
        return S
    ax, ay = about
    return translation(ax, ay) @ S @ translation(-ax, -ay)


def reflection(axis: str = "x") -> np.ndarray:
    """Reflect across the ``"x"`` axis, ``"y"`` axis, or the line ``"y=x"``."""
    if axis == "x":
        return np.diag([1.0, -1.0, 1.0])
    if axis == "y":
        return np.diag([-1.0, 1.0, 1.0])
    if axis in ("y=x", "yx"):
        return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    raise ModelError("axis must be 'x', 'y', or 'y=x'")


def reflection_about_line(theta: float) -> np.ndarray:
    """Reflect across a line through the origin at angle ``theta`` to the x-axis."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])


def shear(kx: float = 0.0, ky: float = 0.0) -> np.ndarray:
    """Shear by ``kx`` (x by y) and ``ky`` (y by x)."""
    return np.array([[1.0, kx, 0.0], [ky, 1.0, 0.0], [0.0, 0.0, 1.0]])


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms; the rightmost is applied to points first."""
    M = np.eye(3)
    for T in transforms:
        M = M @ T
    return M


def apply(transform: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 transform to point(s) of shape ``(2,)`` or ``(n, 2)``."""
    P = np.asarray(points, dtype=float)
    single = P.ndim == 1
    if single:
        P = P[None, :]
    if P.shape[1] != 2:
        raise ModelError("points must have shape (2,) or (n, 2)")
    homo = np.hstack([P, np.ones((len(P), 1))])
    out = (transform @ homo.T).T[:, :2]
    return out[0] if single else out


# --- 3-D rotations ---------------------------------------------------------

def rotation_3d(axis: str, theta: float) -> np.ndarray:
    """3x3 rotation matrix about a principal axis ``'x'``, ``'y'`` or ``'z'``."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ModelError("axis must be 'x', 'y', or 'z'")


def rotation_axis_angle(axis, theta: float) -> np.ndarray:
    """3x3 rotation about an arbitrary unit axis by ``theta`` (Rodrigues' formula)."""
    k = np.asarray(axis, dtype=float)
    n = np.linalg.norm(k)
    if n <= 1e-12:
        raise ModelError("rotation axis must be non-zero")
    k = k / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
