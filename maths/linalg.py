r"""Linear algebra — decompositions, orthogonalization, and structure.

Thin, well-typed conveniences on top of NumPy/SciPy that add the things a
postgraduate user actually reaches for: Gram-Schmidt orthonormalization,
null spaces and rank, conditioning diagnostics, the standard matrix
factorizations (LU, QR, Cholesky, SVD, eigen), and the matrix exponential.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from core.exceptions import ModelError

ArrayLike = np.ndarray | list


def _as_matrix(A: ArrayLike) -> np.ndarray:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ModelError(f"expected a 2-D matrix, got shape {M.shape}")
    return M


def gram_schmidt(vectors: ArrayLike, *, normalize: bool = True, tol: float = 1e-12) -> np.ndarray:
    r"""Orthogonalize the *columns* of ``vectors`` (modified Gram-Schmidt).

    Returns a matrix whose columns are mutually orthogonal (orthonormal if
    ``normalize``). Linearly dependent columns are dropped, so the result may
    have fewer columns than the input — its column count is the rank.
    """
    A = _as_matrix(vectors)
    out: list[np.ndarray] = []
    for j in range(A.shape[1]):
        v = A[:, j].astype(float).copy()
        for q in out:
            v -= np.dot(q, v) * q if normalize else (np.dot(q, v) / np.dot(q, q)) * q
        norm = np.linalg.norm(v)
        if norm <= tol:
            continue  # dependent column
        out.append(v / norm if normalize else v)
    if not out:
        return np.zeros((A.shape[0], 0))
    return np.column_stack(out)


def null_space(A: ArrayLike, *, tol: float | None = None) -> np.ndarray:
    """Orthonormal basis for the null space ``{x : A x = 0}`` (columns)."""
    M = _as_matrix(A)
    u, s, vh = np.linalg.svd(M)
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    rank = int((s > tol).sum())
    return vh[rank:].conj().T


def rank(A: ArrayLike, *, tol: float | None = None) -> int:
    """Numerical rank via SVD."""
    return int(np.linalg.matrix_rank(_as_matrix(A), tol=tol))


def condition_number(A: ArrayLike) -> float:
    """2-norm condition number ``sigma_max / sigma_min`` (``inf`` if singular)."""
    return float(np.linalg.cond(_as_matrix(A)))


def is_positive_definite(A: ArrayLike, *, tol: float = 1e-12) -> bool:
    """True if ``A`` is symmetric positive definite."""
    M = _as_matrix(A)
    if not np.allclose(M, M.T, atol=1e-10):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


@dataclass(slots=True)
class EigenResult:
    """Eigenvalues and (column) eigenvectors."""

    values: np.ndarray
    vectors: np.ndarray


def eig(A: ArrayLike, *, symmetric: bool | None = None) -> EigenResult:
    """Eigendecomposition. Uses the symmetric solver when ``A`` is symmetric.

    For symmetric inputs eigenvalues are real and returned in ascending order.
    """
    M = _as_matrix(A)
    if symmetric is None:
        symmetric = bool(np.allclose(M, M.T, atol=1e-10))
    if symmetric:
        w, v = np.linalg.eigh(M)
    else:
        w, v = np.linalg.eig(M)
    return EigenResult(values=w, vectors=v)


def solve(A: ArrayLike, b: ArrayLike, *, assume_a: str = "gen") -> np.ndarray:
    """Solve ``A x = b``.

    ``assume_a`` may be ``"gen"``, ``"sym"``, ``"her"`` or ``"pos"`` to dispatch
    to a specialized (faster, more stable) LAPACK routine.

    Raises
    ------
    ModelError
        If ``A`` is singular.
    """
    M = _as_matrix(A)
    rhs = np.asarray(b, dtype=float)
    try:
        return sla.solve(M, rhs, assume_a=assume_a)
    except (np.linalg.LinAlgError, sla.LinAlgError) as exc:
        raise ModelError(f"could not solve linear system: {exc}") from exc


def lu(A: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU factorization with partial pivoting: returns ``(P, L, U)`` with ``P A = L U``."""
    return sla.lu(_as_matrix(A))


def qr(A: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR factorization: returns ``(Q, R)`` with ``A = Q R``."""
    return np.linalg.qr(_as_matrix(A))


def cholesky(A: ArrayLike) -> np.ndarray:
    """Lower Cholesky factor ``L`` with ``A = L L^T`` (requires SPD)."""
    M = _as_matrix(A)
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise ModelError(f"matrix is not positive definite: {exc}") from exc


def svd(A: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition: returns ``(U, s, Vh)`` with ``A = U diag(s) Vh``."""
    return np.linalg.svd(_as_matrix(A), full_matrices=False)


def expm(A: ArrayLike) -> np.ndarray:
    """Matrix exponential ``e^A`` (Padé approximation, via SciPy)."""
    return sla.expm(_as_matrix(A))
