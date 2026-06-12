"""Exception hierarchy for the simulator engine."""

from __future__ import annotations


class MathSciError(Exception):
    """Base class for all engine errors."""


class ModelError(MathSciError):
    """Raised when a physical/mathematical model is ill-posed.

    Examples: a Lagrangian with no generalized coordinates, mismatched
    initial-condition dimensions, or a singular mass matrix.
    """


class ConvergenceError(MathSciError):
    """Raised when an iterative or adaptive numerical routine fails to converge."""
