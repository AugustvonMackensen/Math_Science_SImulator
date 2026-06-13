"""Code-execution sandbox: Docker (secure) with a dev-only local fallback."""

from __future__ import annotations

from .runner import ExecutionResult, active_executor, run_code

__all__ = ["ExecutionResult", "run_code", "active_executor"]
