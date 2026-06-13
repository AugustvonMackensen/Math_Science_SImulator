"""Executor selection and the shared result type.

``run_code`` dispatches to the Docker sandbox when available (the secure path)
or, in development, to a local subprocess fallback. The choice is driven by
:data:`backend.app.config.settings.executor` (``"auto"``/``"docker"``/``"local"``).
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from dataclasses import dataclass, field

from ..config import Settings, settings


@dataclass(slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool
    executor: str
    images: list[str] = field(default_factory=list)


@functools.lru_cache(maxsize=1)
def docker_available() -> bool:
    """True if a working Docker CLI + daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        res = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=8,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def active_executor(cfg: Settings = settings) -> str:
    """Resolve which executor will actually be used right now."""
    if cfg.executor == "docker":
        return "docker"
    if cfg.executor == "local":
        return "local"
    # auto
    return "docker" if docker_available() else "local"


def run_code(code: str, timeout: float | None = None, cfg: Settings = settings) -> ExecutionResult:
    """Execute ``code`` in the selected sandbox and return its result."""
    limit = timeout if timeout is not None else cfg.exec_timeout_seconds
    choice = active_executor(cfg)
    if choice == "docker":
        from .docker_runner import run_docker
        return run_docker(code, limit, cfg)
    if not cfg.allow_local_executor:
        raise RuntimeError(
            "Docker is unavailable and the local executor is disabled "
            "(set MSS_ALLOW_LOCAL_EXECUTOR=1 for development)."
        )
    from .local import run_local
    return run_local(code, limit, cfg)
