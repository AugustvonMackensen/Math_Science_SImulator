"""Runtime configuration, read from environment variables.

All knobs have safe defaults so the app runs out of the box for local
development. Override via environment (e.g. ``MSS_EXECUTOR=docker``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    """Application settings."""

    # "auto" picks Docker when available, else the dev local executor.
    executor: str = field(default_factory=lambda: os.getenv("MSS_EXECUTOR", "auto"))
    docker_image: str = field(
        default_factory=lambda: os.getenv("MSS_SANDBOX_IMAGE", "mss-sandbox:latest")
    )
    # Resource limits for a single code run.
    exec_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("MSS_EXEC_TIMEOUT", "10"))
    )
    exec_memory_mb: int = field(default_factory=lambda: int(os.getenv("MSS_EXEC_MEMORY_MB", "256")))
    exec_max_output_bytes: int = field(
        default_factory=lambda: int(os.getenv("MSS_EXEC_MAX_OUTPUT", str(256 * 1024)))
    )
    # Allow the unsafe local subprocess fallback (dev only).
    allow_local_executor: bool = field(
        default_factory=lambda: _env_bool("MSS_ALLOW_LOCAL_EXECUTOR", True)
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("MSS_CORS_ORIGINS", ["http://localhost:5173"])
    )


settings = Settings()
