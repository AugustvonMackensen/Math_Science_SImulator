"""Docker executor — the secure path for running untrusted user code.

Each run launches a throwaway container from the sandbox image with the
network disabled and CPU/memory/PID limits applied, mounting a temporary work
directory that holds the user's code and receives the captured figures. The
container is force-killed if it overruns the timeout.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from ..config import Settings
from .runner import ExecutionResult


def _truncate(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="replace") + "\n...[output truncated]"


def run_docker(code: str, timeout: float, cfg: Settings) -> ExecutionResult:
    start = time.perf_counter()
    timed_out = False
    images: list[str] = []
    container = f"mss-run-{uuid.uuid4().hex[:12]}"

    with tempfile.TemporaryDirectory(prefix="mss-docker-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "code.py").write_text(code, encoding="utf-8")

        cmd = [
            "docker", "run", "--rm", "--name", container,
            "--network", "none",
            "--memory", f"{cfg.exec_memory_mb}m",
            "--memory-swap", f"{cfg.exec_memory_mb}m",
            "--cpus", "1.0",
            "--pids-limit", "128",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{tmpdir.as_posix()}:/work",
            "-w", "/work",
            cfg.docker_image,
            "python", "/opt/harness.py", "/work/code.py", "/work/images.json",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            subprocess.run(["docker", "kill", container], capture_output=True)
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
            stderr += f"\n[timed out after {timeout:g}s]"
            exit_code = -1

        images_path = tmpdir / "images.json"
        if images_path.exists():
            try:
                images = json.loads(images_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                images = []

    return ExecutionResult(
        stdout=_truncate(stdout, cfg.exec_max_output_bytes),
        stderr=_truncate(stderr, cfg.exec_max_output_bytes),
        exit_code=exit_code,
        duration_seconds=time.perf_counter() - start,
        timed_out=timed_out,
        executor="docker",
        images=images,
    )
