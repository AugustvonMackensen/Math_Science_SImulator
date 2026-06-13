"""Dev-only local executor — runs user code in a subprocess on the host.

⚠️  NOT a security boundary. It enforces a timeout and output cap but does NOT
isolate the filesystem, network, or system calls. Use it only for trusted,
single-user local development; install Docker before exposing the app.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..config import Settings
from .runner import ExecutionResult

_HARNESS = Path(__file__).with_name("harness.py")


def _truncate(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="replace") + "\n...[output truncated]"


def run_local(code: str, timeout: float, cfg: Settings) -> ExecutionResult:
    start = time.perf_counter()
    timed_out = False
    images: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mss-run-") as tmp:
        tmpdir = Path(tmp)
        code_path = tmpdir / "code.py"
        images_path = tmpdir / "images.json"
        code_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(_HARNESS), str(code_path), str(images_path)],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[timed out after {timeout:g}s]"
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            exit_code = -1

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
        executor="local",
        images=images,
    )
