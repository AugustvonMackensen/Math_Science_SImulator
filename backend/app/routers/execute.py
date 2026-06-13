"""Code-execution endpoint — runs user Python in the sandbox."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..sandbox import run_code
from ..schemas import ExecuteRequest, ExecuteResponse

router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Run a Python snippet and return stdout/stderr plus any captured figures."""
    try:
        result = run_code(req.code, req.timeout)
    except RuntimeError as exc:  # no executor available / disabled
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExecuteResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        timed_out=result.timed_out,
        executor=result.executor,
        images=result.images,
    )
