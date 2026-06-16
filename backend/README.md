# Backend — Math & Science Simulator API

FastAPI service exposing the compute engine to the web app: **code execution**
(sandboxed), **formula** rendering/calculus, and **geometry** drawing.

## Run

```bash
# from the repo root, with the project venv active
.venv\Scripts\python -m pip install -r backend/requirements.txt
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

## Endpoints

| Method | Path                     | Purpose                                              |
| ------ | ------------------------ | ---------------------------------------------------- |
| GET    | `/health`                | Liveness + which executor (docker/local) is active   |
| POST   | `/api/execute`           | Run a Python snippet; returns stdout/stderr + figures |
| POST   | `/api/formula/evaluate`  | Parse → LaTeX + simplified form + optional value      |
| POST   | `/api/formula/calculus`  | derivative / integral / limit / series               |
| POST   | `/api/geometry/scene`    | Shape specs → drawable primitives + metrics + SVG     |
| POST   | `/api/geometry/plot`     | `y = f(x)` expression → sampled points + SVG curve     |

Formula/calculus/plot endpoints accept `input_format: "text" | "latex" | "auto"`,
so the frontend's MathLive editor can send LaTeX directly. A bare `e` is treated
as Euler's number. LaTeX parsing needs `antlr4-python3-runtime==4.11` (in
`requirements.txt`).

## Code execution & security

`/api/execute` runs code in one of two backends, chosen by `MSS_EXECUTOR`:

- **`docker`** (secure) — a throwaway container with `--network none`, dropped
  capabilities, `no-new-privileges`, and CPU/memory/PID limits. Build the image:

  ```bash
  docker build -f backend/Dockerfile.sandbox -t mss-sandbox:latest .
  ```

- **`local`** (⚠️ dev only) — a host subprocess with a timeout and output cap,
  but **no isolation**. Fine for trusted single-user development; never expose it
  publicly. Disable with `MSS_ALLOW_LOCAL_EXECUTOR=0`.

`auto` (default) uses Docker when the daemon is reachable, otherwise falls back
to local.

## Configuration (environment variables)

| Variable                    | Default            | Meaning                              |
| --------------------------- | ------------------ | ------------------------------------ |
| `MSS_EXECUTOR`              | `auto`             | `auto` / `docker` / `local`          |
| `MSS_SANDBOX_IMAGE`         | `mss-sandbox:latest` | Docker image for execution         |
| `MSS_EXEC_TIMEOUT`          | `10`               | Per-run timeout (seconds)            |
| `MSS_EXEC_MEMORY_MB`        | `256`              | Container memory limit               |
| `MSS_EXEC_MAX_OUTPUT`       | `262144`           | Max stdout/stderr bytes returned     |
| `MSS_ALLOW_LOCAL_EXECUTOR`  | `1`                | Permit the dev local fallback        |
| `MSS_CORS_ORIGINS`          | `http://localhost:5173` | Comma-separated allowed origins |
