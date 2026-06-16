# Frontend — Math & Science Simulator

React + TypeScript (Vite) web UI for the simulator. Three panels:

- **Code IDE** — Monaco editor; runs Python via `/api/execute`, shows the
  terminal output and any matplotlib figures.
- **Formulas** — a **MathLive** math editor (type math directly, with a virtual
  keyboard); rendered with KaTeX; simplify and run calculus (derivative /
  integral / limit / Taylor series). Sent to the backend as LaTeX.
- **Geometry** — three input modes:
  - **Formula** — a MathLive editor for `y = f(x)`, plotted as an SVG curve.
  - **Code** — a Monaco editor; write Python (matplotlib / engine helpers) and
    see the rendered figures.
  - **Shapes (JSON)** — declarative shapes → SVG plus derived constructs
    (circumcircle / incircle / centroid) and metrics.

## Develop

Start the backend first (see [../backend/README.md](../backend/README.md)):

```bash
# terminal 1 — backend on :8000
.venv\Scripts\python -m uvicorn backend.app.main:app --reload

# terminal 2 — frontend on :5173 (proxies /api and /health to :8000)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Build / typecheck

```bash
npm run build       # tsc -b + vite build  ->  dist/
npm run typecheck
```

## Notes

- The Monaco editor is loaded by `@monaco-editor/react` at runtime (CDN loader),
  so the first editor render needs network access in dev.
- API calls use same-origin relative URLs; the Vite dev proxy
  (`vite.config.ts`) forwards them to the FastAPI backend. For production, serve
  the built `dist/` behind the same origin as the API or set `MSS_CORS_ORIGINS`.
