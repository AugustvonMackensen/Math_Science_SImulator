// Typed client for the Math & Science Simulator backend.
// Uses same-origin relative URLs; Vite proxies /api and /health to FastAPI.

export interface ExecuteResponse {
  stdout: string;
  stderr: string;
  exit_code: number;
  duration_seconds: number;
  timed_out: boolean;
  executor: string;
  images: string[]; // base64-encoded PNGs
}

export interface FormulaResponse {
  input: string;
  latex: string;
  simplified: string;
  simplified_latex: string;
  value: number | null;
  free_symbols: string[];
}

export interface CalculusResponse {
  operation: string;
  result: string;
  result_latex: string;
}

export interface FunctionPlotResponse {
  expression_latex: string;
  variable: string;
  x: number[];
  y: (number | null)[];
  svg: string | null;
}

export interface Drawable {
  kind: string;
  data: Record<string, unknown>;
  label: string | null;
  style: Record<string, unknown>;
}

export interface GeometryResponse {
  drawables: Drawable[];
  bounds: Record<string, number>;
  metrics: Record<string, unknown>;
  svg: string | null;
}

export interface HealthResponse {
  status: string;
  executor: string;
  docker_available: boolean;
  configured_executor: string;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (): Promise<HealthResponse> => fetch("/health").then((r) => r.json()),

  execute: (code: string, timeout?: number): Promise<ExecuteResponse> =>
    postJSON("/api/execute", { code, timeout }),

  evaluateFormula: (
    expression: string,
    opts?: { input_format?: "auto" | "text" | "latex"; variables?: Record<string, number> },
  ): Promise<FormulaResponse> =>
    postJSON("/api/formula/evaluate", {
      expression,
      input_format: opts?.input_format ?? "auto",
      variables: opts?.variables ?? null,
    }),

  calculus: (req: {
    expression: string;
    input_format?: "auto" | "text" | "latex";
    variable: string;
    operation: "derivative" | "integral" | "limit" | "series";
    order?: number;
    point?: string;
    lower?: string;
    upper?: string;
  }): Promise<CalculusResponse> => postJSON("/api/formula/calculus", req),

  plotFunction: (req: {
    expression: string;
    input_format?: "auto" | "text" | "latex";
    variable?: string;
    x_min?: number;
    x_max?: number;
    samples?: number;
  }): Promise<FunctionPlotResponse> => postJSON("/api/geometry/plot", req),

  geometryScene: (req: {
    shapes: unknown[];
    derive?: boolean;
    render_svg?: boolean;
  }): Promise<GeometryResponse> =>
    postJSON("/api/geometry/scene", {
      shapes: req.shapes,
      derive: req.derive ?? false,
      render_svg: req.render_svg ?? true,
    }),
};
