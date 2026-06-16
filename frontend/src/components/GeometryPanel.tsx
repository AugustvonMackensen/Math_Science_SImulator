import Editor from "@monaco-editor/react";
import { useState } from "react";
import { api, type ExecuteResponse, type FunctionPlotResponse, type GeometryResponse } from "../api";
import MathInput from "./MathInput";
import TeX from "./TeX";

type Mode = "formula" | "code" | "shapes";

const SHAPES_SAMPLE = JSON.stringify(
  [
    { kind: "triangle", points: [[0, 0], [4, 0], [1, 3]], label: "T" },
    { kind: "circle", center: [2, 1], radius: 0.5, label: "c" },
  ],
  null,
  2,
);

const CODE_SAMPLE = `# Draw anything with matplotlib (or the engine's geometry helpers).
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(t), np.sin(t), label="unit circle")
tri = np.array([[1, 0], [-0.5, 0.87], [-0.5, -0.87], [1, 0]])
ax.plot(tri[:, 0], tri[:, 1], label="triangle")
ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)
print("drew a circle + inscribed triangle")
`;

export default function GeometryPanel() {
  const [mode, setMode] = useState<Mode>("formula");
  return (
    <div>
      <div className="mode-switch">
        <button className={mode === "formula" ? "chip active" : "chip"} onClick={() => setMode("formula")}>
          Formula (y = f(x))
        </button>
        <button className={mode === "code" ? "chip active" : "chip"} onClick={() => setMode("code")}>
          Code
        </button>
        <button className={mode === "shapes" ? "chip active" : "chip"} onClick={() => setMode("shapes")}>
          Shapes (JSON)
        </button>
      </div>
      {mode === "formula" && <FormulaPlot />}
      {mode === "code" && <CodeDraw />}
      {mode === "shapes" && <ShapeScene />}
    </div>
  );
}

// --- Mode 1: plot a function from the math editor ---------------------------

function FormulaPlot() {
  const [expr, setExpr] = useState("\\sin(x)\\cdot e^{-x^2/8}");
  const [xmin, setXmin] = useState(-10);
  const [xmax, setXmax] = useState(10);
  const [res, setRes] = useState<FunctionPlotResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function plot() {
    setError(null);
    try {
      setRes(await api.plotFunction({ expression: expr, input_format: "latex", x_min: xmin, x_max: xmax }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="panel-split">
      <div className="editor-col">
        <div className="section-title">f(x)</div>
        <div className="math-input-wrap">
          <MathInput value={expr} onChange={setExpr} placeholder="e.g. \sin(x)" />
        </div>
        <div className="calc-controls">
          <label>x from<input className="text-input narrow mono" type="number" value={xmin} onChange={(e) => setXmin(+e.target.value)} /></label>
          <label>to<input className="text-input narrow mono" type="number" value={xmax} onChange={(e) => setXmax(+e.target.value)} /></label>
          <button className="primary" onClick={plot}>Plot</button>
        </div>
        {res && (
          <div className="result-card">
            <TeX expr={`f(${res.variable}) = ${res.expression_latex}`} display />
          </div>
        )}
        {error && <div className="err">⚠ {error}</div>}
      </div>
      <div className="output-col">
        <div className="section-title">Canvas</div>
        {res?.svg ? (
          <div className="svg-wrap" dangerouslySetInnerHTML={{ __html: res.svg }} />
        ) : (
          <div className="dim">Plot a function to see it here…</div>
        )}
      </div>
    </div>
  );
}

// --- Mode 2: draw with code -------------------------------------------------

function CodeDraw() {
  const [code, setCode] = useState(CODE_SAMPLE);
  const [result, setResult] = useState<ExecuteResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.execute(code));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="panel-split">
      <div className="editor-col">
        <div className="toolbar">
          <button className="primary" onClick={run} disabled={running}>
            {running ? "Running…" : "▶ Run & draw"}
          </button>
          {result && <span className="meta">{result.executor} · {result.duration_seconds.toFixed(2)}s</span>}
        </div>
        <Editor
          height="55vh"
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={(v) => setCode(v ?? "")}
          options={{ fontSize: 13, minimap: { enabled: false }, scrollBeyondLastLine: false }}
        />
      </div>
      <div className="output-col">
        <div className="section-title">Figures</div>
        {error && <div className="err">⚠ {error}</div>}
        {result?.images?.length ? (
          result.images.map((img, i) => (
            <img key={i} src={`data:image/png;base64,${img}`} alt={`figure ${i + 1}`} />
          ))
        ) : (
          <div className="dim">Run code that calls matplotlib to see figures…</div>
        )}
        {result && (result.stdout || result.stderr) && (
          <>
            <div className="section-title">Terminal</div>
            <pre className="terminal small">
              {result.stdout && <span>{result.stdout}</span>}
              {result.stderr && <span className="err">{result.stderr}</span>}
            </pre>
          </>
        )}
      </div>
    </div>
  );
}

// --- Mode 3: declarative shapes (JSON) -------------------------------------

function ShapeScene() {
  const [text, setText] = useState(SHAPES_SAMPLE);
  const [derive, setDerive] = useState(true);
  const [scene, setScene] = useState<GeometryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function draw() {
    setError(null);
    let shapes: unknown[];
    try {
      shapes = JSON.parse(text);
      if (!Array.isArray(shapes)) throw new Error("shapes must be a JSON array");
    } catch (e) {
      setError(`Invalid JSON: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    try {
      setScene(await api.geometryScene({ shapes, derive, render_svg: true }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="panel-split">
      <div className="editor-col">
        <div className="section-title">Shapes (JSON)</div>
        <textarea className="text-input mono shapes" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="toolbar">
          <label className="check">
            <input type="checkbox" checked={derive} onChange={(e) => setDerive(e.target.checked)} />
            derive (circumcircle / incircle / centroid)
          </label>
          <button className="primary" onClick={draw}>Draw</button>
        </div>
        <div className="dim small">kinds: point · segment · line · circle · triangle · polygon</div>
        {error && <div className="err">⚠ {error}</div>}
      </div>
      <div className="output-col">
        <div className="section-title">Canvas</div>
        {scene?.svg ? (
          <div className="svg-wrap" dangerouslySetInnerHTML={{ __html: scene.svg }} />
        ) : (
          <div className="dim">Draw shapes to see them here…</div>
        )}
        {scene && Object.keys(scene.metrics).length > 0 && (
          <>
            <div className="section-title">Metrics</div>
            <pre className="terminal small">{JSON.stringify(scene.metrics, null, 2)}</pre>
          </>
        )}
      </div>
    </div>
  );
}
