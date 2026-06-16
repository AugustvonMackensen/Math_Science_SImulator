import Editor from "@monaco-editor/react";
import { useState } from "react";
import { api, type ExecuteResponse } from "../api";

const SAMPLE = `# The simulator engine is importable inside the sandbox.
from physics.mechanics import LagrangianSystem
import matplotlib.pyplot as plt

pendulum = LagrangianSystem(
    coordinates=["theta"],
    parameters=["m", "l", "g"],
    lagrangian="m*l**2*theta_dot**2/2 + m*g*l*cos(theta)",
)
res = pendulum.simulate(
    initial={"theta": (1.0, 0.0)},
    t_span=(0.0, 10.0),
    parameters={"m": 1.0, "l": 1.0, "g": 9.81},
)
print("theta_ddot =", pendulum.acceleration_expressions()["theta_ddot"])

plt.plot(res.t, res.component(0))
plt.xlabel("t"); plt.ylabel("theta"); plt.title("Pendulum")
`;

export default function CodeRunner() {
  const [code, setCode] = useState(SAMPLE);
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
            {running ? "Running…" : "▶ Run"}
          </button>
          {result && (
            <span className="meta">
              {result.executor} · {result.duration_seconds.toFixed(2)}s
              {result.timed_out && " · timed out"}
            </span>
          )}
        </div>
        <Editor
          height="60vh"
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={(v) => setCode(v ?? "")}
          options={{ fontSize: 13, minimap: { enabled: false }, scrollBeyondLastLine: false }}
        />
      </div>

      <div className="output-col">
        <div className="section-title">Terminal</div>
        <pre className="terminal">
          {error && <span className="err">Request error: {error}</span>}
          {result?.stdout && <span>{result.stdout}</span>}
          {result?.stderr && <span className="err">{result.stderr}</span>}
          {!error && !result && <span className="dim">Run code to see output…</span>}
        </pre>

        {result && result.images.length > 0 && (
          <div className="figures">
            <div className="section-title">Figures</div>
            {result.images.map((img, i) => (
              <img key={i} src={`data:image/png;base64,${img}`} alt={`figure ${i + 1}`} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
