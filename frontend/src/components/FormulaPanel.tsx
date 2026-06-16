import { useState } from "react";
import { api, type CalculusResponse, type FormulaResponse } from "../api";
import MathInput from "./MathInput";
import TeX from "./TeX";

type Op = "derivative" | "integral" | "limit" | "series";

export default function FormulaPanel() {
  const [expr, setExpr] = useState("\\sin^2(x)+\\cos^2(x)");
  const [formula, setFormula] = useState<FormulaResponse | null>(null);
  const [calc, setCalc] = useState<CalculusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [variable, setVariable] = useState("x");
  const [op, setOp] = useState<Op>("derivative");
  const [point, setPoint] = useState("0");
  const [lower, setLower] = useState("0");
  const [upper, setUpper] = useState("1");

  async function evaluate() {
    setError(null);
    try {
      setFormula(await api.evaluateFormula(expr, { input_format: "latex" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runCalculus() {
    setError(null);
    try {
      setCalc(
        await api.calculus({
          expression: expr,
          input_format: "latex",
          variable,
          operation: op,
          point: op === "limit" || op === "series" ? point : undefined,
          lower: op === "integral" ? lower : undefined,
          upper: op === "integral" ? upper : undefined,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="formula-panel">
      <div className="section-title">Expression</div>
      <div className="math-input-wrap">
        <MathInput value={expr} onChange={setExpr} placeholder="type a formula…" />
      </div>
      <div className="dim small">
        Type math directly (e.g. <code>x^2</code>, <code>\sqrt</code>, <code>\frac</code>, integrals). Sent as LaTeX.
      </div>
      <div className="toolbar">
        <button className="primary" onClick={evaluate}>
          Render &amp; simplify
        </button>
      </div>

      {error && <div className="err">⚠ {error}</div>}

      {formula && (
        <div className="result-card">
          <div className="row">
            <span className="dim">input</span>
            <TeX expr={formula.latex} display />
          </div>
          <div className="row">
            <span className="dim">simplified</span>
            <TeX expr={formula.simplified_latex} display />
          </div>
          {formula.value !== null && (
            <div className="row">
              <span className="dim">value</span>
              <code>{formula.value}</code>
            </div>
          )}
          <div className="dim small">
            free symbols: {formula.free_symbols.join(", ") || "—"}
          </div>
        </div>
      )}

      <div className="section-title">Calculus</div>
      <div className="calc-controls">
        <select value={op} onChange={(e) => setOp(e.target.value as Op)}>
          <option value="derivative">d/dx</option>
          <option value="integral">∫ dx</option>
          <option value="limit">limit</option>
          <option value="series">Taylor series</option>
        </select>
        <label>
          var
          <input className="text-input narrow mono" value={variable} onChange={(e) => setVariable(e.target.value)} />
        </label>
        {op === "integral" && (
          <>
            <label>
              from
              <input className="text-input narrow mono" value={lower} onChange={(e) => setLower(e.target.value)} />
            </label>
            <label>
              to
              <input className="text-input narrow mono" value={upper} onChange={(e) => setUpper(e.target.value)} />
            </label>
          </>
        )}
        {(op === "limit" || op === "series") && (
          <label>
            at
            <input className="text-input narrow mono" value={point} onChange={(e) => setPoint(e.target.value)} />
          </label>
        )}
        <button className="primary" onClick={runCalculus}>
          Compute
        </button>
      </div>

      {calc && (
        <div className="result-card">
          <div className="dim small">{calc.operation}</div>
          <TeX expr={calc.result_latex} display />
        </div>
      )}
    </div>
  );
}
