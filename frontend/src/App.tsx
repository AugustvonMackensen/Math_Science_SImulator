import { useEffect, useState } from "react";
import { api, type HealthResponse } from "./api";
import CodeRunner from "./components/CodeRunner";
import FormulaPanel from "./components/FormulaPanel";
import GeometryPanel from "./components/GeometryPanel";

type Tab = "code" | "formula" | "geometry";

const TABS: { id: Tab; label: string }[] = [
  { id: "code", label: "Code IDE" },
  { id: "formula", label: "Formulas" },
  { id: "geometry", label: "Geometry" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("code");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Math &amp; Science Simulator</h1>
        <div className="health">
          {health ? (
            <span className={health.docker_available ? "ok" : "warn"}>
              executor: {health.executor}
              {!health.docker_available && " (dev — no Docker)"}
            </span>
          ) : (
            <span className="warn">backend offline</span>
          )}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {tab === "code" && <CodeRunner />}
        {tab === "formula" && <FormulaPanel />}
        {tab === "geometry" && <GeometryPanel />}
      </main>
    </div>
  );
}
