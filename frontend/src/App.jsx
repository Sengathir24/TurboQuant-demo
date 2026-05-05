/* =========================================================
   App.jsx — TurboQuant Demo Root
   Cursor: start here. Routes between the 5 demo pages.
   ========================================================= */

import { useState, useEffect } from "react"
import ModelSelector    from "./components/ModelSelector"
import LiveMetricsBar   from "./components/LiveMetricsBar"
import GenerateDemo     from "./pages/GenerateDemo"
import VectorSearchDemo from "./pages/VectorSearchDemo"
import RAGDemo          from "./pages/RAGDemo"
import AttentionDemo    from "./pages/AttentionDemo"
import CostDemo         from "./pages/CostDemo"
import { useModelStore } from "./store/modelStore"
import "./index.css"

const TABS = [
  { id: "generate",  label: "⚡ Generate",      desc: "Stream tokens + KV compression" },
  { id: "vector",    label: "🔍 Vector Search",  desc: "FAISS + compressed embeddings"  },
  { id: "rag",       label: "📚 RAG",            desc: "Retrieve + Generate"             },
  { id: "attention", label: "🧠 Attention",      desc: "KV fidelity per layer"          },
  { id: "cost",      label: "💰 Cloud Cost",     desc: "$/1M tokens calculator"         },
]

export default function App() {
  const [activeTab, setActiveTab] = useState("generate")
  const { modelLoaded, modelName } = useModelStore()

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">TQ</div>
          <div>
            <h1 className="app-title">TurboQuant</h1>
            <p className="app-sub">Real-time LLM memory compression demo</p>
          </div>
        </div>
        <div className="header-right">
          {modelLoaded && (
            <span className="model-pill">
              <span className="model-pill-dot" />
              {modelName}
            </span>
          )}
        </div>
      </header>

      {/* ── Model selector (always visible) ── */}
      <div className="model-bar">
        <ModelSelector />
      </div>

      {/* ── Live metrics strip ── */}
      {modelLoaded && <LiveMetricsBar />}

      {/* ── Tab navigation ── */}
      <nav className="tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${activeTab === t.id ? "active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            <span className="tab-label">{t.label}</span>
            <span className="tab-desc">{t.desc}</span>
          </button>
        ))}
      </nav>

      {/* ── Page content ── */}
      <main className="page-content">
        {!modelLoaded && (
          <div className="empty-state">
            <div className="empty-icon">⬆</div>
            <h2>Select a model above to start</h2>
            <p>Pick any HuggingFace model, configure TurboQuant bits, and run live benchmarks.</p>
          </div>
        )}
        {modelLoaded && (
          <>
            {activeTab === "generate"  && <GenerateDemo />}
            {activeTab === "vector"    && <VectorSearchDemo />}
            {activeTab === "rag"       && <RAGDemo />}
            {activeTab === "attention" && <AttentionDemo />}
            {activeTab === "cost"      && <CostDemo />}
          </>
        )}
      </main>
    </div>
  )
}
