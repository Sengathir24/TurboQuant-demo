/* ModelSelector.jsx — HuggingFace model picker + TurboQuant config */
import { useState, useEffect } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

const FAMILY_COLORS = {
  "GPT-2":  "#4F86C6",
  "FLAN-T5":"#2E9E6B",
  "OPT":    "#D4782A",
  "Pythia": "#8B5CF6",
  "BLOOM":  "#DB2777",
}

export default function ModelSelector() {
  const [models, setModels]       = useState([])
  const [selected, setSelected]   = useState("gpt2-medium")
  const [filterFamily, setFilter] = useState("All")
  const { bits, mode, setBits, setMode, loadingModel, setLoading, setModelLoaded, setError, loadError } = useModelStore()

  useEffect(() => {
    api.models().then(d => setModels(d.models || []))
  }, [])

  const families = ["All", ...new Set(models.map(m => m.family))]
  const visible   = filterFamily === "All" ? models : models.filter(m => m.family === filterFamily)
  const selModel  = models.find(m => m.id === selected)

  async function handleLoad() {
    setLoading(true)
    try {
      const result = await api.loadModel({ model_id: selected, bits, mode })
      if (result.status === "loaded") setModelLoaded(result)
      else setError(result.detail || "Load failed")
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="model-selector">
      {/* ── Family filter ── */}
      <div className="family-pills">
        {families.map(f => (
          <button key={f}
            className={`family-pill ${filterFamily === f ? "on" : ""}`}
            onClick={() => setFilter(f)}
          >{f}</button>
        ))}
      </div>

      {/* ── Model grid ── */}
      <div className="model-grid">
        {visible.map(m => (
          <div key={m.id}
            className={`model-card ${selected === m.id ? "selected" : ""}`}
            onClick={() => setSelected(m.id)}
          >
            <div className="model-card-header">
              <span className="family-dot"
                style={{ background: FAMILY_COLORS[m.family] || "#888" }} />
              <span className="model-card-name">{m.name}</span>
              <span className="model-card-params">{m.params}</span>
            </div>
            <p className="model-card-desc">{m.description}</p>
            <div className="model-card-tags">
              {m.tags.map(t => <span key={t} className="tag">{t}</span>)}
            </div>
            <div className="model-card-meta">
              <span>ctx: {m.context_len.toLocaleString()}</span>
              <span>VRAM: {m.vram_fp16_gb}GB</span>
              <span>rec: {m.recommended_bits}-bit</span>
            </div>
          </div>
        ))}
      </div>

      {/* ── Config row ── */}
      <div className="config-row">
        <div className="config-group">
          <label>Bit-width</label>
          <div className="bit-pills">
            {[1,2,3,4].map(b => (
              <button key={b}
                className={`bit-pill ${bits === b ? "on" : ""}`}
                onClick={() => setBits(b)}
              >
                {b}-bit
                <span className="bit-ratio"> {Math.round(32/b)}×</span>
              </button>
            ))}
          </div>
        </div>

        <div className="config-group">
          <label>Mode</label>
          <div className="mode-toggle">
            <button className={mode === "mse" ? "on" : ""} onClick={() => setMode("mse")}>
              MSE only
            </button>
            <button className={mode === "ip" ? "on" : ""} onClick={() => setMode("ip")}>
              IP + QJL
            </button>
          </div>
        </div>

        <div className="config-group">
          <label>Compression</label>
          <div className="compression-preview">
            <span className="ratio-big">{Math.round(32 / (bits - (mode==="ip"?0:0)))}×</span>
            <span className="ratio-sub">vs 32-bit</span>
          </div>
        </div>

        <button
          className={`load-btn ${loadingModel ? "loading" : ""}`}
          onClick={handleLoad}
          disabled={loadingModel || !selected}
        >
          {loadingModel ? (
            <><span className="spinner" /> Loading model...</>
          ) : (
            `Load ${selModel?.name || "model"}`
          )}
        </button>
      </div>

      {loadError && <div className="error-bar">⚠ {loadError}</div>}
    </div>
  )
}
