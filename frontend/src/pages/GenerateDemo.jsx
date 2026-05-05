/* GenerateDemo.jsx — side-by-side streaming: baseline vs compressed */
import { useState, useRef } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

const PRESETS = [
  "Explain how transformer attention mechanisms work in detail.",
  "Write a short story about an AI that learns to compress memories.",
  "What are the key differences between vector quantization methods?",
  "Describe the future of large language model inference optimization.",
]

export default function GenerateDemo() {
  const { bits, mode } = useModelStore()
  const [prompt, setPrompt]         = useState(PRESETS[0])
  const [maxTokens, setMaxTokens]   = useState(150)
  const [temperature, setTemp]      = useState(0.7)
  const [compareMode, setCompare]   = useState(true)
  const [running, setRunning]       = useState(false)

  const [baselineText, setBaseline] = useState("")
  const [compText, setComp]         = useState("")
  const [baselineTps, setBaseTps]   = useState(null)
  const [compTps, setCompTps]       = useState(null)
  const [compMse, setCompMse]       = useState(null)
  const [bytesSaved, setBytesSaved] = useState(0)
  const [tokenCount, setTokenCount] = useState({baseline:0, compressed:0})

  const baseRef = useRef(null)
  const compRef = useRef(null)

  function scrollBottom(ref) {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }

  function run() {
    setRunning(true)
    setBaseline(""); setComp("")
    setBaseTps(null); setCompTps(null); setCompMse(null)
    setBytesSaved(0); setTokenCount({baseline:0,compressed:0})

    api.streamGenerate(
      { prompt, max_new_tokens: maxTokens, temperature, compare_mode: compareMode },
      (chunk) => {
        if (chunk.run === "baseline") {
          setBaseline(p => p + chunk.token)
          setBaseTps(chunk.tps)
          setTokenCount(c => ({...c, baseline: chunk.tokens_so_far}))
          scrollBottom(baseRef)
        }
        if (chunk.run === "compressed") {
          setComp(p => p + chunk.token)
          setCompTps(chunk.tps)
          setCompMse(chunk.kv_mse)
          setBytesSaved(chunk.bytes_saved)
          setTokenCount(c => ({...c, compressed: chunk.tokens_so_far}))
          scrollBottom(compRef)
        }
        if (chunk.run === "summary") {
          setBytesSaved(chunk.total_bytes_saved)
        }
      },
      () => setRunning(false)
    )
  }

  const mbSaved = (bytesSaved / 1e6).toFixed(2)

  return (
    <div className="demo-page">
      <div className="demo-header">
        <h2>⚡ Streaming Generation</h2>
        <p>Tokens stream in real time. Left = uncompressed baseline. Right = TurboQuant KV cache compression.</p>
      </div>

      {/* ── Controls ── */}
      <div className="controls-card">
        <div className="ctrl-row">
          <div className="ctrl-group">
            <label>Prompt</label>
            <div className="preset-row">
              {PRESETS.map((p,i) => (
                <button key={i} className={`preset-btn ${prompt===p?"on":""}`}
                  onClick={() => setPrompt(p)}>Preset {i+1}</button>
              ))}
            </div>
            <textarea className="prompt-input" value={prompt}
              onChange={e => setPrompt(e.target.value)} rows={3} />
          </div>
        </div>
        <div className="ctrl-row three-col">
          <div className="ctrl-group">
            <label>Max tokens: {maxTokens}</label>
            <input type="range" min={50} max={400} value={maxTokens}
              onChange={e => setMaxTokens(+e.target.value)} />
          </div>
          <div className="ctrl-group">
            <label>Temperature: {temperature}</label>
            <input type="range" min={0} max={1} step={0.1} value={temperature}
              onChange={e => setTemp(+e.target.value)} />
          </div>
          <div className="ctrl-group">
            <label>Compare mode</label>
            <label className="toggle-switch">
              <input type="checkbox" checked={compareMode}
                onChange={e => setCompare(e.target.checked)} />
              <span className="toggle-track" />
            </label>
          </div>
        </div>
        <button className={`run-btn ${running?"running":""}`} onClick={run} disabled={running}>
          {running ? <><span className="spinner"/>Generating...</> : "▶  Run generation"}
        </button>
      </div>

      {/* ── Live stats bar ── */}
      {(baselineTps || compTps) && (
        <div className="live-stats-row">
          <div className="stat-chip">
            <span className="sc-label">Baseline TPS</span>
            <span className="sc-val">{baselineTps?.toFixed(1) || "—"}</span>
          </div>
          <div className="stat-chip compressed">
            <span className="sc-label">Compressed TPS</span>
            <span className="sc-val">{compTps?.toFixed(1) || "—"}</span>
          </div>
          <div className="stat-chip highlight">
            <span className="sc-label">KV MSE</span>
            <span className="sc-val">{compMse?.toFixed(5) || "—"}</span>
          </div>
          <div className="stat-chip highlight">
            <span className="sc-label">MB saved</span>
            <span className="sc-val">{mbSaved}</span>
          </div>
          <div className="stat-chip">
            <span className="sc-label">Compression</span>
            <span className="sc-val">{Math.round(32/bits)}× ({bits}-bit)</span>
          </div>
        </div>
      )}

      {/* ── Side by side output ── */}
      <div className="output-grid">
        <div className="output-panel">
          <div className="output-panel-header baseline">
            <span>Uncompressed (32-bit)</span>
            <span className="token-count">{tokenCount.baseline} tokens</span>
          </div>
          <div className="output-body" ref={baseRef}>
            {baselineText || <span className="placeholder">Output appears here...</span>}
            {running && baselineText && <span className="cursor-blink">▊</span>}
          </div>
        </div>

        <div className="output-panel">
          <div className="output-panel-header compressed">
            <span>TurboQuant {bits}-bit ({mode === "ip" ? "IP+QJL" : "MSE"})</span>
            <span className="token-count">{tokenCount.compressed} tokens</span>
          </div>
          <div className="output-body" ref={compRef}>
            {compText || <span className="placeholder">Output appears here...</span>}
            {running && compText && <span className="cursor-blink">▊</span>}
          </div>
        </div>
      </div>
    </div>
  )
}
