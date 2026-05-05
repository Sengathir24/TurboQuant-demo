/* VectorSearchDemo.jsx */
import { useState } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

export default function VectorSearchDemo() {
  const { bits } = useModelStore()
  const [query, setQuery]   = useState("How does TurboQuant reduce KV cache memory?")
  const [topK, setTopK]     = useState(3)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  async function run() {
    setLoading(true)
    const r = await api.vectorSearch({ query, top_k: topK, bits })
    setResult(r)
    setLoading(false)
  }

  return (
    <div className="demo-page">
      <div className="demo-header">
        <h2>🔍 Vector Search</h2>
        <p>Embeds a corpus, compresses with TurboQuant, and searches with inner product. Compare recall vs baseline.</p>
      </div>
      <div className="controls-card">
        <div className="ctrl-group">
          <label>Query</label>
          <input className="text-input" value={query} onChange={e=>setQuery(e.target.value)} />
        </div>
        <div className="ctrl-group">
          <label>Top-K: {topK}</label>
          <input type="range" min={1} max={5} value={topK} onChange={e=>setTopK(+e.target.value)} />
        </div>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? <><span className="spinner"/>Searching...</> : "▶  Run search"}
        </button>
      </div>

      {result && (
        <div className="results-grid">
          <div className="result-col">
            <div className="result-col-header baseline">Baseline (32-bit)</div>
            {result.baseline_results?.map((r,i) => (
              <div key={i} className="result-doc">
                <span className="rank">#{i+1}</span>
                <span className="score">{r.score.toFixed(4)}</span>
                <p>{r.doc}</p>
              </div>
            ))}
          </div>
          <div className="result-col">
            <div className="result-col-header compressed">
              TurboQuant {bits}-bit — Recall@{topK}: {(result.recall_at_k*100).toFixed(0)}%
            </div>
            {result.compressed_results?.map((r,i) => (
              <div key={i} className="result-doc">
                <span className="rank">#{i+1}</span>
                <span className="score">{r.score.toFixed(4)}</span>
                <p>{r.doc}</p>
              </div>
            ))}
          </div>
          <div className="stats-sidebar">
            <Stat label="Recall@K" value={`${(result.recall_at_k*100).toFixed(0)}%`} />
            <Stat label="Compression" value={`${result.compression_ratio}×`} />
            <Stat label="Baseline lat" value={`${result.baseline_lat_ms?.toFixed(2)}ms`} />
            <Stat label="Compressed lat" value={`${result.compressed_lat_ms?.toFixed(2)}ms`} />
            <Stat label="Memory saved" value={`${result.memory_saved_mb?.toFixed(2)}MB`} />
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({label,value}) {
  return (
    <div className="stat-box">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  )
}
