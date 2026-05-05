/* LiveMetricsBar.jsx — WebSocket real-time metrics strip */
import { useEffect, useState } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

function Metric({ label, value, unit, good, warn }) {
  const cls = value !== null && good !== undefined
    ? (value <= good ? "good" : value <= warn ? "warn" : "bad")
    : ""
  return (
    <div className={`live-metric ${cls}`}>
      <span className="lm-label">{label}</span>
      <span className="lm-value">
        {value !== null && value !== undefined ? value : "—"}
        {unit && <span className="lm-unit"> {unit}</span>}
      </span>
    </div>
  )
}

export default function LiveMetricsBar() {
  const [m, setM] = useState({})
  const { compressionRatio } = useModelStore()

  useEffect(() => {
    const ws = api.connectMetricsWS(setM)
    return () => ws.close()
  }, [])

  const mbSaved = m.mb_saved || 0
  const ratio   = m.compression_ratio || compressionRatio || 1

  return (
    <div className="metrics-strip">
      <Metric label="Compression" value={`${ratio}×`} />
      <Metric label="KV MSE (K)" value={m.avg_mse_k?.toFixed(5)} good={0.05} warn={0.15} />
      <Metric label="KV MSE (V)" value={m.avg_mse_v?.toFixed(5)} good={0.05} warn={0.15} />
      <Metric label="Compress" value={m.avg_compress_ms?.toFixed(2)} unit="ms" good={1} warn={5} />
      <Metric label="MB Saved" value={mbSaved.toFixed(1)} unit="MB" />
      <Metric label="VRAM" value={m.vram_gb?.toFixed(2)} unit="GB" />
      <Metric label="Layers" value={m.n_layers} />
      <div className="live-dot-wrap">
        <span className="live-dot" />
        <span className="live-label">live</span>
      </div>
    </div>
  )
}
