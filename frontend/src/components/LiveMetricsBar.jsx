/* LiveMetricsBar.jsx — WebSocket real-time metrics strip */
import { useEffect, useState } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

/** Avoid tiny MSE rounding to 0.00000 at fixed decimals */
function fmtMse(x) {
  if (!Number.isFinite(x)) return null
  if (x === 0) return "0"
  if (Math.abs(x) < 1e-4) return x.toExponential(3)
  return x.toFixed(5)
}

function Metric({ label, value, unit, good, warn }) {
  const numVal = typeof value === "number" ? value : parseFloat(value)
  const cls =
    !isNaN(numVal) && numVal > 0 && good !== undefined
      ? numVal <= good ? "good" : numVal <= warn ? "warn" : "bad"
      : ""
  const display = (value !== null && value !== undefined && value !== "")
    ? value
    : "—"
  return (
    <div className={`live-metric ${cls}`}>
      <span className="lm-label">{label}</span>
      <span className="lm-value">
        {display}
        {display !== "—" && unit && <span className="lm-unit"> {unit}</span>}
      </span>
    </div>
  )
}

export default function LiveMetricsBar() {
  const [m, setM] = useState({})
  const { bits, compressionRatio, setLiveMetrics } = useModelStore()

  useEffect(() => {
    const ws = api.connectMetricsWS((data) => {
      setM(data)
      setLiveMetrics(data)   // keep store in sync so AttentionDemo gets per_layer
    })
    return () => ws.close()
  }, [setLiveMetrics])

  const mbSaved  = m.mb_saved ?? 0
  const ratioEff = m.compression_ratio ?? compressionRatio ?? 1
  const ratioTheory = Number.isFinite(bits) && bits > 0 ? (32 / bits) : null
  const mseK     = Number.isFinite(m.avg_mse_k) ? fmtMse(m.avg_mse_k) : null
  const mseV     = Number.isFinite(m.avg_mse_v) ? fmtMse(m.avg_mse_v) : null
  const compMs   = Number.isFinite(m.avg_compress_ms) ? m.avg_compress_ms.toFixed(2) : null
  const baseTps  = Number.isFinite(m.baseline_tps) ? m.baseline_tps.toFixed(1) : null
  const compTps  = Number.isFinite(m.compressed_tps) ? m.compressed_tps.toFixed(1) : null

  return (
    <div className="metrics-strip">
      <Metric label="Baseline TPS" value={baseTps} unit="tok/s" />
      <Metric label="Compressed TPS" value={compTps} unit="tok/s" />
      <Metric label="Comp (Theory)" value={ratioTheory != null ? `${ratioTheory.toFixed(1)}×` : null} />
      <Metric label="Comp (Runtime)" value={`${Number(ratioEff).toFixed(2)}×`} />
      <Metric label="KV nMSE (K)"  value={mseK}   good={0.10} warn={0.25} />
      <Metric label="KV nMSE (V)"  value={mseV}   good={0.10} warn={0.25} />
      <Metric label="Compress"    value={compMs} unit="ms" good={15} warn={35} />
      <Metric
        label="MB Saved"
        value={Number.isFinite(mbSaved) ? mbSaved.toFixed(1) : null}
        unit="MB"
      />
      <Metric
        label="VRAM"
        value={Number.isFinite(m.vram_gb) ? m.vram_gb.toFixed(2) : null}
        unit="GB"
      />
      <Metric label="Layers"      value={m.n_layers ?? null} />
      <div className="live-dot-wrap">
        <span className="live-dot" />
        <span className="live-label">live</span>
      </div>
    </div>
  )
}
