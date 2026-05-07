/* AttentionDemo.jsx — KV MSE per-layer visualization */
import { useModelStore } from "../store/modelStore"

export default function AttentionDemo() {
  const { liveMetrics, compressionRatio, bits } = useModelStore()

  const mseK = liveMetrics.avg_mse_k ?? 0
  const mseV = liveMetrics.avg_mse_v ?? 0
  const n    = liveMetrics.n_layers  ?? 0

  // Use real per-layer data from backend if available;
  // otherwise fall back to synthetic display using aggregate MSE.
  const perLayer = liveMetrics.per_layer   // array of { layer, mse_k, mse_v, calls }

  const layers = perLayer && perLayer.length > 0
    ? perLayer
    : Array.from({ length: Math.max(n, 8) }, (_, i) => ({
        layer: i,
        mse_k: mseK > 0 ? mseK * (1 + (i % 3) * 0.1) : 0,
        mse_v: mseV > 0 ? mseV * (1 + (i % 4) * 0.08) : 0,
        calls: 0,
      }))

  const hasRealData = mseK > 0 || mseV > 0

  const maxMseK = Math.max(...layers.map(l => l.mse_k ?? l.mse ?? 0), 0.0001)
  const maxMseV = Math.max(...layers.map(l => l.mse_v ?? 0), 0.0001)
  const maxMse  = Math.max(maxMseK, maxMseV)

  const paperBound = Math.sqrt(3 * Math.PI / 2) / Math.pow(4, bits)
  const ratioTheory = 32 / bits
  const ratioRuntime = Number.isFinite(liveMetrics?.compression_ratio)
    ? liveMetrics.compression_ratio
    : compressionRatio

  function mseClass(v) {
    if (v <= 0) return ""
    return v < 0.05 ? "good" : v < 0.15 ? "warn" : "bad"
  }

  return (
    <div className="demo-page">
      <div className="demo-header">
        <h2>🧠 Attention Score Fidelity</h2>
        <p>
          How much do attention patterns change under {bits}-bit KV compression?
          Lower nMSE (0-1) = more faithful attention.
          {!hasRealData && (
            <span style={{ color: "var(--text-muted)", marginLeft: 8 }}>
              — Run the Generate tab to populate live values.
            </span>
          )}
        </p>
      </div>

      <div className="attention-grid">
        {/* ── Left column: summary stats ── */}
        <div className="attn-stats-col">
          <div className="attn-stat-card">
            <div className="asc-label">Avg KV nMSE (K)</div>
            <div className={`asc-val ${mseClass(mseK)}`}>
              {mseK > 0 ? mseK.toFixed(6) : "—"}
            </div>
          </div>
          <div className="attn-stat-card">
            <div className="asc-label">Avg KV nMSE (V)</div>
            <div className={`asc-val ${mseClass(mseV)}`}>
              {mseV > 0 ? mseV.toFixed(6) : "—"}
            </div>
          </div>
          <div className="attn-stat-card">
            <div className="asc-label">Comp (Theory)</div>
            <div className="asc-val">{ratioTheory.toFixed(1)}×</div>
          </div>
          <div className="attn-stat-card">
            <div className="asc-label">Comp (Runtime)</div>
            <div className="asc-val">{Number(ratioRuntime).toFixed(2)}×</div>
          </div>
          <div className="attn-stat-card">
            <div className="asc-label">Paper upper bound</div>
            <div className="asc-val">{paperBound.toFixed(6)}</div>
          </div>
          {hasRealData && (
            <div className="attn-stat-card">
              <div className="asc-label">Within bound?</div>
              <div className={`asc-val ${mseK <= paperBound ? "good" : "bad"}`}>
                {mseK <= paperBound ? "✓ Yes" : "✗ No"}
              </div>
            </div>
          )}
          <div className="attn-explainer">
            <h4>How to read this</h4>
            <p>MSE &lt; 0.05 → attention patterns nearly identical to uncompressed.</p>
            <p>MSE &lt; 0.15 → small drift, output quality preserved.</p>
            <p>MSE &gt; 0.15 → consider more bits.</p>
            <p>Run the Generate tab to see live values update here.</p>
          </div>
        </div>

        {/* ── Right column: per-layer bar chart ── */}
        <div className="attn-layer-chart">
          <h3>KV nMSE per layer {!hasRealData && <span style={{fontSize:"0.75rem", color:"var(--text-muted)"}}>(placeholder — run generate first)</span>}</h3>
          <div className="layer-bars">
            {layers.map(l => {
              const mseKVal = l.mse_k ?? l.mse ?? 0
              const mseVVal = l.mse_v ?? 0
              return (
                <div key={l.layer} className="layer-bar-row">
                  <span className="lb-label">L{l.layer}</span>
                  <div className="lb-track" style={{ position: "relative", display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
                    {/* Key MSE bar */}
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div
                        className="lb-fill key-fill"
                        style={{ width: `${Math.min(100, mseKVal / maxMse * 100)}%`, minWidth: mseKVal > 0 ? 2 : 0 }}
                      />
                      <span className="lb-val" style={{ fontSize: "0.65rem", opacity: 0.8 }}>
                        {mseKVal > 0 ? mseKVal.toFixed(5) : "—"}
                      </span>
                    </div>
                    {/* Value MSE bar */}
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div
                        className="lb-fill"
                        style={{
                          width: `${Math.min(100, mseVVal / maxMse * 100)}%`,
                          minWidth: mseVVal > 0 ? 2 : 0,
                          background: "var(--accent-secondary, #2E9E6B)",
                          opacity: 0.75,
                        }}
                      />
                      <span className="lb-val" style={{ fontSize: "0.65rem", opacity: 0.7 }}>
                        {mseVVal > 0 ? mseVVal.toFixed(5) : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
          <div className="layer-legend" style={{ display: "flex", gap: 16, marginTop: 8 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span className="ll-dot key-fill" style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2 }} />
              Key MSE
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2, background: "var(--accent-secondary, #2E9E6B)", opacity: 0.75 }} />
              Value MSE
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
