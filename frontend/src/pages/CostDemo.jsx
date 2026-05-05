/* CostDemo.jsx — interactive cloud cost calculator */
import { useState } from "react"
import { useModelStore } from "../store/modelStore"

const GPU_SPECS = {
  "T4 (16GB)":   { vram: 16,  price_hr: 0.53  },
  "A10G (24GB)": { vram: 24,  price_hr: 1.10  },
  "A100 (40GB)": { vram: 40,  price_hr: 3.20  },
  "H100 (80GB)": { vram: 80,  price_hr: 4.50  },
}

const MODEL_TPS = {
  "GPT-2 Small (124M)":    { base_tps: 120, kv_bytes: 2*12*12*64*2 },
  "GPT-2 Medium (345M)":   { base_tps: 80,  kv_bytes: 2*24*16*64*2 },
  "OPT 1.3B":              { base_tps: 35,  kv_bytes: 2*24*32*64*2 },
  "LLaMA-3.1 8B":          { base_tps: 18,  kv_bytes: 2*32*32*128*2 },
  "LLaMA-3.1 70B":         { base_tps: 4,   kv_bytes: 2*80*64*128*2 },
}

export default function CostDemo() {
  const { bits: globalBits } = useModelStore()
  const [gpu, setGpu]         = useState("T4 (16GB)")
  const [llm, setLlm]         = useState("LLaMA-3.1 8B")
  const [bits, setBits]       = useState(globalBits || 3)
  const [monthly_tokens, setMonthly] = useState(100)  // millions

  const gspec = GPU_SPECS[gpu]
  const mspec = MODEL_TPS[llm]
  const ratio = 32 / bits

  const tps_comp = mspec.base_tps * Math.pow(ratio, 0.4)
  const cost_base = (1_000_000 / mspec.base_tps / 3600) * gspec.price_hr
  const cost_comp = (1_000_000 / tps_comp   / 3600) * gspec.price_hr

  const monthly_cost_base = cost_base * monthly_tokens
  const monthly_cost_comp = cost_comp * monthly_tokens
  const monthly_savings   = monthly_cost_base - monthly_cost_comp

  const model_vram_gb = 2.0
  const max_ctx_base = Math.floor((gspec.vram - model_vram_gb) * 1e9 / mspec.kv_bytes)
  const max_ctx_comp = Math.floor(max_ctx_base * ratio)

  return (
    <div className="demo-page">
      <div className="demo-header">
        <h2>💰 Cloud Cost Calculator</h2>
        <p>See real USD savings from TurboQuant compression on any GPU × model combination.</p>
      </div>

      <div className="cost-layout">
        {/* ── Config panel ── */}
        <div className="cost-config">
          <div className="ctrl-group">
            <label>GPU instance</label>
            <div className="pill-group">
              {Object.keys(GPU_SPECS).map(g => (
                <button key={g} className={`pill ${gpu===g?"on":""}`} onClick={()=>setGpu(g)}>
                  {g}
                  <span className="pill-sub">${GPU_SPECS[g].price_hr}/hr</span>
                </button>
              ))}
            </div>
          </div>

          <div className="ctrl-group">
            <label>Model</label>
            <div className="pill-group vertical">
              {Object.keys(MODEL_TPS).map(m => (
                <button key={m} className={`pill ${llm===m?"on":""}`} onClick={()=>setLlm(m)}>
                  {m}
                  <span className="pill-sub">{MODEL_TPS[m].base_tps} tok/s base</span>
                </button>
              ))}
            </div>
          </div>

          <div className="ctrl-group">
            <label>TurboQuant bits: {bits}-bit ({ratio.toFixed(0)}× compression)</label>
            <input type="range" min={1} max={4} value={bits} onChange={e=>setBits(+e.target.value)} />
          </div>

          <div className="ctrl-group">
            <label>Monthly traffic: {monthly_tokens}M tokens</label>
            <input type="range" min={1} max={1000} value={monthly_tokens}
              onChange={e=>setMonthly(+e.target.value)} />
          </div>
        </div>

        {/* ── Results panel ── */}
        <div className="cost-results">
          <div className="cost-compare-row">
            <div className="cost-box baseline">
              <div className="cost-box-label">Baseline (32-bit)</div>
              <div className="cost-big">${cost_base.toFixed(4)}</div>
              <div className="cost-sub">per 1M tokens</div>
              <div className="cost-row"><span>Throughput</span><strong>{mspec.base_tps} tok/s</strong></div>
              <div className="cost-row"><span>Max context</span><strong>{(max_ctx_base/1000).toFixed(0)}K tokens</strong></div>
              <div className="cost-row"><span>Monthly cost</span><strong>${monthly_cost_base.toFixed(0)}</strong></div>
            </div>

            <div className="cost-arrow">→</div>

            <div className="cost-box compressed">
              <div className="cost-box-label">TurboQuant {bits}-bit</div>
              <div className="cost-big green">${cost_comp.toFixed(4)}</div>
              <div className="cost-sub">per 1M tokens</div>
              <div className="cost-row"><span>Throughput</span><strong>{tps_comp.toFixed(0)} tok/s</strong></div>
              <div className="cost-row"><span>Max context</span><strong>{(max_ctx_comp/1000).toFixed(0)}K tokens</strong></div>
              <div className="cost-row"><span>Monthly cost</span><strong>${monthly_cost_comp.toFixed(0)}</strong></div>
            </div>
          </div>

          <div className="savings-banner">
            <div className="sb-item">
              <span className="sb-label">Monthly savings</span>
              <span className="sb-val green">${monthly_savings.toFixed(0)}</span>
            </div>
            <div className="sb-item">
              <span className="sb-label">Cost reduction</span>
              <span className="sb-val green">{((1-cost_comp/cost_base)*100).toFixed(0)}%</span>
            </div>
            <div className="sb-item">
              <span className="sb-label">Context gain</span>
              <span className="sb-val green">{ratio.toFixed(0)}×</span>
            </div>
            <div className="sb-item">
              <span className="sb-label">Annual savings</span>
              <span className="sb-val green">${(monthly_savings*12).toFixed(0)}</span>
            </div>
          </div>

          <div className="cost-note">
            * Throughput estimate uses TPS × compression_ratio^0.4 scaling (empirical).
              Actual gains depend on memory bandwidth and batch size.
          </div>
        </div>
      </div>
    </div>
  )
}
