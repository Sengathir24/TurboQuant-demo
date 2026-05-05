// lib/api.js — all backend calls in one place
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

export const api = {
  health:    () => fetch(`${BASE}/health`).then(r => r.json()),
  models:    () => fetch(`${BASE}/models`).then(r => r.json()),
  loadModel: (payload) =>
    fetch(`${BASE}/load-model`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json()),

  vectorSearch: (payload) =>
    fetch(`${BASE}/vector-search`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json()),

  rag: (payload) =>
    fetch(`${BASE}/rag`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json()),

  liveMetrics: () => fetch(`${BASE}/metrics/live`).then(r => r.json()),

  // SSE streaming generator
  streamGenerate: (payload, onToken, onDone) => {
    fetch(`${BASE}/generate/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(resp => {
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      const read = () => reader.read().then(({ done, value }) => {
        if (done) { onDone?.(); return }
        const text = decoder.decode(value)
        text.split("\n").forEach(line => {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim()
            if (data === "[DONE]") { onDone?.(); return }
            try { onToken(JSON.parse(data)) } catch {}
          }
        })
        read()
      })
      read()
    })
  },

  // WebSocket for live metrics
  connectMetricsWS: (onMessage) => {
    const ws = new WebSocket(`${BASE.replace("http","ws")}/ws/metrics`)
    ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)) } catch {} }
    return ws
  }
}
