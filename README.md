# TurboQuant Demo
**Real-time LLM memory compression — full-stack demo with 12 HuggingFace models**

> Implementation of [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
> FastAPI backend · React frontend · Live KV cache compression · 5 benchmarks

---

## What this does

| Demo tab | What runs |
|----------|-----------|
| ⚡ Generate | Side-by-side streaming — baseline vs TurboQuant KV cache |
| 🔍 Vector Search | FAISS recall@k with compressed embeddings |
| 📚 RAG | Retrieve compressed docs → generate answer |
| 🧠 Attention | KV MSE per layer, live from WebSocket |
| 💰 Cloud Cost | USD/1M tokens calculator across GPU × model |

---

## Cursor: open this in 3 steps

### Step 1 — open the repo
```
File → Open Folder → select turboquant-demo/
```
Cursor will show you `backend/` and `frontend/` side by side.

### Step 2 — start the backend
Open a Cursor terminal and run:
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit http://localhost:8000/docs to see the auto-generated API docs.

### Step 3 — start the frontend
Open a second Cursor terminal:
```bash
cd frontend
npm install
npm run dev
```
Visit http://localhost:5173

---

## Architecture

```
turboquant-demo/
├── backend/
│   ├── main.py              ← FastAPI app, all routes
│   ├── turbo_engine.py      ← TurboQuant math + KV hooks (READ THIS FIRST)
│   ├── model_registry.py    ← 12 HuggingFace models, add more here
│   ├── metrics.py           ← Prometheus-ready metrics collector
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── App.jsx           ← Root, tab routing
    │   ├── store/
    │   │   └── modelStore.js ← Zustand global state
    │   ├── lib/
    │   │   └── api.js        ← All API calls + SSE + WebSocket
    │   ├── components/
    │   │   ├── ModelSelector.jsx   ← HF model picker + bit config
    │   │   └── LiveMetricsBar.jsx  ← WebSocket metrics strip
    │   ├── pages/
    │   │   ├── GenerateDemo.jsx    ← SSE streaming comparison
    │   │   ├── VectorSearchDemo.jsx
    │   │   ├── RAGDemo.jsx
    │   │   ├── AttentionDemo.jsx
    │   │   └── CostDemo.jsx
    │   └── index.css         ← All styles (dark, IBM Plex Mono)
    ├── package.json
    └── vite.config.js
```

---

## How TurboQuant works (3 lines)

```
1. Rotate vector with random orthogonal matrix Π   → values spread evenly
2. Round each coordinate to nearest codebook entry  → MSE compression
3. Store sign(S·residual) — 1 bit per dim           → fixes inner product bias
```

The math is in `backend/turbo_engine.py` → `TurboQuantGPU`.

---

## Adding a new HuggingFace model

Edit `backend/model_registry.py`:
```python
"my-model": ModelConfig(
    id="my-model",
    name="My Model Name",
    hf_id="org/model-name",       # exact HuggingFace ID
    params="1.5B",
    context_len=4096,
    vram_fp16_gb=6.0,
    task="causal-lm",             # or "seq2seq"
    family="MyFamily",
    description="What this model is good for.",
    recommended_bits=3,
    tags=["gpu", "long-context"]
)
```
It auto-appears in the frontend dropdown. No other changes needed.

---

## Environment variables

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000   # backend URL

# backend — no env vars needed for local dev
# For production, set HF_HOME to a fast disk for model cache
```

---

## GPU vs CPU

| Hardware | Works? | Recommended models |
|----------|--------|--------------------|
| CPU only | ✅ | distilgpt2, gpt2, flan-t5-small |
| T4 GPU (16GB) | ✅ | gpt2-medium, flan-t5-base, opt-350m |
| A100 (40GB) | ✅ | opt-1.3b, bloom-560m, pythia-410m |

The backend auto-detects CUDA and uses fp16 on GPU, fp32 on CPU.

---

## Paper claims validated in this demo

| Theorem | Claim | Where to see it |
|---------|-------|-----------------|
| Lemma 1 | Coordinates follow N(0,1/d) after rotation | turbo_engine.py → TurboQuantGPU |
| Theorem 1 | MSE ≤ √(3π/2) / 4^b | Attention tab → avg KV MSE |
| Theorem 2 | QJL correction is unbiased | Generate tab → KV MSE stays low |
| Theorem 3 | Within 2.7× of theoretical best | Cloud Cost tab → compression ratio |

---

## License
MIT — use freely for research and demos.

Based on: *TurboQuant: Near-Optimal Vector Quantization* (arXiv:2504.19874)
