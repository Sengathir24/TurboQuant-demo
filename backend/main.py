"""
TurboQuant Demo — FastAPI Backend
Real-time KV cache compression for HuggingFace LLMs.
Cursor: open this file first and run `uvicorn main:app --reload`
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio, json, time, gc, psutil, torch, numpy as np
from contextlib import asynccontextmanager

from turbo_engine import TurboEngine
from model_registry import MODEL_REGISTRY, ModelConfig
from metrics import MetricsCollector

# ── App lifecycle ──────────────────────────────────────────
engine: Optional[TurboEngine] = None
metrics = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = TurboEngine()
    yield
    if engine:
        engine.cleanup()

app = FastAPI(title="TurboQuant Demo API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request / Response Models ──────────────────────────────
class LoadModelRequest(BaseModel):
    model_id: str          # e.g. "gpt2-medium"
    bits: int = 3          # 1-4
    mode: str = "ip"       # "mse" | "ip"
    outlier_ratio: float = 0.25

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    compare_mode: bool = True   # run baseline + compressed side-by-side

class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    bits: int = 3

class RAGRequest(BaseModel):
    question: str
    bits: int = 3


# ── Routes ────────────────────────────────────────────────
@app.get("/models")
async def list_models():
    """Return all available HuggingFace models with metadata."""
    return {"models": [m.dict() for m in MODEL_REGISTRY.values()]}

@app.get("/health")
async def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "device": device,
        "vram_used_gb": round(vram, 3),
        "ram_used_gb": round(psutil.virtual_memory().used/1e9, 2),
        "model_loaded": engine.model_id if engine else None,
    }

@app.post("/load-model")
async def load_model(req: LoadModelRequest):
    """Load a HuggingFace model with TurboQuant compression configured."""
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model {req.model_id} not in registry")
    result = await asyncio.to_thread(engine.load_model, req.model_id, req.bits, req.mode)
    return result

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Stream token generation with live compression metrics."""
    if not engine.is_loaded():
        raise HTTPException(400, "No model loaded. Call /load-model first.")

    async def token_stream() -> AsyncGenerator[str, None]:
        async for chunk in engine.stream_generate(
            req.prompt, req.max_new_tokens,
            req.temperature, req.compare_mode
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.post("/vector-search")
async def vector_search(req: VectorSearchRequest):
    """Demo: compress embeddings and search with FAISS."""
    if not engine.is_loaded():
        raise HTTPException(400, "Load a model first")
    return await asyncio.to_thread(engine.vector_search, req.query, req.top_k, req.bits)

@app.post("/rag")
async def rag_query(req: RAGRequest):
    """Demo: retrieve + generate with compressed KV cache."""
    if not engine.is_loaded():
        raise HTTPException(400, "Load a model first")
    return await asyncio.to_thread(engine.rag_pipeline, req.question, req.bits)

@app.get("/metrics/live")
async def live_metrics():
    """Real-time compression stats for the dashboard."""
    return engine.get_live_metrics() if engine.is_loaded() else {}

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """WebSocket: push live metrics every 500ms."""
    await websocket.accept()
    try:
        while True:
            data = engine.get_live_metrics() if engine and engine.is_loaded() else {}
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
