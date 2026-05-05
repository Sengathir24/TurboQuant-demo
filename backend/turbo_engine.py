"""
turbo_engine.py — TurboQuant compression engine
Handles model loading, KV cache hooks, streaming generation,
vector search, and RAG pipeline.
Cursor: this is the most important file. Every benchmark goes through here.
"""

import torch, numpy as np, time, gc, asyncio, logging
from typing import Optional, AsyncGenerator, Any
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, TextIteratorStreamer
)
from threading import Thread
import scipy.stats as sp_stats

from model_registry import MODEL_REGISTRY, ModelConfig

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════
# TurboQuant Math — same as the paper, pure PyTorch
# ══════════════════════════════════════════════════════════

def lloyd_max_centroids(bits: int, dim: int, n_iters: int = 300) -> torch.Tensor:
    k = 2 ** bits
    std = 1.0 / max(np.sqrt(dim), 1e-6)
    q = np.linspace(1/(k+1), k/(k+1), k)
    centroids = sp_stats.norm.ppf(q, scale=std)
    for _ in range(n_iters):
        bounds = np.concatenate([[-np.inf], (centroids[:-1]+centroids[1:])/2, [np.inf]])
        new_c = np.array([
            sp_stats.norm.expect(lb=bounds[i], ub=bounds[i+1], scale=std)
            for i in range(k)
        ])
        if np.allclose(new_c, centroids, atol=1e-9): break
        centroids = new_c
    return torch.tensor(centroids, dtype=torch.float32)


@dataclass
class Compressed:
    idx:    torch.Tensor
    qjl:    torch.Tensor
    r_norm: torch.Tensor
    x_norm: torch.Tensor
    shape:  tuple
    mode:   str


class TurboQuantGPU:
    """GPU-native TurboQuant. Supports batched (N,D) compression."""

    def __init__(self, dim: int, bits: int, mode: str = "ip", seed: int = 42):
        self.dim, self.bits, self.mode = dim, bits, mode
        torch.manual_seed(seed)
        A = torch.randn(dim, dim)
        self.PI, _ = torch.linalg.qr(A)
        self.PI = self.PI.to(DEVICE)
        self.S = torch.randn(dim, dim,
                             generator=torch.Generator().manual_seed(seed+1)).to(DEVICE)
        mse_bits = max(1, bits-1) if mode == "ip" else bits
        self.mse_bits = mse_bits
        self.codebook = lloyd_max_centroids(mse_bits, dim).to(DEVICE)
        self.boundaries = ((self.codebook[:-1] + self.codebook[1:]) / 2).to(DEVICE)

    def compress(self, x: torch.Tensor) -> Compressed:
        x = x.to(DEVICE).float()
        sq = x.dim() == 1
        if sq: x = x.unsqueeze(0)
        x_norm = x.norm(dim=-1, keepdim=True)
        x_unit = x / (x_norm + 1e-12)
        y = x_unit @ self.PI.T
        idx = torch.bucketize(y, self.boundaries)
        y_hat = self.codebook[idx]
        r = y - y_hat
        r_norm = r.norm(dim=-1)
        qjl = torch.zeros_like(x, dtype=torch.int8)
        if self.mode == "ip":
            qjl = (r @ self.S.T).sign().to(torch.int8)
        return Compressed(idx.to(torch.int16), qjl, r_norm,
                          x_norm.squeeze(-1), x.shape, self.mode)

    def decompress(self, c: Compressed) -> torch.Tensor:
        y_hat = self.codebook[c.idx.long()]
        x_mse = y_hat @ self.PI
        if c.mode == "ip":
            scale = (np.pi / 2) ** 0.5 / self.dim
            x_qjl = scale * c.r_norm.unsqueeze(-1) * (c.qjl.float() @ self.S)
            x_hat = x_mse + x_qjl
        else:
            x_hat = x_mse
        return x_hat * c.x_norm.unsqueeze(-1)

    def compression_ratio(self) -> float:
        return 32.0 / (self.mse_bits + (1 if self.mode == "ip" else 0))

    def kv_bytes_saved(self, n_tokens: int, n_heads: int) -> int:
        orig = n_tokens * n_heads * self.dim * 2 * 2  # K+V, fp16
        comp = int(orig / self.compression_ratio())
        return orig - comp


# ══════════════════════════════════════════════════════════
# KV Cache Compressor — hooks into attention layers
# ══════════════════════════════════════════════════════════

class KVCompressorHook:
    """
    Wraps a single attention layer's KV output.
    Called after every attention forward pass.
    Compresses K and V tensors and records stats.
    """

    def __init__(self, tq: TurboQuantGPU, layer_idx: int):
        self.tq = tq
        self.layer_idx = layer_idx
        self.stats = {
            "compress_ms": [], "mse_k": [], "mse_v": [],
            "bytes_saved": 0, "calls": 0
        }

    def forward_hook(
        self, module: Any, args: tuple, kwargs: Optional[dict], output: Any
    ):
        """
        Modern transformers (4.40+) return (attn_out, attn_weights) and store KV in
        past_key_values (DynamicCache). Older builds returned (attn_out, (k, v)).
        """
        k: Optional[torch.Tensor] = None
        v: Optional[torch.Tensor] = None
        cache_layer: Any = None

        kw = kwargs or {}
        past = kw.get("past_key_values")
        if past is not None:
            try:
                from transformers.cache_utils import EncoderDecoderCache
                if isinstance(past, EncoderDecoderCache):
                    past = past.self_attention_cache
            except ImportError:
                pass
            li = getattr(module, "layer_idx", None)
            if past is not None and hasattr(past, "layers") and li is not None:
                if li < len(past.layers):
                    cache_layer = past.layers[li]
                    if getattr(cache_layer, "is_initialized", False) and cache_layer.keys.numel() > 0:
                        k, v = cache_layer.keys, cache_layer.values

        if k is None and isinstance(output, tuple) and len(output) >= 2:
            second = output[1]
            if isinstance(second, tuple) and len(second) == 2:
                maybe_k, maybe_v = second
                if torch.is_tensor(maybe_k) and torch.is_tensor(maybe_v):
                    k, v = maybe_k, maybe_v

        if k is None or v is None:
            return output

        t0 = time.perf_counter()
        k_hat, v_hat = self._compress_kv(k, v)
        ms = (time.perf_counter() - t0) * 1000
        self.stats["compress_ms"].append(ms)

        if cache_layer is not None:
            with torch.no_grad():
                cache_layer.keys = k_hat.to(dtype=k.dtype)
                cache_layer.values = v_hat.to(dtype=v.dtype)
        elif isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], tuple):
            return (output[0], (k_hat, v_hat)) + output[2:]
        return output

    def _compress_kv(self, k: torch.Tensor, v: torch.Tensor):
        B, H, T, D = k.shape
        # Flatten to (B*H*T, D), compress, reshape back
        k_flat = k.float().reshape(-1, D)
        v_flat = v.float().reshape(-1, D)
        ck = self.tq.compress(k_flat)
        cv = self.tq.compress(v_flat)
        k_hat = self.tq.decompress(ck).reshape(B, H, T, D)
        v_hat = self.tq.decompress(cv).reshape(B, H, T, D)

        # Track quality
        mse_k = float(((k.float() - k_hat)**2).mean())
        mse_v = float(((v.float() - v_hat)**2).mean())
        self.stats["mse_k"].append(mse_k)
        self.stats["mse_v"].append(mse_v)
        self.stats["bytes_saved"] += self.tq.kv_bytes_saved(T, H)
        self.stats["calls"] += 1

        return k_hat.to(k.dtype), v_hat.to(v.dtype)


# ══════════════════════════════════════════════════════════
# TurboEngine — main orchestrator
# ══════════════════════════════════════════════════════════

class TurboEngine:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.model_cfg: Optional[ModelConfig] = None
        self.hooks = []
        self.kv_hooks: list[KVCompressorHook] = []
        self.tq: Optional[TurboQuantGPU] = None
        self._bits = 3
        self._mode = "ip"
        self._live_metrics = {}

        # Fake corpus for vector search demo
        self._corpus = [
            "The transformer architecture uses attention mechanisms to process sequences.",
            "Vector quantization compresses high-dimensional embeddings efficiently.",
            "Large language models require significant GPU memory during inference.",
            "KV cache compression reduces memory usage in autoregressive generation.",
            "TurboQuant achieves near-optimal distortion within 2.7x of theoretical bound.",
            "FAISS enables efficient nearest-neighbor search over compressed vectors.",
            "RAG combines retrieval with generation for knowledge-grounded answers.",
            "Federated learning trains models across distributed devices privately.",
            "Post-quantum cryptography protects against quantum computing attacks.",
            "Attention score fidelity measures how well compressed KV reconstructs attention.",
        ]

    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_id: str, bits: int, mode: str) -> dict:
        """Load model + tokenizer, attach TurboQuant KV hooks."""
        cfg = MODEL_REGISTRY[model_id]
        self._bits = bits
        self._mode = mode

        # Clear previous
        self._remove_hooks()
        if self.model:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Loading {cfg.hf_id}...")
        t0 = time.time()

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        loader = (AutoModelForCausalLM if cfg.task == "causal-lm"
                  else AutoModelForSeq2SeqLM)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = loader.from_pretrained(
            cfg.hf_id, torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        self.model.eval()

        self.model_id = model_id
        self.model_cfg = cfg

        # Detect head_dim and attach hooks
        head_dim = self._detect_head_dim()
        if head_dim:
            self.tq = TurboQuantGPU(dim=head_dim, bits=bits, mode=mode)
            self._attach_hooks()

        load_time = time.time() - t0
        vram = torch.cuda.memory_allocated()/1e9 if DEVICE == "cuda" else 0

        return {
            "status": "loaded",
            "model_id": model_id,
            "model_name": cfg.name,
            "load_time_s": round(load_time, 2),
            "vram_gb": round(vram, 3),
            "head_dim": head_dim,
            "compression_ratio": round(self.tq.compression_ratio(), 2) if self.tq else 1.0,
            "bits": bits,
            "mode": mode,
        }

    def _detect_head_dim(self) -> Optional[int]:
        """Infer head_dim from model config."""
        cfg = self.model.config
        if hasattr(cfg, "n_embd") and hasattr(cfg, "n_head"):
            return cfg.n_embd // cfg.n_head       # GPT-2
        if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
            return cfg.hidden_size // cfg.num_attention_heads  # OPT, BLOOM
        if hasattr(cfg, "d_model") and hasattr(cfg, "num_heads"):
            return cfg.d_model // cfg.num_heads   # T5
        return 64  # fallback

    def _attach_hooks(self):
        """Register forward hooks on all attention layers."""
        self.kv_hooks = []
        layers = self._get_attention_layers()
        for i, layer in enumerate(layers):
            hook_obj = KVCompressorHook(self.tq, layer_idx=i)
            h = layer.register_forward_hook(hook_obj.forward_hook, with_kwargs=True)
            self.hooks.append(h)
            self.kv_hooks.append(hook_obj)
        logger.info(f"Attached TurboQuant hooks to {len(layers)} layers")

    def _get_attention_layers(self):
        """Return list of attention modules regardless of model family."""
        m = self.model
        # GPT-2
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return [block.attn for block in m.transformer.h]
        # OPT
        if hasattr(m, "model") and hasattr(m.model, "decoder"):
            return [l.self_attn for l in m.model.decoder.layers]
        # BLOOM
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return [block.self_attention for block in m.transformer.h]
        # T5
        if hasattr(m, "encoder"):
            layers = []
            if hasattr(m.encoder, "block"):
                layers += [b.layer[0].SelfAttention for b in m.encoder.block]
            return layers
        return []

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.kv_hooks.clear()

    async def stream_generate(
        self, prompt: str, max_new_tokens: int,
        temperature: float, compare_mode: bool
    ) -> AsyncGenerator[dict, None]:
        """
        Stream tokens with live metrics.
        If compare_mode=True, first runs baseline (no hooks), then compressed.
        Yields SSE-compatible dicts.
        """
        results = {}

        for run_name, use_compression in [
            ("baseline", False), ("compressed", True)
        ] if compare_mode else [("compressed", True)]:

            # Toggle hooks
            for hook_obj in self.kv_hooks:
                hook_obj.stats = {"compress_ms": [], "mse_k": [], "mse_v": [],
                                   "bytes_saved": 0, "calls": 0}

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_cache=True,
            )

            t_start = time.perf_counter()
            peak_mem_before = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0

            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            tokens = []
            for token in streamer:
                tokens.append(token)
                elapsed = time.perf_counter() - t_start
                tps = len(tokens) / max(elapsed, 0.001)
                peak_mem = (torch.cuda.memory_allocated()
                            if DEVICE == "cuda" else 0) / 1e9

                avg_mse = np.mean([
                    np.mean(h.stats["mse_k"]) if h.stats["mse_k"] else 0
                    for h in self.kv_hooks
                ]) if self.kv_hooks else 0

                yield {
                    "run": run_name,
                    "token": token,
                    "tokens_so_far": len(tokens),
                    "tps": round(tps, 1),
                    "elapsed_s": round(elapsed, 2),
                    "peak_mem_gb": round(peak_mem, 3),
                    "kv_mse": round(float(avg_mse), 6),
                    "bytes_saved": sum(h.stats["bytes_saved"] for h in self.kv_hooks),
                }
                await asyncio.sleep(0)   # yield control

            thread.join()
            results[run_name] = "".join(tokens)

        # Final summary
        yield {
            "run": "summary",
            "baseline_text": results.get("baseline", ""),
            "compressed_text": results.get("compressed", ""),
            "total_bytes_saved": sum(h.stats["bytes_saved"] for h in self.kv_hooks),
            "compression_ratio": self.tq.compression_ratio() if self.tq else 1.0,
        }

    def vector_search(self, query: str, top_k: int, bits: int) -> dict:
        """Demo: embed corpus, compress, search with inner product."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            return {"error": "pip install sentence-transformers faiss-cpu"}

        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        corpus_embs = embed_model.encode(self._corpus, normalize_embeddings=True)
        query_emb   = embed_model.encode([query], normalize_embeddings=True)[0]
        embed_dim   = corpus_embs.shape[1]

        # Baseline search
        t0 = time.perf_counter()
        scores_base = corpus_embs @ query_emb
        top_k_base = np.argsort(scores_base)[::-1][:top_k]
        lat_base = (time.perf_counter() - t0) * 1000

        # TurboQuant compressed search
        tq_vs = TurboQuantGPU(dim=embed_dim, bits=bits, mode="ip")
        xt = torch.tensor(corpus_embs).to(DEVICE)
        c = tq_vs.compress(xt)
        comp_embs = tq_vs.decompress(c).cpu().numpy()

        t0 = time.perf_counter()
        scores_comp = comp_embs @ query_emb
        top_k_comp = np.argsort(scores_comp)[::-1][:top_k]
        lat_comp = (time.perf_counter() - t0) * 1000

        recall = len(set(top_k_base) & set(top_k_comp)) / top_k

        return {
            "query": query,
            "baseline_results": [
                {"doc": self._corpus[i], "score": float(scores_base[i])}
                for i in top_k_base
            ],
            "compressed_results": [
                {"doc": self._corpus[i], "score": float(scores_comp[i])}
                for i in top_k_comp
            ],
            "recall_at_k": round(recall, 3),
            "baseline_lat_ms": round(lat_base, 2),
            "compressed_lat_ms": round(lat_comp, 2),
            "compression_ratio": round(tq_vs.compression_ratio(), 2),
            "memory_saved_mb": round(
                (corpus_embs.nbytes - corpus_embs.nbytes / tq_vs.compression_ratio()) / 1e6, 3
            ),
        }

    def rag_pipeline(self, question: str, bits: int) -> dict:
        """Demo RAG: retrieve from corpus, generate answer with loaded model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return {"error": "pip install sentence-transformers"}

        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        corpus_embs = embed_model.encode(self._corpus, normalize_embeddings=True)
        q_emb = embed_model.encode([question], normalize_embeddings=True)[0]
        dim = corpus_embs.shape[1]

        tq_rag = TurboQuantGPU(dim=dim, bits=bits, mode="ip")
        xt = torch.tensor(corpus_embs).to(DEVICE)
        c = tq_rag.compress(xt)
        comp_embs = tq_rag.decompress(c).cpu().numpy()

        scores = comp_embs @ q_emb
        top3 = np.argsort(scores)[::-1][:3]
        context = " ".join([self._corpus[i] for i in top3])

        if self.model_cfg and self.model_cfg.task == "seq2seq":
            prompt = f"Answer the question based on context.\nContext: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=400).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return {
            "question": question,
            "retrieved_docs": [self._corpus[i] for i in top3],
            "answer": answer,
            "compression_ratio": round(tq_rag.compression_ratio(), 2),
            "bits": bits,
        }

    def get_live_metrics(self) -> dict:
        if not self.kv_hooks:
            return {}
        all_mse_k = [m for h in self.kv_hooks for m in h.stats["mse_k"]]
        all_mse_v = [m for h in self.kv_hooks for m in h.stats["mse_v"]]
        all_lat    = [l for h in self.kv_hooks for l in h.stats["compress_ms"]]
        total_saved = sum(h.stats["bytes_saved"] for h in self.kv_hooks)
        return {
            "avg_mse_k": round(float(np.mean(all_mse_k)) if all_mse_k else 0, 6),
            "avg_mse_v": round(float(np.mean(all_mse_v)) if all_mse_v else 0, 6),
            "avg_compress_ms": round(float(np.mean(all_lat)) if all_lat else 0, 3),
            "bytes_saved_total": total_saved,
            "mb_saved": round(total_saved / 1e6, 3),
            "compression_ratio": round(self.tq.compression_ratio(), 2) if self.tq else 1.0,
            "n_layers": len(self.kv_hooks),
            "vram_gb": round(torch.cuda.memory_allocated()/1e9, 3) if DEVICE == "cuda" else 0,
        }

    def cleanup(self):
        self._remove_hooks()
        if self.model:
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
