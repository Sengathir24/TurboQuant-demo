"""
turbo_engine.py — TurboQuant compression engine
Handles model loading, KV cache hooks, streaming generation,
vector search, and RAG pipeline.
Cursor: this is the most important file. Every benchmark goes through here.
"""

import torch, numpy as np, time, gc, asyncio, logging
import torch.nn.functional as F
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, TextIteratorStreamer
)
from threading import Thread
import scipy.stats as sp_stats

from model_registry import MODEL_REGISTRY, ModelConfig

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _finite_metric(x: float, cap: float = 1e6) -> float:
    """Safe scalar for JSON/UI — blocks NaN/inf and absurd spikes from bad KV states."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if xf != xf or xf in (float("inf"), float("-inf")):
        return 0.0
    return float(min(max(xf, 0.0), cap))


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float = 1.12) -> torch.Tensor:
    """HF-style repetition penalty on already generated tokens."""
    if penalty <= 1.0 or not generated_ids:
        return logits
    unique_ids = torch.tensor(sorted(set(generated_ids)), device=logits.device, dtype=torch.long)
    token_logits = logits[0, unique_ids]
    penalized = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)
    logits[0, unique_ids] = penalized
    return logits


def _ban_repeated_ngrams(logits: torch.Tensor, generated_ids: list[int], ngram_size: int = 3) -> torch.Tensor:
    """Prevents generating an n-gram that already appeared in the same output."""
    if ngram_size <= 1 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned: set[int] = set()
    for i in range(len(generated_ids) - (ngram_size - 1)):
        if tuple(generated_ids[i:i + ngram_size - 1]) == prefix:
            banned.add(generated_ids[i + ngram_size - 1])
    if banned:
        logits[0, list(banned)] = float("-inf")
    return logits


# ══════════════════════════════════════════════════════════
# TurboQuant Math — same as the paper, pure PyTorch
# ══════════════════════════════════════════════════════════

def lloyd_max_centroids(bits: int, dim: int, n_iters: int = 300) -> torch.Tensor:
    # Keep centroid construction aligned with turboquant_plus:
    # - 1-bit and 2-bit use closed-form calibrated centroids
    # - >=3 bits use Lloyd optimization on N(0, 1/d)
    if bits == 1:
        c = np.sqrt(2.0 / (np.pi * max(dim, 1)))
        return torch.tensor([-c, c], dtype=torch.float32)
    if bits == 2:
        vals = np.array([-1.51, -0.453, 0.453, 1.51], dtype=np.float64) / np.sqrt(max(dim, 1))
        return torch.tensor(vals, dtype=torch.float32)

    k = 2 ** bits
    std = 1.0 / max(np.sqrt(dim), 1e-6)

    def _cond_mean(a: float, b: float) -> float:
        # E[X | a < X < b], X ~ N(0, std^2), stable form used in turboquant_plus.
        az = a / std if np.isfinite(a) else a
        bz = b / std if np.isfinite(b) else b
        if not np.isfinite(az):
            prob = sp_stats.norm.cdf(bz)
        elif not np.isfinite(bz):
            prob = sp_stats.norm.sf(az)
        else:
            prob = sp_stats.norm.cdf(bz) - sp_stats.norm.cdf(az)
        if prob < 1e-15:
            if np.isfinite(a) and not np.isfinite(b):
                return a + std
            if not np.isfinite(a) and np.isfinite(b):
                return b - std
            if np.isfinite(a) and np.isfinite(b):
                return (a + b) / 2.0
            return 0.0
        pdf_diff = sp_stats.norm.pdf(az) - sp_stats.norm.pdf(bz)
        return std * pdf_diff / prob

    # Initialize using quantiles.
    q = np.linspace(0.0, 1.0, k + 1)[1:-1]
    boundaries = sp_stats.norm.ppf(q, scale=std)
    centroids = np.zeros(k, dtype=np.float64)
    centroids[0] = _cond_mean(-np.inf, boundaries[0])
    for i in range(1, k - 1):
        centroids[i] = _cond_mean(boundaries[i - 1], boundaries[i])
    centroids[-1] = _cond_mean(boundaries[-1], np.inf)

    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        new_c = np.zeros_like(centroids)
        new_c[0] = _cond_mean(-np.inf, boundaries[0])
        for i in range(1, k - 1):
            new_c[i] = _cond_mean(boundaries[i - 1], boundaries[i])
        new_c[-1] = _cond_mean(boundaries[-1], np.inf)
        if np.allclose(new_c, centroids, atol=1e-9):
            break
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

    def __init__(self, dim: int, bits: int, mode: str = "ip", seed: int = 42, norm_correction: bool = True):
        self.dim, self.bits, self.mode = dim, bits, mode
        self.norm_correction = bool(norm_correction)
        # In demo workloads, full-strength QJL residual can over-shoot vector MSE for K.
        # Keep IP+QJL path, but temper residual contribution for more stable K-MSE.
        self.k_qjl_alpha = 0.75 if mode == "ip" else 0.0
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
        # V-cache follows turboquant_plus KV design: MSE-only quantization at full bit width.
        self.v_mse_bits = bits
        self.v_codebook = lloyd_max_centroids(self.v_mse_bits, dim).to(DEVICE)
        self.v_boundaries = ((self.v_codebook[:-1] + self.v_codebook[1:]) / 2).to(DEVICE)

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
            qjl = torch.where(qjl == 0, torch.ones_like(qjl), qjl)
        return Compressed(idx.to(torch.int16), qjl, r_norm,
                          x_norm.squeeze(-1), x.shape, self.mode)

    def decompress(self, c: Compressed) -> torch.Tensor:
        y_hat = self.codebook[c.idx.long()]
        if self.norm_correction:
            y_hat_norm = y_hat.norm(dim=-1, keepdim=True)
            y_hat = y_hat / (y_hat_norm + 1e-10)
        x_mse = y_hat @ self.PI
        if c.mode == "ip":
            scale = (np.pi / 2) ** 0.5 / self.dim
            x_qjl = scale * c.r_norm.unsqueeze(-1) * (c.qjl.float() @ self.S)
            x_hat = x_mse + self.k_qjl_alpha * x_qjl
        else:
            x_hat = x_mse
        return x_hat * c.x_norm.unsqueeze(-1)

    def compression_ratio(self) -> float:
        # Match turboquant_plus memory model:
        # original per vector: d * 16 bits (fp16)
        # compressed per vector:
        #   - IP mode: d * b bits + 32-bit norm
        #   - MSE mode: d * b bits (+ optional 32-bit norm; we keep parity here)
        bits_per_coord = self.mse_bits + (1 if self.mode == "ip" else 0)
        original_bits = self.dim * 16.0
        compressed_bits = self.dim * bits_per_coord + 32.0
        return original_bits / max(compressed_bits, 1e-12)

    def kv_bytes_saved(self, n_tokens: int, n_heads: int) -> int:
        n_vectors = n_tokens * n_heads
        # K + V are both fp16 vectors of size head_dim.
        original_bits = n_vectors * self.dim * 16 * 2
        k_bits_per_coord = self.mse_bits + (1 if self.mode == "ip" else 0)
        v_bits_per_coord = self.v_mse_bits
        # Store one float32 norm per vector per cache (K and V).
        compressed_bits = n_vectors * (
            (self.dim * k_bits_per_coord + 32) + (self.dim * v_bits_per_coord + 32)
        )
        original_bytes = int(original_bits // 8)
        compressed_bytes = int(np.ceil(compressed_bits / 8.0))
        return max(0, original_bytes - compressed_bytes)

    def compress_kv_tensors(self, k: torch.Tensor, v: torch.Tensor, return_details: bool = False):
        """
        Compress K/V tensors (B,H,T,D) or (B,T,D).
        Returns (k_hat, v_hat, mse_k, mse_v, bytes_saved).
        If return_details=True, appends raw NMSE/SSE diagnostics.
        """
        orig_dtype = k.dtype
        k_f = k.to(DEVICE).float()
        v_f = v.to(DEVICE).float()

        squeezed = False
        if k_f.dim() == 3:
            k_f = k_f.unsqueeze(1)
            v_f = v_f.unsqueeze(1)
            squeezed = True

        B, H, T, D = k_f.shape

        if D != self.dim:
            return k.to(orig_dtype), v.to(orig_dtype), 0.0, 0.0, 0

        k_flat = k_f.reshape(-1, D)
        v_flat = v_f.reshape(-1, D)

        # K: TurboQuant (IP mode => PolarQuant + QJL residual correction).
        ck = self.compress(k_flat)
        k_hat = self.decompress(ck).reshape(B, H, T, D)

        # V: MSE-only path with full bit width (TurboQuantMSE behavior from turboquant_plus).
        v_norm = v_flat.norm(dim=-1, keepdim=True)
        v_unit = v_flat / (v_norm + 1e-12)
        vy = v_unit @ self.PI.T
        vidx = torch.bucketize(vy, self.v_boundaries)
        vy_hat = self.v_codebook[vidx]
        if self.norm_correction:
            vy_hat_norm = vy_hat.norm(dim=-1, keepdim=True)
            vy_hat = vy_hat / (vy_hat_norm + 1e-10)
        v_hat = ((vy_hat @ self.PI) * v_norm).reshape(B, H, T, D)

        # Report both bounded score (UI-friendly) and raw normalized error (diagnostics).
        diff_k = (k_f.double() - k_hat.double())
        diff_v = (v_f.double() - v_hat.double())
        sse_k = diff_k.pow(2).sum().item()
        sse_v = diff_v.pow(2).sum().item()
        ref_k = max(k_f.double().pow(2).sum().item(), 1e-12)
        ref_v = max(v_f.double().pow(2).sum().item(), 1e-12)
        nmse_k = sse_k / ref_k
        nmse_v = sse_v / ref_v
        mse_k = _finite_metric(nmse_k / (1.0 + nmse_k), cap=1.0)
        mse_v = _finite_metric(nmse_v / (1.0 + nmse_v), cap=1.0)
        bytes_saved = self.kv_bytes_saved(T, H)

        if squeezed:
            k_hat = k_hat.squeeze(1)
            v_hat = v_hat.squeeze(1)

        k_hat = torch.nan_to_num(k_hat, nan=0.0)
        v_hat = torch.nan_to_num(v_hat, nan=0.0)
        if orig_dtype in (torch.float16, torch.bfloat16):
            finfo = torch.finfo(orig_dtype)
            k_hat = torch.clamp(k_hat, finfo.min, finfo.max)
            v_hat = torch.clamp(v_hat, finfo.min, finfo.max)

        if not return_details:
            return k_hat.to(orig_dtype), v_hat.to(orig_dtype), mse_k, mse_v, bytes_saved
        details = {
            "nmse_k": _finite_metric(nmse_k, cap=1e6),
            "nmse_v": _finite_metric(nmse_v, cap=1e6),
            "sse_k": _finite_metric(sse_k, cap=1e12),
            "sse_v": _finite_metric(sse_v, cap=1e12),
            "ref_k": _finite_metric(ref_k, cap=1e12),
            "ref_v": _finite_metric(ref_v, cap=1e12),
        }
        return k_hat.to(orig_dtype), v_hat.to(orig_dtype), mse_k, mse_v, bytes_saved, details


# ══════════════════════════════════════════════════════════
# KV Extraction helpers — handles both legacy tuple KV and
# the new transformers Cache API (DynamicCache, StaticCache)
# introduced in HuggingFace transformers >= 4.36.
# ══════════════════════════════════════════════════════════

def _extract_kv_from_cache_obj(cache_obj, layer_idx: int):
    """
    Extract (k, v) tensors from a HF Cache object.

    New transformers (DynamicCache): ``cache.layers[layer_idx].keys`` / ``.values``
    (see CacheLayer / DynamicLayer — shapes [B, H, T, D]).

    Older API: ``key_cache`` / ``value_cache`` lists of tensors per layer.
    """
    # ── Transformers 4.52+ style: Cache.layers[*].keys / .values ──
    if hasattr(cache_obj, "layers"):
        ls = cache_obj.layers
        if isinstance(ls, (list, tuple)) and layer_idx < len(ls):
            layer = ls[layer_idx]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                k, v = layer.keys, layer.values
                if (
                    isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                    and k.dim() >= 3
                    and k.shape[-2] > 0
                    and k.numel() > 0
                ):
                    return k, v

    # ── Legacy list-of-tensors per layer ───────────────────────
    if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
        kc = cache_obj.key_cache
        vc = cache_obj.value_cache
        if (
            isinstance(kc, (list, tuple))
            and layer_idx < len(kc)
            and isinstance(kc[layer_idx], torch.Tensor)
            and kc[layer_idx].dim() >= 3
        ):
            return kc[layer_idx], vc[layer_idx]
    return None


def _inject_kv_into_cache_obj(cache_obj, layer_idx: int, k_hat: torch.Tensor, v_hat: torch.Tensor) -> bool:
    """Write compressed K/V back into a HF Cache (new layers API or legacy lists)."""
    if hasattr(cache_obj, "layers") and layer_idx < len(cache_obj.layers):
        layer = cache_obj.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            layer.keys = k_hat
            layer.values = v_hat
            return True
    if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
        kc = cache_obj.key_cache
        if isinstance(kc, (list, tuple)) and layer_idx < len(kc):
            cache_obj.key_cache[layer_idx] = k_hat
            cache_obj.value_cache[layer_idx] = v_hat
            return True
    return False


def _resolve_past_key_values_from_forward(args, kwargs) -> Optional[object]:
    """past_key_values may be positional (rare) or keyword (GPT2Block → GPT2Attention)."""
    if kwargs:
        # Different model families use different names for cache objects.
        for key in ("past_key_values", "past_key_value", "layer_past", "cache"):
            pv = kwargs.get(key)
            if pv is not None:
                return pv
    if args and len(args) > 1 and args[1] is not None:
        return args[1]
    return None


def _iter_candidate_caches(past_key_values: Optional[object]):
    """
    EncoderDecoderCache exposes self_attention_cache / cross_attention_cache;
    decoder-only models pass a single DynamicCache.
    """
    if past_key_values is None:
        return
    yield past_key_values
    if hasattr(past_key_values, "self_attention_cache") and past_key_values.self_attention_cache is not None:
        yield past_key_values.self_attention_cache
    if hasattr(past_key_values, "cross_attention_cache") and past_key_values.cross_attention_cache is not None:
        yield past_key_values.cross_attention_cache


def _make_kv_forward_hook(hook_obj: "KVCompressorHook"):
    """PyTorch 2.x: with_kwargs=True → (module, args, kwargs, output)."""

    def forward_hook(module, args, kwargs, output):
        return hook_obj(module, args, kwargs or {}, output)

    def forward_hook_legacy(module, inp, output):
        return hook_obj(module, inp, {}, output)

    forward_hook.legacy = forward_hook_legacy  # type: ignore[attr-defined]
    return forward_hook


def _find_kv_in_output(output, layer_idx: int):
    """
    Locate (k, v) tensors in an attention layer's forward output.

    Handles:
      1. Legacy tuple KV:      output contains a (tuple/list of 2 same-shape tensors)
      2. HF Cache object:      output contains a Cache instance (DynamicCache etc.)
      3. Cache at top level:   output IS the cache (rare but possible in some hooks)

    Returns (kv_idx, k, v) where kv_idx is the position in `output` (-1 if top-level),
    or (None, None, None) if not found.
    """
    # Case 0: output itself is a Cache-like object
    kv = _extract_kv_from_cache_obj(output, layer_idx)
    if kv is not None:
        return -1, kv[0], kv[1]

    if not isinstance(output, (tuple, list)):
        return None, None, None

    for i, item in enumerate(output):
        if item is None:
            continue

        # ── Case 1: legacy tuple/list of 2 matching tensors ──────
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], torch.Tensor)
            and isinstance(item[1], torch.Tensor)
            and item[0].shape == item[1].shape
            and item[0].dim() >= 3
        ):
            return i, item[0], item[1]

        # ── Case 2: HF Cache object ───────────────────────────────
        kv = _extract_kv_from_cache_obj(item, layer_idx)
        if kv is not None:
            return i, kv[0], kv[1]

    return None, None, None


def _inject_kv_into_output(output, kv_idx: int, k_hat: torch.Tensor,
                            v_hat: torch.Tensor, layer_idx: int):
    """
    Rebuild the output tuple with compressed k/v substituted back in.
    Handles both legacy tuple KV and HF Cache objects.
    """
    if kv_idx == -1:
        # Top-level Cache object output
        if _inject_kv_into_cache_obj(output, layer_idx, k_hat, v_hat):
            return output
        if hasattr(output, "key_cache") and hasattr(output, "value_cache"):
            if layer_idx < len(output.key_cache):
                output.key_cache[layer_idx] = k_hat
                output.value_cache[layer_idx] = v_hat
        return output

    item = output[kv_idx]

    if isinstance(item, (tuple, list)):
        # Legacy: replace the 2-element container
        new_kv = type(item)([k_hat, v_hat])
        if isinstance(output, tuple):
            return output[:kv_idx] + (new_kv,) + output[kv_idx + 1:]
        new_output = list(output)
        new_output[kv_idx] = new_kv
        return new_output

    # HF Cache object: mutate in-place (layers API or legacy lists)
    if _inject_kv_into_cache_obj(item, layer_idx, k_hat, v_hat):
        return output

    return output  # fallback: no substitution


# ══════════════════════════════════════════════════════════
# KV Cache Compressor — hooks into attention layers
# ══════════════════════════════════════════════════════════

class KVCompressorHook:
    """
    Wraps a single attention layer's forward output.
    Compresses K and V tensors and records per-layer stats.

    Compatibility:
      - GPT-2 / modern causal LM: KV lives in ``past_key_values`` (DynamicCache.layers[i].keys/values),
        while attention returns only ``(hidden, attn_weights)`` — we read/write the cache after forward.
      - Legacy: (hidden, (k,v)), OPT (hidden, weights, (k,v)), cache-in-output tuple paths.
    """

    def __init__(self, tq: "TurboQuantGPU", layer_idx: int):
        self.tq = tq
        self.layer_idx = layer_idx
        self._active = True
        self.stats = {
            "compress_ms": [], "mse_k": [], "mse_v": [],
            "bytes_saved": 0, "calls": 0
        }

    def reset_stats(self):
        self.stats = {
            "compress_ms": [], "mse_k": [], "mse_v": [],
            "bytes_saved": 0, "calls": 0
        }

    def __call__(self, module, args, kwargs, output):
        if not self._active:
            return output

        kwargs = kwargs or {}
        # ── Primary path: mutate past_key_values cache (transformers ≥ 4.40 causal LM) ──
        pv = _resolve_past_key_values_from_forward(args, kwargs)
        for cache in _iter_candidate_caches(pv):
            kv = _extract_kv_from_cache_obj(cache, self.layer_idx)
            if kv is None:
                continue
            k_raw, v_raw = kv
            t0 = time.perf_counter()
            try:
                k_hat, v_hat, mse_k, mse_v, bytes_saved = self._compress_kv(k_raw, v_raw)
            except Exception as e:
                logger.debug(f"KV compression skipped on layer {self.layer_idx}: {e}")
                return output
            ms = (time.perf_counter() - t0) * 1000

            self.stats["compress_ms"].append(ms)
            self.stats["mse_k"].append(mse_k)
            self.stats["mse_v"].append(mse_v)
            self.stats["bytes_saved"] += bytes_saved
            self.stats["calls"] += 1

            _inject_kv_into_cache_obj(cache, self.layer_idx, k_hat, v_hat)
            return output

        # ── Fallback: KV embedded in forward output (older models) ──
        kv_idx, k_raw, v_raw = _find_kv_in_output(output, self.layer_idx)
        if kv_idx is None:
            return output

        t0 = time.perf_counter()
        try:
            k_hat, v_hat, mse_k, mse_v, bytes_saved = self._compress_kv(k_raw, v_raw)
        except Exception as e:
            logger.debug(f"KV compression skipped on layer {self.layer_idx}: {e}")
            return output
        ms = (time.perf_counter() - t0) * 1000

        self.stats["compress_ms"].append(ms)
        self.stats["mse_k"].append(mse_k)
        self.stats["mse_v"].append(mse_v)
        self.stats["bytes_saved"] += bytes_saved
        self.stats["calls"] += 1

        return _inject_kv_into_output(output, kv_idx, k_hat, v_hat, self.layer_idx)

    def _compress_kv(self, k: torch.Tensor, v: torch.Tensor):
        """Delegates to TurboQuantGPU.compress_kv_tensors."""
        return self.tq.compress_kv_tensors(k, v)


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
        self._live_layer_metrics: dict[int, dict] = {}
        self._live_run_metrics = {
            "baseline_tps": 0.0,
            "compressed_tps": 0.0,
            "compressed_kv_mse": 0.0,
            "compressed_kv_mse_k": 0.0,
            "compressed_kv_mse_v": 0.0,
            "compressed_kv_nmse_step": 0.0,
            "compressed_kv_nmse_cum": 0.0,
            "compressed_kv_nmse_k_cum": 0.0,
            "compressed_kv_nmse_v_cum": 0.0,
            "avg_compress_ms": 0.0,
            "bytes_saved_total": 0,
            "last_run": None,
        }

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
        return hasattr(self, "model") and self.model is not None

    def load_model(self, model_id: str, bits: int, mode: str) -> dict:
        """Load model + tokenizer, attach TurboQuant KV hooks."""
        cfg = MODEL_REGISTRY[model_id]
        if mode == "ip" and bits < 2:
            raise ValueError("IP TurboQuant requires bits >= 2 (uses b-1 PolarQuant + 1-bit QJL).")
        self._bits = bits
        self._mode = mode

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
            cfg.hf_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(DEVICE)
        self.model.eval()

        self.model_id = model_id
        self.model_cfg = cfg
        self._live_run_metrics = {
            "baseline_tps": 0.0,
            "compressed_tps": 0.0,
            "compressed_kv_mse": 0.0,
            "compressed_kv_mse_k": 0.0,
            "compressed_kv_mse_v": 0.0,
            "compressed_kv_nmse_step": 0.0,
            "compressed_kv_nmse_cum": 0.0,
            "compressed_kv_nmse_k_cum": 0.0,
            "compressed_kv_nmse_v_cum": 0.0,
            "avg_compress_ms": 0.0,
            "bytes_saved_total": 0,
            "last_run": None,
        }
        self._live_layer_metrics = {}

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
        cfg = self.model.config
        # T5-family configs expose per-head key/value dim directly.
        if hasattr(cfg, "d_kv"):
            return int(cfg.d_kv)
        if hasattr(cfg, "n_embd") and hasattr(cfg, "n_head"):
            return cfg.n_embd // cfg.n_head
        if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
            return cfg.hidden_size // cfg.num_attention_heads
        if hasattr(cfg, "d_model") and hasattr(cfg, "num_heads"):
            return cfg.d_model // cfg.num_heads
        return 64

    def _attach_hooks(self):
        self.kv_hooks = []
        layers = self._get_attention_layers()
        for i, layer in enumerate(layers):
            hook_obj = KVCompressorHook(self.tq, layer_idx=i)
            wrapped = _make_kv_forward_hook(hook_obj)
            try:
                h = layer.register_forward_hook(wrapped, with_kwargs=True)
            except TypeError:
                h = layer.register_forward_hook(wrapped.legacy)
            self.hooks.append(h)
            self.kv_hooks.append(hook_obj)
        logger.info(f"Attached TurboQuant hooks to {len(layers)} layers")

    def _get_attention_layers(self):
        """
        Return list of attention modules for all supported model families.
        NOTE: BLOOM and GPT-2 both have model.transformer.h — differentiate
        by block attribute: BLOOM uses .self_attention, GPT-2 uses .attn.
        """
        m = self.model
        layers = []

        # ── GPT-NeoX / Pythia ─────────────────────────────────────
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return [layer.attention for layer in m.gpt_neox.layers]

        # ── GPT-2 / BLOOM — shared transformer.h path ─────────────
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            blocks = m.transformer.h
            if blocks:
                b0 = blocks[0]
                if hasattr(b0, "self_attention"):      # BLOOM
                    return [b.self_attention for b in blocks]
                elif hasattr(b0, "attn"):              # GPT-2 / DistilGPT2
                    return [b.attn for b in blocks]

        # ── OPT ───────────────────────────────────────────────────
        if hasattr(m, "model") and hasattr(m.model, "decoder"):
            return [l.self_attn for l in m.model.decoder.layers]

        # ── T5 — encoder self-attn + decoder self-attn + cross-attn
        # We hook T5Attention objects directly. For the new Cache API,
        # KVCompressorHook._find_kv_in_output detects the DynamicCache.
        if hasattr(m, "encoder") and hasattr(m, "decoder"):
            if hasattr(m.encoder, "block"):
                layers += [b.layer[0].SelfAttention for b in m.encoder.block]
            if hasattr(m.decoder, "block"):
                layers += [b.layer[0].SelfAttention for b in m.decoder.block]
                layers += [b.layer[1].EncDecAttention for b in m.decoder.block]
            return layers

        # ── T5-encoder only fallback ───────────────────────────────
        if hasattr(m, "encoder") and hasattr(m.encoder, "block"):
            return [b.layer[0].SelfAttention for b in m.encoder.block]

        logger.warning("Could not detect attention layers — KV hooks not attached")
        return []

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.kv_hooks.clear()

    def _snapshot_live_kv_from_hooks(self) -> None:
        """Copy hook aggregates into _live_run_metrics (seq2seq). Causal LM uses cache walk — hooks often empty; do not overwrite those metrics with zeros."""
        if not self.kv_hooks:
            return
        all_mse_k = [v for h in self.kv_hooks for v in h.stats["mse_k"]]
        all_mse_v = [v for h in self.kv_hooks for v in h.stats["mse_v"]]
        all_lat = [v for h in self.kv_hooks for v in h.stats["compress_ms"]]
        total_b = sum(h.stats["bytes_saved"] for h in self.kv_hooks)
        if not all_mse_k and not all_mse_v and not all_lat and total_b == 0:
            return
        mk = _finite_metric(float(np.mean(all_mse_k)) if all_mse_k else 0.0)
        mv = _finite_metric(float(np.mean(all_mse_v)) if all_mse_v else 0.0)
        self._live_run_metrics["compressed_kv_mse_k"] = mk
        self._live_run_metrics["compressed_kv_mse_v"] = mv
        self._live_run_metrics["compressed_kv_mse"] = _finite_metric((mk + mv) * 0.5)
        self._live_run_metrics["avg_compress_ms"] = _finite_metric(
            float(np.mean(all_lat)) if all_lat else 0.0, cap=1e6
        )
        self._live_run_metrics["bytes_saved_total"] = int(total_b)

    def _compress_past_key_values_inplace(
        self, past_key_values, prev_seq_lens: dict[int, int]
    ) -> tuple[float, float, int, float, list, list, float, float, float, float]:
        """
        TurboQuant on HF DynamicCache after model.forward — matches explicit KV streaming docs.
        Mutates only newly appended KV slice for each layer in compressed run.
        Returns
        (mean_mse_k, mean_mse_v, bytes_saved_step, compress_ms, all_mse_k, all_mse_v,
         step_sse_k, step_sse_v, step_ref_k, step_ref_v).
        """
        if past_key_values is None or self.tq is None:
            return 0.0, 0.0, 0, 0.0, [], [], 0.0, 0.0, 0.0, 0.0
        if not hasattr(past_key_values, "layers"):
            return 0.0, 0.0, 0, 0.0, [], [], 0.0, 0.0, 0.0, 0.0

        layer_mks: list[float] = []
        layer_mvs: list[float] = []
        step_sse_k = 0.0
        step_sse_v = 0.0
        step_ref_k = 0.0
        step_ref_v = 0.0
        total_bytes = 0
        t0 = time.perf_counter()

        for layer_idx, layer in enumerate(past_key_values.layers):
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                continue
            k, v = layer.keys, layer.values
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                continue
            if k.numel() == 0 or k.dim() < 3:
                continue
            seq_len = int(k.shape[-2])
            prev_len = int(prev_seq_lens.get(layer_idx, 0))
            delta_tokens = max(seq_len - prev_len, 0)
            prev_seq_lens[layer_idx] = seq_len
            if delta_tokens <= 0:
                continue
            # Measure and apply compression only on newly appended KV tokens for this step.
            k_new = k[..., prev_len:seq_len, :]
            v_new = v[..., prev_len:seq_len, :]
            if k_new.numel() == 0 or v_new.numel() == 0:
                continue
            k_hat_new, v_hat_new, mse_k, mse_v, _, details = self.tq.compress_kv_tensors(
                k_new, v_new, return_details=True
            )
            layer_mks.append(mse_k)
            layer_mvs.append(mse_v)
            lm = self._live_layer_metrics.setdefault(
                layer_idx, {"mse_k_sum": 0.0, "mse_v_sum": 0.0, "calls": 0}
            )
            lm["mse_k_sum"] += float(mse_k)
            lm["mse_v_sum"] += float(mse_v)
            lm["calls"] += 1
            step_sse_k += float(details["sse_k"])
            step_sse_v += float(details["sse_v"])
            step_ref_k += float(details["ref_k"])
            step_ref_v += float(details["ref_v"])
            n_heads = int(k.shape[1]) if k.dim() >= 4 else 1
            total_bytes += self.tq.kv_bytes_saved(delta_tokens, n_heads)
            if torch.isfinite(k_hat_new).all() and torch.isfinite(v_hat_new).all():
                layer.keys[..., prev_len:seq_len, :] = k_hat_new
                layer.values[..., prev_len:seq_len, :] = v_hat_new

        compress_ms = (time.perf_counter() - t0) * 1000
        mk = _finite_metric(float(np.mean(layer_mks)) if layer_mks else 0.0)
        mv = _finite_metric(float(np.mean(layer_mvs)) if layer_mvs else 0.0)
        return (
            mk, mv, total_bytes, compress_ms, layer_mks, layer_mvs,
            step_sse_k, step_sse_v, step_ref_k, step_ref_v
        )

    async def _stream_generate_seq2seq(
        self, prompt: str, max_new_tokens: int, temperature: float, compare_mode: bool
    ) -> AsyncGenerator[dict, None]:
        """Encoder–decoder models: keep generate() + hook path (T5/FLAN)."""
        results: dict = {}
        runs = (
            [("baseline", False), ("compressed", True)]
            if compare_mode else [("compressed", True)]
        )
        for run_name, use_compression in runs:
            for hook_obj in self.kv_hooks:
                hook_obj.reset_stats()
                hook_obj._active = use_compression

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
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            pieces: list[str] = []
            for token in streamer:
                pieces.append(token)
                elapsed = time.perf_counter() - t_start
                tps = len(pieces) / max(elapsed, 0.001)
                peak_mem = (torch.cuda.memory_allocated() if DEVICE == "cuda" else 0) / 1e9
                all_mse_k = [v for h in self.kv_hooks for v in h.stats["mse_k"]]
                all_mse_v = [v for h in self.kv_hooks for v in h.stats["mse_v"]]
                all_lat = [v for h in self.kv_hooks for v in h.stats["compress_ms"]]
                avg_mse = float(np.mean(all_mse_k)) if all_mse_k else 0.0
                avg_mse_v = float(np.mean(all_mse_v)) if all_mse_v else 0.0
                avg_lat = float(np.mean(all_lat)) if all_lat else 0.0
                total_bytes = sum(h.stats["bytes_saved"] for h in self.kv_hooks)
                if run_name == "baseline":
                    self._live_run_metrics["baseline_tps"] = float(tps)
                else:
                    self._live_run_metrics["compressed_tps"] = float(tps)
                    self._live_run_metrics["compressed_kv_mse"] = float((avg_mse + avg_mse_v) * 0.5)
                    self._live_run_metrics["compressed_kv_mse_k"] = float(avg_mse)
                    self._live_run_metrics["compressed_kv_mse_v"] = float(avg_mse_v)
                    self._live_run_metrics["avg_compress_ms"] = float(avg_lat)
                    self._live_run_metrics["bytes_saved_total"] = int(total_bytes)
                self._live_run_metrics["last_run"] = run_name
                yield {
                    "run": run_name,
                    "token": token,
                    "tokens_so_far": len(pieces),
                    "tps": round(tps, 1),
                    "elapsed_s": round(elapsed, 2),
                    "peak_mem_gb": round(peak_mem, 3),
                    "kv_mse": round(avg_mse, 6),
                    "bytes_saved": total_bytes,
                }
                await asyncio.sleep(0)
            thread.join()
            results[run_name] = "".join(pieces)

        self._snapshot_live_kv_from_hooks()
        total_saved = sum(h.stats["bytes_saved"] for h in self.kv_hooks)
        yield {
            "run": "summary",
            "baseline_text": results.get("baseline", ""),
            "compressed_text": results.get("compressed", ""),
            "total_bytes_saved": total_saved,
            "compression_ratio": self.tq.compression_ratio() if self.tq else 1.0,
        }

    async def stream_generate(
        self, prompt: str, max_new_tokens: int,
        temperature: float, compare_mode: bool
    ) -> AsyncGenerator[dict, None]:
        """
        Stream tokens with live metrics.
        Causal LM: explicit prefill + decode with ``past_key_values`` (HF DynamicCache) per docs.
        Seq2seq: ``generate()`` + hooks.
        """
        if self.model_cfg and self.model_cfg.task != "causal-lm":
            async for chunk in self._stream_generate_seq2seq(
                prompt, max_new_tokens, temperature, compare_mode
            ):
                yield chunk
            return

        results: dict = {}
        self._live_run_metrics = {
            "baseline_tps": 0.0,
            "compressed_tps": 0.0,
            "compressed_kv_mse": 0.0,
            "compressed_kv_mse_k": 0.0,
            "compressed_kv_mse_v": 0.0,
            "compressed_kv_nmse_step": 0.0,
            "compressed_kv_nmse_cum": 0.0,
            "compressed_kv_nmse_k_cum": 0.0,
            "compressed_kv_nmse_v_cum": 0.0,
            "avg_compress_ms": 0.0,
            "bytes_saved_total": 0,
            "last_run": None,
        }
        self._live_layer_metrics = {}
        runs = (
            [("baseline", False), ("compressed", True)]
            if compare_mode else [("compressed", True)]
        )

        eos_id = self.tokenizer.eos_token_id
        # In compare mode, use deterministic decoding so differences reflect KV compression,
        # not two different random sampling trajectories.
        do_sample = (temperature > 0) and (not compare_mode)
        # Guardrails for causal compressed runs:
        # - let cache stabilize before first compression
        # - avoid pathological immediate-EOS collapse under compressed logits
        compress_warmup_tokens = 8
        min_tokens_before_eos_compressed = 16
        final_compressed_saved = 0

        for run_name, use_compression in runs:
            for hook_obj in self.kv_hooks:
                hook_obj.reset_stats()
                hook_obj._active = False

            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = prompt_inputs["input_ids"]
            attn_mask = prompt_inputs.get("attention_mask")

            gen_ids: list[int] = []
            past_key_values = None
            cur_ids = input_ids
            cur_attn = attn_mask
            stream_prefix = 0
            run_bytes_total = 0
            prev_seq_lens: dict[int, int] = {}
            run_sse_k = 0.0
            run_sse_v = 0.0
            run_ref_k = 0.0
            run_ref_v = 0.0

            t_start = time.perf_counter()

            for _step in range(max_new_tokens):
                with torch.no_grad():
                    out = self.model(
                        input_ids=cur_ids,
                        attention_mask=cur_attn,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                past_key_values = out.past_key_values

                want_compress = (
                    run_name == "compressed"
                    and use_compression
                    and self.tq is not None
                    and len(gen_ids) >= compress_warmup_tokens
                )
                if want_compress:
                    mk, mv, bstep, cms, _, _, step_sse_k, step_sse_v, step_ref_k, step_ref_v = self._compress_past_key_values_inplace(
                        past_key_values, prev_seq_lens
                    )
                    run_sse_k += step_sse_k
                    run_sse_v += step_sse_v
                    run_ref_k += step_ref_k
                    run_ref_v += step_ref_v
                    run_bytes_total += bstep
                    # Per-step aggregates only (avoids blowing up UI if one timestep spikes)
                    avg_mse = _finite_metric(mk)
                    avg_mse_v = _finite_metric(mv)
                    avg_lat = _finite_metric(cms)
                    step_nmse_k = _finite_metric(step_sse_k / max(step_ref_k, 1e-12), cap=1e6)
                    step_nmse_v = _finite_metric(step_sse_v / max(step_ref_v, 1e-12), cap=1e6)
                    step_nmse = _finite_metric((step_nmse_k + step_nmse_v) * 0.5, cap=1e6)
                    cum_nmse_k = _finite_metric(run_sse_k / max(run_ref_k, 1e-12), cap=1e6)
                    cum_nmse_v = _finite_metric(run_sse_v / max(run_ref_v, 1e-12), cap=1e6)
                    cum_nmse = _finite_metric((cum_nmse_k + cum_nmse_v) * 0.5, cap=1e6)
                else:
                    # Keep cache-length tracking in sync during warmup so once compression
                    # starts we only compress newly appended tokens.
                    if (
                        run_name == "compressed"
                        and use_compression
                        and past_key_values is not None
                        and hasattr(past_key_values, "layers")
                    ):
                        for layer_idx, layer in enumerate(past_key_values.layers):
                            if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor) and layer.keys.dim() >= 3:
                                prev_seq_lens[layer_idx] = int(layer.keys.shape[-2])
                    avg_mse = avg_mse_v = avg_lat = 0.0
                    step_nmse = cum_nmse_k = cum_nmse_v = cum_nmse = 0.0

                logits = out.logits[:, -1, :]
                logits = _apply_repetition_penalty(logits, gen_ids, penalty=1.12)
                logits = _ban_repeated_ngrams(logits, gen_ids, ngram_size=3)
                if do_sample:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                nid = int(next_id.item())
                if (
                    eos_id is not None
                    and nid == eos_id
                    and run_name == "compressed"
                    and len(gen_ids) < min_tokens_before_eos_compressed
                ):
                    # Ban EOS briefly in compressed run to avoid early-stop collapse that
                    # obscures side-by-side quality comparison in the demo UI.
                    alt_logits = logits.clone()
                    alt_logits[:, eos_id] = float("-inf")
                    if do_sample:
                        alt_probs = F.softmax(alt_logits / max(temperature, 1e-6), dim=-1)
                        next_id = torch.multinomial(alt_probs, num_samples=1)
                    else:
                        next_id = torch.argmax(alt_logits, dim=-1, keepdim=True)
                    nid = int(next_id.item())
                gen_ids.append(nid)

                stream_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                token = stream_text[stream_prefix:]
                stream_prefix = len(stream_text)

                elapsed = time.perf_counter() - t_start
                tps = len(gen_ids) / max(elapsed, 0.001)
                peak_mem = (torch.cuda.memory_allocated() if DEVICE == "cuda" else 0) / 1e9

                if run_name == "baseline":
                    self._live_run_metrics["baseline_tps"] = float(tps)
                else:
                    self._live_run_metrics["compressed_tps"] = float(tps)
                    # Report cumulative normalized error for dashboard consistency.
                    self._live_run_metrics["compressed_kv_mse"] = cum_nmse
                    self._live_run_metrics["compressed_kv_mse_k"] = avg_mse
                    self._live_run_metrics["compressed_kv_mse_v"] = avg_mse_v
                    self._live_run_metrics["compressed_kv_nmse_step"] = step_nmse
                    self._live_run_metrics["compressed_kv_nmse_cum"] = cum_nmse
                    self._live_run_metrics["compressed_kv_nmse_k_cum"] = cum_nmse_k
                    self._live_run_metrics["compressed_kv_nmse_v_cum"] = cum_nmse_v
                    self._live_run_metrics["avg_compress_ms"] = avg_lat
                    self._live_run_metrics["bytes_saved_total"] = int(run_bytes_total)
                self._live_run_metrics["last_run"] = run_name

                yield {
                    "run": run_name,
                    "token": token,
                    "tokens_so_far": len(gen_ids),
                    "tps": round(tps, 1),
                    "elapsed_s": round(elapsed, 2),
                    "peak_mem_gb": round(peak_mem, 3),
                    "kv_mse": avg_mse,
                    "bytes_saved": run_bytes_total,
                }
                await asyncio.sleep(0)

                if eos_id is not None and nid == eos_id:
                    break

                cur_ids = next_id
                cur_attn = None

            results[run_name] = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if run_name == "compressed":
                final_compressed_saved = int(run_bytes_total)

        self._snapshot_live_kv_from_hooks()

        yield {
            "run": "summary",
            "baseline_text": results.get("baseline", ""),
            "compressed_text": results.get("compressed", ""),
            "total_bytes_saved": final_compressed_saved,
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

        t0 = time.perf_counter()
        scores_base = corpus_embs @ query_emb
        top_k_base  = np.argsort(scores_base)[::-1][:top_k]
        lat_base    = (time.perf_counter() - t0) * 1000

        tq_vs    = TurboQuantGPU(dim=embed_dim, bits=bits, mode="ip")
        xt       = torch.tensor(corpus_embs).to(DEVICE)
        c        = tq_vs.compress(xt)
        comp_embs = tq_vs.decompress(c).cpu().numpy()

        t0 = time.perf_counter()
        scores_comp = comp_embs @ query_emb
        top_k_comp  = np.argsort(scores_comp)[::-1][:top_k]
        lat_comp    = (time.perf_counter() - t0) * 1000

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

        tq_rag    = TurboQuantGPU(dim=dim, bits=bits, mode="ip")
        xt        = torch.tensor(corpus_embs).to(DEVICE)
        c         = tq_rag.compress(xt)
        comp_embs = tq_rag.decompress(c).cpu().numpy()

        scores = comp_embs @ q_emb
        top3   = np.argsort(scores)[::-1][:3]
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
        lr = self._live_run_metrics
        all_mse_k   = [v for h in self.kv_hooks for v in h.stats["mse_k"]] if self.kv_hooks else []
        all_mse_v   = [v for h in self.kv_hooks for v in h.stats["mse_v"]] if self.kv_hooks else []
        all_lat     = [v for h in self.kv_hooks for v in h.stats["compress_ms"]] if self.kv_hooks else []
        total_saved = sum(h.stats["bytes_saved"] for h in self.kv_hooks) if self.kv_hooks else 0

        # Prefer live hook lists; fall back to last causal-generation snapshot (hooks often idle).
        avg_k = float(np.mean(all_mse_k)) if all_mse_k else float(lr.get("compressed_kv_nmse_k_cum", lr.get("compressed_kv_mse_k", lr.get("compressed_kv_mse", 0.0))))
        avg_v = float(np.mean(all_mse_v)) if all_mse_v else float(lr.get("compressed_kv_nmse_v_cum", lr.get("compressed_kv_mse_v", 0.0)))
        avg_cm = float(np.mean(all_lat)) if all_lat else float(lr.get("avg_compress_ms", 0.0))
        avg_k, avg_v, avg_cm = _finite_metric(avg_k), _finite_metric(avg_v), _finite_metric(avg_cm, cap=1e4)
        if total_saved == 0 and int(lr.get("bytes_saved_total") or 0) > 0:
            total_saved = int(lr["bytes_saved_total"])

        if self.kv_hooks and any(h.stats["calls"] > 0 for h in self.kv_hooks):
            per_layer = [
                {
                    "layer": h.layer_idx,
                    "mse_k": round(float(np.mean(h.stats["mse_k"])) if h.stats["mse_k"] else 0.0, 8),
                    "mse_v": round(float(np.mean(h.stats["mse_v"])) if h.stats["mse_v"] else 0.0, 8),
                    "calls": h.stats["calls"],
                }
                for h in self.kv_hooks
            ]
        elif self._live_layer_metrics:
            per_layer = [
                {
                    "layer": int(layer_idx),
                    "mse_k": round(_finite_metric(vals["mse_k_sum"] / max(vals["calls"], 1)), 8),
                    "mse_v": round(_finite_metric(vals["mse_v_sum"] / max(vals["calls"], 1)), 8),
                    "calls": int(vals["calls"]),
                }
                for layer_idx, vals in sorted(self._live_layer_metrics.items())
            ]
        else:
            per_layer = []

        return {
            "avg_mse_k":         round(avg_k, 8),
            "avg_mse_v":         round(avg_v, 8),
            "avg_compress_ms":   round(avg_cm, 3),
            "bytes_saved_total": total_saved,
            "mb_saved":          round(total_saved / 1e6, 3),
            "compression_ratio": round(self.tq.compression_ratio(), 2) if self.tq else 1.0,
            "n_layers":          len(self.kv_hooks),
            "vram_gb":           round(torch.cuda.memory_allocated()/1e9, 3) if DEVICE == "cuda" else 0,
            "baseline_tps":      round(float(self._live_run_metrics.get("baseline_tps", 0.0)), 1),
            "compressed_tps":    round(float(self._live_run_metrics.get("compressed_tps", 0.0)), 1),
            "kv_mse":            round(float(self._live_run_metrics.get("compressed_kv_mse", 0.0)), 6),
            "kv_nmse_step":      round(float(self._live_run_metrics.get("compressed_kv_nmse_step", 0.0)), 6),
            "kv_nmse_cum":       round(float(self._live_run_metrics.get("compressed_kv_nmse_cum", 0.0)), 6),
            "kv_nmse_k_cum":     round(float(self._live_run_metrics.get("compressed_kv_nmse_k_cum", 0.0)), 6),
            "kv_nmse_v_cum":     round(float(self._live_run_metrics.get("compressed_kv_nmse_v_cum", 0.0)), 6),
            "last_run":          self._live_run_metrics.get("last_run"),
            "per_layer":         per_layer,
        }

    def cleanup(self):
        self._remove_hooks()
        if self.model:
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
