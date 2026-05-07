"""
One-command TurboQuant backend smoke test.

Covers:
- Causal LM load + generate(compare) + per-layer attention metrics
- Vector search
- RAG (causal)
- Seq2Seq load + generate + RAG

Usage:
    python smoke_test.py
    python smoke_test.py --max-new-tokens 8 --top-k 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from typing import Any

from turbo_engine import TurboEngine


def _preview(text: str | None, n: int = 120) -> str:
    if not text:
        return ""
    return text[:n]


async def _run_generate(
    engine: TurboEngine,
    prompt: str,
    max_new_tokens: int,
    compare_mode: bool,
) -> tuple[int, dict[str, Any] | None]:
    chunks = 0
    summary = None
    async for chunk in engine.stream_generate(prompt, max_new_tokens, 0.0, compare_mode):
        if chunk.get("run") == "summary":
            summary = chunk
            break
        chunks += 1
    return chunks, summary


def run_smoke(max_new_tokens: int, top_k: int, bits: int) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "checks": {}}

    # ---- Causal LM flows ----
    eng = TurboEngine()
    try:
        info = eng.load_model("gpt2", bits, "ip")
        out["checks"]["gpt2_load"] = {
            "ok": True,
            "model": info.get("model_id"),
            "head_dim": info.get("head_dim"),
            "compression_ratio": info.get("compression_ratio"),
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["gpt2_load"] = {"ok": False, "error": str(exc)}
        return out

    try:
        chunks, summary = asyncio.run(
            _run_generate(
                eng,
                "TurboQuant one-command smoke test prompt:",
                max_new_tokens,
                True,
            )
        )
        metrics = eng.get_live_metrics()
        nonzero_layers = len(
            [x for x in metrics.get("per_layer", []) if x.get("calls", 0) > 0]
        )
        out["checks"]["gpt2_generate_compare"] = {
            "ok": True,
            "chunks": chunks,
            "summary_keys": sorted(list(summary.keys())) if summary else [],
            "nonzero_layers": nonzero_layers,
            "kv_mse": metrics.get("kv_mse"),
        }
    except Exception:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["gpt2_generate_compare"] = {
            "ok": False,
            "error": traceback.format_exc(),
        }

    try:
        vs = eng.vector_search("How does KV cache compression work?", top_k, bits)
        ok = "error" not in vs
        out["checks"]["vector_search"] = {
            "ok": ok,
            "error": vs.get("error"),
            "recall_at_k": vs.get("recall_at_k") if ok else None,
            "compression_ratio": vs.get("compression_ratio") if ok else None,
        }
        out["ok"] = out["ok"] and ok
    except Exception:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["vector_search"] = {
            "ok": False,
            "error": traceback.format_exc(),
        }

    try:
        rag = eng.rag_pipeline("What does TurboQuant improve?", bits)
        ok = "error" not in rag
        out["checks"]["rag_causal"] = {
            "ok": ok,
            "error": rag.get("error"),
            "answer_preview": _preview(rag.get("answer")) if ok else "",
        }
        out["ok"] = out["ok"] and ok
    except Exception:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["rag_causal"] = {"ok": False, "error": traceback.format_exc()}
    finally:
        eng.cleanup()

    # ---- Seq2Seq flows ----
    eng2 = TurboEngine()
    try:
        info2 = eng2.load_model("flan-t5-small", bits, "ip")
        out["checks"]["flan_load"] = {
            "ok": True,
            "model": info2.get("model_id"),
            "head_dim": info2.get("head_dim"),
            "compression_ratio": info2.get("compression_ratio"),
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["flan_load"] = {"ok": False, "error": str(exc)}
        return out

    try:
        chunks2, summary2 = asyncio.run(
            _run_generate(
                eng2,
                "Explain KV compression briefly.",
                max_new_tokens,
                False,
            )
        )
        metrics2 = eng2.get_live_metrics()
        out["checks"]["flan_generate"] = {
            "ok": True,
            "chunks": chunks2,
            "summary_keys": sorted(list(summary2.keys())) if summary2 else [],
            "n_layers": metrics2.get("n_layers"),
        }
    except Exception:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["flan_generate"] = {"ok": False, "error": traceback.format_exc()}

    try:
        rag2 = eng2.rag_pipeline("What is attention cache?", bits)
        ok = "error" not in rag2
        out["checks"]["rag_seq2seq"] = {
            "ok": ok,
            "error": rag2.get("error"),
            "answer_preview": _preview(rag2.get("answer")) if ok else "",
        }
        out["ok"] = out["ok"] and ok
    except Exception:  # pragma: no cover - runtime dependent
        out["ok"] = False
        out["checks"]["rag_seq2seq"] = {"ok": False, "error": traceback.format_exc()}
    finally:
        eng2.cleanup()

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TurboQuant backend smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    result = run_smoke(args.max_new_tokens, args.top_k, args.bits)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
