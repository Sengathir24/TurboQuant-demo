"""
model_registry.py — All supported HuggingFace models
Cursor: add new models here. They auto-appear in the frontend dropdown.
"""

from pydantic import BaseModel
from typing import Optional

class ModelConfig(BaseModel):
    id: str
    name: str
    hf_id: str               # actual HuggingFace model string
    params: str              # human-readable size
    context_len: int         # max tokens
    vram_fp16_gb: float      # VRAM needed at fp16
    task: str                # "causal-lm" | "seq2seq"
    family: str              # GPT-2, LLaMA, etc.
    description: str
    recommended_bits: int    # best TurboQuant bits for this model
    tags: list[str]

MODEL_REGISTRY: dict[str, ModelConfig] = {

    # ── GPT-2 family ─────────────────────────────────────
    "gpt2": ModelConfig(
        id="gpt2", name="GPT-2 Small", hf_id="gpt2",
        params="124M", context_len=1024, vram_fp16_gb=0.5,
        task="causal-lm", family="GPT-2",
        description="Fastest demo model. Good for CPU.",
        recommended_bits=2,
        tags=["fast","cpu-ok","demo"]
    ),
    "gpt2-medium": ModelConfig(
        id="gpt2-medium", name="GPT-2 Medium", hf_id="gpt2-medium",
        params="345M", context_len=1024, vram_fp16_gb=1.2,
        task="causal-lm", family="GPT-2",
        description="Good balance of speed and quality for KV cache demo.",
        recommended_bits=3,
        tags=["recommended","kv-demo","attention"]
    ),
    "gpt2-large": ModelConfig(
        id="gpt2-large", name="GPT-2 Large", hf_id="gpt2-large",
        params="774M", context_len=1024, vram_fp16_gb=2.8,
        task="causal-lm", family="GPT-2",
        description="Larger GPT-2. Needs GPU.",
        recommended_bits=3,
        tags=["gpu","quality"]
    ),

    # ── FLAN-T5 family ───────────────────────────────────
    "flan-t5-small": ModelConfig(
        id="flan-t5-small", name="FLAN-T5 Small", hf_id="google/flan-t5-small",
        params="80M", context_len=512, vram_fp16_gb=0.3,
        task="seq2seq", family="FLAN-T5",
        description="Best for RAG demos. Instruction-tuned.",
        recommended_bits=2,
        tags=["rag","cpu-ok","instruction-tuned"]
    ),
    "flan-t5-base": ModelConfig(
        id="flan-t5-base", name="FLAN-T5 Base", hf_id="google/flan-t5-base",
        params="250M", context_len=512, vram_fp16_gb=0.9,
        task="seq2seq", family="FLAN-T5",
        description="Recommended for RAG + QA demos.",
        recommended_bits=3,
        tags=["rag","recommended","instruction-tuned"]
    ),
    "flan-t5-large": ModelConfig(
        id="flan-t5-large", name="FLAN-T5 Large", hf_id="google/flan-t5-large",
        params="780M", context_len=512, vram_fp16_gb=2.8,
        task="seq2seq", family="FLAN-T5",
        description="Best quality RAG. Needs GPU.",
        recommended_bits=3,
        tags=["rag","gpu","quality"]
    ),

    # ── OPT family ───────────────────────────────────────
    "opt-125m": ModelConfig(
        id="opt-125m", name="OPT 125M", hf_id="facebook/opt-125m",
        params="125M", context_len=2048, vram_fp16_gb=0.5,
        task="causal-lm", family="OPT",
        description="Meta OPT. Longer context than GPT-2.",
        recommended_bits=2,
        tags=["fast","cpu-ok","long-context"]
    ),
    "opt-350m": ModelConfig(
        id="opt-350m", name="OPT 350M", hf_id="facebook/opt-350m",
        params="350M", context_len=2048, vram_fp16_gb=1.3,
        task="causal-lm", family="OPT",
        description="Good for long-context KV cache demo.",
        recommended_bits=3,
        tags=["long-context","kv-demo"]
    ),
    "opt-1.3b": ModelConfig(
        id="opt-1.3b", name="OPT 1.3B", hf_id="facebook/opt-1.3b",
        params="1.3B", context_len=2048, vram_fp16_gb=4.8,
        task="causal-lm", family="OPT",
        description="1B-scale model. Needs T4 GPU.",
        recommended_bits=3,
        tags=["gpu","1b-scale","long-context"]
    ),

    # ── DistilGPT2 ───────────────────────────────────────
    "distilgpt2": ModelConfig(
        id="distilgpt2", name="DistilGPT-2", hf_id="distilgpt2",
        params="82M", context_len=1024, vram_fp16_gb=0.3,
        task="causal-lm", family="GPT-2",
        description="Smallest model. Perfect for quick CPU demos.",
        recommended_bits=2,
        tags=["tiny","cpu-ok","fast"]
    ),

    # ── Pythia family ────────────────────────────────────
    "pythia-160m": ModelConfig(
        id="pythia-160m", name="Pythia 160M", hf_id="EleutherAI/pythia-160m",
        params="160M", context_len=2048, vram_fp16_gb=0.6,
        task="causal-lm", family="Pythia",
        description="EleutherAI Pythia. Good for research demos.",
        recommended_bits=2,
        tags=["research","cpu-ok","long-context"]
    ),
    "pythia-410m": ModelConfig(
        id="pythia-410m", name="Pythia 410M", hf_id="EleutherAI/pythia-410m",
        params="410M", context_len=2048, vram_fp16_gb=1.5,
        task="causal-lm", family="Pythia",
        description="Mid-size Pythia. Recommended for benchmarks.",
        recommended_bits=3,
        tags=["research","benchmark","long-context"]
    ),

    # ── Bloom ────────────────────────────────────────────
    "bloom-560m": ModelConfig(
        id="bloom-560m", name="BLOOM 560M", hf_id="bigscience/bloom-560m",
        params="560M", context_len=2048, vram_fp16_gb=2.1,
        task="causal-lm", family="BLOOM",
        description="Multilingual model. Great for diverse text demos.",
        recommended_bits=3,
        tags=["multilingual","gpu","diverse"]
    ),
}
