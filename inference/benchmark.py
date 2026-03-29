"""NeuroCache inference benchmark.

Run from the kage/ root:
    python -m inference.benchmark

Expected on M4 + Qwen3.5-9B-4bit:
  Load time:       8–12 s
  Inference speed: 15–25 tok/s
  RAM (model):     +~4.5 GB

Exits non-zero if any hard threshold is missed.
"""
from __future__ import annotations

import sys
import time
import os

# Ensure the kage root is on the path when running as __main__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0.0


def _fmt(label: str, value: str, width: int = 22) -> str:
    return f"  {label:<{width}} {value}"


def benchmark(model_id: str | None = None) -> bool:
    """Run the benchmark. Returns True if all thresholds pass."""
    from inference.local_llm import LocalLLM, DEFAULT_MODEL_ID

    model = model_id or DEFAULT_MODEL_ID
    llm = LocalLLM(model)

    # ── Model load ───────────────────────────────────────────────────
    ram_before = _ram_gb()
    t0 = time.perf_counter()
    llm.load()
    load_time = time.perf_counter() - t0
    ram_after_load = _ram_gb()
    model_ram = ram_after_load - ram_before

    # ── Inference ────────────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": (
                "Summarize the key ideas behind transformer self-attention "
                "in exactly 200 words."
            ),
        }
    ]
    t1 = time.perf_counter()
    response = llm.chat(messages, max_tokens=300, stream=False)
    infer_time = time.perf_counter() - t1
    ram_after_infer = _ram_gb()
    kv_ram = ram_after_infer - ram_after_load

    stats = llm.last_stats
    tok_per_sec = stats.get("tok_per_sec", 0.0)
    tokens = stats.get("tokens", 0)

    # ── Report ───────────────────────────────────────────────────────
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  NeuroCache Benchmark — {llm.model_id}")
    print(sep)
    print(_fmt("Load time:", f"{load_time:.1f} s"))
    print(_fmt("Model RAM:", f"+{model_ram:.2f} GB"))
    print(_fmt("KV RAM delta:", f"+{kv_ram:.3f} GB"))
    print(_fmt("Inference speed:", f"{tok_per_sec:.1f} tok/s  ({tokens} tokens)"))
    print(_fmt("Inference time:", f"{infer_time:.1f} s"))
    print()
    print("  Response preview:")
    print(f"    {str(response)[:120]}...")
    print(sep)

    # ── Thresholds ───────────────────────────────────────────────────
    failures: list[str] = []

    if model_ram > 6.0:
        failures.append(f"Model RAM {model_ram:.2f} GB > 6.0 GB threshold")
    if tok_per_sec > 0 and tok_per_sec < 5.0:
        failures.append(f"Inference speed {tok_per_sec:.1f} tok/s < 5.0 tok/s threshold")

    if failures:
        print("\n  FAILED:")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        return False

    print("\n  All thresholds passed.\n")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuroCache inference benchmark")
    parser.add_argument("--model", default=None, help="HF model ID or local path (default: mlx-community/Qwen3.5-9B-MLX-4bit)")
    args = parser.parse_args()

    ok = benchmark(model_id=args.model)
    sys.exit(0 if ok else 1)
