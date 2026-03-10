#!/usr/bin/env python3
"""
Kage model benchmark — real measured numbers on all available MLX + Ollama models.
Run with:  /Users/imadeddine/micromamba/envs/m4-ml/bin/python scripts/benchmark_models.py
"""
from __future__ import annotations

import gc
import json
import statistics
import subprocess
import time
import urllib.request
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MLX_MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
]

OLLAMA_MODELS = [
    "qwen3:8b",
    "qwen3.5:9b",
]

# Representative Kage prompts — short voice response (target: 40-80 tok output)
PROMPTS = [
    {
        "name": "factual",
        "user": "What is the capital of France?",
        "system": "You are Kage, a helpful voice assistant. Keep answers short and natural for speech. No markdown.",
        "max_tokens": 60,
    },
    {
        "name": "conversational",
        "user": "Set a reminder for my dentist appointment next Tuesday at 3pm.",
        "system": "You are Kage, a helpful voice assistant. Keep answers short and natural for speech. No markdown.",
        "max_tokens": 80,
    },
    {
        "name": "routing_8tok",  # simulates the 8-token routing classifier call
        "user": "Search for the latest Python 3.13 release notes",
        "system": "You are a routing classifier. Reply with exactly one word: 'yes' or 'no'.",
        "max_tokens": 8,
    },
]

REPEATS = 2  # how many times to run each prompt per model (averaged)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_row(label: str, value: str) -> None:
    print(f"  {label:<35} {value}")


# ──────────────────────────────────────────────────────────────────────────────
# MLX benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def bench_mlx_model(model_id: str) -> dict[str, Any]:
    from mlx_lm import load, stream_generate  # type: ignore

    _print_header(f"MLX: {model_id}")

    # ── Load ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_s = time.perf_counter() - t0
    _print_row("Cold load time", f"{load_s:.2f}s")

    # ── Warmup ────────────────────────────────────────────────────────────────
    warmup_prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    try:
        warmup_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        pass

    t_wu = time.perf_counter()
    for _ in stream_generate(model, tokenizer, prompt=warmup_prompt, max_tokens=5):
        pass
    warmup_s = time.perf_counter() - t_wu
    _print_row("Warmup (5 tok)", f"{warmup_s:.2f}s")

    results: dict[str, Any] = {
        "model": model_id,
        "backend": "mlx",
        "load_s": round(load_s, 2),
        "warmup_s": round(warmup_s, 2),
        "prompts": {},
    }

    # ── Per-prompt benchmarks ─────────────────────────────────────────────────
    for pdef in PROMPTS:
        pname = pdef["name"]
        max_tokens = pdef["max_tokens"]

        try:
            messages = [
                {"role": "system", "content": pdef["system"]},
                {"role": "user", "content": pdef["user"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # Suppress thinking for Qwen3 models
            if "Qwen3" in model_id or "qwen3" in model_id.lower():
                prompt += "<think>\n\n</think>\n"
        except Exception:
            prompt = f"<|im_start|>system\n{pdef['system']}<|im_end|>\n<|im_start|>user\n{pdef['user']}<|im_end|>\n<|im_start|>assistant\n"

        run_tps: list[float] = []
        run_ttft: list[float] = []
        run_texts: list[str] = []

        for _ in range(REPEATS):
            text_parts: list[str] = []
            ttft: float | None = None
            t_start = time.perf_counter()

            last_chunk: Any = None
            for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                text_parts.append(text)
                last_chunk = chunk

            t_end = time.perf_counter()
            elapsed = t_end - t_start

            gen_tps = 0.0
            if last_chunk is not None and hasattr(last_chunk, "generation_tps"):
                v = last_chunk.generation_tps
                if v:
                    gen_tps = float(v)
            if gen_tps == 0.0 and elapsed > 0:
                n_tok = len(text_parts)  # rough estimate
                if last_chunk is not None and hasattr(last_chunk, "generation_tokens"):
                    n_tok = int(last_chunk.generation_tokens)
                gen_tps = n_tok / elapsed

            run_tps.append(gen_tps)
            run_ttft.append(ttft or elapsed)
            run_texts.append("".join(text_parts).strip())

        avg_tps = round(statistics.mean(run_tps), 1)
        avg_ttft = round(statistics.mean(run_ttft) * 1000, 0)  # ms

        results["prompts"][pname] = {
            "tok_per_sec": avg_tps,
            "ttft_ms": avg_ttft,
            "sample_response": run_texts[-1][:120],
        }
        _print_row(
            f"  [{pname}] {max_tokens}tok | tps / TTFT",
            f"{avg_tps:.1f} tok/s  |  TTFT {avg_ttft:.0f}ms",
        )
        print(f"    Response: {run_texts[-1][:100]!r}")

    # cleanup
    del model
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore
        mx.metal.clear_cache()
    except Exception:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Ollama benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def _ollama_generate(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def bench_ollama_model(model_name: str) -> dict[str, Any]:
    _print_header(f"Ollama: {model_name}")

    results: dict[str, Any] = {
        "model": model_name,
        "backend": "ollama",
        "prompts": {},
    }

    load_measured = False
    for pdef in PROMPTS:
        pname = pdef["name"]
        max_tokens = pdef["max_tokens"]

        # Build a simple prompt string (Ollama handles chat templates internally)
        prompt_str = (
            f"<|im_start|>system\n{pdef['system']}<|im_end|>\n"
            f"<|im_start|>user\n{pdef['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Suppress thinking for Qwen3
        if "qwen3" in model_name.lower():
            prompt_str += "<think>\n\n</think>\n"

        run_tps: list[float] = []
        run_ttft: list[float] = []
        run_texts: list[str] = []
        first_load_s: float | None = None

        for i in range(REPEATS):
            try:
                r = _ollama_generate(model_name, prompt_str, max_tokens)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            ec = r.get("eval_count", 0)
            ed = r.get("eval_duration", 1)  # nanoseconds
            pd = r.get("prompt_eval_duration", 0)  # nanoseconds
            ld = r.get("load_duration", 0)  # nanoseconds

            tps = ec / ed * 1e9 if ed > 0 else 0.0
            # TTFT = load + prompt processing (model already warm after first call)
            ttft_ms = (ld + pd) / 1e6

            if i == 0 and not load_measured:
                first_load_s = ld / 1e9
                load_measured = True

            run_tps.append(tps)
            run_ttft.append(ttft_ms)
            run_texts.append(r.get("response", "").strip())

        if not run_tps:
            continue

        avg_tps = round(statistics.mean(run_tps), 1)
        avg_ttft = round(statistics.mean(run_ttft), 0)

        results["prompts"][pname] = {
            "tok_per_sec": avg_tps,
            "ttft_ms": avg_ttft,
            "sample_response": run_texts[-1][:120],
        }
        _print_row(
            f"  [{pname}] {max_tokens}tok | tps / TTFT",
            f"{avg_tps:.1f} tok/s  |  TTFT {avg_ttft:.0f}ms",
        )
        print(f"    Response: {run_texts[-1][:100]!r}")

    if first_load_s is not None:
        results["load_s"] = round(first_load_s, 2)
        _print_row("Load duration (cold)", f"{first_load_s:.2f}s")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Speculative decoding benchmark (9B + 4B draft)
# ──────────────────────────────────────────────────────────────────────────────

def bench_speculative(main_id: str, draft_id: str) -> dict[str, Any]:
    from mlx_lm import load, stream_generate  # type: ignore

    _print_header(f"Speculative: {main_id.split('/')[-1]} + {draft_id.split('/')[-1]}")

    t0 = time.perf_counter()
    main_model, tokenizer = load(main_id)
    draft_model, _ = load(draft_id)
    load_s = time.perf_counter() - t0
    _print_row("Load time (both models)", f"{load_s:.2f}s")

    # warmup
    warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for _ in stream_generate(main_model, tokenizer, prompt=warmup_prompt, max_tokens=5, draft_model=draft_model):
        pass
    _print_row("Warmed up", "ok")

    results: dict[str, Any] = {
        "model": f"{main_id}+draft:{draft_id}",
        "backend": "mlx_speculative",
        "load_s": round(load_s, 2),
        "prompts": {},
    }

    for pdef in PROMPTS:
        pname = pdef["name"]
        max_tokens = pdef["max_tokens"]

        messages = [
            {"role": "system", "content": pdef["system"]},
            {"role": "user", "content": pdef["user"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt += "<think>\n\n</think>\n"  # suppress thinking for Qwen3.5

        run_tps: list[float] = []
        run_ttft: list[float] = []
        run_texts: list[str] = []

        for _ in range(REPEATS):
            text_parts: list[str] = []
            ttft: float | None = None
            t_start = time.perf_counter()
            last_chunk: Any = None

            for chunk in stream_generate(
                main_model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, draft_model=draft_model
            ):
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                text_parts.append(chunk.text if hasattr(chunk, "text") else str(chunk))
                last_chunk = chunk

            elapsed = time.perf_counter() - t_start
            gen_tps = 0.0
            if last_chunk and hasattr(last_chunk, "generation_tps") and last_chunk.generation_tps:
                gen_tps = float(last_chunk.generation_tps)
            if gen_tps == 0.0 and elapsed > 0:
                n = int(last_chunk.generation_tokens) if last_chunk and hasattr(last_chunk, "generation_tokens") else len(text_parts)
                gen_tps = n / elapsed

            run_tps.append(gen_tps)
            run_ttft.append((ttft or elapsed) * 1000)
            run_texts.append("".join(text_parts).strip())

        avg_tps = round(statistics.mean(run_tps), 1)
        avg_ttft = round(statistics.mean(run_ttft), 0)
        results["prompts"][pname] = {
            "tok_per_sec": avg_tps,
            "ttft_ms": avg_ttft,
            "sample_response": run_texts[-1][:120],
        }
        _print_row(
            f"  [{pname}] {max_tokens}tok | tps / TTFT",
            f"{avg_tps:.1f} tok/s  |  TTFT {avg_ttft:.0f}ms",
        )
        print(f"    Response: {run_texts[-1][:100]!r}")

    del main_model, draft_model
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore
        mx.metal.clear_cache()
    except Exception:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict[str, Any]]) -> None:
    _print_header("SUMMARY — Kage model benchmark (M4 16GB)")
    print(f"  {'Model':<45} {'Backend':<12} {'Conv tps':>9} {'Conv TTFT':>10} {'Route tps':>10}")
    print(f"  {'-'*45} {'-'*12} {'-'*9} {'-'*10} {'-'*10}")
    for r in all_results:
        name = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
        backend = r["backend"]
        conv = r["prompts"].get("conversational", {})
        route = r["prompts"].get("routing_8tok", {})
        tps = conv.get("tok_per_sec", 0)
        ttft = conv.get("ttft_ms", 0)
        rtps = route.get("tok_per_sec", 0)
        print(f"  {name:<45} {backend:<12} {tps:>8.1f} {ttft:>9.0f}ms {rtps:>9.1f}")

    print()
    print("  Voice UX targets: conversational tps > 60, TTFT < 600ms")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results: list[dict[str, Any]] = []

    # Ollama models first (server already running, no env setup)
    print("\n>>> Benchmarking Ollama models...")
    for m in OLLAMA_MODELS:
        try:
            r = bench_ollama_model(m)
            all_results.append(r)
        except Exception as e:
            print(f"  SKIP {m}: {e}")

    # MLX models
    print("\n>>> Benchmarking MLX models...")
    for m in MLX_MODELS:
        try:
            r = bench_mlx_model(m)
            all_results.append(r)
        except Exception as e:
            print(f"  SKIP {m}: {e}")

    # Speculative decoding: 9B + 4B draft
    print("\n>>> Benchmarking speculative decoding (9B + 4B draft)...")
    try:
        r = bench_speculative(
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        all_results.append(r)
    except Exception as e:
        print(f"  SKIP speculative: {e}")

    # Save results
    out_path = "scripts/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
