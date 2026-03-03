from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import requests

import config
from core.memory import MemoryStore

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

_SYSTEM_PROMPT = """You are Kage (影), a personal AI assistant for {name}.

Your name means "Shadow" in Japanese — always present, always aware, working quietly in the background.

Personality:
- Direct and honest, but warm. Never robotic or corporate.
- Conversational: responses are spoken aloud. No bullet points, no markdown, no lists. Plain natural speech only.
- Concise by default: 2–3 sentences maximum. If more detail is needed, give the key point and ask if they want more.

Today is {date}.
"""


class BrainService:
    def __init__(self, *, settings: config.Settings | None = None, memory: MemoryStore | None = None) -> None:
        self.settings = settings or config.get()
        self.memory = memory or MemoryStore()
        self.last_stats: dict = {}

        if self.settings.llm_backend == "mlx_vlm":
            self._init_mlx_vlm()
        elif self.settings.llm_backend == "mlx":
            self._init_mlx()
        else:
            self.session = requests.Session()

    # ── MLX-VLM (Qwen3.5 and other VLMs — text-only inference, no image needed) ──

    def _init_mlx_vlm(self) -> None:
        # Load model weights and tokenizer separately — bypasses the broken
        # AutoVideoProcessor.from_pretrained path in transformers (torchvision absent
        # + Qwen3.5 video processor class not yet in transformers' registry = crash).
        # stream_generate works fine with a plain tokenizer + detokenizer for text-only
        # inference (no images/video), which is all we need for voice.
        from mlx_vlm.utils import load_model, get_model_path  # type: ignore[import]
        from mlx_vlm.tokenizer_utils import load_tokenizer as _load_det  # type: ignore[import]
        from mlx_vlm.utils import StoppingCriteria  # type: ignore[import]
        from mlx_vlm import stream_generate  # type: ignore[import]
        from transformers import AutoTokenizer

        model_path = get_model_path(self.settings.mlx_model)
        print(f"  Loading {self.settings.mlx_model}…", flush=True)
        model = load_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Attach the detokenizer and stopping criteria that stream_generate expects
        detokenizer_class = _load_det(model_path, return_tokenizer=False)
        tokenizer.detokenizer = detokenizer_class(tokenizer)
        # Prefer tokenizer's eos_token_id; model.config may be None for new architectures
        eos_ids = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        tokenizer.stopping_criteria = StoppingCriteria(eos_ids, tokenizer)

        self._vlm_model = model
        self._tokenizer = tokenizer
        self._vlm_processor = tokenizer  # stream_generate uses processor directly for text
        self._vlm_stream = stream_generate

        self._mlx_draft_model = None  # speculative decoding not supported in mlx_vlm

        print("  Warming up (compiling MLX graphs)…", flush=True)
        warmup = self._apply_chat_template(self._build_messages("Hello"))
        for _ in self._vlm_stream(self._vlm_model, self._vlm_processor,
                                   prompt=warmup, max_tokens=5):
            pass
        print("  Ready.\n", flush=True)

    def _stream_sentences_mlx_vlm(self, user_input: str) -> Iterator[str]:
        prompt = self._apply_chat_template(self._build_messages(user_input))
        buffer = ""
        total_tokens = 0
        pure_gen_s = 0.0
        gen_iter = iter(self._vlm_stream(
            self._vlm_model,
            self._vlm_processor,
            prompt=prompt,
            max_tokens=self.settings.mlx_max_tokens,
        ))
        while True:
            t_tok = time.perf_counter()
            try:
                chunk = next(gen_iter)
            except StopIteration:
                break
            pure_gen_s += time.perf_counter() - t_tok
            text: str = chunk.text if hasattr(chunk, "text") else str(chunk)
            total_tokens = getattr(chunk, "generation_tokens", total_tokens + 1)
            buffer += text
            parts = _SENTENCE_END.split(buffer)
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buffer = parts[-1]
        if buffer.strip():
            yield buffer.strip()
        self.last_stats = {
            "backend": "mlx_vlm",
            "tokens": total_tokens,
            "gen_seconds": pure_gen_s,
            "tok_per_sec": total_tokens / pure_gen_s if pure_gen_s > 0 else 0,
        }

    # ── MLX-LM (text-only models — Qwen3, Qwen2.5, Llama, etc.) ─────────────────

    def _init_mlx(self) -> None:
        from mlx_lm import load, stream_generate  # type: ignore[import]

        print(f"  Loading {self.settings.mlx_model}…", flush=True)
        model, tokenizer = load(self.settings.mlx_model)
        self._mlx_model = model
        self._tokenizer = tokenizer
        self._mlx_stream = stream_generate

        self._mlx_draft_model = None
        if self.settings.mlx_draft_model:
            print(f"  Loading draft model {self.settings.mlx_draft_model}…", flush=True)
            draft_model, _ = load(self.settings.mlx_draft_model)
            self._mlx_draft_model = draft_model

        print("  Warming up (compiling MLX graphs)…", flush=True)
        warmup = self._apply_chat_template(self._build_messages("Hello"))
        for _ in self._mlx_stream(self._mlx_model, self._tokenizer,
                                   prompt=warmup, max_tokens=5,
                                   draft_model=self._mlx_draft_model):
            pass
        print("  Ready.\n", flush=True)

    def _stream_sentences_mlx(self, user_input: str) -> Iterator[str]:
        prompt = self._apply_chat_template(self._build_messages(user_input))
        buffer = ""
        total_tokens = 0
        pure_gen_s = 0.0
        gen_iter = iter(self._mlx_stream(
            self._mlx_model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.settings.mlx_max_tokens,
            draft_model=self._mlx_draft_model,
        ))
        while True:
            t_tok = time.perf_counter()
            try:
                chunk = next(gen_iter)
            except StopIteration:
                break
            pure_gen_s += time.perf_counter() - t_tok
            text: str = chunk.text if hasattr(chunk, "text") else str(chunk)
            total_tokens = getattr(chunk, "generation_tokens", total_tokens + 1)
            buffer += text
            parts = _SENTENCE_END.split(buffer)
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buffer = parts[-1]
        if buffer.strip():
            yield buffer.strip()
        self.last_stats = {
            "backend": "mlx",
            "tokens": total_tokens,
            "gen_seconds": pure_gen_s,
            "tok_per_sec": total_tokens / pure_gen_s if pure_gen_s > 0 else 0,
        }

    # ── Shared helpers ────────────────────────────────────────────────────────────

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template. Passes enable_thinking=False for Qwen3/Qwen3.5."""
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3/Qwen3.5: skip <think>…</think> for voice
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _system_prompt(self) -> str:
        today = datetime.now().strftime("%A, %B %d %Y at %H:%M")
        return _SYSTEM_PROMPT.format(name=self.settings.user_name, date=today)

    def _build_messages(self, user_input: str) -> list[dict]:
        system = self._system_prompt()
        memory_ctx = self.memory.recall(user_input)
        if memory_ctx:
            system += f"\n\nMemory:\n{memory_ctx}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]

    # ── Ollama ────────────────────────────────────────────────────────────────────

    def _stream_sentences_ollama(self, user_input: str) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": self._build_messages(user_input),
            "stream": True,
            "options": {
                "num_gpu": 99,   # offload all layers to Metal GPU
                "f16_kv": True,  # fp16 KV cache — faster, lower memory
            },
        }
        url = f"{self.settings.ollama_base_url}/api/chat"
        resp = self.session.post(url, json=payload, timeout=self.settings.ollama_timeout_seconds, stream=True)
        resp.raise_for_status()

        buffer = ""
        final_data: dict = {}
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            buffer += data.get("message", {}).get("content", "")
            parts = _SENTENCE_END.split(buffer)
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buffer = parts[-1]
            if data.get("done"):
                final_data = data
                break
        if buffer.strip():
            yield buffer.strip()
        gen_tokens = final_data.get("eval_count", 0)
        gen_ns = final_data.get("eval_duration", 0)
        self.last_stats = {
            "backend": "ollama",
            "tokens": gen_tokens,
            "gen_seconds": gen_ns / 1e9 if gen_ns else 0,
            "tok_per_sec": (gen_tokens / (gen_ns / 1e9)) if gen_ns else 0,
        }

    # ── Dispatch ──────────────────────────────────────────────────────────────────

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
        if self.settings.llm_backend == "mlx_vlm":
            return self._stream_sentences_mlx_vlm(user_input)
        if self.settings.llm_backend == "mlx":
            return self._stream_sentences_mlx(user_input)
        return self._stream_sentences_ollama(user_input)

    def think_stream(self, user_input: str) -> Iterator[str]:
        """Stream LLM response sentence by sentence. Persists exchange when done."""
        parts: list[str] = []
        for sentence in self._stream_sentences(user_input):
            parts.append(sentence)
            yield sentence
        reply = " ".join(parts)
        if reply:
            try:
                self.memory.store_exchange(user_input, reply)
            except Exception:
                logger.exception("Failed to persist exchange")
