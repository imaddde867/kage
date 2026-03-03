from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

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

Grounding rules:
- Only state facts that are explicitly in this conversation or in Memory. Never invent, assume, or extrapolate.
- If you don't know something, say so plainly. Do not guess.

Today is {date}.
"""


class BrainService:
    def __init__(self, *, settings: config.Settings | None = None, memory: MemoryStore | None = None) -> None:
        self.settings = settings or config.get()
        self.memory = memory or MemoryStore()
        self.last_stats: dict[str, Any] = {}
        self._backend = self.settings.llm_backend.strip().lower()

        if self._backend == "mlx_vlm":
            self._init_mlx_vlm()
        elif self._backend == "mlx":
            self._init_mlx()
        else:
            raise ValueError(
                f"Unsupported LLM_BACKEND '{self.settings.llm_backend}'. "
                "Use one of: mlx_vlm, mlx."
            )

    # ── MLX-VLM (Qwen3.5 and other VLMs — text-only inference) ──

    def _init_mlx_vlm(self) -> None:
        from mlx_vlm import stream_generate  # type: ignore[import]
        from mlx_vlm.tokenizer_utils import load_tokenizer as load_detokenizer  # type: ignore[import]
        from mlx_vlm.utils import StoppingCriteria  # type: ignore[import]
        from mlx_vlm.utils import get_model_path, load_model  # type: ignore[import]
        from transformers import AutoTokenizer

        model_path = get_model_path(self.settings.mlx_model)
        print(f"  Loading {self.settings.mlx_model}…", flush=True)
        model = load_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        detokenizer_class = load_detokenizer(model_path, return_tokenizer=False)
        tokenizer.detokenizer = detokenizer_class(tokenizer)
        eos_ids = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        tokenizer.stopping_criteria = StoppingCriteria(eos_ids, tokenizer)

        self._vlm_model = model
        self._tokenizer = tokenizer
        self._vlm_processor = tokenizer
        self._vlm_stream = stream_generate

        print("  Warming up (compiling MLX graphs)…", flush=True)
        warmup = self._apply_chat_template(self._build_messages("Hello"))
        for _ in self._vlm_stream(
            self._vlm_model,
            self._vlm_processor,
            prompt=warmup,
            max_tokens=5,
        ):
            pass
        print("  Ready.\n", flush=True)

    # ── MLX-LM (text-only models) ────────────────────────────────────────────────

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
        for _ in self._mlx_stream(
            self._mlx_model,
            self._tokenizer,
            prompt=warmup,
            max_tokens=5,
            draft_model=self._mlx_draft_model,
        ):
            pass
        print("  Ready.\n", flush=True)

    def _stream_sentences_mlx_vlm(self, user_input: str) -> Iterator[str]:
        prompt = self._apply_chat_template(self._build_messages(user_input))
        gen_iter = iter(
            self._vlm_stream(
                self._vlm_model,
                self._vlm_processor,
                prompt=prompt,
                max_tokens=self.settings.mlx_max_tokens,
                temperature=self.settings.temperature,
            )
        )
        yield from self._stream_chunked_sentences(gen_iter, backend="mlx_vlm")

    def _stream_sentences_mlx(self, user_input: str) -> Iterator[str]:
        prompt = self._apply_chat_template(self._build_messages(user_input))
        gen_iter = iter(
            self._mlx_stream(
                self._mlx_model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.settings.mlx_max_tokens,
                draft_model=self._mlx_draft_model,
                temp=self.settings.temperature,
            )
        )
        yield from self._stream_chunked_sentences(gen_iter, backend="mlx")

    def _stream_chunked_sentences(self, gen_iter: Iterator[Any], *, backend: str) -> Iterator[str]:
        buffer = ""
        total_tokens = 0
        pure_gen_s = 0.0

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
            "backend": backend,
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

    # ── Dispatch ──────────────────────────────────────────────────────────────────

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
        if self._backend == "mlx_vlm":
            return self._stream_sentences_mlx_vlm(user_input)
        if self._backend == "mlx":
            return self._stream_sentences_mlx(user_input)
        raise RuntimeError(f"Unsupported LLM backend: {self._backend}")

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
