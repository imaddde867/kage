from __future__ import annotations

import json
import logging
import re
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
- Concise by default. Elaborate only when asked.

Today is {date}.
"""


class BrainService:
    def __init__(self, *, settings: config.Settings | None = None, memory: MemoryStore | None = None) -> None:
        self.settings = settings or config.get()
        self.memory = memory or MemoryStore()

        if self.settings.llm_backend == "mlx":
            self._init_mlx()
        else:
            self.session = requests.Session()

    def _init_mlx(self) -> None:
        logger.info("Loading MLX model %s (Apple Neural Engine + GPU)…", self.settings.mlx_model)
        from mlx_lm import load, stream_generate  # type: ignore[import]
        self._mlx_model, self._mlx_tokenizer = load(self.settings.mlx_model)
        self._mlx_stream_generate = stream_generate
        logger.info("MLX model loaded.")

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

    def _stream_sentences_mlx(self, user_input: str) -> Iterator[str]:
        messages = self._build_messages(user_input)
        prompt: str = self._mlx_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        buffer = ""
        for chunk in self._mlx_stream_generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=prompt,
            max_tokens=self.settings.mlx_max_tokens,
        ):
            # mlx-lm >= 0.19 yields GenerationResponse with .text; older yields str
            text: str = chunk.text if hasattr(chunk, "text") else str(chunk)
            buffer += text
            parts = _SENTENCE_END.split(buffer)
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buffer = parts[-1]
        if buffer.strip():
            yield buffer.strip()

    def _stream_sentences_ollama(self, user_input: str) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": self._build_messages(user_input),
            "stream": True,
            "options": {
                "num_gpu": 99,   # offload all layers to Metal GPU
                "f16_kv": True,  # fp16 KV cache — faster, lower VRAM
            },
        }
        url = f"{self.settings.ollama_base_url}/api/chat"
        resp = self.session.post(url, json=payload, timeout=self.settings.ollama_timeout_seconds, stream=True)
        resp.raise_for_status()

        buffer = ""
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
                break
        if buffer.strip():
            yield buffer.strip()

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
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
