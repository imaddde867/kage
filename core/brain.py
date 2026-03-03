from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime

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
        self.session = requests.Session()

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

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
        payload = {
            "model": self.settings.ollama_model,
            "messages": self._build_messages(user_input),
            "stream": True,
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
