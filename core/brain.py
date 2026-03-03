from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

import requests

import config
from connectors import ConnectorManager, get_default_manager
from core.memory import MemoryStore, get_default_store

logger = logging.getLogger(__name__)

_LIVE_CONTEXT_HINTS = {
    "calendar", "event", "events",
    "meeting", "meetings",
    "schedule", "scheduled",
    "reminder", "reminders",
    "todo", "todos", "task", "tasks",
    "note", "notes",
    "today", "tomorrow",
    "project", "projects",
}

# Matches a sentence boundary: punctuation followed by whitespace, or at end of string.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

_SYSTEM_PROMPT = """You are Kage (影), a personal AI assistant for {name}.

Your name means "Shadow" in Japanese — you are always present, always aware, working quietly in the background of {name}'s life.

You are not a generic assistant. You know {name}'s life, goals, habits, and context. \
You are part life coach, part best friend, part technical collaborator.

Personality:
- Direct and honest, but warm. Never robotic or corporate.
- Proactive: if you notice something relevant from the context below, bring it up naturally.
- Conversational: your responses are spoken aloud. No bullet points, no markdown, no lists. \
  Plain natural speech only.
- Concise by default. Elaborate only when asked.
- Smart enough to push back gently when {name} is making a questionable call.

Rules:
- Speak like a real person talking, not like a document being read.
- Use live context and memory naturally — never announce that you're referencing them.
- Today is {date}.
"""


class BrainService:
    def __init__(
        self,
        *,
        settings: config.Settings | None = None,
        memory_store: MemoryStore | None = None,
        connector_manager: ConnectorManager | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or config.get_settings()
        self.memory_store = memory_store or get_default_store()
        self.connector_manager = connector_manager or get_default_manager()
        self.session = session or requests.Session()

    def build_system_prompt(self) -> str:
        today = datetime.now().strftime("%A, %B %d %Y at %H:%M")
        return _SYSTEM_PROMPT.format(name=self.settings.user_name, date=today)

    def collect_live_context(self) -> str:
        try:
            return self.connector_manager.get_all_context()
        except Exception:
            logger.exception("Failed to collect live connector context")
            return ""

    def collect_memory_context(self, query: str) -> str:
        try:
            return self.memory_store.recall(query)
        except Exception:
            logger.exception("Failed to recall memory for query")
            return ""

    @staticmethod
    def _needs_live_context(user_input: str) -> bool:
        tokens = {t.strip(".,!?;:()[]{}\"'").lower() for t in user_input.split()}
        return bool(tokens & _LIVE_CONTEXT_HINTS)

    def _build_payload(self, user_input: str, *, stream: bool = False) -> dict[str, Any]:
        system_prompt = self.build_system_prompt()

        if self._needs_live_context(user_input):
            live = self.collect_live_context()
            if live:
                system_prompt += f"\n\nLive context from your apps:\n{live}"

        memory = self.collect_memory_context(user_input)
        if memory:
            system_prompt += f"\n\nMemory context:\n{memory}"

        return {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "think": self.settings.ollama_think,
            "stream": stream,
        }

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
        """POST to Ollama, retrying without 'think' if the server rejects it."""
        url = f"{self.settings.ollama_base_url}/api/chat"
        resp = self.session.post(url, json=payload, timeout=self.settings.ollama_timeout_seconds, stream=stream)

        if resp.status_code == 400 and "think" in payload:
            try:
                error_text = (resp.json() if resp.content else {}).get("error", "")
            except ValueError:
                error_text = resp.text or ""
            lowered = str(error_text).lower()
            if "think" in lowered and any(t in lowered for t in ("unknown", "unsupported", "invalid")):
                logger.warning("Ollama does not support 'think'. Retrying without it.")
                payload = {k: v for k, v in payload.items() if k != "think"}
                resp = self.session.post(url, json=payload, timeout=self.settings.ollama_timeout_seconds, stream=stream)

        resp.raise_for_status()
        return resp

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
        payload = self._build_payload(user_input, stream=True)
        buffer = ""
        for raw_line in self._post(payload, stream=True).iter_lines():
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

    def think_stream(self, user_input: str, on_sentence: Callable[[str], None]) -> str:
        """Stream the LLM response sentence by sentence, calling on_sentence() for each.
        Falls back to a single non-streaming call if streaming fails."""
        try:
            parts: list[str] = []
            for sentence in self._stream_sentences(user_input):
                on_sentence(sentence)
                parts.append(sentence)
            reply = " ".join(parts)
        except Exception as exc:
            logger.warning("Streaming failed (%s), falling back to non-streaming", exc)
            reply = self.think(user_input)
            on_sentence(reply)
            return reply

        if reply:
            try:
                self.memory_store.store_exchange(user_input, reply)
            except Exception:
                logger.exception("Failed to persist conversation exchange")

        return reply

    def think(self, user_input: str) -> str:
        payload = self._build_payload(user_input)
        try:
            resp = self._post(payload)
            data = resp.json()
            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError("Invalid Ollama response: missing message object")
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Invalid Ollama response: empty content")
            reply = content.strip()
        except requests.exceptions.ConnectionError:
            return "I can't reach Ollama. Make sure it's running."
        except requests.exceptions.Timeout:
            return "Ollama took too long to respond. Try again in a second."
        except requests.exceptions.RequestException:
            logger.exception("Ollama request failed")
            return "Something went wrong while talking to Ollama."
        except Exception:
            logger.exception("Unexpected brain error")
            return "Something went wrong."

        try:
            self.memory_store.store_exchange(user_input, reply)
        except Exception:
            logger.exception("Failed to persist conversation exchange")

        return reply


_DEFAULT_BRAIN: BrainService | None = None


def get_default_brain() -> BrainService:
    global _DEFAULT_BRAIN
    if _DEFAULT_BRAIN is None:
        _DEFAULT_BRAIN = BrainService()
    return _DEFAULT_BRAIN


def think(user_input: str) -> str:
    return get_default_brain().think(user_input)


__all__ = ["BrainService", "get_default_brain", "think"]
