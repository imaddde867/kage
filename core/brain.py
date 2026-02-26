from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

import config
from connectors import ConnectorManager, get_default_manager
from core.memory import MemoryStore, get_default_store

logger = logging.getLogger(__name__)

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

    def build_payload(self, user_input: str) -> dict[str, Any]:
        system_prompt = self.build_system_prompt()

        live_context = self.collect_live_context()
        if live_context:
            system_prompt += f"\n\nLive context from your apps:\n{live_context}"

        memory_context = self.collect_memory_context(user_input)
        if memory_context:
            system_prompt += f"\n\nMemory context:\n{memory_context}"

        return {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "stream": False,
        }

    def call_ollama(self, payload: dict[str, Any]) -> str:
        response = self.session.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError("Invalid Ollama response: missing message object")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Invalid Ollama response: empty message content")

        return content.strip()

    def think(self, user_input: str) -> str:
        payload = self.build_payload(user_input)

        try:
            reply = self.call_ollama(payload)
        except requests.exceptions.ConnectionError:
            return "I can't reach Ollama. Make sure it's running — just type: ollama serve"
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out after %ss", self.settings.ollama_timeout_seconds)
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
    """
    Send user input to Ollama with live connector context + memory injected.
    Returns Kage's response as plain text.
    """
    return get_default_brain().think(user_input)


__all__ = ["BrainService", "get_default_brain", "think"]
