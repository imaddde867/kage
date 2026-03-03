from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterator
from typing import Any

import config
from core.brain_generation import GenerationRuntime
from core.brain_guardrails import (
    derive_policy_note as _derive_policy_note,
    deterministic_response,
    update_policy_state,
)
from core.brain_prompting import apply_chat_template, build_messages, build_system_prompt, collect_recent_turns
from core.memory import MemoryStore

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")


class BrainService:
    def __init__(self, *, settings: config.Settings | None = None, memory: MemoryStore | None = None) -> None:
        self.settings = settings or config.get()
        self.memory = memory or MemoryStore()
        self._recent_turns: deque[tuple[str, str]] = deque(maxlen=max(0, self.settings.recent_turns))
        self._prefers_honesty = False
        self._prefers_forced_yes = False

        self._runtime = GenerationRuntime(settings=self.settings)
        self.last_stats = self._runtime.last_stats
        self._warmup()

    def _warmup(self) -> None:
        prompt = self._build_prompt("Hello", text_mode=False)
        self._runtime.warmup(prompt, max_tokens=5)

    def _system_prompt(self, *, text_mode: bool = False) -> str:
        return build_system_prompt(self.settings.user_name, text_mode=text_mode)

    def _collect_recent_turns(self) -> list[tuple[str, str]]:
        return collect_recent_turns(
            memory=self.memory,
            live_turns=self._recent_turns,
            limit=max(0, self.settings.recent_turns),
        )

    def _build_messages(self, user_input: str, *, text_mode: bool) -> list[dict[str, str]]:
        recent_turns = self._collect_recent_turns()
        user_messages = [user_text for user_text, _ in recent_turns if user_text]
        user_messages.append(user_input)
        policy_note = _derive_policy_note(user_messages)
        return build_messages(
            user_input=user_input,
            user_name=self.settings.user_name,
            text_mode=text_mode,
            memory=self.memory,
            recent_turns=recent_turns,
            policy_note=policy_note,
        )

    def _build_prompt(self, user_input: str, *, text_mode: bool) -> str:
        return apply_chat_template(self._runtime.tokenizer, self._build_messages(user_input, text_mode=text_mode))

    def _update_policy_state(self, user_input: str) -> None:
        self._prefers_honesty, self._prefers_forced_yes = update_policy_state(
            user_input,
            prefers_honesty=self._prefers_honesty,
            prefers_forced_yes=self._prefers_forced_yes,
        )

    def _deterministic_response(self, user_input: str) -> str | None:
        return deterministic_response(
            user_input,
            prefers_honesty=self._prefers_honesty,
            prefers_forced_yes=self._prefers_forced_yes,
        )

    def _stream_sentences(self, user_input: str) -> Iterator[str]:
        prompt = self._build_prompt(user_input, text_mode=False)
        buffer = ""
        for text in self._runtime.stream_raw(prompt):
            buffer += text
            parts = _SENTENCE_END.split(buffer)
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buffer = parts[-1]

        if buffer.strip():
            yield buffer.strip()

    def _stream_text(self, user_input: str) -> Iterator[str]:
        prompt = self._build_prompt(user_input, text_mode=True)
        yield from self._runtime.stream_raw(prompt)

    def _persist_exchange(self, user_input: str, reply: str) -> None:
        if not reply:
            return
        if self.settings.recent_turns > 0:
            self._recent_turns.append((user_input, reply))
        try:
            self.memory.store_exchange(user_input, reply)
        except Exception:
            logger.exception("Failed to persist exchange")

    def think_stream(self, user_input: str) -> Iterator[str]:
        self._update_policy_state(user_input)
        deterministic = self._deterministic_response(user_input)
        if deterministic:
            reply = deterministic.strip()
            if reply:
                yield reply
                self._persist_exchange(user_input, reply)
            return

        parts: list[str] = []
        for sentence in self._stream_sentences(user_input):
            parts.append(sentence)
            yield sentence
        self._persist_exchange(user_input, " ".join(parts).strip())

    def think_text_stream(self, user_input: str) -> Iterator[str]:
        self._update_policy_state(user_input)
        deterministic = self._deterministic_response(user_input)
        if deterministic:
            reply = deterministic.strip()
            if reply:
                yield reply
                self._persist_exchange(user_input, reply)
            return

        parts: list[str] = []
        for chunk in self._stream_text(user_input):
            parts.append(chunk)
            yield chunk
        self._persist_exchange(user_input, "".join(parts).strip())

