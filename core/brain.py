from __future__ import annotations

import logging
import re
import time
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

        if getattr(self.settings, "second_brain_enabled", True):
            from core.second_brain.entity_store import EntityStore
            from core.second_brain.llm_extractor import LLMEntityExtractor
            from core.second_brain.planner import IntentRouter
            from core.second_brain.proactive import ProactiveEngine

            self._entity_store = EntityStore(self.memory.db_path)
            self._router = IntentRouter()
            self._proactive = ProactiveEngine(self._entity_store, self.settings)
            # Initialized after warmup so tokenizer is ready
            self._llm_extractor: LLMEntityExtractor | None = None

        self._warmup()

    def _warmup(self) -> None:
        prompt = self._build_prompt("Hello", text_mode=False)
        self._runtime.warmup(prompt, max_tokens=5)
        # Tokenizer is ready after warmup — initialize LLM extractor now
        if hasattr(self, "_llm_extractor") and self._llm_extractor is None:
            from core.second_brain.llm_extractor import LLMEntityExtractor
            self._llm_extractor = LLMEntityExtractor(self._runtime, self._runtime.tokenizer)

    def _system_prompt(self, *, text_mode: bool = False) -> str:
        return build_system_prompt(self.settings.user_name, text_mode=text_mode)

    def _collect_recent_turns(self) -> list[tuple[str, str]]:
        return collect_recent_turns(
            memory=self.memory,
            live_turns=self._recent_turns,
            limit=max(0, self.settings.recent_turns),
        )

    def _build_messages(
        self,
        user_input: str,
        *,
        text_mode: bool,
        route: Any = None,
    ) -> list[dict[str, str]]:
        recent_turns = self._collect_recent_turns()
        user_messages = [user_text for user_text, _ in recent_turns if user_text]
        user_messages.append(user_input)
        policy_note = _derive_policy_note(user_messages)

        entity_context = ""
        if hasattr(self, "_entity_store"):
            if route is not None and route.inject_entities:
                # Full context: tasks + commitments + profile + preferences
                entity_context = self._entity_store.recall_for_prompt(
                    char_budget=getattr(self.settings, "entity_recall_budget", 400)
                )
            else:
                # Always inject profile + preferences for personal coherence
                entity_context = self._entity_store.recall_personal_context()

        return build_messages(
            user_input=user_input,
            user_name=self.settings.user_name,
            text_mode=text_mode,
            memory=self.memory,
            recent_turns=recent_turns,
            policy_note=policy_note,
            entity_context=entity_context,
        )

    def _build_prompt(self, user_input: str, *, text_mode: bool, route: Any = None) -> str:
        return apply_chat_template(
            self._runtime.tokenizer,
            self._build_messages(user_input, text_mode=text_mode, route=route),
        )

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

    def _stream_sentences(self, user_input: str, *, route: Any = None) -> Iterator[str]:
        prompt = self._build_prompt(user_input, text_mode=False, route=route)
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

    def _stream_text(self, user_input: str, *, route: Any = None) -> Iterator[str]:
        prompt = self._build_prompt(user_input, text_mode=True, route=route)
        yield from self._runtime.stream_raw(prompt)

    def _extract_and_store(self, user_input: str, exchange_id: str) -> None:
        if not self._llm_extractor:
            return
        t0 = time.perf_counter()
        try:
            entities = self._llm_extractor.extract(user_input)
            for entity in entities:
                self._entity_store.upsert(
                    entity.kind,
                    entity.key,
                    entity.value,
                    due_date=entity.due_date,
                    source_id=exchange_id,
                )
        except Exception:
            logger.exception("Entity extraction failed")
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug("Entity extraction completed in %.0fms", elapsed_ms)

    def _persist_exchange(self, user_input: str, reply: str, *, route: Any = None) -> None:
        if not reply:
            return
        if self.settings.recent_turns > 0:
            self._recent_turns.append((user_input, reply))
        try:
            exchange_id: str | None = self.memory.store_exchange(user_input, reply)
        except Exception:
            logger.exception("Failed to persist exchange")
            exchange_id = None

        if (
            exchange_id is not None
            and getattr(self.settings, "extraction_enabled", True)
            and hasattr(self, "_entity_store")
        ):
            self._extract_and_store(user_input, exchange_id)

    def think_stream(self, user_input: str) -> Iterator[str]:
        self._update_policy_state(user_input)
        deterministic = self._deterministic_response(user_input)
        if deterministic:
            reply = deterministic.strip()
            if reply:
                yield reply
                self._persist_exchange(user_input, reply)
            return

        route = self._router.classify(user_input) if hasattr(self, "_router") else None

        parts: list[str] = []
        for sentence in self._stream_sentences(user_input, route=route):
            parts.append(sentence)
            yield sentence

        reply = " ".join(parts).strip()
        self._persist_exchange(user_input, reply, route=route)

        if route is not None and route.proactive_ok and hasattr(self, "_proactive"):
            suggestion = self._proactive.suggest(reply, proactive_ok=route.proactive_ok)
            if suggestion:
                yield suggestion

    def think_text_stream(self, user_input: str) -> Iterator[str]:
        self._update_policy_state(user_input)
        deterministic = self._deterministic_response(user_input)
        if deterministic:
            reply = deterministic.strip()
            if reply:
                yield reply
                self._persist_exchange(user_input, reply)
            return

        route = self._router.classify(user_input) if hasattr(self, "_router") else None

        parts: list[str] = []
        for chunk in self._stream_text(user_input, route=route):
            parts.append(chunk)
            yield chunk

        reply = "".join(parts).strip()
        self._persist_exchange(user_input, reply, route=route)

        if route is not None and route.proactive_ok and hasattr(self, "_proactive"):
            suggestion = self._proactive.suggest(reply, proactive_ok=route.proactive_ok)
            if suggestion:
                yield suggestion
