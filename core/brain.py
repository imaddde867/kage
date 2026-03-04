from __future__ import annotations

import logging
import re
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

import config
from core.agent.loop import AgentLoop
from core.agent.tool_registry import ToolRegistry
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

        # Agent loop and registry are initialized in _warmup() after the tokenizer
        # is ready.  Both start as None and are set to real instances only when
        # agent_enabled=True.  Code that uses them must check for None first.
        self._agent_loop: AgentLoop | None = None
        self._tool_registry: ToolRegistry | None = None

        self._warmup()

    def _warmup(self) -> None:
        prompt = self._build_prompt("Hello", text_mode=False)
        self._runtime.warmup(prompt, max_tokens=5)
        # Tokenizer is ready after warmup — initialize LLM extractor now
        if hasattr(self, "_llm_extractor") and self._llm_extractor is None:
            from core.second_brain.llm_extractor import LLMEntityExtractor
            self._llm_extractor = LLMEntityExtractor(self._runtime, self._runtime.tokenizer)
        # Initialize the agent loop after warmup so the tokenizer is available.
        # _build_tool_registry() assembles all connectors that are installed.
        if self.settings.agent_enabled and self._agent_loop is None:
            self._tool_registry = self._build_tool_registry()
            self._agent_loop = AgentLoop(
                runtime=self._runtime,
                tokenizer=self._runtime.tokenizer,
                registry=self._tool_registry,
                settings=self.settings,
            )

    def _build_tool_registry(self) -> ToolRegistry:
        """Assemble a ToolRegistry from all available connectors.

        Core connectors (always registered — no extra dependencies):
            web_search, notify, speak

        Memory connectors (registered only when second_brain is enabled —
        they need a live EntityStore):
            mark_task_done, update_fact, list_open_tasks

        Optional connectors (registered when extra packages are installed;
        silently skipped with a debug log if the import fails):
            web_fetch         — requires httpx + trafilatura
            shell             — no extra deps, always available
            calendar_read     — macOS only (osascript)
            reminder_add      — macOS only (osascript)

        To add a new tool: import it here and call registry.register().
        """
        registry = ToolRegistry()

        # --- Core connectors (always available) ---
        from connectors.web_search import WebSearchTool
        from connectors.notify import NotifyTool, SpeakTool
        registry.register(WebSearchTool())
        registry.register(NotifyTool())
        registry.register(SpeakTool())

        # --- Memory connectors (requires second_brain EntityStore) ---
        if hasattr(self, "_entity_store"):
            from connectors.memory_ops import MarkTaskDoneTool, UpdateFactTool, ListOpenTasksTool
            db_path = self.memory.db_path
            registry.register(MarkTaskDoneTool(db_path))
            registry.register(UpdateFactTool(db_path))
            registry.register(ListOpenTasksTool(db_path))

        # --- Optional connectors (gracefully skipped if deps missing) ---
        try:
            from connectors.web_fetch import WebFetchTool
            registry.register(WebFetchTool())
        except ImportError:
            logger.debug("web_fetch unavailable (install httpx + trafilatura to enable)")

        try:
            from connectors.shell import ShellTool
            registry.register(ShellTool())
        except ImportError:
            logger.debug("shell connector unavailable")

        try:
            from connectors.apple_calendar import CalendarReadTool, ReminderAddTool
            registry.register(CalendarReadTool())
            registry.register(ReminderAddTool())
        except ImportError:
            logger.debug("apple_calendar connector unavailable")

        return registry

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

    def _routing_prompt(self, user_input: str) -> str:
        """Build the chat-template prompt for the tool-need routing classifier.

        The system message instructs the model to answer with a single word
        ("yes" or "no") so we can cap generation at 8 tokens and keep the
        routing call fast.
        """
        system = (
            "You are a routing classifier. Answer with a single word: 'yes' or 'no'.\n"
            "Output 'yes' if the request requires using external tools such as web search, "
            "calendar, shell commands, or memory operations.\n"
            "Output 'no' if it is a simple conversational question answerable from knowledge alone."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        return apply_chat_template(self._runtime.tokenizer, messages)

    def _needs_tools(self, user_input: str) -> bool:
        """Ask the LLM whether this request requires tool use.

        Caps generation at 8 tokens (enough for "yes" or "no" plus any BOS/EOS
        tokens) to minimise added latency.  track_stats=False avoids overwriting
        brain.last_stats, which is used by the timing display in main.py.

        Returns True when the LLM's answer starts with "yes" (case-insensitive).
        Any other output (including empty strings) is treated as False so the
        fast conversational path is used by default.
        """
        prompt = self._routing_prompt(user_input)
        answer = "".join(
            self._runtime.stream_raw(prompt, max_tokens=8, track_stats=False)
        ).strip().lower()
        return answer.startswith("yes")

    def agent_stream(self, user_input: str) -> Iterator[str]:
        """Run the agentic multi-step path for a single user request.

        Called by think_stream / think_text_stream when _needs_tools() returns
        True.  Assembles the entity context from EntityStore and delegates to
        AgentLoop.run(), then persists the final reply to memory.

        Yields string chunks exactly as think_stream does so callers
        (respond() in main.py) need no special handling for agent responses.
        """
        assert self._agent_loop is not None
        self._update_policy_state(user_input)
        entity_context = ""
        if hasattr(self, "_entity_store"):
            entity_context = self._entity_store.recall_for_prompt(
                char_budget=getattr(self.settings, "entity_recall_budget", 400)
            )
        parts: list[str] = []
        for chunk in self._agent_loop.run(user_input, context=entity_context):
            parts.append(chunk)
            yield chunk
        reply = "".join(parts).strip()
        self._persist_exchange(user_input, reply)

    def think_stream(self, user_input: str) -> Iterator[str]:
        # Routing check: if the agent is enabled and the LLM decides this
        # request needs tools, delegate to the agentic path immediately.
        # The fast conversational path continues below when tools are not needed.
        if self._agent_loop is not None and self._needs_tools(user_input):
            yield from self.agent_stream(user_input)
            return

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
        # Same routing check as think_stream — text mode also benefits from
        # the agentic path when tool use is needed.
        if self._agent_loop is not None and self._needs_tools(user_input):
            yield from self.agent_stream(user_input)
            return

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
