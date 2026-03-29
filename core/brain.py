"""BrainService — the main entry point for all LLM requests.

Acts as a compatibility facade over RequestOrchestrator and the underlying
runtime components (GenerationRuntime, MemoryStore, AgentLoop, EntityStore).
Callers (app_runner, SessionController, heartbeat) interact only with this
class; nothing below it is part of the public API.

Public interface:
  think_stream(text)      — voice mode: yields sentence-split chunks for TTS
  think_text_stream(text) — text mode: yields raw token chunks for display
"""
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
from core.brain_prompting import (
    apply_chat_template,
    build_messages,
    build_system_prompt,
    collect_recent_turns,
    derive_topic_hint,
)
from core.intent_signals import DEFAULT_SIGNALS
from core.memory import MemoryStore
from core.platform import (
    ActionExecutor,
    ContextPlanner,
    ExecutionPlanner,
    PolicyEngine,
    ProactivePolicyEngine,
    Request,
    RequestOrchestrator,
)
from core.platform.storage import ApprovalStore, AutonomyTaskStore, EvidenceStore, TraceStore
from core.runtime_events import RuntimeObserver, StatusUpdate

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")
_ROUTING_SIGNAL_WEIGHTS: dict[str, float] = {
    "capability_query": -2.0,
    "calendar_lookup": 2.0,
    "live_web": 1.5,
    "needs_tools": 1.0,
}
_ROUTING_SCORE_TOOLS_THRESHOLD = 1.0
_ROUTING_SCORE_NO_TOOLS_THRESHOLD = -1.0
_TOOL_HEALTH_QUERY_RE = re.compile(
    r"\b(degraded|failing|failures?|broken|health|status)\b.*\b(connectors?|tools?)\b"
    r"|\b(connectors?|tools?)\b.*\b(degraded|failing|failures?|broken|health|status)\b",
    re.IGNORECASE,
)
_WEB_REQUEST_RE = re.compile(r"\b(web|internet|online|search|fetch|source|url|latest|current|right now)\b", re.IGNORECASE)
_LOCAL_REQUEST_RE = re.compile(
    r"\b(shell|terminal|os|operating system|cpu|architecture|uname|sw_vers|sysctl|pwd|directory|file|files|pdf|docx|xlsx|csv|download|downloads|desktop|document)\b",
    re.IGNORECASE,
)
_CALENDAR_REQUEST_RE = re.compile(r"\b(calendar|appointment|schedule|meeting|reminder)\b", re.IGNORECASE)
_MEMORY_REQUEST_RE = re.compile(
    r"\b(remember|memory|profile|preference|task|tasks|commitment|commitments|mark .* done|open tasks?)\b",
    re.IGNORECASE,
)
_TOOL_CATEGORY_BY_NAME: dict[str, str] = {
    "mark_task_done": "memory",
    "update_fact": "memory",
    "list_open_tasks": "memory",
    "forget_fact": "memory",
    "schedule_reminder": "memory",
    "list_scheduled_reminders": "memory",
    "cancel_reminder": "memory",
    "shell": "local",
    "shell_mutation": "local",
    "local_find_files": "local",
    "local_read_text": "local",
    "local_extract_pdf": "local",
    "local_extract_docx": "local",
    "local_extract_sheet": "local",
    "web_search": "web",
    "web_fetch": "web",
    "browser_fetch": "web",
    "notify": "external action",
    "speak": "external action",
    "calendar_read": "external action",
    "reminder_add": "external action",
}
_TOOL_CATEGORY_ORDER: tuple[str, ...] = ("memory", "local", "web", "external action", "other")


class BrainService:
    def __init__(self, *, settings: config.Settings | None = None, memory: MemoryStore | None = None) -> None:
        self.settings = settings or config.get()
        self.memory = memory or MemoryStore()
        self._trace_store = TraceStore(self.memory.db_path)
        self._evidence_store = EvidenceStore(self.memory.db_path)
        self._approval_store = ApprovalStore(self.memory.db_path)
        self._task_store = AutonomyTaskStore(self.memory.db_path)
        self._recent_turns: deque[tuple[str, str]] = deque(maxlen=max(0, self.settings.recent_turns))
        self._prefers_honesty = False
        self._prefers_forced_yes = False
        self._execution_planner = ExecutionPlanner()
        self._context_planner = ContextPlanner()
        self._proactive_policy = ProactivePolicyEngine()
        self._policy_engine = PolicyEngine(
            settings=self.settings,
            approval_store=self._approval_store,
        )
        self._action_executor = ActionExecutor(task_store=self._task_store)
        self._active_context_plan = None
        self._observers: list[RuntimeObserver] = []
        self._orchestrator = RequestOrchestrator(
            settings=self.settings,
            execution_planner=self._execution_planner,
            context_planner=self._context_planner,
            action_executor=self._action_executor,
        )

        self._runtime = GenerationRuntime(settings=self.settings)
        self.last_stats = self._runtime.last_stats
        self._llm_client: Any = None
        self._last_vault_injected: bool = False

        if self.settings.second_brain_enabled:
            from core.second_brain.entity_store import EntityStore
            from core.second_brain.llm_extractor import LLMEntityExtractor
            from core.second_brain.planner import IntentRouter
            from core.second_brain.proactive import ProactiveEngine

            self._entity_store = EntityStore(self.memory.db_path)
            self._router = IntentRouter()
            self._proactive = ProactiveEngine(self._entity_store, self.settings)
            # Initialized after warmup so tokenizer is ready
            self._llm_extractor: LLMEntityExtractor | None = None

        # The tool registry is always initialized in _warmup() for truthful
        # capability reporting and planning. The agent loop itself is still
        # created only when agent_enabled=True.
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
        # Build registry regardless of agent mode so capability queries remain
        # accurate even when AGENT_ENABLED=false.
        if self._tool_registry is None:
            self._tool_registry = self._build_tool_registry()

        # LLM router — LOCAL (Qwen3.5-9B) for vault/personal, CLOUD for general
        if self.settings.neurocache_enabled and self._llm_client is None:
            try:
                from inference.client import KageLLMClient
                self._llm_client = KageLLMClient(cloud_runtime=self._runtime)
                logger.info("[llm_client] KageLLMClient initialized (LOCAL Qwen3.5-9B lazy)")
            except Exception:
                logger.debug("[llm_client] KageLLMClient unavailable, using cloud runtime", exc_info=True)

        # Initialize the agent loop after warmup so the tokenizer is available.
        if self.settings.agent_enabled and self._agent_loop is None:
            self._agent_loop = AgentLoop(
                runtime=self._runtime,
                tokenizer=self._runtime.tokenizer,
                registry=self._tool_registry,
                settings=self.settings,
            )

    def _build_tool_registry(self) -> ToolRegistry:
        """Assemble a ToolRegistry from all available connectors.

        Core connectors (always registered — no extra dependencies):
            web_search, notify

        Memory connectors (registered only when second_brain is enabled —
        they need a live EntityStore):
            mark_task_done, update_fact, list_open_tasks

        Optional connectors (registered when extra packages are installed;
        silently skipped with a debug log if the import fails):
            web_fetch         — prefers scrapling[fetchers], falls back to httpx
            shell             — no extra deps, always available
            local_artifacts   — read-only local file discovery + extraction (txt/pdf/docx/csv/xlsx)
            calendar_read     — macOS only (osascript)
            reminder_add      — macOS only (osascript)

        To add a new tool: import it here and call registry.register().
        """
        registry = ToolRegistry(
            trace_store=self._trace_store,
            evidence_store=self._evidence_store,
            on_tool_start=self._handle_tool_started,
            on_tool_finish=self._handle_tool_finished,
            policy_engine=self._policy_engine,
        )

        # --- Core connectors (always available) ---
        from connectors.web_search import WebSearchTool
        from connectors.notify import NotifyTool
        registry.register(WebSearchTool())
        registry.register(NotifyTool())

        # --- Memory connectors (requires second_brain EntityStore) ---
        if hasattr(self, "_entity_store"):
            from connectors.memory_ops import (
                ForgetFactTool,
                ListOpenTasksTool,
                MarkTaskDoneTool,
                UpdateFactTool,
            )
            from connectors.scheduled_reminders import (
                CancelReminderTool,
                ListScheduledTool,
                ScheduleReminderTool,
            )
            db_path = self.memory.db_path
            registry.register(MarkTaskDoneTool(db_path))
            registry.register(UpdateFactTool(db_path))
            registry.register(ListOpenTasksTool(db_path))
            registry.register(ForgetFactTool(db_path))
            registry.register(ScheduleReminderTool(db_path))
            registry.register(ListScheduledTool(db_path))
            registry.register(CancelReminderTool(db_path))

        # --- Optional connectors (gracefully skipped if deps missing) ---
        try:
            from connectors.web_fetch import WebFetchTool
            registry.register(WebFetchTool())
        except ImportError:
            logger.debug('web_fetch unavailable (install "scrapling[fetchers]" + httpx to enable)')

        try:
            from connectors.browser_fetch import BrowserFetchTool
            registry.register(BrowserFetchTool())
        except ImportError:
            logger.debug("browser_fetch unavailable (install playwright + run: playwright install chromium)")

        try:
            from connectors.shell import ShellTool
            registry.register(ShellTool())
        except ImportError:
            logger.debug("shell connector unavailable")

        try:
            from connectors.local_artifacts import (
                LocalExtractDocxTool,
                LocalExtractPdfTool,
                LocalExtractSheetTool,
                LocalFindFilesTool,
                LocalReadTextTool,
            )

            registry.register(LocalFindFilesTool())
            registry.register(LocalReadTextTool())
            registry.register(LocalExtractPdfTool())
            registry.register(LocalExtractDocxTool())
            registry.register(LocalExtractSheetTool())
        except ImportError:
            logger.debug("local_artifacts connector unavailable")

        try:
            from connectors.apple_calendar import CalendarReadTool, ReminderAddTool
            registry.register(CalendarReadTool())
            registry.register(ReminderAddTool())
        except ImportError:
            logger.debug("apple_calendar connector unavailable")

        if self.settings.neurocache_enabled:
            try:
                from connectors.neurocache_connector import VaultSearchTool
                registry.register(VaultSearchTool(
                    self.settings.neurocache_api_url,
                    self.settings.neurocache_vault_inbox,
                ))
            except Exception:
                logger.debug("neurocache vault_search tool unavailable")

        return registry

    def _system_prompt(self, *, text_mode: bool = False) -> str:
        assistant_name = self.settings.assistant_name
        return build_system_prompt(
            self.settings.user_name,
            assistant_name=assistant_name,
            text_mode=text_mode,
        )

    def add_observer(self, observer: RuntimeObserver) -> None:
        if observer in self._observers:
            return
        self._observers.append(observer)

    def remove_observer(self, observer: RuntimeObserver) -> None:
        self._observers = [candidate for candidate in self._observers if candidate is not observer]

    def _notify_status(self, status: str, *, detail: str = "", metadata: dict[str, Any] | None = None) -> None:
        update = StatusUpdate(status=status, detail=detail, metadata=metadata or {})
        for observer in list(self._observers):
            callback = getattr(observer, "on_status_changed", None)
            if callable(callback):
                try:
                    callback(update)
                except Exception:
                    logger.debug("Status observer callback failed")

    def _notify_error(
        self,
        message: str,
        *,
        recoverable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for observer in list(self._observers):
            callback = getattr(observer, "on_error", None)
            if callable(callback):
                try:
                    callback(message, recoverable=recoverable, metadata=metadata or {})
                except Exception:
                    logger.debug("Error observer callback failed")

    def _handle_tool_started(self, tool_name: str, args: dict[str, Any]) -> None:
        self._notify_status("checking_tools", detail=f"Running {tool_name}")
        for observer in list(self._observers):
            callback = getattr(observer, "on_tool_started", None)
            if callable(callback):
                try:
                    callback(tool_name, args)
                except Exception:
                    logger.debug("Tool start observer callback failed")

    def _handle_tool_finished(self, result: Any) -> None:
        for observer in list(self._observers):
            callback = getattr(observer, "on_tool_finished", None)
            if callable(callback):
                try:
                    callback(result)
                except Exception:
                    logger.debug("Tool finish observer callback failed")

        outcome = getattr(result, "outcome", None)
        for source in list(getattr(outcome, "sources", []) or []):
            for observer in list(self._observers):
                callback = getattr(observer, "on_source_added", None)
                if callable(callback):
                    try:
                        callback(source, tool_name=result.tool_name)
                    except Exception:
                        logger.debug("Source observer callback failed")

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
        active_plan = getattr(self, "_active_context_plan", None)

        entity_context = ""
        if hasattr(self, "_entity_store"):
            planned_mode = str(getattr(active_plan, "entity_mode", "")).strip().lower()
            if planned_mode == "full":
                entity_context = self._entity_store.recall_for_prompt(
                    char_budget=getattr(active_plan, "char_budget", self.settings.entity_recall_budget)
                )
            elif planned_mode == "personal_only":
                entity_context = self._entity_store.recall_personal_context()
            elif route is not None and route.inject_entities:
                # Full context: tasks + commitments + profile + preferences
                entity_context = self._entity_store.recall_for_prompt(
                    char_budget=self.settings.entity_recall_budget
                )
            else:
                # Always inject profile + preferences for personal coherence
                entity_context = self._entity_store.recall_personal_context()

        memory_recall_enabled = self.settings.recent_turns > 0
        if active_plan is not None:
            memory_recall_enabled = bool(
                getattr(active_plan, "include_memory_recall", memory_recall_enabled)
            )

        vault_context = ""
        if self.settings.neurocache_enabled:
            try:
                from connectors.neurocache_connector import get_connector
                vault_context = get_connector(
                    self.settings.neurocache_api_url,
                    self.settings.neurocache_vault_inbox,
                ).retrieve_context(user_input)
            except Exception:
                logger.debug("[neurocache] context retrieval failed", exc_info=True)

        self._last_vault_injected = bool(vault_context)

        return build_messages(
            user_input=user_input,
            user_name=self.settings.user_name,
            assistant_name=self.settings.assistant_name,
            text_mode=text_mode,
            memory=self.memory,
            recent_turns=recent_turns,
            policy_note=policy_note,
            entity_context=entity_context,
            vault_context=vault_context,
            topic_hint=derive_topic_hint(recent_turns),
            memory_recall_enabled=memory_recall_enabled,
        )

    def _wants_task_context(self, user_input: str) -> bool:
        return DEFAULT_SIGNALS.has(user_input, "task_context")

    def _agent_entity_context(
        self,
        user_input: str,
        *,
        mode_override: str | None = None,
        budget_override: int | None = None,
    ) -> str:
        if not hasattr(self, "_entity_store"):
            return ""

        mode = (
            str(mode_override).strip().lower()
            if mode_override
            else self.settings.agent_entity_mode.strip().lower()
        )
        budget = (
            int(budget_override)
            if budget_override is not None
            else self.settings.entity_recall_budget
        )

        if mode == "full":
            return self._entity_store.recall_for_prompt(char_budget=budget)
        if mode == "personal_only":
            return self._entity_store.recall_personal_context()
        if self._wants_task_context(user_input):
            return self._entity_store.recall_for_prompt(char_budget=budget)
        return self._entity_store.recall_personal_context()

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

    def _maybe_handle_task_completion(self, user_input: str) -> str | None:
        if not hasattr(self, "_entity_store"):
            return None
        try:
            from connectors.memory_ops import auto_mark_task_done_from_text

            result = auto_mark_task_done_from_text(self.memory.db_path, user_input)
        except Exception:
            logger.debug("Deterministic task completion handler failed", exc_info=True)
            return None
        if result is None or result.is_error:
            return None
        return result.content.strip() or None

    def _stream_sentences(self, user_input: str, *, route: Any = None) -> Iterator[str]:
        prompt = self._build_prompt(user_input, text_mode=False, route=route)
        intent = route.intent if route is not None else "GENERAL"
        runtime = self._llm_client or self._runtime
        stream_kwargs: dict[str, Any] = {}
        if self._llm_client is not None:
            stream_kwargs = {
                "intent": intent,
                "vault_context_injected": self._last_vault_injected,
            }
        buffer = ""
        for text in runtime.stream_raw(prompt, **stream_kwargs):
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
        intent = route.intent if route is not None else "GENERAL"
        runtime = self._llm_client or self._runtime
        stream_kwargs: dict[str, Any] = {}
        if self._llm_client is not None:
            stream_kwargs = {
                "intent": intent,
                "vault_context_injected": self._last_vault_injected,
            }
        yield from runtime.stream_raw(prompt, **stream_kwargs)

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

        should_extract = True
        if route is not None and hasattr(route, "should_extract"):
            should_extract = bool(route.should_extract)

        if (
            exchange_id is not None
            and self.settings.extraction_enabled
            and hasattr(self, "_entity_store")
            and should_extract
        ):
            self._extract_and_store(user_input, exchange_id)

    def _routing_prompt(self, user_input: str) -> str:
        """Build the chat-template prompt for the tool-need routing classifier.

        The system message instructs the model to answer with a single word
        ("yes" or "no") so we can cap generation at 8 tokens and keep the
        routing call fast.
        """
        system = (
            "You are a routing classifier. Reply with exactly one word: 'yes' or 'no'.\n\n"
            "Output 'yes' if the user request requires external tools such as:\n"
            "  - Web search or fetching URLs\n"
            "  - Calendar or reminder operations\n"
            "  - Shell commands or system information\n"
            "  - Local file discovery or document extraction (txt/pdf/docx/csv/xlsx)\n"
            "  - Memory write operations (storing facts or tasks)\n"
            "  - Product/spec comparisons that need current web data or local machine inspection\n"
            "  - Time-sensitive facts, current events, prices, or live data\n\n"
            "Output 'no' only for simple conversational questions answerable from knowledge alone.\n\n"
            "Examples:\n"
            "  'What is 2+2?' → no\n"
            "  'Search for the latest Bitcoin price' → yes\n"
            "  'Add a reminder to call mom tomorrow' → yes\n"
            "  'What events do I have today?' → yes\n"
            "  'Who wrote Hamlet?' → no\n"
            "  'Run ls in my home folder' → yes\n"
            "  'What is the weather right now?' → yes\n"
            "  'Compare the new MacBook Neo to my local machine' → yes\n"
            "  'There is a PDF in Downloads called Lasku, check specs' → yes"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        return apply_chat_template(self._runtime.tokenizer, messages)

    def _capability_response(self, user_input: str) -> str | None:
        registry = getattr(self, "_tool_registry", None)
        if registry is None:
            return None
        if not DEFAULT_SIGNALS.has(user_input, "capability_query"):
            return None

        names = registry.names()
        if not names:
            return "I don't have any connectors enabled right now."

        grouped: dict[str, list[str]] = {}
        for name in names:
            category = _TOOL_CATEGORY_BY_NAME.get(name, "other")
            grouped.setdefault(category, []).append(name)

        parts: list[str] = []
        for category in _TOOL_CATEGORY_ORDER:
            tools = grouped.get(category, [])
            if tools:
                parts.append(f"{category}: {', '.join(tools)}")
        mode = "enabled" if self.settings.agent_enabled else "disabled"
        mode_note = (
            "I can execute them now."
            if self.settings.agent_enabled
            else "Execution is currently off because AGENT_ENABLED=false."
        )

        degraded: list[str] = []
        for name in names:
            health = self.tool_health(name)
            if health < 0.6:
                degraded.append(f"{name} ({health:.0%} recent success)")

        wants_health = bool(_TOOL_HEALTH_QUERY_RE.search(user_input))
        if degraded:
            health_note = f"Degraded right now: {', '.join(degraded)}."
        elif wants_health:
            health_note = "No degraded connectors detected in recent tool traces."
        else:
            health_note = ""

        body = " | ".join(parts)
        message = f"Agent mode: {mode}. {mode_note} {body}."
        if health_note:
            message = f"{message} {health_note}"
        return message.strip()

    def _requested_connector_domains(self, user_input: str) -> list[str]:
        domains: list[str] = []
        if _MEMORY_REQUEST_RE.search(user_input):
            domains.append("memory")
        if _LOCAL_REQUEST_RE.search(user_input):
            domains.append("local machine")
        if _WEB_REQUEST_RE.search(user_input):
            domains.append("web")
        if _CALENDAR_REQUEST_RE.search(user_input):
            domains.append("calendar/reminders")
        return domains

    def tooling_unavailable_response(self, user_input: str, *, decision: Any, catalog: Any) -> str:
        del decision, catalog
        requested = self._requested_connector_domains(user_input)
        requested_scope = (
            f" ({', '.join(requested)})"
            if requested
            else ""
        )
        follow_up = "Enable AGENT_ENABLED=true in .env and restart Kage."
        if "memory" in requested and not self.settings.second_brain_enabled:
            follow_up += " For memory/task tools also set SECOND_BRAIN_ENABLED=true."
        return (
            "I can't execute connector actions"
            f"{requested_scope} in this session because AGENT_ENABLED=false. "
            f"{follow_up}"
        )

    def _heuristic_needs_tools(self, user_input: str) -> bool | None:
        text = user_input.strip()
        if not text:
            return False
        score = DEFAULT_SIGNALS.weighted_score(text, _ROUTING_SIGNAL_WEIGHTS)
        if score >= _ROUTING_SCORE_TOOLS_THRESHOLD:
            return True
        if score <= _ROUTING_SCORE_NO_TOOLS_THRESHOLD:
            return False
        return None

    def _classify_ambiguous_tool_need(self, user_input: str) -> bool:
        prompt = self._routing_prompt(user_input)
        for attempt in range(2):
            answer = "".join(
                self._runtime.stream_raw(
                    prompt, max_tokens=8, track_stats=False, temperature=0.0
                )
            ).strip().lower()
            if answer.startswith("yes"):
                return True
            if answer.startswith("no"):
                return False
            logger.debug(
                "Routing attempt %d inconclusive (got %r) — retrying", attempt + 1, answer
            )
        logger.debug("Routing inconclusive after retry — defaulting to tools")
        return True

    def _needs_tools(self, user_input: str) -> bool:
        """Decide whether tools are needed using unified execution planning."""
        planner = getattr(self, "_execution_planner", None)
        if planner is None:
            heuristic = self._heuristic_needs_tools(user_input)
            if heuristic is not None:
                return heuristic
            return self._classify_ambiguous_tool_need(user_input)
        return planner.needs_tools(
            user_input=user_input,
            classify_ambiguous=self._classify_ambiguous_tool_need,
        )

    def agent_stream(self, user_input: str) -> Iterator[str]:
        """Run the agentic multi-step path for a single user request.

        Called by think_stream / think_text_stream when _needs_tools() returns
        True.  Assembles the entity context from EntityStore and delegates to
        AgentLoop.run(), then persists the final reply to memory.

        Yields string chunks exactly as think_stream does so callers
        (respond() in main.py) need no special handling for agent responses.
        """
        if self._agent_loop is None:
            logger.error("agent_stream called but _agent_loop is None — agent not initialized")
            return
        self._update_policy_state(user_input)
        self._notify_status("checking_tools", detail="Running agent workflow")
        entity_context = self._agent_entity_context(user_input)
        topic_hint = derive_topic_hint(self._collect_recent_turns())
        if topic_hint:
            topic_line = f"Conversation topic: {topic_hint}"
            entity_context = f"{entity_context}\n{topic_line}".strip() if entity_context else topic_line
        parts: list[str] = []
        for chunk in self._agent_loop.run(user_input, context=entity_context):
            parts.append(chunk)
            yield chunk
        reply = "".join(parts).strip()
        self._persist_exchange(user_input, reply)

    # ------------------------------------------------------------------
    # Orchestrator runtime adapter
    # ------------------------------------------------------------------

    def available_tool_names(self) -> list[str]:
        registry = getattr(self, "_tool_registry", None)
        if registry is None:
            return []
        return registry.names()

    def tool_health(self, tool_name: str) -> float:
        trace = getattr(self, "_trace_store", None)
        if trace is None:
            return 1.0
        try:
            return float(trace.tool_health(tool_name))
        except Exception:
            return 1.0

    def agent_enabled(self) -> bool:
        return self._agent_loop is not None

    def classify_ambiguous_tool_need(self, user_input: str) -> bool:
        return self._classify_ambiguous_tool_need(user_input)

    def required_approval_tiers(self) -> tuple[str, ...]:
        engine = getattr(self, "_policy_engine", None)
        if engine is None:
            return ()
        try:
            return tuple(sorted(str(tier) for tier in engine.required_tiers()))
        except Exception:
            return ()

    def report_status(self, status: str, *, detail: str = "", metadata: dict[str, Any] | None = None) -> None:
        self._notify_status(status, detail=detail, metadata=metadata)

    def record_decision_trace(self, decision: Any, context_plan: Any) -> None:
        trace = getattr(self, "_trace_store", None)
        if trace is None:
            return
        try:
            trace.record(
                event_kind="decision",
                event_name=str(getattr(decision, "strategy", "unknown")),
                status="ok",
                payload={
                    "reason_codes": list(getattr(decision, "reason_codes", ())),
                    "context_sources": list(getattr(context_plan, "sources", ())),
                    "context_reason_codes": list(getattr(context_plan, "reason_codes", ())),
                    "entity_mode": getattr(context_plan, "entity_mode", ""),
                },
            )
        except Exception:
            logger.debug("Failed to record decision trace")

    def capability_response(self, user_input: str) -> str | None:
        return self._capability_response(user_input)

    def agent_context(self, user_input: str, context_plan: Any) -> str:
        mode = getattr(context_plan, "entity_mode", None)
        budget = getattr(context_plan, "char_budget", None)
        entity_context = self._agent_entity_context(
            user_input,
            mode_override=mode,
            budget_override=budget,
        )
        topic_hint = derive_topic_hint(self._collect_recent_turns())
        if topic_hint:
            topic_line = f"Conversation topic: {topic_hint}"
            entity_context = f"{entity_context}\n{topic_line}".strip() if entity_context else topic_line

        if self.settings.neurocache_enabled:
            try:
                from connectors.neurocache_connector import get_connector
                vault_context = get_connector(
                    self.settings.neurocache_api_url,
                    self.settings.neurocache_vault_inbox,
                ).retrieve_context(user_input)
                if vault_context:
                    entity_context = f"{entity_context}\n\n{vault_context}".strip() if entity_context else vault_context
            except Exception:
                logger.debug("[neurocache] agent context retrieval failed", exc_info=True)

        return entity_context

    def agent_runner(self, task: str, entity_context: str) -> Iterator[str]:
        if self._agent_loop is None:
            logger.error("agent_runner called but _agent_loop is None — agent not initialized")
            return
        self._update_policy_state(task)
        self._notify_status("checking_tools", detail="Executing tool plan")
        yield from self._agent_loop.run(task, context=entity_context)

    def agent_run_summary(self) -> Any:
        if self._agent_loop is None:
            return None
        try:
            return self._agent_loop.last_run_summary()
        except Exception:
            return None

    def persist_exchange(self, user_input: str, reply: str) -> None:
        self._persist_exchange(user_input, reply)

    def direct_response_stream(self, user_input: str, *, text_mode: bool, context_plan: Any) -> Iterator[str]:
        self._active_context_plan = context_plan
        try:
            self._update_policy_state(user_input)
            self._notify_status("drafting_answer", detail="Preparing response")
            task_completion = self._maybe_handle_task_completion(user_input)
            if task_completion:
                yield task_completion
                self._persist_exchange(user_input, task_completion)
                self._notify_status("done", detail="Reply complete")
                return
            deterministic = self._deterministic_response(user_input)
            if deterministic:
                reply = deterministic.strip()
                if reply:
                    yield reply
                    self._persist_exchange(user_input, reply)
                    self._notify_status("done", detail="Reply complete")
                return

            route = self._router.classify(user_input) if hasattr(self, "_router") else None
            parts: list[str] = []
            if text_mode:
                self._notify_status("drafting_answer", detail="Streaming response")
                for chunk in self._stream_text(user_input, route=route):
                    parts.append(chunk)
                    yield chunk
                reply = "".join(parts).strip()
            else:
                for sentence in self._stream_sentences(user_input, route=route):
                    parts.append(sentence)
                    yield sentence
                reply = " ".join(parts).strip()

            self._persist_exchange(user_input, reply, route=route)
            self._notify_status("done", detail="Reply complete")
            if route is not None and route.proactive_ok and hasattr(self, "_entity_store"):
                suggestion = self._proactive_policy.suggest_from_reply(
                    entity_store=self._entity_store,
                    settings=self.settings,
                    reply=reply,
                    proactive_ok=route.proactive_ok,
                )
                if suggestion:
                    yield suggestion
        except Exception as exc:
            self._notify_error(str(exc), recoverable=False)
            raise
        finally:
            self._active_context_plan = None

    def think_stream(self, user_input: str) -> Iterator[str]:
        yield from self._orchestrator.handle(
            Request(text=user_input, text_mode=False, source="voice"),
            runtime=self,
        )

    def think_text_stream(self, user_input: str) -> Iterator[str]:
        yield from self._orchestrator.handle(
            Request(text=user_input, text_mode=True, source="text"),
            runtime=self,
        )

    def vault_stats_line(self) -> str:
        """One-line vault summary for startup displays. Empty string if unavailable."""
        if not self.settings.neurocache_enabled:
            return ""
        try:
            from connectors.neurocache_connector import get_connector
            stats = get_connector(
                self.settings.neurocache_api_url,
                self.settings.neurocache_vault_inbox,
            ).get_stats()
            if stats.get("status") == "offline":
                return "vault: offline"
            return f"vault: {stats.get('notes', 0)} notes, {stats.get('entities', 0)} entities"
        except Exception:
            return "vault: unavailable"
