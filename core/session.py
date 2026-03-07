from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from core.agent.tool_base import ToolResult
from core.runtime_events import StatusUpdate


_SENTINEL_KIND = "__closed__"


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    session_id: str
    turn_id: int | None = None
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SessionController:
    def __init__(self, *, brain: Any) -> None:
        self._brain = brain
        self._queue: queue.Queue[SessionEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._cancel_requested = threading.Event()
        self._closed = False
        self._session_id = self._new_session_id()
        self._turn_id = 0
        self._last_answer = ""
        self._latest_sources: list[str] = []
        self._source_keys: set[tuple[str, str]] = set()
        self._active_status = "idle"
        self._emit(
            "session_started",
            data={
                "assistant_name": getattr(self.settings, "assistant_name", "Kage"),
                "llm_backend": getattr(self.settings, "llm_backend", "unknown"),
                "model": getattr(self.settings, "mlx_model", "unknown"),
            },
        )

    @property
    def settings(self) -> Any:
        return getattr(self._brain, "settings", None)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            worker = self._worker
        return worker is not None and worker.is_alive()

    @property
    def last_answer(self) -> str:
        return self._last_answer

    def current_sources(self) -> list[str]:
        return list(self._latest_sources)

    def reset(self) -> None:
        if self.is_busy:
            raise RuntimeError("Cannot reset while a request is running.")
        self._session_id = self._new_session_id()
        self._turn_id = 0
        self._last_answer = ""
        self._latest_sources = []
        self._source_keys.clear()
        self._active_status = "idle"
        self._emit(
            "session_started",
            data={
                "assistant_name": getattr(self.settings, "assistant_name", "Kage"),
                "llm_backend": getattr(self.settings, "llm_backend", "unknown"),
                "model": getattr(self.settings, "mlx_model", "unknown"),
            },
        )

    def submit(self, text: str) -> None:
        prompt = (text or "").strip()
        if not prompt:
            raise ValueError("Prompt must be non-empty.")
        if self.is_busy:
            raise RuntimeError("A request is already running.")

        self._turn_id += 1
        turn_id = self._turn_id
        self._cancel_requested = threading.Event()
        self._latest_sources = []
        self._source_keys.clear()
        self._last_answer = ""
        self._emit("user_message", turn_id=turn_id, text=prompt)
        self._emit_status("thinking", turn_id=turn_id, detail="Planning response")

        worker = threading.Thread(
            target=self._run_turn,
            args=(turn_id, prompt),
            daemon=True,
            name=f"kage-session-{turn_id}",
        )
        with self._lock:
            self._worker = worker
        worker.start()

    def cancel(self) -> None:
        if not self.is_busy:
            return
        self._cancel_requested.set()
        self._emit_status("cancelled", turn_id=self._turn_id, detail="Stopped streaming this reply")

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def next_event(self, timeout: float | None = None) -> SessionEvent | None:
        try:
            event = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if event.kind == _SENTINEL_KIND:
            return None
        return event

    def events(self) -> Iterator[SessionEvent]:
        while True:
            event = self._queue.get()
            if event.kind == _SENTINEL_KIND:
                return
            yield event

    def close(self) -> None:
        self._closed = True
        self._queue.put(SessionEvent(kind=_SENTINEL_KIND, session_id=self._session_id))

    def recent_history(self, *, limit: int = 8) -> list[tuple[str, str]]:
        memory = getattr(self._brain, "memory", None)
        if memory is None:
            return []
        recent = getattr(memory, "recent_turns", None)
        if not callable(recent):
            return []
        try:
            return list(recent(limit=limit))
        except Exception:
            return []

    def memory_summary(self, *, char_budget: int = 700) -> str:
        if not getattr(self.settings, "second_brain_enabled", False):
            return "Structured memory is disabled. Enable SECOND_BRAIN_ENABLED=true to populate this panel."

        entity_store = getattr(self._brain, "_entity_store", None)
        if entity_store is None:
            return "Structured memory is enabled but no entity store is available yet."

        try:
            summary = entity_store.recall_for_prompt(char_budget=char_budget)
        except Exception:
            return "Unable to load structured memory right now."
        return summary or "No structured memory entries yet."

    def connector_summary(self) -> str:
        names_fn = getattr(self._brain, "available_tool_names", None)
        health_fn = getattr(self._brain, "tool_health", None)
        if not callable(names_fn):
            return "Connectors unavailable."
        names = list(names_fn())
        if not names:
            return "No connectors registered."

        lines: list[str] = []
        for name in names:
            health = 1.0
            if callable(health_fn):
                try:
                    health = float(health_fn(name))
                except Exception:
                    health = 1.0
            lines.append(f"- {name}: {health:.0%} recent success")
        agent_enabled = bool(getattr(self.settings, "agent_enabled", False))
        prefix = "Agent mode is enabled." if agent_enabled else "Agent mode is disabled."
        return prefix + "\n" + "\n".join(lines)

    def on_status_changed(self, update: StatusUpdate) -> None:
        self._emit_status(update.status, turn_id=self._turn_id, detail=update.detail, metadata=update.metadata)

    def on_tool_started(self, tool_name: str, args: dict[str, Any]) -> None:
        self._emit(
            "tool_started",
            turn_id=self._turn_id,
            text=tool_name,
            data={"tool_name": tool_name, "args": dict(args)},
        )

    def on_tool_finished(self, result: ToolResult) -> None:
        outcome = result.outcome
        self._emit(
            "tool_finished",
            turn_id=self._turn_id,
            text=result.tool_name,
            data={
                "tool_name": result.tool_name,
                "status": getattr(outcome, "status", "error" if result.is_error else "ok"),
                "latency_ms": getattr(outcome, "latency_ms", None),
                "is_error": result.is_error,
                "preview": result.content[:220],
            },
        )
        if result.is_error:
            self.on_error(
                f"{result.tool_name} failed: {result.content[:220]}",
                recoverable=True,
                metadata={"tool_name": result.tool_name},
            )

    def on_source_added(self, source: str, *, tool_name: str = "") -> None:
        key = (tool_name, source)
        if key in self._source_keys:
            return
        self._source_keys.add(key)
        self._latest_sources.append(source)
        self._emit(
            "source_added",
            turn_id=self._turn_id,
            text=source,
            data={"tool_name": tool_name, "source": source},
        )

    def on_error(
        self,
        message: str,
        *,
        recoverable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "error",
            turn_id=self._turn_id,
            text=message,
            data={"recoverable": recoverable, **(metadata or {})},
        )

    def _run_turn(self, turn_id: int, prompt: str) -> None:
        self._register_observer()
        parts: list[str] = []
        cancelled = False
        stream = None
        try:
            stream = self._brain.think_text_stream(prompt)
            for chunk in stream:
                if self._cancel_requested.is_set():
                    cancelled = True
                    break
                text = str(chunk)
                parts.append(text)
                self._emit("assistant_chunk", turn_id=turn_id, text=text)

            if self._cancel_requested.is_set():
                cancelled = True

            final_text = "".join(parts).strip()
            self._last_answer = final_text
            if cancelled:
                self._emit(
                    "assistant_done",
                    turn_id=turn_id,
                    text=final_text,
                    data={"cancelled": True},
                )
            else:
                self._emit("assistant_done", turn_id=turn_id, text=final_text)
                stats = dict(getattr(self._brain, "last_stats", {}) or {})
                if stats:
                    self._emit("metrics_updated", turn_id=turn_id, data=stats)
                self._emit_status("done", turn_id=turn_id, detail="Reply complete")
        except Exception as exc:
            self.on_error(str(exc), recoverable=False)
            self._emit_status("error", turn_id=turn_id, detail="Request failed")
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._remove_observer()
            with self._lock:
                self._worker = None

    def _register_observer(self) -> None:
        add = getattr(self._brain, "add_observer", None)
        if callable(add):
            add(self)

    def _remove_observer(self) -> None:
        remove = getattr(self._brain, "remove_observer", None)
        if callable(remove):
            remove(self)

    def _emit_status(
        self,
        status: str,
        *,
        turn_id: int | None,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._active_status = status
        self._emit(
            "status_changed",
            turn_id=turn_id,
            text=detail,
            data={"status": status, **(metadata or {})},
        )

    def _emit(
        self,
        kind: str,
        *,
        turn_id: int | None = None,
        text: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        self._queue.put(
            SessionEvent(
                kind=kind,
                session_id=self._session_id,
                turn_id=turn_id,
                text=text,
                data=data or {},
            )
        )

    @staticmethod
    def _new_session_id() -> str:
        return str(uuid.uuid4())
