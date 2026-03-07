"""Registry for agent tools — register, dispatch, and schema generation.

ToolRegistry is the single point of contact between AgentLoop and all tools.
It handles three responsibilities:

    1. Registration  — tools are added once at startup via register().
    2. Schema gen    — schema_block() produces the tool list injected into the
                       agent system prompt so the LLM knows what's available.
    3. Dispatch      — execute() looks up the tool by name, calls it, and
                       catches any unexpected exceptions so the loop never crashes.

Usage::

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    result = registry.execute(ToolCall(name="web_search", args={"query": "..."}))
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.agent.tool_base import Tool, ToolCall, ToolOutcome, ToolResult

logger = logging.getLogger(__name__)

_TOOL_ALIASES: dict[str, str] = {
    "search": "web_search",
    "fetch": "web_fetch",
}


class ToolRegistry:
    """Maintains a name → Tool mapping and routes ToolCall objects to the right tool.

    Tools are stored in insertion order (Python dict preserves order since 3.7),
    so schema_block() lists them in the order they were registered.
    """

    def __init__(
        self,
        *,
        trace_store: Any | None = None,
        evidence_store: Any | None = None,
        on_tool_start: Any | None = None,
        on_tool_finish: Any | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._trace_store = trace_store
        self._evidence_store = evidence_store
        self._on_tool_start = on_tool_start
        self._on_tool_finish = on_tool_finish

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        If a tool with the same name is already registered it is silently
        replaced, which makes it straightforward to override a default tool
        with a custom implementation.
        """
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        """Return the names of all registered tools in registration order."""
        return list(self._tools.keys())

    def schema_block(self) -> str:
        """Return a newline-separated list of tool schema lines.

        Injected into the agent system prompt so the LLM knows what tools
        exist and what arguments each one accepts.  Each line is produced
        by Tool.schema_line().
        """
        return "\n".join(t.schema_line() for t in self._tools.values())

    def _normalize_name(self, name: str) -> str:
        key = (name or "").strip().lower()
        return _TOOL_ALIASES.get(key, key)

    def _repair_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        repaired = dict(args)

        if tool_name == "web_fetch" and "url" not in repaired:
            for candidate in ("href", "link"):
                if isinstance(repaired.get(candidate), str) and repaired[candidate].strip():
                    repaired["url"] = repaired[candidate]
                    break

        if tool_name == "web_search" and "query" not in repaired:
            if isinstance(repaired.get("q"), str) and repaired["q"].strip():
                repaired["query"] = repaired["q"]

        for key in ("max_results", "max_chars", "days"):
            value = repaired.get(key)
            if isinstance(value, str):
                try:
                    repaired[key] = int(value.strip())
                except ValueError:
                    pass

        return repaired

    def _example_for_tool(self, tool_name: str) -> str:
        if tool_name == "web_fetch":
            return 'Example: <tool>web_fetch</tool><input>{"url":"https://example.com"}</input>'
        if tool_name == "web_search":
            return 'Example: <tool>web_search</tool><input>{"query":"latest AI news"}</input>'
        return ""

    def _validate_args(self, tool: Tool, args: dict[str, Any]) -> str | None:
        schema = getattr(tool, "parameters", {}) or {}
        if not isinstance(schema, dict):
            return None

        required = schema.get("required") or []
        if not isinstance(required, list):
            required = []
        missing = [key for key in required if key not in args or args.get(key) in (None, "")]
        if missing:
            details = f"{tool.name} requires: {', '.join(missing)}."
            example = self._example_for_tool(tool.name)
            return f"{details} {example}".strip()

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return None

        type_errors: list[str] = []
        for key, value in args.items():
            schema_entry = properties.get(key)
            if not isinstance(schema_entry, dict):
                continue
            expected = schema_entry.get("type")
            if expected == "string" and not isinstance(value, str):
                type_errors.append(f"'{key}' must be a string")
            elif expected == "integer" and not isinstance(value, int):
                type_errors.append(f"'{key}' must be an integer")
            elif expected == "number" and not isinstance(value, (int, float)):
                type_errors.append(f"'{key}' must be a number")
            elif expected == "boolean" and not isinstance(value, bool):
                type_errors.append(f"'{key}' must be a boolean")
            elif expected == "object" and not isinstance(value, dict):
                type_errors.append(f"'{key}' must be an object")
            elif expected == "array" and not isinstance(value, list):
                type_errors.append(f"'{key}' must be an array")

        if type_errors:
            details = "; ".join(type_errors) + "."
            example = self._example_for_tool(tool.name)
            return f"{details} {example}".strip()
        return None

    def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch a ToolCall to the matching tool and return its result.

        If the tool name is not registered, returns an error result listing
        the available tools — this gets fed back to the model as an
        observation so it can try an alternative approach.

        Any exception raised by tool.execute() is caught here and converted
        to an error ToolResult so the AgentLoop always keeps running.
        """
        started = time.perf_counter()
        tool_name = self._normalize_name(call.name)
        if tool_name == "invalid_tool_call":
            raw = ""
            if isinstance(call.args, dict):
                raw = str(call.args.get("raw", "")).strip()
            return self._finalize(
                ToolResult(
                tool_name="invalid_tool_call",
                content=(
                    "Malformed tool output. Use canonical format: "
                    "<tool>tool_name</tool><input>{...}</input>. "
                    f"Raw: {raw[:180]}"
                ),
                is_error=True,
                ),
                query_text=str(call.args) if isinstance(call.args, dict) else None,
                started=started,
            )

        tool = self._tools.get(tool_name)
        if tool is None:
            return self._finalize(
                ToolResult(
                tool_name=tool_name,
                content=f"Unknown tool '{tool_name}'. Available: {', '.join(self.names())}",
                is_error=True,
                ),
                query_text=str(call.args),
                started=started,
            )

        if not isinstance(call.args, dict):
            return self._finalize(
                ToolResult(
                tool_name=tool_name,
                content=f"{tool_name} expects a JSON object of arguments.",
                is_error=True,
                ),
                query_text=None,
                started=started,
            )

        repaired_args = self._repair_args(tool_name, call.args)
        if callable(self._on_tool_start):
            try:
                self._on_tool_start(tool_name, dict(repaired_args))
            except Exception:
                logger.debug("Tool start callback failed for '%s'", tool_name)
        validation_error = self._validate_args(tool, repaired_args)
        if validation_error:
            return self._finalize(
                ToolResult(tool_name=tool_name, content=validation_error, is_error=True),
                query_text=str(repaired_args),
                started=started,
            )

        try:
            result = tool.execute(**repaired_args)
            return self._finalize(result, query_text=str(repaired_args), started=started)
        except TypeError as exc:
            logger.exception("Tool '%s' argument error", tool_name)
            example = self._example_for_tool(tool_name)
            msg = f"Invalid arguments for '{tool_name}': {exc}."
            if example:
                msg = f"{msg} {example}"
            return self._finalize(
                ToolResult(tool_name=tool_name, content=msg, is_error=True),
                query_text=str(repaired_args),
                started=started,
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an error", tool_name)
            return self._finalize(
                ToolResult(tool_name=tool_name, content=str(exc), is_error=True),
                query_text=str(repaired_args),
                started=started,
            )

    def _finalize(self, result: ToolResult, *, query_text: str | None, started: float) -> ToolResult:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if result.outcome is None:
            result.outcome = ToolOutcome(
                status="error" if result.is_error else "ok",
                structured=None,
                sources=[],
                retryable=result.is_error,
                latency_ms=elapsed_ms,
                side_effects=False,
            )
        elif result.outcome.latency_ms is None:
            result.outcome.latency_ms = elapsed_ms

        if self._trace_store is not None:
            try:
                self._trace_store.record_tool_result(
                    tool_name=result.tool_name,
                    is_error=result.is_error,
                    latency_ms=result.outcome.latency_ms,
                    content_preview=result.content,
                )
            except Exception:
                logger.debug("Trace recording failed for tool '%s'", result.tool_name)

        if self._evidence_store is not None and result.outcome is not None:
            try:
                self._evidence_store.record(
                    tool_name=result.tool_name,
                    status=result.outcome.status,
                    query_text=query_text,
                    content=result.content,
                    structured=result.outcome.structured,
                    sources=result.outcome.sources,
                    latency_ms=result.outcome.latency_ms,
                )
            except Exception:
                logger.debug("Evidence recording failed for tool '%s'", result.tool_name)

        if callable(self._on_tool_finish):
            try:
                self._on_tool_finish(result)
            except Exception:
                logger.debug("Tool finish callback failed for '%s'", result.tool_name)

        return result
