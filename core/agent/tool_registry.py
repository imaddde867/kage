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

from core.agent.tool_base import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Maintains a name → Tool mapping and routes ToolCall objects to the right tool.

    Tools are stored in insertion order (Python dict preserves order since 3.7),
    so schema_block() lists them in the order they were registered.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

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

    def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch a ToolCall to the matching tool and return its result.

        If the tool name is not registered, returns an error result listing
        the available tools — this gets fed back to the model as an
        observation so it can try an alternative approach.

        Any exception raised by tool.execute() is caught here and converted
        to an error ToolResult so the AgentLoop always keeps running.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_name=call.name,
                content=f"Unknown tool '{call.name}'. Available: {', '.join(self.names())}",
                is_error=True,
            )
        try:
            return tool.execute(**call.args)
        except Exception as exc:
            logger.exception("Tool '%s' raised an error", call.name)
            return ToolResult(tool_name=call.name, content=str(exc), is_error=True)
