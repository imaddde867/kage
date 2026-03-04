"""Registry for agent tools — register, dispatch, and schema generation."""
from __future__ import annotations

import logging

from core.agent.tool_base import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schema_block(self) -> str:
        """Formatted tool list for injection into the agent system prompt."""
        return "\n".join(t.schema_line() for t in self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
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
