"""Base classes for agent tools."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    tool_name: str
    content: str
    is_error: bool = False


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any] = {}  # JSON Schema; subclasses override as class var

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...

    def schema_line(self) -> str:
        """Single-line description injected into the agent system prompt."""
        props = self.parameters.get("properties", {})
        args_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        return f"- {self.name}({args_str}): {self.description}"
