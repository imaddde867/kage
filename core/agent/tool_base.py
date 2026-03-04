"""Base classes shared by every agent tool.

Three public types are defined here:

    ToolCall    — what the LLM requests  (name + args dict)
    ToolResult  — what the tool returns  (content string + error flag)
    Tool        — abstract base class that all connectors must subclass

Concrete tools live in connectors/ and are registered in
BrainService._build_tool_registry().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation parsed from the model's XML output.

    Produced by parser.parse_step(); consumed by ToolRegistry.execute().

    Attributes:
        name: The tool identifier, e.g. "web_search".
        args: Keyword arguments to pass to tool.execute(), e.g. {"query": "..."}.
    """
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of a tool invocation.

    Always carries a string content field so the AgentLoop can inject the
    result back into the conversation as a plain-text observation, regardless
    of whether the tool returned structured or unstructured data.

    Attributes:
        tool_name: Name of the tool that produced this result (for logging).
        content:   Human-readable result text injected as the next user message.
        is_error:  True if the tool failed. The loop continues regardless —
                   the model decides whether to retry or give up.
    """
    tool_name: str
    content: str
    is_error: bool = False


class Tool(ABC):
    """Abstract base class for all agent tools.

    Subclass this to add a new capability to Kage. Three class attributes
    must be set on every concrete subclass:

        name        — unique identifier used in XML tool calls, e.g. "web_search"
        description — one-line description shown to the LLM in the system prompt
        parameters  — JSON Schema "object" describing accepted arguments;
                      drives the schema_line() method and LLM guidance

    Example minimal implementation::

        class MyTool(Tool):
            name = "my_tool"
            description = "Does something useful"
            parameters = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }

            def execute(self, *, text: str, **kwargs) -> ToolResult:
                return ToolResult(tool_name=self.name, content=text.upper())

    Then register it::

        registry.register(MyTool())
    """

    name: str
    description: str
    # JSON Schema describing accepted kwargs; subclasses set this as a class var.
    # The empty dict default means the tool accepts no arguments.
    parameters: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a result.

        Implementations should never raise — catch all exceptions and return
        a ToolResult with is_error=True instead.  The AgentLoop will feed
        the error content back to the model as an observation and continue.
        """
        ...

    def schema_line(self) -> str:
        """Return a single line describing this tool, injected into the system prompt.

        Format: ``- name(arg: type, ...): description``

        The ToolRegistry collects these lines into a block that teaches the
        model what tools are available and how to call them.
        """
        props = self.parameters.get("properties", {})
        args_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        return f"- {self.name}({args_str}): {self.description}"
