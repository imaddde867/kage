"""Unit tests for core/agent/tool_registry.py — no LLM required."""
import unittest
from typing import Any

from core.agent.tool_base import Tool, ToolCall, ToolResult
from core.agent.tool_registry import ToolRegistry


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes the message back"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, *, message: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content=message)


class _CountTool(Tool):
    name = "count"
    description = "Returns the length of text"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, *, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content=str(len(text)))


class _BrokenTool(Tool):
    name = "broken"
    description = "Always raises"
    parameters: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("tool is broken")


class TestToolRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(_EchoTool())
        self.registry.register(_CountTool())
        self.registry.register(_BrokenTool())

    def test_names_contains_registered_tools(self) -> None:
        names = self.registry.names()
        self.assertIn("echo", names)
        self.assertIn("count", names)
        self.assertIn("broken", names)

    def test_execute_success(self) -> None:
        result = self.registry.execute(ToolCall(name="echo", args={"message": "hello"}))
        self.assertEqual(result.content, "hello")
        self.assertFalse(result.is_error)
        self.assertEqual(result.tool_name, "echo")

    def test_execute_second_tool(self) -> None:
        result = self.registry.execute(ToolCall(name="count", args={"text": "abcde"}))
        self.assertEqual(result.content, "5")
        self.assertFalse(result.is_error)

    def test_execute_unknown_tool_returns_error(self) -> None:
        result = self.registry.execute(ToolCall(name="nonexistent", args={}))
        self.assertTrue(result.is_error)
        self.assertIn("nonexistent", result.content)

    def test_execute_tool_exception_returns_error(self) -> None:
        result = self.registry.execute(ToolCall(name="broken", args={}))
        self.assertTrue(result.is_error)
        self.assertIn("broken", result.content)

    def test_schema_block_contains_tool_info(self) -> None:
        schema = self.registry.schema_block()
        self.assertIn("echo", schema)
        self.assertIn("Echoes", schema)
        self.assertIn("count", schema)

    def test_schema_block_lists_args(self) -> None:
        schema = self.registry.schema_block()
        self.assertIn("message", schema)
        self.assertIn("text", schema)

    def test_register_replaces_same_name(self) -> None:
        class _EchoV2(Tool):
            name = "echo"
            description = "Echoes v2"
            parameters: dict[str, Any] = {}

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(tool_name=self.name, content="v2")

        self.registry.register(_EchoV2())
        result = self.registry.execute(ToolCall(name="echo", args={}))
        self.assertEqual(result.content, "v2")


if __name__ == "__main__":
    unittest.main()
