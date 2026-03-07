"""Unit tests for core/agent/tool_registry.py — no LLM required.

Tests use three minimal Tool subclasses defined here as test doubles:
    _EchoTool   — returns the message arg unchanged
    _CountTool  — returns the length of a text arg as a string
    _BrokenTool — always raises RuntimeError (tests exception handling)

All tests are pure Python; no model, no network, no file I/O.
"""
import unittest
from typing import Any

from core.agent.tool_base import Tool, ToolCall, ToolResult
from core.agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test doubles — minimal Tool subclasses used as fixtures
# ---------------------------------------------------------------------------

class _EchoTool(Tool):
    """Returns the message argument unchanged — useful for verifying dispatch."""
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
    """Returns len(text) as a string — a second distinct tool for multi-tool tests."""
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
    """Always raises RuntimeError — tests that the registry catches exceptions."""
    name = "broken"
    description = "Always raises"
    parameters: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("tool is broken")


class _WebFetchDouble(Tool):
    name = "web_fetch"
    description = "Fetch URL"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["url"],
    }

    def execute(self, *, url: str, max_chars: int = 0, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content=f"url={url}|max_chars={max_chars}")


class _WebSearchDouble(Tool):
    name = "web_search"
    description = "Search web"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }

    def execute(self, *, query: str, max_results: int = 0, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, content=f"query={query}|max_results={max_results}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):
    """Verify ToolRegistry registration, dispatch, error handling, and schema output."""

    def setUp(self) -> None:
        """Register all three test-double tools before each test."""
        self.registry = ToolRegistry()
        self.registry.register(_EchoTool())
        self.registry.register(_CountTool())
        self.registry.register(_BrokenTool())
        self.registry.register(_WebFetchDouble())
        self.registry.register(_WebSearchDouble())

    def test_names_contains_registered_tools(self) -> None:
        """names() returns the tool name of every registered tool."""
        names = self.registry.names()
        self.assertIn("echo", names)
        self.assertIn("count", names)
        self.assertIn("broken", names)

    def test_execute_success(self) -> None:
        """A valid call dispatches to the correct tool and returns its ToolResult."""
        result = self.registry.execute(ToolCall(name="echo", args={"message": "hello"}))
        self.assertEqual(result.content, "hello")
        self.assertFalse(result.is_error)
        self.assertEqual(result.tool_name, "echo")

    def test_execute_second_tool(self) -> None:
        """A second registered tool is also dispatched correctly."""
        result = self.registry.execute(ToolCall(name="count", args={"text": "abcde"}))
        self.assertEqual(result.content, "5")
        self.assertFalse(result.is_error)

    def test_execute_unknown_tool_returns_error(self) -> None:
        """Calling an unregistered tool name returns an error ToolResult (no crash)."""
        result = self.registry.execute(ToolCall(name="nonexistent", args={}))
        self.assertTrue(result.is_error)
        self.assertIn("nonexistent", result.content)

    def test_execute_tool_exception_returns_error(self) -> None:
        """An exception raised inside execute() is caught and returned as error ToolResult."""
        result = self.registry.execute(ToolCall(name="broken", args={}))
        self.assertTrue(result.is_error)
        self.assertIn("broken", result.content)

    def test_missing_required_arg_returns_error_without_raise(self) -> None:
        result = self.registry.execute(ToolCall(name="web_fetch", args={}))
        self.assertTrue(result.is_error)
        self.assertIn("requires", result.content)
        self.assertIn("url", result.content)

    def test_auto_repair_href_to_url(self) -> None:
        result = self.registry.execute(ToolCall(name="web_fetch", args={"href": "https://example.com"}))
        self.assertFalse(result.is_error)
        self.assertIn("url=https://example.com", result.content)

    def test_alias_name_and_numeric_string_are_repaired(self) -> None:
        result = self.registry.execute(
            ToolCall(name="search", args={"q": "agadir restaurants", "max_results": "7"})
        )
        self.assertFalse(result.is_error)
        self.assertIn("query=agadir restaurants", result.content)
        self.assertIn("max_results=7", result.content)

    def test_invalid_tool_call_name_returns_actionable_error(self) -> None:
        result = self.registry.execute(
            ToolCall(name="invalid_tool_call", args={"raw": "<tool name=>"})
        )
        self.assertTrue(result.is_error)
        self.assertIn("Malformed tool output", result.content)

    def test_schema_block_contains_tool_info(self) -> None:
        """schema_block() includes each tool's name and description for the system prompt."""
        schema = self.registry.schema_block()
        self.assertIn("echo", schema)
        self.assertIn("Echoes", schema)
        self.assertIn("count", schema)

    def test_schema_block_lists_args(self) -> None:
        """schema_block() includes parameter names so the model knows what args to supply."""
        schema = self.registry.schema_block()
        self.assertIn("message", schema)
        self.assertIn("text", schema)

    def test_register_replaces_same_name(self) -> None:
        """Registering a tool with an existing name replaces the old registration."""
        class _EchoV2(Tool):
            name = "echo"
            description = "Echoes v2"
            parameters: dict[str, Any] = {}

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(tool_name=self.name, content="v2")

        self.registry.register(_EchoV2())
        result = self.registry.execute(ToolCall(name="echo", args={}))
        self.assertEqual(result.content, "v2")

    def test_tool_start_callback_runs_only_after_validation(self) -> None:
        starts: list[tuple[str, dict[str, Any]]] = []
        registry = ToolRegistry(on_tool_start=lambda name, args: starts.append((name, dict(args))))
        registry.register(_WebFetchDouble())

        # Missing required "url" should fail validation and skip on_tool_start.
        failed = registry.execute(ToolCall(name="web_fetch", args={}))
        self.assertTrue(failed.is_error)
        self.assertEqual(starts, [])

        ok = registry.execute(ToolCall(name="web_fetch", args={"url": "https://example.com"}))
        self.assertFalse(ok.is_error)
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][0], "web_fetch")


if __name__ == "__main__":
    unittest.main()
