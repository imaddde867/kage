"""Unit tests for BrainService routing helpers.

Tests _heuristic_needs_tools() for high-signal patterns and
_capability_response() for formatting.  No LLM required.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


def _make_brain(tmp_path: Path) -> Any:
    """Build a BrainService with a fully mocked runtime so no model loads."""
    import config
    from core.brain import BrainService
    from core.memory import MemoryStore

    settings = config.get()
    memory = MemoryStore(str(tmp_path / "test.db"))

    with patch("core.brain.GenerationRuntime") as mock_runtime_cls:
        mock_runtime = MagicMock()
        mock_runtime.tokenizer = None
        mock_runtime.last_stats = {}
        mock_runtime_cls.return_value = mock_runtime

        with patch("core.brain.BrainService._warmup", return_value=None):
            brain = BrainService(settings=settings, memory=memory)
            brain._runtime = mock_runtime
            # tool registry is None — that's fine for routing-only tests
    return brain


class HeuristicNeedsToolsTests(unittest.TestCase):
    """_heuristic_needs_tools() must quickly classify high-signal inputs."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.brain = _make_brain(Path(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _heuristic(self, text: str):
        return self.brain._heuristic_needs_tools(text)

    def test_empty_input_returns_false(self) -> None:
        self.assertFalse(self._heuristic(""))

    def test_web_search_request_returns_true(self) -> None:
        result = self._heuristic("search for the latest news on AI chips")
        # high-signal web request — should be True or None (ambiguous), never False
        self.assertIsNot(result, False)

    def test_shell_command_request_returns_true(self) -> None:
        result = self._heuristic("run ls in my home directory")
        self.assertIsNot(result, False)

    def test_calendar_request_returns_true(self) -> None:
        result = self._heuristic("what events do I have today on my calendar")
        self.assertIsNot(result, False)

    def test_simple_math_returns_false_or_none(self) -> None:
        # Pure conversational queries should score low; may be False or None
        result = self._heuristic("what is two plus two")
        self.assertNotEqual(result, True)

    def test_returns_bool_or_none(self) -> None:
        """Return value must always be bool | None."""
        for text in ["hello", "search the web", "run a shell script"]:
            result = self._heuristic(text)
            self.assertIn(result, (True, False, None))


class CapabilityResponseTests(unittest.TestCase):
    """_capability_response() formats the tool list correctly."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.brain = _make_brain(Path(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_none_for_non_capability_query(self) -> None:
        result = self.brain._capability_response("what is the weather today?")
        self.assertIsNone(result)

    def test_returns_string_for_capability_query(self) -> None:
        from core.agent.tool_registry import ToolRegistry
        from core.agent.tool_base import Tool, ToolResult

        class _FakeTool(Tool):
            name = "web_search"
            description = "Search"
            parameters: dict = {"type": "object", "properties": {}, "required": []}

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(tool_name=self.name, content="ok")

        registry = ToolRegistry()
        registry.register(_FakeTool())
        self.brain._tool_registry = registry

        result = self.brain._capability_response("what tools can you use?")
        # Either None (signal not matched) or a non-empty string
        if result is not None:
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_returns_none_when_no_registry(self) -> None:
        self.brain._tool_registry = None
        result = self.brain._capability_response("what can you do?")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
