from __future__ import annotations

import unittest
from types import SimpleNamespace

import config
from core.platform.models import Request
from core.platform.orchestrator import RequestOrchestrator


class _Runtime:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, str]] = []
        self.last_trace = None
        self._agent_enabled = True

    def available_tool_names(self) -> list[str]:
        return ["web_search", "web_fetch", "shell"]

    def tool_health(self, _name: str) -> float:
        return 1.0

    def agent_enabled(self) -> bool:
        return self._agent_enabled

    def classify_ambiguous_tool_need(self, _user_input: str) -> bool:
        return False

    def record_decision_trace(self, decision, context_plan) -> None:
        self.last_trace = (decision, context_plan)

    def capability_response(self, user_input: str) -> str | None:
        if "connectors" in user_input.lower():
            return "capability list"
        return None

    def tooling_unavailable_response(self, _user_input: str, *, decision, catalog) -> str:
        _ = (decision, catalog)
        return "tools unavailable fallback"

    def persist_exchange(self, user_input: str, reply: str) -> None:
        self.persisted.append((user_input, reply))

    def agent_context(self, _user_input: str, _context_plan) -> str:
        return "context"

    def agent_runner(self, _task: str, _context: str):
        yield "agent reply"

    def direct_response_stream(self, _user_input: str, *, text_mode: bool, context_plan):
        del text_mode, context_plan
        yield "direct reply"


class RequestOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = config.get()
        self.orchestrator = RequestOrchestrator(settings=self.settings)

    def test_capability_path(self) -> None:
        runtime = _Runtime()
        out = list(
            self.orchestrator.handle(
                Request(text="What connectors can you use?", text_mode=True),
                runtime=runtime,
            )
        )
        self.assertEqual(out, ["capability list"])
        self.assertEqual(runtime.persisted[0][1], "capability list")

    def test_agent_path(self) -> None:
        runtime = _Runtime()
        runtime.classify_ambiguous_tool_need = lambda _text: True
        out = list(
            self.orchestrator.handle(
                Request(text="Search the web for latest updates", text_mode=True),
                runtime=runtime,
            )
        )
        self.assertEqual(out, ["agent reply"])
        self.assertTrue(runtime.persisted)

    def test_direct_path_when_agent_disabled(self) -> None:
        runtime = _Runtime()
        runtime._agent_enabled = False
        out = list(
            self.orchestrator.handle(
                Request(text="Tell me a short joke", text_mode=True),
                runtime=runtime,
            )
        )
        self.assertEqual(out, ["direct reply"])

    def test_tooling_unavailable_path_when_agent_disabled(self) -> None:
        runtime = _Runtime()
        runtime._agent_enabled = False
        out = list(
            self.orchestrator.handle(
                Request(text="Use shell to show current directory and date", text_mode=True),
                runtime=runtime,
            )
        )
        self.assertEqual(out, ["tools unavailable fallback"])
        self.assertEqual(runtime.persisted[-1][1], "tools unavailable fallback")


if __name__ == "__main__":
    unittest.main()
