from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from core.agent.tool_base import ToolOutcome, ToolResult
from core.runtime_events import StatusUpdate
from core.session import SessionController


class _FakeBrain:
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.settings = SimpleNamespace(
            assistant_name="Kage",
            llm_backend="fake",
            mlx_model="fake/model",
            second_brain_enabled=False,
            agent_enabled=True,
        )
        self.last_stats = {"backend": "fake", "tokens": 12, "tok_per_sec": 42.0}
        self._observers = []
        self.memory = SimpleNamespace(recent_turns=lambda limit=4: [("u1", "a1")][:limit])

    def add_observer(self, observer) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        self._observers = [candidate for candidate in self._observers if candidate is not observer]

    def available_tool_names(self) -> list[str]:
        return ["web_search"]

    def tool_health(self, _name: str) -> float:
        return 1.0

    def think_text_stream(self, prompt: str):
        if self.mode == "error":
            raise RuntimeError("boom")

        for observer in list(self._observers):
            observer.on_status_changed(StatusUpdate(status="checking_tools", detail="Looking up sources"))
            observer.on_tool_started("web_search", {"query": prompt})

        time.sleep(0.01)
        result = ToolResult(
            tool_name="web_search",
            content='{"results": 1}',
            outcome=ToolOutcome(
                status="ok",
                structured={"results": 1},
                sources=["https://example.com"],
                retryable=False,
                latency_ms=18.0,
                side_effects=False,
            ),
        )
        for observer in list(self._observers):
            observer.on_tool_finished(result)
            observer.on_source_added("https://example.com", tool_name="web_search")
            observer.on_status_changed(StatusUpdate(status="drafting_answer", detail="Streaming reply"))

        chunks = ["Hello", " world"]
        if self.mode == "slow":
            chunks = ["alpha", " beta", " gamma", " delta"]
        for chunk in chunks:
            time.sleep(0.02)
            yield chunk


class SessionControllerTests(unittest.TestCase):
    def _collect_events(self, controller: SessionController) -> list:
        events = []
        while True:
            event = controller.next_event(timeout=0.0)
            if event is None:
                return events
            events.append(event)

    def test_event_ordering_and_sources(self) -> None:
        controller = SessionController(brain=_FakeBrain())
        self._collect_events(controller)  # drop session_started

        controller.submit("find something")
        self.assertTrue(controller.wait_until_idle(1.0))

        events = self._collect_events(controller)
        kinds = [event.kind for event in events]
        self.assertIn("user_message", kinds)
        self.assertIn("tool_started", kinds)
        self.assertIn("tool_finished", kinds)
        self.assertIn("source_added", kinds)
        self.assertIn("assistant_chunk", kinds)
        self.assertIn("assistant_done", kinds)
        self.assertIn("metrics_updated", kinds)
        self.assertLess(kinds.index("tool_started"), kinds.index("tool_finished"))
        self.assertLess(kinds.index("tool_finished"), kinds.index("assistant_done"))
        self.assertEqual(controller.current_sources(), ["https://example.com"])

    def test_cancel_marks_turn_as_cancelled(self) -> None:
        controller = SessionController(brain=_FakeBrain(mode="slow"))
        self._collect_events(controller)

        controller.submit("slow request")
        time.sleep(0.05)
        controller.cancel()
        self.assertTrue(controller.wait_until_idle(1.0))

        events = self._collect_events(controller)
        done_events = [event for event in events if event.kind == "assistant_done"]
        self.assertTrue(done_events)
        self.assertTrue(done_events[-1].data.get("cancelled"))

    def test_error_event_is_emitted(self) -> None:
        controller = SessionController(brain=_FakeBrain(mode="error"))
        self._collect_events(controller)

        controller.submit("explode")
        self.assertTrue(controller.wait_until_idle(1.0))

        events = self._collect_events(controller)
        errors = [event for event in events if event.kind == "error"]
        statuses = [event for event in events if event.kind == "status_changed"]
        self.assertTrue(errors)
        self.assertIn("boom", errors[-1].text)
        self.assertTrue(any(event.data.get("status") == "error" for event in statuses))

    def test_reset_clears_stale_events(self) -> None:
        controller = SessionController(brain=_FakeBrain())
        self._collect_events(controller)

        controller.submit("find something")
        self.assertTrue(controller.wait_until_idle(1.0))

        # Intentionally do not drain queue before reset.
        controller.reset()
        events = self._collect_events(controller)
        self.assertEqual([event.kind for event in events], ["session_started"])


if __name__ == "__main__":
    unittest.main()
