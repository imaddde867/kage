from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.session import SessionEvent

try:
    from core.textual_chat import Composer, EmptyState, KageChatApp
except ImportError:  # pragma: no cover - optional dependency in local env
    Composer = None  # type: ignore[assignment]
    EmptyState = None  # type: ignore[assignment]
    KageChatApp = None  # type: ignore[assignment]


class _FakeController:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            assistant_name="Kage",
            llm_backend="fake",
            mlx_model="fake/model",
            agent_enabled=True,
            second_brain_enabled=True,
        )
        self.is_busy = False
        self.last_answer = ""
        self._events: list[SessionEvent] = []
        self._sources: list[str] = []
        self.reset_calls = 0

    def submit(self, text: str) -> None:
        self.is_busy = True
        self._sources = ["https://example.com"]
        self._events.extend(
            [
                SessionEvent(kind="status_changed", session_id="s", text="Planning", data={"status": "thinking"}),
                SessionEvent(kind="tool_started", session_id="s", text="web_search", data={"tool_name": "web_search", "args": {"query": text}}),
                SessionEvent(kind="tool_finished", session_id="s", text="web_search", data={"tool_name": "web_search", "status": "ok", "preview": "1 result"}),
                SessionEvent(kind="source_added", session_id="s", text="https://example.com", data={"tool_name": "web_search", "source": "https://example.com"}),
                SessionEvent(kind="assistant_chunk", session_id="s", text="Hello"),
                SessionEvent(kind="assistant_chunk", session_id="s", text=" world"),
                SessionEvent(kind="assistant_done", session_id="s", text="Hello world"),
                SessionEvent(kind="metrics_updated", session_id="s", data={"tok_per_sec": 55.0}),
            ]
        )
        self.last_answer = "Hello world"

    def next_event(self, timeout: float | None = None):
        del timeout
        if not self._events:
            self.is_busy = False
            return None
        event = self._events.pop(0)
        if event.kind == "assistant_done":
            self.is_busy = False
        return event

    def current_sources(self) -> list[str]:
        return list(self._sources)

    def recent_history(self, *, limit: int = 8) -> list[tuple[str, str]]:
        return [("What happened?", "A concise answer.")][:limit]

    def memory_summary(self, *, char_budget: int = 700) -> str:
        del char_budget
        return "Tasks: review design."

    def connector_summary(self) -> str:
        return "Agent mode is enabled.\n- web_search: 100% recent success"

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_answer = ""
        self._sources = []

    def close(self) -> None:
        return None


@unittest.skipUnless(KageChatApp is not None, "textual not installed")
class TextualChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_state_is_rendered_on_startup(self) -> None:
        app = KageChatApp(controller=_FakeController())
        async with app.run_test() as pilot:
            transcript = pilot.app.query_one("#transcript")
            self.assertTrue(any(isinstance(child, EmptyState) for child in transcript.children))

    async def test_message_submission_streams_reply_and_tool_rows(self) -> None:
        controller = _FakeController()
        app = KageChatApp(controller=controller, timing=True)
        async with app.run_test() as pilot:
            composer = pilot.app.query_one(Composer)
            composer.insert("hello")
            await pilot.press("enter")
            await asyncio.sleep(0.25)
            self.assertIsNotNone(pilot.app._turn_widgets.assistant)
            self.assertEqual(pilot.app._turn_widgets.assistant._content, "Hello world")
            self.assertEqual(len(pilot.app._tool_rows), 1)

    async def test_sidebar_toggle_and_copy_action(self) -> None:
        controller = _FakeController()
        controller.last_answer = "Ready to copy"
        app = KageChatApp(controller=controller)
        async with app.run_test() as pilot:
            sidebar = pilot.app.query_one("#sidebar")
            self.assertTrue(sidebar.has_class("-hidden"))
            pilot.app.action_toggle_sidebar()
            self.assertFalse(sidebar.has_class("-hidden"))
            with patch("core.textual_chat.copy_to_clipboard") as clipboard:
                pilot.app.action_copy_last_answer()
            clipboard.assert_called_once_with("Ready to copy")

    async def test_new_chat_clears_transcript(self) -> None:
        controller = _FakeController()
        app = KageChatApp(controller=controller)
        async with app.run_test() as pilot:
            composer = pilot.app.query_one(Composer)
            composer.insert("hello")
            await pilot.press("enter")
            await asyncio.sleep(0.25)
            pilot.app.action_new_chat()
            transcript = pilot.app.query_one("#transcript")
            self.assertTrue(any(isinstance(child, EmptyState) for child in transcript.children))
            self.assertEqual(controller.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
