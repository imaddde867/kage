"""Tests for BrainService streaming and sentence splitting."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from core.brain import BrainService, _SENTENCE_END


def _stream_response(*tokens: str) -> mock.MagicMock:
    """Build a mock streaming requests.Response from token strings."""
    lines = [
        json.dumps({"message": {"content": t}, "done": i == len(tokens) - 1}).encode()
        for i, t in enumerate(tokens)
    ]
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


def _make_brain() -> BrainService:
    settings = SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_timeout_seconds=10,
        ollama_think=False,
        user_name="Tester",
    )
    memory = mock.MagicMock()
    memory.recall.return_value = ""
    connectors = mock.MagicMock()
    connectors.get_all_context.return_value = ""
    return BrainService(settings=settings, memory_store=memory, connector_manager=connectors, session=mock.MagicMock())


class SentenceRegexTests(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self) -> None:
        parts = _SENTENCE_END.split("Hello world. How are you? Fine!")
        self.assertEqual(parts[:-1], ["Hello world.", "How are you?", "Fine!"])
        self.assertEqual(parts[-1], "")

    def test_incomplete_tail_kept(self) -> None:
        parts = _SENTENCE_END.split("Hello world. Still thinking")
        self.assertEqual(parts[:-1], ["Hello world."])
        self.assertEqual(parts[-1], "Still thinking")

    def test_no_split_mid_sentence(self) -> None:
        self.assertEqual(_SENTENCE_END.split("Just one sentence"), ["Just one sentence"])

    def test_trailing_sentence_end(self) -> None:
        parts = _SENTENCE_END.split("Done.")
        self.assertEqual(parts[0], "Done.")
        self.assertEqual(parts[-1], "")


class ThinkStreamTests(unittest.TestCase):
    def test_yields_sentences_to_callback(self) -> None:
        brain = _make_brain()
        brain.session.post.return_value = _stream_response("Hello ", "world. ", "How ", "are ", "you? ", "Fine!")
        received: list[str] = []
        brain.think_stream("hi", received.append)
        self.assertEqual(received, ["Hello world.", "How are you?", "Fine!"])

    def test_returns_full_reply(self) -> None:
        brain = _make_brain()
        brain.session.post.return_value = _stream_response("One. ", "Two.")
        self.assertEqual(brain.think_stream("hi", lambda _: None), "One. Two.")

    def test_persists_memory(self) -> None:
        brain = _make_brain()
        brain.session.post.return_value = _stream_response("Answer.")
        brain.think_stream("question", lambda _: None)
        brain.memory_store.store_exchange.assert_called_once_with("question", "Answer.")

    def test_fallback_on_error(self) -> None:
        brain = _make_brain()
        brain.session.post.side_effect = Exception("network error")
        with mock.patch.object(brain, "think", return_value="fallback") as mock_think:
            received: list[str] = []
            result = brain.think_stream("hi", received.append)
        mock_think.assert_called_once_with("hi")
        self.assertEqual(received, ["fallback"])
        self.assertEqual(result, "fallback")

    def test_no_context_for_unrelated_input(self) -> None:
        brain = _make_brain()
        brain.session.post.return_value = _stream_response("Sure.")
        brain.think_stream("tell me a joke", lambda _: None)
        brain.connector_manager.get_all_context.assert_not_called()

    def test_fetches_context_on_hint(self) -> None:
        brain = _make_brain()
        brain.connector_manager.get_all_context.return_value = "Meeting at 3pm"
        brain.session.post.return_value = _stream_response("You have a meeting.")
        brain.think_stream("what's on my calendar today", lambda _: None)
        brain.connector_manager.get_all_context.assert_called_once()


if __name__ == "__main__":
    unittest.main()
