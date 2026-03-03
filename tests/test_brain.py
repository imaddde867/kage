"""Tests for BrainService streaming and sentence splitting."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from core.brain import BrainService, _SENTENCE_END


def _make_stream_response(*tokens: str, done_on_last: bool = True) -> mock.MagicMock:
    """Build a mock streaming requests.Response from a list of token strings."""
    lines = []
    for i, token in enumerate(tokens):
        is_done = done_on_last and i == len(tokens) - 1
        lines.append(json.dumps({"message": {"content": token}, "done": is_done}).encode())
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


class SentenceRegexTests(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self) -> None:
        # String ending with '!' → all three are complete sentences; tail is ''.
        parts = _SENTENCE_END.split("Hello world. How are you? Fine!")
        self.assertEqual(parts[:-1], ["Hello world.", "How are you?", "Fine!"])
        self.assertEqual(parts[-1], "")  # empty tail — full sentence at end

    def test_incomplete_tail_kept(self) -> None:
        # String NOT ending with .!? → last fragment is the incomplete buffer tail.
        parts = _SENTENCE_END.split("Hello world. Still thinking")
        self.assertEqual(parts[:-1], ["Hello world."])
        self.assertEqual(parts[-1], "Still thinking")

    def test_no_split_mid_sentence(self) -> None:
        parts = _SENTENCE_END.split("Just one sentence")
        self.assertEqual(parts, ["Just one sentence"])

    def test_trailing_sentence_end(self) -> None:
        # Sentence ending at the string boundary → complete part + empty tail.
        parts = _SENTENCE_END.split("Done.")
        self.assertEqual(parts[0], "Done.")
        self.assertEqual(parts[-1], "")


class ThinkStreamTests(unittest.TestCase):
    def _make_brain(self) -> BrainService:
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
        session = mock.MagicMock()
        brain = BrainService(settings=settings, memory_store=memory, connector_manager=connectors, session=session)
        return brain

    def test_think_stream_yields_sentences_to_callback(self) -> None:
        brain = self._make_brain()
        tokens = ["Hello ", "world. ", "How ", "are ", "you? ", "Fine!"]
        resp = _make_stream_response(*tokens)
        brain.session.post.return_value = resp
        resp.status_code = 200

        received: list[str] = []
        brain.think_stream("hi", received.append)

        self.assertEqual(received, ["Hello world.", "How are you?", "Fine!"])

    def test_think_stream_returns_full_reply(self) -> None:
        brain = self._make_brain()
        tokens = ["One. ", "Two."]
        resp = _make_stream_response(*tokens)
        brain.session.post.return_value = resp

        result = brain.think_stream("hi", lambda _: None)
        self.assertEqual(result, "One. Two.")

    def test_think_stream_persists_memory(self) -> None:
        brain = self._make_brain()
        resp = _make_stream_response("Answer.")
        brain.session.post.return_value = resp

        brain.think_stream("question", lambda _: None)
        brain.memory_store.store_exchange.assert_called_once_with("question", "Answer.")

    def test_think_stream_fallback_on_error(self) -> None:
        brain = self._make_brain()
        brain.session.post.side_effect = Exception("network error")

        # think() itself will also fail and return its error string
        with mock.patch.object(brain, "think", return_value="fallback reply") as mock_think:
            received: list[str] = []
            result = brain.think_stream("hi", received.append)

        mock_think.assert_called_once_with("hi")
        self.assertEqual(received, ["fallback reply"])
        self.assertEqual(result, "fallback reply")

    def test_think_stream_no_context_when_unrelated(self) -> None:
        brain = self._make_brain()
        resp = _make_stream_response("Sure.")
        brain.session.post.return_value = resp

        brain.think_stream("tell me a joke", lambda _: None)
        brain.connector_manager.get_all_context.assert_not_called()

    def test_think_stream_fetches_context_on_hint(self) -> None:
        brain = self._make_brain()
        brain.connector_manager.get_all_context.return_value = "Meeting at 3pm"
        resp = _make_stream_response("You have a meeting.")
        brain.session.post.return_value = resp

        brain.think_stream("what's on my calendar today", lambda _: None)
        brain.connector_manager.get_all_context.assert_called_once()


if __name__ == "__main__":
    unittest.main()
