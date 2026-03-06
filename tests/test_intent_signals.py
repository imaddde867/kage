from __future__ import annotations

import unittest

from core.intent_signals import DEFAULT_SIGNALS, IntentSignals, SignalRule


class TestIntentSignals(unittest.TestCase):
    def test_weighted_scoring(self) -> None:
        signals = IntentSignals(
            (
                SignalRule("x", r"hello", 0.6),
                SignalRule("x", r"world", 0.6),
            )
        )
        self.assertFalse(signals.has("hello there", "x"))
        self.assertTrue(signals.has("hello world", "x"))

    def test_default_capability_query(self) -> None:
        self.assertTrue(DEFAULT_SIGNALS.has("What connectors can you use?", "capability_query"))

    def test_default_calendar_lookup(self) -> None:
        self.assertTrue(DEFAULT_SIGNALS.has("When is my next dentist visit?", "calendar_lookup"))

    def test_default_live_web(self) -> None:
        self.assertTrue(DEFAULT_SIGNALS.has("Any latest updates right now?", "live_web"))

    def test_default_hardware_compare_needs_tools(self) -> None:
        text = "Is the new MacBook Neo or my local machine better?"
        self.assertTrue(DEFAULT_SIGNALS.has(text, "needs_tools"))

    def test_shell_system_request_needs_tools(self) -> None:
        text = "Use shell to show current directory and date."
        self.assertTrue(DEFAULT_SIGNALS.has(text, "needs_tools"))

    def test_reminder_request_needs_tools(self) -> None:
        text = "Add a reminder: Pay rent due 2026-03-09."
        self.assertTrue(DEFAULT_SIGNALS.has(text, "needs_tools"))

    def test_memory_write_request_needs_tools(self) -> None:
        text = "Remember this preference: I prefer concise answers."
        self.assertTrue(DEFAULT_SIGNALS.has(text, "needs_tools"))

    def test_degraded_connector_query_counts_as_capability_query(self) -> None:
        text = "Which connectors look degraded right now based on recent failures?"
        self.assertTrue(DEFAULT_SIGNALS.has(text, "capability_query"))

    def test_weighted_score_supports_mixed_intents(self) -> None:
        score = DEFAULT_SIGNALS.weighted_score(
            "what connectors can you use right now",
            {"capability_query": -2.0, "live_web": 1.0},
        )
        self.assertLess(score, 0.0)


if __name__ == "__main__":
    unittest.main()
