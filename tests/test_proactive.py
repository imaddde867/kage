from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.second_brain.entity_store import EntityStore
from core.second_brain.proactive import ProactiveEngine


class ProactiveEngineTests(unittest.TestCase):
    def _make_engine(self, debounce: int = 0) -> tuple[ProactiveEngine, EntityStore]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        store = EntityStore(db_path=Path(tmpdir.name) / "test.db")
        settings = SimpleNamespace(proactive_debounce_seconds=debounce)
        return ProactiveEngine(store, settings), store

    def test_suggest_returns_due_today_task(self) -> None:
        engine, store = self._make_engine(debounce=0)
        store.upsert("task", "standup", "daily standup")
        result = engine.suggest("Let me know if you have questions.", proactive_ok=True)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("standup", result.lower())
        self.assertTrue(result.startswith("By the way,"))

    def test_suggest_skipped_when_already_mentioned_in_reply(self) -> None:
        engine, store = self._make_engine(debounce=0)
        store.upsert("task", "standup", "daily standup")
        result = engine.suggest("You have a daily standup to attend.", proactive_ok=True)
        self.assertIsNone(result)

    def test_suggest_debounce_prevents_rapid_repeat(self) -> None:
        engine, store = self._make_engine(debounce=60)
        store.upsert("task", "standup", "daily standup")
        result1 = engine.suggest("Sure!", proactive_ok=True)
        result2 = engine.suggest("Sure again!", proactive_ok=True)
        self.assertIsNotNone(result1)
        self.assertIsNone(result2)

    def test_suggest_returns_none_when_no_open_entities(self) -> None:
        engine, store = self._make_engine(debounce=0)
        result = engine.suggest("Let me check.", proactive_ok=True)
        self.assertIsNone(result)

    def test_suggest_none_for_general_intent(self) -> None:
        engine, store = self._make_engine(debounce=0)
        store.upsert("task", "standup", "daily standup")
        result = engine.suggest("Let me know!", proactive_ok=False)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
