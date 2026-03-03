from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.second_brain.entity_store import EntityStore


class EntityStoreTests(unittest.TestCase):
    def _store(self) -> EntityStore:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return EntityStore(db_path=Path(tmpdir.name) / "test.db")

    def test_upsert_creates_new_entity(self) -> None:
        store = self._store()
        eid = store.upsert("task", "review_pr", "review the PR")
        self.assertIsNotNone(eid)
        entity = store.get_by_key("task", "review_pr")
        self.assertIsNotNone(entity)
        assert entity is not None
        self.assertEqual(entity.value, "review the PR")
        self.assertEqual(entity.status, "active")

    def test_upsert_updates_existing_entity_by_kind_and_key(self) -> None:
        store = self._store()
        eid1 = store.upsert("task", "review_pr", "review the PR")
        eid2 = store.upsert("task", "review_pr", "review the PR again")
        self.assertEqual(eid1, eid2)
        entity = store.get_by_key("task", "review_pr")
        assert entity is not None
        self.assertEqual(entity.value, "review the PR again")

    def test_get_by_kind_filters_status(self) -> None:
        store = self._store()
        store.upsert("task", "task1", "task one")
        store.upsert("task", "task2", "task two")
        eid3 = store.upsert("task", "task3", "task three")
        store.mark_done(eid3)

        active = store.get_by_kind("task", status="active")
        done = store.get_by_kind("task", status="done")
        self.assertEqual(len(active), 2)
        self.assertEqual(len(done), 1)

    def test_mark_done_changes_status(self) -> None:
        store = self._store()
        eid = store.upsert("task", "finish_report", "finish the report")
        store.mark_done(eid)
        entity = store.get_by_key("task", "finish_report")
        assert entity is not None
        self.assertEqual(entity.status, "done")

    def test_recall_for_prompt_respects_char_budget(self) -> None:
        store = self._store()
        for i in range(20):
            store.upsert("task", f"task_{i}", f"do something important number {i}")
        result = store.recall_for_prompt(char_budget=100)
        self.assertLessEqual(len(result), 100)

    def test_recall_for_prompt_excludes_done_entities(self) -> None:
        store = self._store()
        eid = store.upsert("task", "done_task", "a finished task")
        store.mark_done(eid)
        store.upsert("task", "active_task", "an active task")
        result = store.recall_for_prompt(char_budget=400)
        self.assertIn("an active task", result)
        self.assertNotIn("a finished task", result)


if __name__ == "__main__":
    unittest.main()
