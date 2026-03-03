from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def _store(self) -> MemoryStore:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return MemoryStore(db_path=Path(tmpdir.name) / "memory.db")

    def test_recent_turns_returns_latest_in_chronological_order(self) -> None:
        store = self._store()
        store.store_exchange("u1", "a1")
        store.store_exchange("u2", "a2")
        store.store_exchange("u3", "a3")

        turns = store.recent_turns(limit=2)
        self.assertEqual(turns, [("u2", "a2"), ("u3", "a3")])

    def test_recent_turns_truncate_long_entries(self) -> None:
        store = self._store()
        store.store_exchange("abcdefghijklmnopqrstuvwxyz", "reply")

        turns = store.recent_turns(limit=1, max_chars=12)
        self.assertEqual(len(turns), 1)
        user_text, _ = turns[0]
        self.assertEqual(user_text, "abcdefghi...")
        self.assertEqual(len(user_text), 12)

    def test_recall_prefers_overlap_and_deduplicates(self) -> None:
        store = self._store()
        rel_user = "Let's finalize meeting schedule for tomorrow"
        rel_reply = "First meeting is at nine."
        store.store_exchange(rel_user, rel_reply)
        store.store_exchange(rel_user, rel_reply)  # duplicate
        store.store_exchange("What is your favorite color?", "Blue.")
        store.store_exchange("Project schedule draft is in notion", "Please review it tonight.")

        ctx = store.recall("meeting schedule tomorrow", n_results=5, char_budget=1200)
        lower = ctx.lower()

        self.assertIn("meeting schedule for tomorrow", lower)
        self.assertNotIn("favorite color", lower)
        self.assertEqual(lower.count("meeting schedule for tomorrow"), 1)

    def test_recall_respects_character_budget(self) -> None:
        store = self._store()
        store.store_exchange("meeting schedule tomorrow", "done")
        store.store_exchange("meeting schedule tomorrow", "done again")

        ctx = store.recall("meeting schedule tomorrow", n_results=5, char_budget=90)
        self.assertLessEqual(len(ctx), 90)


if __name__ == "__main__":
    unittest.main()
