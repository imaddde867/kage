"""Tests for prune() and count() on platform storage stores.

Uses TemporaryDirectory so no persistent state is left on disk.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.platform.storage import ConversationStore, EvidenceStore, TraceStore


def _db(tmp: str) -> Path:
    return Path(tmp) / "memory.db"


class ConversationStorePruneTests(unittest.TestCase):
    def test_count_empty_is_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            self.assertEqual(store.count(), 0)

    def test_count_reflects_inserts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            for i in range(5):
                store.store_exchange(f"user {i}", f"reply {i}")
            self.assertEqual(store.count(), 5)

    def test_prune_keeps_exact_n(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            for i in range(10):
                store.store_exchange(f"user {i}", f"reply {i}")
            deleted = store.prune(keep_last=4)
            self.assertEqual(deleted, 6)
            self.assertEqual(store.count(), 4)

    def test_prune_keep_more_than_exist_deletes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            for i in range(3):
                store.store_exchange(f"user {i}", f"reply {i}")
            deleted = store.prune(keep_last=100)
            self.assertEqual(deleted, 0)
            self.assertEqual(store.count(), 3)

    def test_prune_keep_zero_deletes_all(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            for i in range(5):
                store.store_exchange(f"user {i}", f"reply {i}")
            deleted = store.prune(keep_last=0)
            self.assertEqual(deleted, 5)
            self.assertEqual(store.count(), 0)

    def test_recall_still_works_after_prune(self) -> None:
        with TemporaryDirectory() as tmp:
            store = ConversationStore(_db(tmp))
            for i in range(8):
                store.store_exchange(f"machine learning query {i}", f"reply about ML {i}")
            store.prune(keep_last=3)
            result = store.recall("machine learning")
            self.assertIsInstance(result, str)


class EvidenceStorePruneTests(unittest.TestCase):
    def _insert(self, store: EvidenceStore, n: int) -> None:
        for i in range(n):
            store.record(
                tool_name="web_search",
                status="ok",
                query_text=f"query {i}",
                content=f"result {i}",
                structured=None,
                sources=[],
                latency_ms=50.0,
            )

    def test_count_and_prune(self) -> None:
        with TemporaryDirectory() as tmp:
            store = EvidenceStore(_db(tmp))
            self._insert(store, 10)
            self.assertEqual(store.count(), 10)
            deleted = store.prune(keep_last=3)
            self.assertEqual(deleted, 7)
            self.assertEqual(store.count(), 3)

    def test_prune_keep_zero_deletes_all(self) -> None:
        with TemporaryDirectory() as tmp:
            store = EvidenceStore(_db(tmp))
            self._insert(store, 5)
            store.prune(keep_last=0)
            self.assertEqual(store.count(), 0)


class TraceStorePruneTests(unittest.TestCase):
    def _insert(self, store: TraceStore, n: int) -> None:
        for i in range(n):
            store.record(
                event_kind="tool_call",
                event_name="web_search",
                status="ok",
                latency_ms=float(i),
            )

    def test_count_and_prune(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TraceStore(_db(tmp))
            self._insert(store, 12)
            self.assertEqual(store.count(), 12)
            deleted = store.prune(keep_last=5)
            self.assertEqual(deleted, 7)
            self.assertEqual(store.count(), 5)

    def test_prune_keep_zero_deletes_all(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TraceStore(_db(tmp))
            self._insert(store, 4)
            store.prune(keep_last=0)
            self.assertEqual(store.count(), 0)

    def test_tool_health_still_works_after_prune(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TraceStore(_db(tmp))
            self._insert(store, 20)
            store.prune(keep_last=5)
            health = store.tool_health("web_search")
            self.assertGreaterEqual(health, 0.0)
            self.assertLessEqual(health, 1.0)


if __name__ == "__main__":
    unittest.main()
