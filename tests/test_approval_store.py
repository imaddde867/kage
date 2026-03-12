from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.platform.storage import ApprovalStore


class ApprovalStoreTests(unittest.TestCase):
    def test_grant_list_revoke_round_trip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = ApprovalStore(db_path)

            key = store.grant(
                scope_kind="tool",
                scope_name="shell",
                granted_by="test",
                note="unit-test",
            )
            self.assertEqual(key, "tool:shell")
            self.assertTrue(store.is_granted(scope_kind="tool", scope_name="shell"))

            entries = store.list_entries(limit=10)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].approval_key, "tool:shell")
            self.assertEqual(store.count_entries(), 1)

            removed = store.revoke(scope_kind="tool", scope_name="shell")
            self.assertTrue(removed)
            self.assertFalse(store.is_granted(scope_kind="tool", scope_name="shell"))
            self.assertEqual(store.count_entries(), 0)


if __name__ == "__main__":
    unittest.main()
