from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from core.platform.policy_engine import PolicyEngine
from core.platform.storage import ApprovalStore


class PolicyEngineTests(unittest.TestCase):
    def _engine(self, *, db_path: Path, mode: str = "strict", tiers: tuple[str, ...] = ("moderate_change", "high_impact")) -> PolicyEngine:
        settings = SimpleNamespace(
            agent_policy_mode=mode,
            agent_approval_required_tiers=tiers,
        )
        store = ApprovalStore(db_path)
        return PolicyEngine(settings=settings, approval_store=store)

    def test_strict_mode_blocks_high_impact_without_approval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="strict")
            decision = engine.evaluate(tool_name="shell", args={"command": "pwd"})
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_approval)
        self.assertIn("awaiting_approval", decision.reason_code)

    def test_tool_approval_unblocks_tool(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = ApprovalStore(db_path)
            store.grant(scope_kind="tool", scope_name="shell", granted_by="test")
            settings = SimpleNamespace(
                agent_policy_mode="strict",
                agent_approval_required_tiers=("moderate_change", "high_impact"),
            )
            engine = PolicyEngine(settings=settings, approval_store=store)
            decision = engine.evaluate(tool_name="shell", args={"command": "pwd"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, "policy_approved")

    def test_owner_fast_allows_moderate_without_explicit_approval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            decision = engine.evaluate(tool_name="update_fact", args={"key": "city", "value": "Helsinki"})
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_hybrid_only_gates_high_impact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="hybrid")
            moderate = engine.evaluate(tool_name="update_fact", args={"key": "city", "value": "Helsinki"})
            high = engine.evaluate(tool_name="shell", args={"command": "pwd"})
        self.assertTrue(moderate.allowed)
        self.assertFalse(moderate.requires_approval)
        self.assertFalse(high.allowed)
        self.assertTrue(high.requires_approval)

    def test_owner_fast_allows_high_impact_without_approval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            decision = engine.evaluate(tool_name="shell", args={"command": "pwd"})
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_strict_mode_allows_local_safe_read_without_approval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="strict")
            decision = engine.evaluate(tool_name="local_extract_pdf", args={"path": "~/Downloads/Lasku.pdf"})
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_unknown_tool_is_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="strict")
            decision = engine.evaluate(tool_name="totally_unknown_tool", args={})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, "policy_unknown_tool")

    # ------------------------------------------------------------------
    # owner_fast mode — all tiers should be pre-approved
    # ------------------------------------------------------------------

    def test_owner_fast_required_tiers_is_empty(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            self.assertEqual(engine.required_tiers(), set())

    def test_owner_fast_allows_safe_read_tools(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            for tool in ("web_search", "web_fetch", "local_read_text"):
                with self.subTest(tool=tool):
                    decision = engine.evaluate(tool_name=tool, args={})
                    self.assertTrue(decision.allowed)
                    self.assertFalse(decision.requires_approval)

    def test_owner_fast_allows_moderate_change_tools(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            for tool in ("reminder_add", "notify", "update_fact"):
                with self.subTest(tool=tool):
                    decision = engine.evaluate(tool_name=tool, args={})
                    self.assertTrue(decision.allowed)
                    self.assertFalse(decision.requires_approval)

    def test_owner_fast_allows_high_impact_without_store_approval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            engine = self._engine(db_path=db_path, mode="owner_fast")
            decision = engine.evaluate(tool_name="shell", args={"command": "echo hi"})
            self.assertTrue(decision.allowed)
            self.assertFalse(decision.requires_approval)
            self.assertEqual(decision.reason_code, "policy_allowed")

    def test_owner_fast_does_not_consult_approval_store(self) -> None:
        """owner_fast bypasses the approval store entirely."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = ApprovalStore(db_path)
            # Do NOT grant any approvals — should still be allowed
            settings = SimpleNamespace(
                agent_policy_mode="owner_fast",
                agent_approval_required_tiers=("moderate_change", "high_impact"),
            )
            engine = PolicyEngine(settings=settings, approval_store=store)
            decision = engine.evaluate(tool_name="shell_mutation", args={})
            self.assertTrue(decision.allowed)
            self.assertFalse(decision.requires_approval)


if __name__ == "__main__":
    unittest.main()
