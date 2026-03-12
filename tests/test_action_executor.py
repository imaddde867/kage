from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.platform.action_executor import ActionExecutor
from core.platform.models import ExecutionIntent, RiskTier
from core.platform.storage import AutonomyTaskStore


class ActionExecutorTests(unittest.TestCase):
    def test_run_agent_marks_task_done_when_output_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = AutonomyTaskStore(db_path)
            executor = ActionExecutor(task_store=store)
            intent = ExecutionIntent(
                intent_id="intent_test",
                action="run_agent",
                risk_tier=RiskTier.SAFE_READ,
                reason_codes=("tools_required",),
            )

            def runner(task: str, context: str):
                _ = (task, context)
                yield "completed"

            out = list(
                executor.run_agent(
                    task="check weather",
                    entity_context="",
                    agent_runner=runner,
                    intent=intent,
                )
            )
            self.assertEqual(out, ["completed"])
            recent = store.list_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].status, "done")

    def test_run_agent_marks_task_failed_on_exception(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = AutonomyTaskStore(db_path)
            executor = ActionExecutor(task_store=store)

            def runner(task: str, context: str):
                _ = (task, context)
                raise RuntimeError("boom")
                yield ""

            with self.assertRaises(RuntimeError):
                list(
                    executor.run_agent(
                        task="do risky thing",
                        entity_context="",
                        agent_runner=runner,
                    )
                )
            recent = store.list_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].status, "failed")
            self.assertIn("boom", recent[0].last_error or "")

    def test_run_agent_marks_awaiting_approval_when_summary_reports_policy_block(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            store = AutonomyTaskStore(db_path)
            executor = ActionExecutor(task_store=store)

            class _Summary:
                awaiting_approval = True
                policy_blocked = True
                failed = False

            def runner(task: str, context: str):
                _ = (task, context)
                yield "Please grant approval to continue."

            out = list(
                executor.run_agent(
                    task="use shell",
                    entity_context="",
                    agent_runner=runner,
                    agent_summary_getter=lambda: _Summary(),
                )
            )
            self.assertEqual(out, ["Please grant approval to continue."])
            recent = store.list_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].status, "awaiting_approval")
            self.assertIn("waiting for policy approval", (recent[0].last_error or "").lower())


if __name__ == "__main__":
    unittest.main()
