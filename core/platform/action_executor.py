from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from core.platform.models import ExecutionIntent, RiskTier, TaskStatus
from core.platform.storage import AutonomyTaskStore


class ActionExecutor:
    def __init__(self, *, task_store: AutonomyTaskStore | None = None) -> None:
        self._task_store = task_store

    def _default_intent(self) -> ExecutionIntent:
        return ExecutionIntent(
            intent_id="intent_default",
            action="run_agent",
            risk_tier=RiskTier.SAFE_READ,
            reason_codes=(),
            requires_approval=False,
            constraints={},
        )

    def run_agent(
        self,
        *,
        task: str,
        entity_context: str,
        agent_runner: Callable[[str, str], Iterator[str]],
        intent: ExecutionIntent | None = None,
        agent_summary_getter: Callable[[], Any] | None = None,
    ) -> Iterator[str]:
        active_intent = intent or self._default_intent()
        task_id: str | None = None

        if self._task_store is not None:
            task_id = self._task_store.create_task(
                task_text=task,
                intent=active_intent,
                status=TaskStatus.PLANNED,
            )
            self._task_store.transition(task_id, status=TaskStatus.IN_PROGRESS)

        produced_output = False
        run_summary: Any = None
        try:
            for chunk in agent_runner(task, entity_context):
                produced_output = True
                yield chunk
        except Exception as exc:
            if self._task_store is not None and task_id is not None:
                self._task_store.transition(
                    task_id,
                    status=TaskStatus.FAILED,
                    last_error=str(exc),
                )
            raise
        finally:
            if callable(agent_summary_getter):
                try:
                    run_summary = agent_summary_getter()
                except Exception:
                    run_summary = None

        if self._task_store is None or task_id is None:
            return

        awaiting_approval = bool(getattr(run_summary, "awaiting_approval", False))
        policy_blocked = bool(getattr(run_summary, "policy_blocked", False))
        failed = bool(getattr(run_summary, "failed", False))

        if awaiting_approval or policy_blocked:
            self._task_store.transition(
                task_id,
                status=TaskStatus.AWAITING_APPROVAL,
                last_error="Execution is waiting for policy approval.",
            )
            return

        if failed:
            self._task_store.transition(
                task_id,
                status=TaskStatus.FAILED,
                last_error="Agent run failed before completion.",
            )
            return

        if produced_output:
            self._task_store.transition(task_id, status=TaskStatus.DONE)
            return

        self._task_store.transition(
            task_id,
            status=TaskStatus.BLOCKED,
            last_error="Agent run produced no output.",
        )
