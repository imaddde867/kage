"""Request orchestration — the decision and control plane for every LLM request.

RequestOrchestrator coordinates the four planning/execution steps that happen
before a response is streamed back to the caller:

  1. CapabilityCatalog build — inspect which tools are registered and healthy.
  2. ExecutionPlanner.plan() — choose strategy (direct / agent / retrieval).
  3. ContextPlanner.plan()  — decide which memory/entity sources to inject.
  4. Dispatch to the right executor (agent loop or direct LLM stream).

BrainService is a compatibility facade that owns the runtime state (model,
memory, tool registry).  All routing decisions belong here.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.platform.action_executor import ActionExecutor
from core.platform.capability_catalog import CapabilityCatalog
from core.platform.context_planner import ContextPlanner
from core.platform.execution_planner import ExecutionPlanner
from core.platform.models import Request, Strategy


class RequestOrchestrator:
    """Decision and control plane for all LLM request handling."""

    def __init__(
        self,
        *,
        settings: Any,
        execution_planner: ExecutionPlanner | None = None,
        context_planner: ContextPlanner | None = None,
        action_executor: ActionExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._execution_planner = execution_planner or ExecutionPlanner()
        self._context_planner = context_planner or ContextPlanner()
        self._action_executor = action_executor or ActionExecutor()

    def _catalog(self, runtime: Any) -> CapabilityCatalog:
        names = runtime.available_tool_names()
        return CapabilityCatalog.build(
            settings=self._settings,
            tool_names=names,
            tool_health_fn=runtime.tool_health,
        )

    def handle(self, request: Request, *, runtime: Any) -> Iterator[str]:
        report_status = getattr(runtime, "report_status", None)
        if callable(report_status):
            report_status("planning", detail="Inspecting request")
        catalog = self._catalog(runtime)
        decision = self._execution_planner.plan(
            user_input=request.text,
            agent_enabled=runtime.agent_enabled(),
            catalog=catalog,
            classify_ambiguous=runtime.classify_ambiguous_tool_need,
        )
        context_plan = self._context_planner.plan(
            user_input=request.text,
            decision=decision,
            entity_recall_budget=getattr(self._settings, "entity_recall_budget", 400),
            recent_turns_enabled=(getattr(self._settings, "recent_turns", 0) > 0),
        )
        runtime.record_decision_trace(decision, context_plan)

        if decision.capability_query:
            if callable(report_status):
                report_status("capability_query", detail="Summarizing available connectors")
            capability = runtime.capability_response(request.text)
            if capability:
                yield capability
                runtime.persist_exchange(request.text, capability)
                return

        if decision.tooling_unavailable:
            if callable(report_status):
                report_status("tools_unavailable", detail="Tooling is disabled for this session")
            fallback = runtime.tooling_unavailable_response(
                request.text,
                decision=decision,
                catalog=catalog,
            )
            if fallback:
                yield fallback
                runtime.persist_exchange(request.text, fallback)
                return

        if decision.use_agent and decision.strategy in {Strategy.TOOL_PLAN, Strategy.MIXED_EVIDENCE}:
            if callable(report_status):
                report_status("checking_tools", detail="Using connectors to gather evidence")
            context = runtime.agent_context(request.text, context_plan)
            required_tiers = ()
            required_tiers_fn = getattr(runtime, "required_approval_tiers", None)
            if callable(required_tiers_fn):
                try:
                    required_tiers = tuple(required_tiers_fn())
                except Exception:
                    required_tiers = ()
            intent = self._execution_planner.build_execution_intent(
                user_input=request.text,
                decision=decision,
                required_approval_tiers=required_tiers,
            )
            summary_getter = getattr(runtime, "agent_run_summary", None)
            if not callable(summary_getter):
                summary_getter = None
            parts: list[str] = []
            for chunk in self._action_executor.run_agent(
                task=request.text,
                entity_context=context,
                agent_runner=runtime.agent_runner,
                intent=intent,
                agent_summary_getter=summary_getter,
            ):
                parts.append(chunk)
                yield chunk
            runtime.persist_exchange(request.text, "".join(parts).strip())
            return

        if callable(report_status):
            report_status("drafting_answer", detail="Streaming response")
        yield from runtime.direct_response_stream(
            request.text,
            text_mode=request.text_mode,
            context_plan=context_plan,
        )
