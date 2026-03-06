from __future__ import annotations

from core.intent_signals import DEFAULT_SIGNALS
from core.platform.models import ContextPlan, DecisionPlan, Strategy


class ContextPlanner:
    def plan(
        self,
        *,
        user_input: str,
        decision: DecisionPlan,
        entity_recall_budget: int,
        recent_turns_enabled: bool,
    ) -> ContextPlan:
        reasons: list[str] = []
        sources: list[str] = []

        include_recent = bool(recent_turns_enabled)
        include_memory = bool(recent_turns_enabled)
        entity_mode = "personal_only"

        if include_recent:
            sources.append("recent_history")
            reasons.append("recent_turns_enabled")
        if include_memory:
            sources.append("episodic_memory")
            reasons.append("memory_recall_enabled")

        if decision.strategy in {Strategy.TOOL_PLAN, Strategy.MIXED_EVIDENCE, Strategy.RETRIEVAL_ONLY}:
            sources.append("facts")
            reasons.append("knowledge_context_required")

        task_context = DEFAULT_SIGNALS.has(user_input, "task_context")
        if task_context:
            entity_mode = "full"
            sources.append("tasks")
            reasons.append("task_context_signal")
        elif decision.strategy in {Strategy.TOOL_PLAN, Strategy.MIXED_EVIDENCE}:
            entity_mode = "relevance_filtered"
            reasons.append("agent_context")
        else:
            entity_mode = "personal_only"

        seen: set[str] = set()
        deduped_sources: list[str] = []
        for source in sources:
            if source in seen:
                continue
            seen.add(source)
            deduped_sources.append(source)

        return ContextPlan(
            sources=tuple(deduped_sources),
            reason_codes=tuple(reasons),
            include_memory_recall=include_memory,
            include_recent_turns=include_recent,
            entity_mode=entity_mode,
            char_budget=max(150, int(entity_recall_budget)),
        )

