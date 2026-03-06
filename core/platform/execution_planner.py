from __future__ import annotations

from collections.abc import Callable

from core.intent_signals import DEFAULT_SIGNALS
from core.platform.capability_catalog import CapabilityCatalog
from core.platform.models import DecisionPlan, Strategy

_ROUTING_SIGNAL_WEIGHTS: dict[str, float] = {
    "capability_query": -2.0,
    "calendar_lookup": 2.0,
    "live_web": 1.5,
    "needs_tools": 1.0,
}
_ROUTING_SCORE_TOOLS_THRESHOLD = 1.0
_ROUTING_SCORE_NO_TOOLS_THRESHOLD = -1.0


class ExecutionPlanner:
    def _heuristic_needs_tools(self, text: str) -> bool | None:
        prompt = text.strip()
        if not prompt:
            return False
        score = DEFAULT_SIGNALS.weighted_score(prompt, _ROUTING_SIGNAL_WEIGHTS)
        if score >= _ROUTING_SCORE_TOOLS_THRESHOLD:
            return True
        if score <= _ROUTING_SCORE_NO_TOOLS_THRESHOLD:
            return False
        return None

    def needs_tools(
        self,
        *,
        user_input: str,
        classify_ambiguous: Callable[[str], bool],
    ) -> bool:
        heuristic = self._heuristic_needs_tools(user_input)
        if heuristic is not None:
            return heuristic
        return classify_ambiguous(user_input)

    def plan(
        self,
        *,
        user_input: str,
        agent_enabled: bool,
        catalog: CapabilityCatalog,
        classify_ambiguous: Callable[[str], bool],
    ) -> DecisionPlan:
        reasons: list[str] = []
        capability_query = DEFAULT_SIGNALS.has(user_input, "capability_query")
        if capability_query:
            reasons.append("capability_query")
            return DecisionPlan(
                strategy=Strategy.DIRECT_ANSWER,
                use_agent=False,
                reason_codes=tuple(reasons),
                capability_query=True,
                requires_fresh_data=False,
            )

        needs_tools = self.needs_tools(
            user_input=user_input,
            classify_ambiguous=classify_ambiguous,
        )
        live_web = DEFAULT_SIGNALS.has(user_input, "live_web")
        task_context = DEFAULT_SIGNALS.has(user_input, "task_context")

        if needs_tools and not agent_enabled:
            reasons.append("tools_required")
            reasons.append("tools_unavailable")
            if live_web:
                reasons.append("fresh_data_required")
            return DecisionPlan(
                strategy=Strategy.DIRECT_ANSWER,
                use_agent=False,
                reason_codes=tuple(reasons),
                requires_fresh_data=live_web,
                tooling_unavailable=True,
            )

        if needs_tools and agent_enabled:
            reasons.append("tools_required")
            if live_web:
                reasons.append("fresh_data_required")
                web_fetch = catalog.get("web_fetch")
                if web_fetch is not None and web_fetch.health < 0.5:
                    reasons.append("web_fetch_degraded")
            if DEFAULT_SIGNALS.has(user_input, "calendar_lookup"):
                calendar = catalog.get("calendar_read")
                if calendar is not None and calendar.health < 0.5:
                    reasons.append("calendar_read_degraded")
            strategy = Strategy.MIXED_EVIDENCE if task_context else Strategy.TOOL_PLAN
            return DecisionPlan(
                strategy=strategy,
                use_agent=True,
                reason_codes=tuple(reasons),
                requires_fresh_data=live_web,
            )

        if task_context and catalog.get("episodic_memory") is not None:
            reasons.append("task_context_memory")
            return DecisionPlan(
                strategy=Strategy.RETRIEVAL_ONLY,
                use_agent=False,
                reason_codes=tuple(reasons),
                requires_fresh_data=False,
            )

        reasons.append("direct_generation")
        return DecisionPlan(
            strategy=Strategy.DIRECT_ANSWER,
            use_agent=False,
            reason_codes=tuple(reasons),
            requires_fresh_data=False,
        )
