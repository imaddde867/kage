from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class Strategy(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    RETRIEVAL_ONLY = "retrieval_only"
    TOOL_PLAN = "tool_plan"
    MIXED_EVIDENCE = "mixed_evidence"


class SideEffectLevel(str, Enum):
    NONE = "none"
    LOCAL_MUTATION = "local_mutation"
    EXTERNAL_MUTATION = "external_mutation"


@dataclass(frozen=True)
class Request:
    text: str
    text_mode: bool = False
    source: str = "cli"
    request_id: str | None = None


ResponseStream = Iterator[str]


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    freshness: str = "static"
    latency_class: str = "low"
    cost_class: str = "low"
    side_effect_level: SideEffectLevel = SideEffectLevel.NONE
    requires_confirmation: bool = False
    supports_citation: bool = False
    health: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionPlan:
    strategy: Strategy
    use_agent: bool
    reason_codes: tuple[str, ...]
    capability_query: bool = False
    requires_fresh_data: bool = False
    tooling_unavailable: bool = False


@dataclass(frozen=True)
class ContextPlan:
    sources: tuple[str, ...]
    reason_codes: tuple[str, ...]
    include_memory_recall: bool
    include_recent_turns: bool
    entity_mode: str = "relevance_filtered"
    char_budget: int = 400


@dataclass(frozen=True)
class ProactiveOpportunity:
    kind: str
    message: str
    reason: str
    due_date: str | None = None
    entity_id: str | None = None
