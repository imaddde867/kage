from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

import config
from core.platform.models import Capability, SideEffectLevel


@dataclass
class CapabilityCatalog:
    _capabilities: dict[str, Capability]

    @classmethod
    def build(
        cls,
        *,
        settings: config.Settings,
        tool_names: Iterable[str],
        tool_health_fn: Callable[[str], float] | None = None,
    ) -> "CapabilityCatalog":
        items: dict[str, Capability] = {}

        items["recent_history"] = Capability(
            name="recent_history",
            kind="memory",
            freshness="session",
            latency_class="low",
            cost_class="low",
        )
        items["episodic_memory"] = Capability(
            name="episodic_memory",
            kind="memory",
            freshness="persistent",
            latency_class="low",
            cost_class="low",
        )
        items["facts"] = Capability(
            name="facts",
            kind="knowledge",
            freshness="persistent",
            latency_class="low",
            cost_class="low",
        )

        for name in tool_names:
            side_effect = SideEffectLevel.NONE
            requires_confirmation = False
            supports_citation = name in {"web_search", "web_fetch"}
            freshness = "static"
            if name in {"web_search", "web_fetch", "calendar_read", "shell"}:
                freshness = "live"
            if name in {"reminder_add", "notify", "speak", "shell_mutation", "mark_task_done", "update_fact"}:
                side_effect = SideEffectLevel.LOCAL_MUTATION
                requires_confirmation = name in {"shell_mutation"}
            if name in {"reminder_add"}:
                side_effect = SideEffectLevel.EXTERNAL_MUTATION
            health = 1.0
            if callable(tool_health_fn):
                try:
                    health = float(tool_health_fn(name))
                except Exception:
                    health = 1.0
            items[name] = Capability(
                name=name,
                kind="tool",
                freshness=freshness,
                latency_class="medium",
                cost_class="low",
                side_effect_level=side_effect,
                requires_confirmation=requires_confirmation,
                supports_citation=supports_citation,
                health=max(0.0, min(1.0, health)),
            )

        if settings.second_brain_enabled:
            items["tasks"] = Capability(
                name="tasks",
                kind="knowledge",
                freshness="persistent",
                latency_class="low",
                cost_class="low",
            )

        return cls(items)

    def get(self, name: str) -> Capability | None:
        return self._capabilities.get(name)

    def names(self) -> list[str]:
        return list(self._capabilities.keys())

    def as_list(self) -> list[Capability]:
        return list(self._capabilities.values())
