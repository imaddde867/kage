from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from core.platform.proactive_policy import ProactivePolicyEngine

if TYPE_CHECKING:
    from core.second_brain.entity_store import EntityStore


class ProactiveEngine:
    def __init__(self, entity_store: "EntityStore", settings: Any) -> None:
        self._entity_store = entity_store
        self._settings = settings
        self._last_suggestion_time: float = 0.0
        self._policy = ProactivePolicyEngine()

    def suggest(self, reply: str, *, proactive_ok: bool) -> Optional[str]:
        if not proactive_ok:
            return None

        debounce = getattr(self._settings, "proactive_debounce_seconds", 60)
        now = time.time()
        if now - self._last_suggestion_time < debounce:
            return None
        suggestion = self._policy.suggest_from_reply(
            entity_store=self._entity_store,
            settings=self._settings,
            reply=reply,
            proactive_ok=proactive_ok,
        )
        if suggestion:
            self._last_suggestion_time = now
        return suggestion
