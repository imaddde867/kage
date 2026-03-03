from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.second_brain.entity_store import EntityStore


class ProactiveEngine:
    def __init__(self, entity_store: "EntityStore", settings: Any) -> None:
        self._entity_store = entity_store
        self._settings = settings
        self._last_suggestion_time: float = 0.0

    def suggest(self, reply: str, *, proactive_ok: bool) -> Optional[str]:
        if not proactive_ok:
            return None

        debounce = getattr(self._settings, "proactive_debounce_seconds", 60)
        now = time.time()
        if now - self._last_suggestion_time < debounce:
            return None

        entities = []
        entities.extend(self._entity_store.get_by_kind("task", status="active"))
        entities.extend(self._entity_store.get_by_kind("commitment", status="active"))

        if not entities:
            return None

        reply_lower = reply.lower()
        for entity in entities:
            if entity.value.lower() in reply_lower:
                continue

            if entity.due_date:
                suggestion = f"By the way, you have {entity.value} due {entity.due_date}."
            else:
                suggestion = f"By the way, you have an open task: {entity.value}."

            self._last_suggestion_time = now
            return suggestion

        return None
