"""HeartbeatAgent — background daemon that proactively speaks due/overdue reminders.

Respects DND hours and AudioState.IDLE to avoid interrupting the user.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from core.audio_coordinator import AudioCoordinator
    from core.brain import BrainService

logger = logging.getLogger(__name__)


class HeartbeatAgent:
    def __init__(
        self,
        brain: BrainService,
        coordinator: AudioCoordinator,
        settings: config.Settings,
    ) -> None:
        self._brain = brain
        self._coordinator = coordinator
        self._settings = settings
        self._last_proactive: float = 0.0

    def start(self) -> None:
        thread = threading.Thread(target=self._loop, name="kage-heartbeat", daemon=True)
        thread.start()
        logger.info(
            "HeartbeatAgent started (interval=%ds)", self._settings.heartbeat_interval_seconds
        )

    def _loop(self) -> None:
        while True:
            try:
                time.sleep(self._settings.heartbeat_interval_seconds)
                self._tick()
            except Exception:
                logger.exception("HeartbeatAgent tick failed")

    def _in_dnd(self) -> bool:
        hour = datetime.now().hour
        start = self._settings.dnd_start_hour
        end = self._settings.dnd_end_hour
        # Handles overnight windows (e.g. 23–7) and same-day windows (e.g. 9–17)
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    def _audio_is_idle(self) -> bool:
        from core.audio_coordinator import AudioState
        return self._coordinator.state == AudioState.IDLE

    def _debounce_ok(self) -> bool:
        elapsed = time.monotonic() - self._last_proactive
        return elapsed >= self._settings.proactive_debounce_seconds

    def _tick(self) -> None:
        if self._in_dnd():
            logger.debug("HeartbeatAgent: DND active, skipping")
            return
        if not self._audio_is_idle():
            logger.debug("HeartbeatAgent: audio not idle, skipping")
            return
        if not self._debounce_ok():
            logger.debug("HeartbeatAgent: debounce active, skipping")
            return

        message = self._compose_message()
        if not message:
            return

        logger.info("HeartbeatAgent speaking: %s", message[:80])
        self._last_proactive = time.monotonic()
        try:
            from core.speaker import speak
            speak(message)
        except Exception:
            logger.exception("HeartbeatAgent speak failed")

    def _compose_message(self) -> str | None:
        """Return a proactive message for due/overdue items, or None if nothing is due."""
        entity_store = getattr(self._brain, "_entity_store", None)
        if entity_store is None:
            return None

        today = date.today().isoformat()
        due_items: list[str] = []
        for kind in ("task", "commitment"):
            try:
                entities = entity_store.get_by_kind(kind, status="active")
            except Exception:
                logger.exception("HeartbeatAgent: failed to fetch %s entities", kind)
                continue
            for entity in entities:
                if not entity.due_date or entity.due_date > today:
                    continue
                suffix = " (overdue)" if entity.due_date < today else " (due today)"
                due_items.append(entity.value + suffix)

        if not due_items:
            return None

        name = self._settings.user_name
        if len(due_items) == 1:
            return f"Hey {name}, just a reminder: {due_items[0]}."
        items_str = "; ".join(due_items[:3])
        return f"Hey {name}, you have {len(due_items)} things due: {items_str}."
