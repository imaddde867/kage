"""CronDaemon — background daemon that fires scheduled reminders when due.

How it works
------------
A single daemon thread (started by CronDaemon.start()) runs _loop() forever.
Every cron_poll_interval_seconds it calls _tick(), which:

    1. Guard: DND hours active → return.
    2. Guard: coordinator.state != IDLE → return (Kage is busy).
    3. Query EntityStore for kind="scheduled_reminder", status="active".
    4. Filter: due_date <= datetime.now().isoformat() AND id not in _fired.
    5. For each due reminder: add to _fired, mark_done in EntityStore, speak().
    6. Sleep 2s between consecutive reminders in the same tick.

Thread safety
-------------
The daemon thread only reads EntityStore (SQLite handles locking) and checks
AudioCoordinator.state.  The _fired set is only written from the daemon thread.
No explicit locking is needed.

Configuration (all via .env / config.py)
-----------------------------------------
    CRON_ENABLED               — master switch (default false)
    CRON_POLL_INTERVAL_SECONDS — seconds between polls (default 60)

Prerequisite: SECOND_BRAIN_ENABLED=true
    CronDaemon returns early from _tick() if _entity_store is not set on brain.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from core.audio_coordinator import AudioCoordinator
    from core.brain import BrainService

logger = logging.getLogger(__name__)

_INTER_REMINDER_SLEEP_SECONDS = 2


class CronDaemon:
    """Background daemon that fires scheduled reminders when their due time arrives.

    Args:
        brain:       BrainService — used to access EntityStore via _entity_store.
        coordinator: AudioCoordinator — checked before speaking.
        settings:    Loaded Settings (cron_poll_interval_seconds, dnd hours).
    """

    def __init__(
        self,
        brain: BrainService,
        coordinator: AudioCoordinator,
        settings: config.Settings,
    ) -> None:
        self._brain = brain
        self._coordinator = coordinator
        self._settings = settings
        # In-memory guard: prevents double-fire if mark_done races with next tick.
        # On restart, entity is already "done" in SQLite so it won't appear in query.
        self._fired: set[str] = set()

    def start(self) -> None:
        """Launch the cron daemon thread (daemon=True, auto-terminates with process)."""
        thread = threading.Thread(target=self._loop, name="kage-cron", daemon=True)
        thread.start()
        logger.info(
            "CronDaemon started (poll_interval=%ds)",
            self._settings.cron_poll_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            try:
                time.sleep(self._settings.cron_poll_interval_seconds)
                self._tick()
            except Exception:
                logger.exception("CronDaemon tick failed")

    def _tick(self) -> None:
        if self._in_dnd():
            logger.debug("CronDaemon: DND active, skipping")
            return
        if not self._audio_is_idle():
            logger.debug("CronDaemon: audio not idle, skipping")
            return

        entity_store = getattr(self._brain, "_entity_store", None)
        if entity_store is None:
            logger.debug("CronDaemon: _entity_store not set (SECOND_BRAIN_ENABLED=false?), skipping")
            return

        try:
            reminders = entity_store.get_by_kind("scheduled_reminder", status="active")
        except Exception:
            logger.exception("CronDaemon: failed to query reminders")
            return

        now_iso = datetime.now().isoformat(timespec="seconds")
        due = [
            r for r in reminders
            if r.due_date is not None
            and r.due_date <= now_iso
            and r.id not in self._fired
        ]

        if not due:
            return

        # Sort by due_date so earliest fires first
        due.sort(key=lambda r: r.due_date or "")

        for i, reminder in enumerate(due):
            self._fired.add(reminder.id)
            try:
                entity_store.mark_done(reminder.id)
            except Exception:
                logger.exception("CronDaemon: failed to mark reminder done: %s", reminder.id)

            message = f"Reminder: {reminder.value}"
            logger.info("CronDaemon firing reminder: %s", message[:80])
            try:
                from core.speaker import speak
                speak(message)
            except Exception:
                logger.exception("CronDaemon: speak failed for reminder %s", reminder.id)

            if i < len(due) - 1:
                time.sleep(_INTER_REMINDER_SLEEP_SECONDS)

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------

    def _in_dnd(self) -> bool:
        hour = datetime.now().hour
        start = self._settings.dnd_start_hour
        end = self._settings.dnd_end_hour
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    def _audio_is_idle(self) -> bool:
        from core.audio_coordinator import AudioState
        return self._coordinator.state == AudioState.IDLE
