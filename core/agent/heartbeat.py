"""HeartbeatAgent — background daemon that proactively speaks due/overdue reminders.

How it works
------------
A single daemon thread (started by HeartbeatAgent.start()) runs _loop()
forever.  On each iteration it:

    1. Sleeps for heartbeat_interval_seconds.
    2. Calls _tick(), which runs three guard checks before doing anything:
         • DND hours   — skip if current hour is inside [dnd_start_hour, dnd_end_hour).
         • Audio state — skip if the audio coordinator is not IDLE (Kage is
                         already speaking or listening).
         • Debounce    — skip if a proactive message was sent less than
                         proactive_debounce_seconds ago.
    3. If all guards pass, calls _compose_message() to look for due/overdue
       tasks and commitments in EntityStore.
    4. If a message is composed, calls speak() directly.

Thread safety
-------------
The daemon thread only reads EntityStore (via SQLite, which handles its own
locking) and reads AudioCoordinator.state (a simple attribute).  The only
mutable instance state is _last_proactive (a float), written only from the
daemon thread.  No explicit locking is needed.

Configuration (all via .env / config.py)
-----------------------------------------
    HEARTBEAT_INTERVAL_SECONDS  — how often to check (default 300 = 5 min)
    HEARTBEAT_ENABLED           — set to false to disable entirely
    DND_START_HOUR              — do-not-disturb window start, 24 h (default 23)
    DND_END_HOUR                — do-not-disturb window end,   24 h (default 7)
    PROACTIVE_DEBOUNCE_SECONDS  — minimum gap between proactive messages (default 60)

Extending
---------
To add a new proactive trigger (e.g. weather alerts, calendar summaries),
add a new helper method and call it from _compose_message().
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import TYPE_CHECKING

import config

# TYPE_CHECKING guard avoids a circular import at runtime:
#   heartbeat → brain → (everything).  The types are only needed for
#   static analysis / IDE support.
if TYPE_CHECKING:
    from core.audio_coordinator import AudioCoordinator
    from core.brain import BrainService

logger = logging.getLogger(__name__)


class HeartbeatAgent:
    """Background daemon that watches for due items and speaks proactively.

    Args:
        brain:       BrainService instance — used to access the EntityStore.
        coordinator: AudioCoordinator — checked before speaking to ensure
                     Kage is not already in the middle of a conversation.
        settings:    Loaded Settings (heartbeat_interval_seconds, dnd hours, etc.).
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
        # Monotonic timestamp of the last proactive speech event.
        # Starts at 0.0 so the debounce check passes immediately on first tick.
        self._last_proactive: float = 0.0

    def start(self) -> None:
        """Launch the heartbeat daemon thread.

        The thread is a daemon so it automatically terminates when the main
        process exits — no explicit shutdown needed.  Call this once after
        BrainService is fully initialised.
        """
        thread = threading.Thread(target=self._loop, name="kage-heartbeat", daemon=True)
        thread.start()
        logger.info(
            "HeartbeatAgent started (interval=%ds)", self._settings.heartbeat_interval_seconds
        )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Sleep → tick → repeat forever.  Exceptions are logged and swallowed
        so a transient error (e.g. DB locked) never kills the daemon thread.
        """
        while True:
            try:
                time.sleep(self._settings.heartbeat_interval_seconds)
                self._tick()
            except Exception:
                logger.exception("HeartbeatAgent tick failed")

    def _tick(self) -> None:
        """One heartbeat cycle: run guards, compose a message, and speak it."""
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
            return  # nothing due right now

        logger.info("HeartbeatAgent speaking: %s", message[:80])
        # Update last_proactive before speaking so a slow TTS call doesn't
        # allow a second tick to slip through before it returns.
        self._last_proactive = time.monotonic()
        try:
            from core.speaker import speak
            speak(message)
        except Exception:
            logger.exception("HeartbeatAgent speak failed")

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------

    def _in_dnd(self) -> bool:
        """Return True if the current hour falls inside the DND window.

        Handles both same-day windows (e.g. 9–17) and overnight windows
        that cross midnight (e.g. 23–7).
        """
        hour = datetime.now().hour
        start = self._settings.dnd_start_hour
        end = self._settings.dnd_end_hour
        if start > end:
            # Overnight: in-DND when hour >= 23 OR hour < 7
            return hour >= start or hour < end
        # Same-day: in-DND when 9 <= hour < 17
        return start <= hour < end

    def _audio_is_idle(self) -> bool:
        """Return True if Kage is not currently speaking or listening."""
        from core.audio_coordinator import AudioState
        return self._coordinator.state == AudioState.IDLE

    def _debounce_ok(self) -> bool:
        """Return True if enough time has passed since the last proactive message."""
        elapsed = time.monotonic() - self._last_proactive
        return elapsed >= self._settings.proactive_debounce_seconds

    # ------------------------------------------------------------------
    # Message composition
    # ------------------------------------------------------------------

    def _compose_message(self) -> str | None:
        """Look up due/overdue tasks and commitments and format a reminder.

        Returns None if:
          - second_brain is disabled (no _entity_store on brain), or
          - no tasks or commitments are due today or overdue.

        Returns a spoken-word string like:
          "Hey Imad, just a reminder: Finish Q1 report (due today)."
          "Hey Imad, you have 3 things due: Task one (overdue); Task two (due today)."
        """
        # _entity_store is only set when second_brain_enabled=True.
        # Use getattr rather than hasattr so static analysis knows the type.
        entity_store = getattr(self._brain, "_entity_store", None)
        if entity_store is None:
            return None

        today = date.today().isoformat()  # "YYYY-MM-DD" — same format as EntityStore.due_date
        due_items: list[str] = []

        for kind in ("task", "commitment"):
            try:
                entities = entity_store.get_by_kind(kind, status="active")
            except Exception:
                logger.exception("HeartbeatAgent: failed to fetch %s entities", kind)
                continue
            for entity in entities:
                if not entity.due_date or entity.due_date > today:
                    continue  # no due date, or due in the future
                suffix = " (overdue)" if entity.due_date < today else " (due today)"
                due_items.append(entity.value + suffix)

        if not due_items:
            return None

        name = self._settings.user_name
        if len(due_items) == 1:
            return f"Hey {name}, just a reminder: {due_items[0]}."
        # Limit to 3 items in speech to keep the message short; the user can
        # ask "what else?" if they want the full list.
        items_str = "; ".join(due_items[:3])
        return f"Hey {name}, you have {len(due_items)} things due: {items_str}."
