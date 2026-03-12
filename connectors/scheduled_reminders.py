"""Scheduled reminder tools — store, list, and cancel time-based reminders.

Reminders are stored in EntityStore with kind="scheduled_reminder" and a
due_date ISO datetime string.  The CronDaemon (core/agent/cron_daemon.py)
polls every 60s and fires reminders that have come due.

Requires SECOND_BRAIN_ENABLED=true (EntityStore must be available).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from core.agent.tool_base import Tool, ToolResult

# Accepted ISO datetime formats (naive local time)
_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M%z",
)


def _parse_when(when: str) -> Optional[str]:
    """Parse a datetime string and return a naive local ISO string, or None on failure.

    Accepts ISO formats with or without timezone offset.  Timezone info is
    stripped so the stored string compares correctly against datetime.now().isoformat().
    Zero-padded ISO format guarantees lexicographic ordering is chronological.
    """
    text = (when or "").strip()
    if not text:
        return None
    for fmt in _ISO_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            # Strip tzinfo — store as naive local time
            return dt.replace(tzinfo=None).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def _store(db_path: Path):  # type: ignore[return]
    from core.second_brain.entity_store import EntityStore
    return EntityStore(db_path)


class ScheduleReminderTool(Tool):
    """Schedule a reminder to fire at a specific date/time.

    The CronDaemon checks every 60 seconds and speaks the reminder when due.
    Requires SECOND_BRAIN_ENABLED=true and CRON_ENABLED=true.
    """

    name = "schedule_reminder"
    description = "Schedule a spoken reminder at a specific date and time"
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The reminder message to speak when due",
            },
            "when": {
                "type": "string",
                "description": (
                    "When to fire the reminder — ISO datetime string: "
                    "YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM, or YYYY-MM-DD"
                ),
            },
        },
        "required": ["message", "when"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, *, message: str, when: str, **kwargs) -> ToolResult:
        due_date = _parse_when(when)
        if due_date is None:
            return ToolResult(
                tool_name=self.name,
                content=(
                    f"Could not parse datetime {when!r}. "
                    "Use ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM"
                ),
                is_error=True,
            )

        key = f"reminder_{due_date.replace(':', '').replace('-', '').replace('T', '_')}"
        _store(self._db_path).upsert(
            "scheduled_reminder",
            key,
            message,
            due_date=due_date,
        )
        return ToolResult(
            tool_name=self.name,
            content=f"Reminder scheduled for {due_date}: {message!r}",
        )


class ListScheduledTool(Tool):
    """List all active scheduled reminders, sorted by due date."""

    name = "list_scheduled_reminders"
    description = "List all pending scheduled reminders"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, **kwargs) -> ToolResult:
        reminders = _store(self._db_path).get_by_kind("scheduled_reminder", status="active")
        if not reminders:
            return ToolResult(tool_name=self.name, content="No pending reminders.")

        sorted_reminders = sorted(reminders, key=lambda e: e.due_date or "")
        lines = [
            f"- [{e.due_date or 'no date'}] {e.value}"
            for e in sorted_reminders
        ]
        return ToolResult(
            tool_name=self.name,
            content="Scheduled reminders:\n" + "\n".join(lines),
        )


class CancelReminderTool(Tool):
    """Cancel a scheduled reminder by message content or key.

    Matching strategy (tried in order, stops at first match):
    1. Exact key match.
    2. Substring match on message value (case-insensitive).
    """

    name = "cancel_reminder"
    description = "Cancel a pending reminder by message content or key"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Reminder key or message substring to cancel",
            },
        },
        "required": ["query"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, *, query: str, **kwargs) -> ToolResult:
        store = _store(self._db_path)
        query_lower = query.lower()

        # Pass 1: exact key match
        entity = store.get_by_key("scheduled_reminder", query)
        if entity and entity.status == "active":
            store.mark_done(entity.id)
            return ToolResult(
                tool_name=self.name,
                content=f"Cancelled reminder: {entity.value!r}",
            )

        # Pass 2: substring match on value
        for entity in store.get_by_kind("scheduled_reminder", status="active"):
            if query_lower in entity.value.lower():
                store.mark_done(entity.id)
                return ToolResult(
                    tool_name=self.name,
                    content=f"Cancelled reminder: {entity.value!r}",
                )

        return ToolResult(
            tool_name=self.name,
            content=f"No active reminder matching {query!r} found.",
            is_error=True,
        )
