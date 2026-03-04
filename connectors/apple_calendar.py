"""macOS Calendar and Reminders connectors via osascript (AppleScript bridge).

Two tools are provided:

    CalendarReadTool  — queries the macOS Calendar app for upcoming events
                        in the next N days and returns them as a text list.

    ReminderAddTool   — creates a new item in the macOS Reminders app with
                        an optional due date.

AppleScript bridge
------------------
Both tools use subprocess + osascript to drive the Calendar and Reminders
applications.  This requires:
  - macOS (osascript is not available on Linux/Windows).
  - Accessibility permissions granted to whichever terminal/app runs Kage.

AppleScript injection prevention
---------------------------------
Any user-supplied text (e.g. reminder titles) is passed through _escape_as()
before being embedded in the AppleScript string.  _escape_as() is defined
locally here (mirroring the implementation in notify.py) rather than shared,
keeping each connector self-contained with no cross-connector imports.

The CalendarReadTool does NOT accept user text in the AppleScript body —
only the integer `days` parameter is interpolated, cast to int() first.
"""
from __future__ import annotations

import subprocess
from datetime import date

from core.agent.tool_base import Tool, ToolResult


def _escape_as(text: str) -> str:
    """Escape a Python string for safe embedding in an AppleScript string literal.

    Only backslashes and double quotes require escaping inside AppleScript
    double-quoted strings.  Backslashes must be doubled first to avoid
    double-processing when the quote escape is applied.

    Args:
        text: Raw Python string to embed in AppleScript.

    Returns:
        Escaped string safe to place between AppleScript double quotes.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(script: str, timeout: int = 10) -> tuple[str, bool]:
    """Execute an AppleScript string and return (output, is_error).

    Args:
        script:  Complete AppleScript text to run.
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        A tuple of (output_string, is_error_bool).
        On success: (stdout.strip(), False).
        On failure: (error_message, True).
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            # osascript writes error messages to stderr
            return result.stderr.strip() or "osascript returned non-zero exit code.", True
        return result.stdout.strip(), False
    except FileNotFoundError:
        return "osascript is not available (non-macOS system).", True
    except subprocess.TimeoutExpired:
        return "osascript timed out.", True


class CalendarReadTool(Tool):
    """Return upcoming Calendar events for the next N days.

    Iterates over all calendars visible in the Calendar app and returns
    event summaries with their start times.  Output is capped at 1500 chars
    to keep the observation size manageable for the model.

    The `days` argument is cast to int() inside the AppleScript interpolation
    to prevent injection via a non-integer value.
    """
    name = "calendar_read"
    description = "Read upcoming events from macOS Calendar for the next N days"
    parameters = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Number of days ahead to check (default 3)"},
        },
        "required": [],
    }

    def execute(self, *, days: int = 3, **kwargs) -> ToolResult:
        """Query Calendar for events in the next `days` days.

        Args:
            days: Look-ahead window in days.  Defaults to 3.

        Returns:
            ToolResult with one "event at datetime" line per event,
            or a "no events found" message if the calendar is clear.
        """
        # int(days) cast prevents AppleScript injection if the model passes a
        # non-integer value.  The AppleScript 'days' keyword is a time unit.
        script = f"""
tell application "Calendar"
    set startDate to (current date)
    set endDate to startDate + ({int(days)} * days)
    set resultText to ""
    repeat with c in every calendar
        set evts to every event of c whose start date >= startDate and start date <= endDate
        repeat with e in evts
            set resultText to resultText & (summary of e) & " at " & (start date of e) & "\\n"
        end repeat
    end repeat
    return resultText
end tell
"""
        output, is_error = _run_osascript(script)
        if is_error:
            return ToolResult(tool_name=self.name, content=output, is_error=True)
        if not output:
            return ToolResult(tool_name=self.name, content=f"No events found in the next {days} days.")
        return ToolResult(tool_name=self.name, content=output[:1500])


class ReminderAddTool(Tool):
    """Add a new item to macOS Reminders with an optional due date.

    The reminder is created in the default Reminders list.  If due_date is
    provided it must be in ISO 8601 format (YYYY-MM-DD); malformed dates are
    silently ignored and the reminder is created without a due date.

    The title is escaped via _escape_as() before AppleScript interpolation.
    """
    name = "reminder_add"
    description = "Add a reminder to macOS Reminders with an optional due date"
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format (optional)"},
        },
        "required": ["title"],
    }

    def execute(self, *, title: str, due_date: str | None = None, **kwargs) -> ToolResult:
        """Create a Reminders item.

        Args:
            title:    The reminder text (will be escaped before AppleScript use).
            due_date: Optional ISO 8601 date string.  If provided and valid,
                      the due date is set on the reminder.  Invalid strings
                      are silently skipped.

        Returns:
            ToolResult confirming the reminder was created, or an error if
            osascript failed (e.g. permission denied, app not running).
        """
        # Build the optional "set due date" AppleScript clause.
        due_clause = ""
        if due_date:
            try:
                d = date.fromisoformat(due_date)
                # AppleScript 'date "Month DD, YYYY"' is the most portable date format.
                due_clause = (
                    f'set due date of newReminder to date "{d.strftime("%B %d, %Y")}"'
                )
            except ValueError:
                pass  # Silently skip — the reminder is still created without a due date

        # title is escaped to prevent AppleScript injection.
        script = f"""
tell application "Reminders"
    set newReminder to make new reminder with properties {{name:"{_escape_as(title)}"}}
    {due_clause}
end tell
"""
        output, is_error = _run_osascript(script)
        if is_error:
            return ToolResult(tool_name=self.name, content=output, is_error=True)
        suffix = f" (due {due_date})" if due_date else ""
        return ToolResult(tool_name=self.name, content=f"Reminder added: {title!r}{suffix}")
