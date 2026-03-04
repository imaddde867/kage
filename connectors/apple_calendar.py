"""macOS Calendar and Reminders connectors via osascript."""
from __future__ import annotations

import subprocess
from datetime import date

from core.agent.tool_base import Tool, ToolResult


def _escape_as(text: str) -> str:
    """Escape a string for use inside AppleScript double-quoted string literals."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(script: str, timeout: int = 10) -> tuple[str, bool]:
    """Run an AppleScript snippet; return (output, is_error)."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return result.stderr.strip() or "osascript returned non-zero exit code.", True
        return result.stdout.strip(), False
    except FileNotFoundError:
        return "osascript is not available (non-macOS system).", True
    except subprocess.TimeoutExpired:
        return "osascript timed out.", True


class CalendarReadTool(Tool):
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
        due_clause = ""
        if due_date:
            try:
                d = date.fromisoformat(due_date)
                due_clause = (
                    f'set due date of newReminder to date "{d.strftime("%B %d, %Y")}"'
                )
            except ValueError:
                pass  # silently skip malformed date

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
