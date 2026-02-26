from __future__ import annotations

from ._apple import format_bulleted_section, non_empty_lines, run_osascript

_MAX_REMINDERS = 25


def get_context() -> str:
    """
    Pull incomplete reminders from Apple Reminders.
    Returns a formatted string for LLM context injection.
    """
    script = f"""
    set output to ""
    set itemCount to 0
    tell application "Reminders"
        set allLists to every list
        repeat with aList in allLists
            if itemCount >= {_MAX_REMINDERS} then exit repeat
            set theReminders to (every reminder of aList whose completed is false)
            repeat with r in theReminders
                if itemCount >= {_MAX_REMINDERS} then exit repeat
                set rName to name of r
                set rDue to ""
                try
                    set rDue to due date of r
                    set rDue to " (due: " & (rDue as string) & ")"
                end try
                set output to output & rName & rDue & "\n"
                set itemCount to itemCount + 1
            end repeat
        end repeat
    end tell
    return output
    """

    raw = run_osascript(script, timeout=10)
    if not raw:
        return ""

    return format_bulleted_section(
        f"[Reminders — open items (top {_MAX_REMINDERS})]",
        non_empty_lines(raw),
    )
