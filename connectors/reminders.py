from __future__ import annotations

from ._apple import format_bulleted_section, non_empty_lines, run_osascript


def get_context() -> str:
    """
    Pull incomplete reminders from Apple Reminders.
    Returns a formatted string for LLM context injection.
    """
    script = """
    set output to ""
    tell application "Reminders"
        set allLists to every list
        repeat with aList in allLists
            set theReminders to (every reminder of aList whose completed is false)
            repeat with r in theReminders
                set rName to name of r
                set rDue to ""
                try
                    set rDue to due date of r
                    set rDue to " (due: " & (rDue as string) & ")"
                end try
                set output to output & rName & rDue & "\n"
            end repeat
        end repeat
    end tell
    return output
    """

    raw = run_osascript(script, timeout=10)
    if not raw:
        return ""

    return format_bulleted_section("[Reminders — open items]", non_empty_lines(raw))
