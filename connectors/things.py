from __future__ import annotations

from ._apple import format_bulleted_section, non_empty_lines, run_osascript


def get_context() -> str:
    """
    Pull today's to-dos from Things 3.
    Returns a formatted string for LLM context injection.
    """
    script = """
    if application "Things3" is not running then return ""
    tell application "Things3"
        set output to ""
        set todayItems to to dos of list "Today"
        repeat with t in todayItems
            set tName to name of t
            set tNotes to notes of t
            if tNotes is not "" then
                set output to output & tName & " (note: " & tNotes & ")" & "\n"
            else
                set output to output & tName & "\n"
            end if
        end repeat
        set inboxItems to to dos of list "Inbox"
        repeat with t in inboxItems
            set output to output & "[Inbox] " & name of t & "\n"
        end repeat
        return output
    end tell
    """

    raw = run_osascript(script, timeout=10)
    if not raw:
        return ""

    return format_bulleted_section("[Things 3 — Today & Inbox]", non_empty_lines(raw))
