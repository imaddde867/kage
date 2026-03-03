from __future__ import annotations

from ._apple import format_bulleted_section, non_empty_lines, run_osascript


def get_context() -> str:
    """
    Pull today's and tomorrow's events from Apple Calendar.
    Returns a formatted string for LLM context injection.
    """
    script = """
    if application "Calendar" is not running then return ""
    set output to ""
    set today to current date
    set startOfDay to today - (time of today)
    set endOfTomorrow to startOfDay + (2 * days) - 1

    tell application "Calendar"
        repeat with aCal in calendars
            set theEvents to (every event of aCal whose start date >= startOfDay and start date <= endOfTomorrow)
            repeat with e in theEvents
                set eTitle to summary of e
                set eStart to start date of e
                set output to output & eTitle & " | " & (eStart as string) & "\n"
            end repeat
        end repeat
    end tell
    return output
    """

    raw = run_osascript(script, timeout=10)
    if not raw:
        return ""

    return format_bulleted_section("[Calendar — today & tomorrow]", non_empty_lines(raw))
