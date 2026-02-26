from __future__ import annotations

from ._apple import run_osascript

# How many notes to pull (iterates Notes app order; not guaranteed sorted)
_MAX_NOTES = 10
# Max characters per note body before truncating
_MAX_BODY_CHARS = 300


def get_context() -> str:
    """
    Pull recent notes from Apple Notes.
    Returns a formatted string for LLM context injection.
    """
    script = f"""
    set output to ""
    tell application "Notes"
        set theNotes to every note
        set count to 0
        repeat with n in theNotes
            if count >= {_MAX_NOTES} then exit repeat
            set nTitle to name of n
            set nBody to plaintext of n
            if length of nBody > {_MAX_BODY_CHARS} then
                set nBody to text 1 thru {_MAX_BODY_CHARS} of nBody & "..."
            end if
            set output to output & "TITLE: " & nTitle & "\n" & nBody & "\n---\n"
            set count to count + 1
        end repeat
    end tell
    return output
    """

    raw = run_osascript(script, timeout=15)
    if not raw:
        return ""

    return f"[Apple Notes — recent {_MAX_NOTES}]\n{raw.strip()}"
