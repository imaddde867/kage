"""Notification connectors — macOS system notification and TTS speech.

Two tools are provided:

    NotifyTool  — displays a macOS notification banner via osascript.
                  Useful when Kage completes a long background task and wants
                  to alert the user without interrupting what they're doing.

    SpeakTool   — calls core.speaker.speak() directly so the agent can trigger
                  speech from within an agentic tool chain (e.g. the heartbeat
                  daemon, or a multi-step workflow that ends with a spoken result).

AppleScript injection prevention
---------------------------------
Both tools escape user-supplied strings before interpolating them into
AppleScript.  _escape_as() handles backslash and double-quote characters,
which are the only characters that can break out of an AppleScript
double-quoted string literal.
"""
from __future__ import annotations

import subprocess

from core.agent.tool_base import Tool, ToolResult


def _escape_as(text: str) -> str:
    """Escape a Python string for safe use inside an AppleScript string literal.

    AppleScript string literals are delimited by double quotes.  The only
    characters that need escaping inside them are:
        \\  →  \\\\   (backslash must be doubled first, before quoting)
        "   →  \\"    (double quote must be escaped)

    Args:
        text: Raw Python string that will be embedded in AppleScript.

    Returns:
        Escaped string safe to place between AppleScript double quotes.

    Example:
        _escape_as('say "hi"')  →  'say \\\\"hi\\\\"'
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


class NotifyTool(Tool):
    """Display a macOS notification banner via osascript.

    The notification appears in the macOS Notification Center and (if the
    user has alerts enabled for Script Editor / osascript) as a banner on
    screen.  Does not require the user to be listening — good for alerting
    the user after a background task completes.
    """
    name = "notify"
    description = "Show a macOS system notification with a title and message"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Notification body text"},
            "title": {"type": "string", "description": "Notification title (optional)"},
        },
        "required": ["message"],
    }

    def execute(self, *, message: str, title: str = "Kage", **kwargs) -> ToolResult:
        """Show a macOS notification.

        Both message and title are escaped before AppleScript interpolation
        to prevent any user-supplied text from breaking the script.

        Args:
            message: Body text of the notification.
            title:   Banner title; defaults to "Kage".

        Returns:
            Success ToolResult, or error if osascript is unavailable or fails.
        """
        # Build the AppleScript one-liner with properly escaped strings.
        script = (
            f'display notification "{_escape_as(message)}" '
            f'with title "{_escape_as(title)}"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                timeout=5,
                check=True,       # raises CalledProcessError on non-zero exit
                capture_output=True,
            )
            return ToolResult(tool_name=self.name, content=f"Notification sent: {message!r}")
        except subprocess.CalledProcessError as exc:
            # stderr may be bytes (Python ≥ 3.12 captures as bytes by default)
            # or a string depending on the subprocess flags used.
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ToolResult(tool_name=self.name, content=f"osascript error: {stderr}", is_error=True)
        except FileNotFoundError:
            # osascript is only available on macOS.
            return ToolResult(
                tool_name=self.name,
                content="osascript not available (non-macOS?)",
                is_error=True,
            )


class SpeakTool(Tool):
    """Speak a message aloud using Kage's TTS system (Kokoro via mlx-audio).

    This tool lets the agent trigger speech mid-chain — for example, the
    heartbeat daemon composes a message and calls speak() directly, bypassing
    the normal voice-loop response path.

    The core.speaker module is imported lazily inside execute() so this
    connector loads cleanly even when mlx-audio is not installed (e.g. in
    CI or text-only environments).
    """
    name = "speak"
    description = "Speak a message aloud using the TTS system"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Text to speak"}},
        "required": ["message"],
    }

    def execute(self, *, message: str, **kwargs) -> ToolResult:
        try:
            from core.speaker import speak
            speak(message)
            return ToolResult(tool_name=self.name, content=f"Spoke: {message!r}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Speak failed: {exc}", is_error=True)
