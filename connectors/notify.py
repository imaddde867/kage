"""Notification connectors — macOS system notification and TTS speak."""
from __future__ import annotations

import subprocess

from core.agent.tool_base import Tool, ToolResult


def _escape_as(text: str) -> str:
    """Escape a string for use inside AppleScript double-quoted string literals."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


class NotifyTool(Tool):
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
        script = (
            f'display notification "{_escape_as(message)}" '
            f'with title "{_escape_as(title)}"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                timeout=5,
                check=True,
                capture_output=True,
            )
            return ToolResult(tool_name=self.name, content=f"Notification sent: {message!r}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ToolResult(tool_name=self.name, content=f"osascript error: {stderr}", is_error=True)
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                content="osascript not available (non-macOS?)",
                is_error=True,
            )


class SpeakTool(Tool):
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
