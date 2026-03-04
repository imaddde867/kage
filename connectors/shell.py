"""Safe shell connector — allowlisted commands only, no sudo, no pipes."""
from __future__ import annotations

import shlex
import subprocess

from core.agent.tool_base import Tool, ToolResult

_ALLOWED: frozenset[str] = frozenset({
    "ls", "cat", "mkdir", "mv", "cp", "open", "pwd", "echo", "date",
})


class ShellTool(Tool):
    name = "shell"
    description = (
        f"Run a safe shell command. Allowed commands: {', '.join(sorted(_ALLOWED))}. "
        "No sudo, no pipes, no redirection."
    )
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The shell command to run"}},
        "required": ["command"],
    }

    def execute(self, *, command: str, **kwargs) -> ToolResult:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return ToolResult(tool_name=self.name, content=f"Invalid command syntax: {exc}", is_error=True)

        if not parts:
            return ToolResult(tool_name=self.name, content="Empty command.", is_error=True)

        cmd_name = parts[0]
        if cmd_name not in _ALLOWED:
            return ToolResult(
                tool_name=self.name,
                content=f"Command '{cmd_name}' is not allowed. Allowed: {', '.join(sorted(_ALLOWED))}",
                is_error=True,
            )

        # Block pipes and redirects in the raw command string
        for char in ("|", ">", "<", "&", ";", "`", "$"):
            if char in command:
                return ToolResult(
                    tool_name=self.name,
                    content=f"Pipes, redirects, and shell operators are not allowed.",
                    is_error=True,
                )

        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return ToolResult(tool_name=self.name, content=output[:1000])
        except subprocess.TimeoutExpired:
            return ToolResult(tool_name=self.name, content="Command timed out after 10 seconds.", is_error=True)
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Command failed: {exc}", is_error=True)
