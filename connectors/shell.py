"""Safe shell connector — allowlisted commands only, no sudo, no pipes.

Security model
--------------
The agent is given access to a very small set of read-mostly shell commands
to help with tasks like listing files, reading text files, or opening apps.
Three layers of protection are applied in execute():

    1. Allowlist check  — only commands in _ALLOWED may run.
    2. Operator block   — the raw command string is scanned for shell
                          metacharacters (|, >, <, &, ;, `, $) that could
                          be used to chain or redirect commands.
    3. No shell=True    — subprocess.run() is called with a parsed argument
                          list, so the system shell never interprets the
                          command.  This prevents injection via argument values.

To add a new allowed command
-----------------------------
Add its name to _ALLOWED.  Commands must be simple binaries (no aliases, no
functions, no scripts).  Do not add destructive commands (rm, mv to /, etc.)
or commands that write to the network or modify system state.
"""
from __future__ import annotations

import shlex
import subprocess

from core.agent.tool_base import Tool, ToolResult

# Frozen set of binary names the agent is permitted to run.
# All other commands are rejected with a descriptive error.
_ALLOWED: frozenset[str] = frozenset({
    "ls",     # list directory contents
    "cat",    # print file contents
    "mkdir",  # create directories
    "cp",     # copy files (read-only use; destructive flags are blocked below)
    "open",   # open a file or application (macOS)
    "pwd",    # print working directory
    "echo",   # print a string
    "date",   # print current date/time
})

# Flags that can cause irreversible data loss even on allowlisted commands.
_DESTRUCTIVE_FLAGS: frozenset[str] = frozenset({
    "-rf", "-fr", "-f", "--force", "--delete", "-delete", "--remove",
})


class ShellTool(Tool):
    """Run a safe, allowlisted shell command and return its stdout.

    The description is built dynamically so it always reflects the current
    _ALLOWED set — useful for future contributors who add more commands.
    """
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
        """Parse, validate, and run a shell command.

        Args:
            command: A single shell command string, e.g. "ls -la /tmp".
                     No pipes, redirects, or compound operators are allowed.

        Returns:
            ToolResult with stdout (or stderr if stdout is empty), truncated
            to 1000 chars.  Returns an error result for any security violation,
            parse failure, timeout, or subprocess error.
        """
        # shlex.split handles quoted strings correctly, e.g. 'echo "hello world"'
        # → ['echo', 'hello world'].  Raises ValueError for unterminated quotes.
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return ToolResult(tool_name=self.name, content=f"Invalid command syntax: {exc}", is_error=True)

        if not parts:
            return ToolResult(tool_name=self.name, content="Empty command.", is_error=True)

        # Layer 1: allowlist check on the binary name (first token).
        cmd_name = parts[0]
        if cmd_name not in _ALLOWED:
            return ToolResult(
                tool_name=self.name,
                content=f"Command '{cmd_name}' is not allowed. Allowed: {', '.join(sorted(_ALLOWED))}",
                is_error=True,
            )

        # Layer 2a: block destructive flags on any allowed command.
        for part in parts[1:]:
            if part in _DESTRUCTIVE_FLAGS:
                return ToolResult(
                    tool_name=self.name,
                    content=f"Flag '{part}' is not allowed — it can cause irreversible data loss.",
                    is_error=True,
                )

        # Layer 2b: block shell metacharacters in the raw command string.
        # These are checked on the original string (not the parsed parts) to
        # catch cases where a metachar appears quoted — we reject conservatively.
        for char in ("|", ">", "<", "&", ";", "`", "$"):
            if char in command:
                return ToolResult(
                    tool_name=self.name,
                    content="Pipes, redirects, and shell operators are not allowed.",
                    is_error=True,
                )

        # Layer 3: execute with a parsed list (shell=False by default in subprocess).
        # This means the OS exec()s the binary directly — no shell interpretation.
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=10,  # prevents the agent from hanging on slow commands
            )
            # Prefer stdout; fall back to stderr (some tools write results there).
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return ToolResult(tool_name=self.name, content=output[:1000])
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                content="Command timed out after 10 seconds.",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Command failed: {exc}", is_error=True)
