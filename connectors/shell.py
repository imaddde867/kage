"""Bounded shell connectors with explicit side-effect separation.

`ShellTool` is the default read-only connector registered in the tool registry.
`ShellMutationTool` is opt-in and requires an explicit confirmation token.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from core.agent.tool_base import Tool, ToolOutcome, ToolResult

_ALLOWED_READONLY: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "pwd",
        "echo",
        "date",
        "uname",
        "sw_vers",
        "sysctl",
        "system_profiler",
    }
)
_ALLOWED_MUTATION: frozenset[str] = frozenset({"mkdir", "cp", "open"})
_DESTRUCTIVE_FLAGS: frozenset[str] = frozenset(
    {"-rf", "-fr", "-f", "--force", "--delete", "-delete", "--remove"}
)
_CONFIRM_TOKEN = "YES_I_UNDERSTAND_LOCAL_MUTATION"


def _has_meta_operators(command: str) -> bool:
    return any(char in command for char in ("|", ">", "<", "&", ";", "`", "$"))


def _normalized_parts(command: str) -> tuple[list[str] | None, str | None]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return None, f"Invalid command syntax: {exc}"
    if not parts:
        return None, "Empty command."
    if _has_meta_operators(command):
        return None, "Pipes, redirects, and shell operators are not allowed."
    return parts, None


def _result(
    *,
    tool_name: str,
    content: str,
    is_error: bool,
    latency_ms: float | None = None,
    side_effects: bool = False,
) -> ToolResult:
    status = "error" if is_error else "ok"
    return ToolResult(
        tool_name=tool_name,
        content=content,
        is_error=is_error,
        outcome=ToolOutcome(
            status=status,
            structured=None,
            sources=[],
            retryable=is_error,
            latency_ms=latency_ms,
            side_effects=side_effects,
        ),
    )


class _BaseShellTool(Tool):
    _allowed: frozenset[str]
    _is_mutating: bool = False

    def _validate(self, parts: list[str], *, command: str) -> str | None:
        cmd_name = parts[0]
        if cmd_name not in self._allowed:
            return f"Command '{cmd_name}' is not allowed. Allowed: {', '.join(sorted(self._allowed))}"
        if not self._is_mutating:
            for part in parts[1:]:
                if part in _DESTRUCTIVE_FLAGS:
                    return f"Flag '{part}' is not allowed."
        return None

    def _run(self, parts: list[str], *, timeout: int = 10) -> ToolResult:
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return _result(
                tool_name=self.name,
                content=output[:1200],
                is_error=False,
                side_effects=self._is_mutating,
            )
        except subprocess.TimeoutExpired:
            return _result(
                tool_name=self.name,
                content=f"Command timed out after {timeout} seconds.",
                is_error=True,
                side_effects=self._is_mutating,
            )
        except Exception as exc:
            return _result(
                tool_name=self.name,
                content=f"Command failed: {exc}",
                is_error=True,
                side_effects=self._is_mutating,
            )


class ShellReadOnlyTool(_BaseShellTool):
    name = "shell"
    description = (
        "Run read-only shell commands. Allowed commands: "
        + ", ".join(sorted(_ALLOWED_READONLY))
        + "."
    )
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Read-only shell command"}},
        "required": ["command"],
    }
    _allowed = _ALLOWED_READONLY
    _is_mutating = False

    def execute(self, *, command: str, **kwargs) -> ToolResult:
        del kwargs
        parts, err = _normalized_parts(command)
        if parts is None:
            return _result(tool_name=self.name, content=str(err), is_error=True)
        err = self._validate(parts, command=command)
        if err:
            return _result(tool_name=self.name, content=err, is_error=True)
        return self._run(parts, timeout=10)


class ShellMutationTool(_BaseShellTool):
    name = "shell_mutation"
    description = (
        "Run local mutating shell commands with explicit confirmation token. "
        f"Allowed commands: {', '.join(sorted(_ALLOWED_MUTATION))}."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Mutating shell command"},
            "confirm_token": {
                "type": "string",
                "description": f"Must equal '{_CONFIRM_TOKEN}'",
            },
            "allowed_root": {
                "type": "string",
                "description": "Optional absolute path that command arguments must stay within",
            },
        },
        "required": ["command", "confirm_token"],
    }
    _allowed = _ALLOWED_MUTATION
    _is_mutating = True

    def _within_root(self, parts: Iterable[str], allowed_root: str) -> bool:
        root = Path(allowed_root).expanduser().resolve()
        for part in parts:
            if part.startswith("-"):
                continue
            if part.startswith("http://") or part.startswith("https://"):
                continue
            candidate = Path(part).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            try:
                resolved = candidate.resolve()
            except Exception:
                return False
            if root not in resolved.parents and resolved != root:
                return False
        return True

    def execute(
        self,
        *,
        command: str,
        confirm_token: str,
        allowed_root: str | None = None,
        **kwargs,
    ) -> ToolResult:
        del kwargs
        if confirm_token != _CONFIRM_TOKEN:
            return _result(
                tool_name=self.name,
                content="Mutation command blocked: missing/invalid confirmation token.",
                is_error=True,
                side_effects=True,
            )
        parts, err = _normalized_parts(command)
        if parts is None:
            return _result(tool_name=self.name, content=str(err), is_error=True, side_effects=True)
        err = self._validate(parts, command=command)
        if err:
            return _result(tool_name=self.name, content=err, is_error=True, side_effects=True)
        if allowed_root and not self._within_root(parts[1:], allowed_root):
            return _result(
                tool_name=self.name,
                content=f"Mutation command blocked: path escapes allowed_root={allowed_root!r}.",
                is_error=True,
                side_effects=True,
            )
        return self._run(parts, timeout=15)


# Backward-compatible alias used by existing tests/imports.
class ShellTool(ShellReadOnlyTool):
    pass
