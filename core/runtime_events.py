from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.agent.tool_base import ToolResult


@dataclass(frozen=True)
class StatusUpdate:
    status: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeObserver(Protocol):
    def on_status_changed(self, update: StatusUpdate) -> None: ...

    def on_tool_started(self, tool_name: str, args: dict[str, Any]) -> None: ...

    def on_tool_finished(self, result: ToolResult) -> None: ...

    def on_source_added(self, source: str, *, tool_name: str = "") -> None: ...

    def on_error(
        self,
        message: str,
        *,
        recoverable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
