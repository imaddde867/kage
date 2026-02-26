"""
Connectors — pull live context from your apps into Jarvis/Kage.

Each connector exposes a get_context() -> str function.
Call get_all_context() to get everything at once, ready for LLM injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import calendar, notes, reminders

logger = logging.getLogger(__name__)

ConnectorCallable = Callable[[], str]


@dataclass
class ConnectorManager:
    connectors: Iterable[ConnectorCallable] = field(
        default_factory=lambda: [calendar.get_context, reminders.get_context, notes.get_context]
    )

    def __post_init__(self) -> None:
        self.connectors = list(self.connectors)

    def get_all_context(self) -> str:
        """
        Run all connectors and return combined context string.
        Silently skips any connector that fails or returns nothing.
        """
        sections: list[str] = []

        for connector in self.connectors:
            try:
                ctx = (connector() or "").strip()
            except Exception:
                logger.exception("Connector failed: %s", getattr(connector, "__name__", repr(connector)))
                continue
            if ctx:
                sections.append(ctx)

        return "\n\n".join(sections)


_DEFAULT_MANAGER: ConnectorManager | None = None


def get_default_manager() -> ConnectorManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = ConnectorManager()
    return _DEFAULT_MANAGER


def get_all_context() -> str:
    return get_default_manager().get_all_context()


__all__ = ["ConnectorManager", "get_default_manager", "get_all_context"]
