from __future__ import annotations

from collections.abc import Callable, Iterator


class ActionExecutor:
    def run_agent(
        self,
        *,
        task: str,
        entity_context: str,
        agent_runner: Callable[[str, str], Iterator[str]],
    ) -> Iterator[str]:
        yield from agent_runner(task, entity_context)

