"""Memory operation tools — mark tasks done, update facts, list open tasks.

These wrap EntityStore so the agent can read and mutate its own memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.agent.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from core.second_brain.entity_store import EntityStore


def _store(db_path: Path) -> EntityStore:
    from core.second_brain.entity_store import EntityStore
    return EntityStore(db_path)


class MarkTaskDoneTool(Tool):
    name = "mark_task_done"
    description = "Mark a task or commitment as completed in memory"
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Task key or description to mark done"},
        },
        "required": ["key"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, *, key: str, **kwargs) -> ToolResult:
        store = _store(self._db_path)
        key_lower = key.lower()

        # Exact key match first (cheap)
        for kind in ("task", "commitment"):
            entity = store.get_by_key(kind, key)
            if entity:
                store.mark_done(entity.id)
                return ToolResult(tool_name=self.name, content=f"Marked '{entity.value}' as done.")

        # Substring match on value (user said "I finished the report" → matches task value)
        for kind in ("task", "commitment"):
            for entity in store.get_by_kind(kind, status="active"):
                if key_lower in entity.value.lower():
                    store.mark_done(entity.id)
                    return ToolResult(tool_name=self.name, content=f"Marked '{entity.value}' as done.")

        return ToolResult(
            tool_name=self.name,
            content=f"No active task or commitment matching '{key}' found.",
            is_error=True,
        )


class UpdateFactTool(Tool):
    name = "update_fact"
    description = "Store or update a fact about the user in memory"
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "Entity kind: profile, preference, task, commitment",
            },
            "key": {"type": "string", "description": "Unique key for this fact"},
            "value": {"type": "string", "description": "The fact value to store"},
        },
        "required": ["kind", "key", "value"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, *, kind: str, key: str, value: str, **kwargs) -> ToolResult:
        _store(self._db_path).upsert(kind, key, value)
        return ToolResult(tool_name=self.name, content=f"Stored {kind}/{key} = {value!r}.")


class ListOpenTasksTool(Tool):
    name = "list_open_tasks"
    description = "List all active tasks and commitments in memory"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, **kwargs) -> ToolResult:
        content = _store(self._db_path).recall_for_prompt(char_budget=800)
        if not content:
            return ToolResult(tool_name=self.name, content="No active tasks or commitments.")
        return ToolResult(tool_name=self.name, content=content)
