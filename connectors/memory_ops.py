"""Memory operation tools — mark tasks done, update facts, list open tasks.

These tools wrap EntityStore so the agent can read and mutate the user's
persistent memory directly from a conversation turn.

Why the agent needs these
--------------------------
Without memory tools the agent can only read entity context (injected into
the system prompt at the start of each request).  With these tools it can:
  - Close the loop: mark a task done when the user says "I finished it".
  - Learn new facts: store a preference or profile item the user mentions.
  - Plan: list open tasks before deciding what to remind the user about.

EntityStore schema (for reference)
------------------------------------
    kind      — "task", "commitment", "profile", "preference"
    key       — unique identifier within a kind
    value     — human-readable description / fact
    status    — "active" or "done"
    due_date  — ISO 8601 date string or None

All three tools take the db_path at construction time because EntityStore
opens a new SQLite connection per operation; passing the path avoids holding
a connection open across the potentially long lifetime of BrainService.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.agent.tool_base import Tool, ToolResult

# TYPE_CHECKING guard: EntityStore is only imported for type hints here.
# The actual runtime import happens inside _store() to avoid loading SQLite
# code at module import time.
if TYPE_CHECKING:
    from core.second_brain.entity_store import EntityStore


def _store(db_path: Path) -> EntityStore:
    """Open (or create) an EntityStore at db_path.

    A fresh EntityStore is constructed on every call.  This is intentional:
    SQLite handles its own file-level locking, and keeping per-call
    connections avoids stale state across long-lived objects.
    """
    from core.second_brain.entity_store import EntityStore
    return EntityStore(db_path)


class MarkTaskDoneTool(Tool):
    """Mark a task or commitment as completed in the entity store.

    Matching strategy (tried in order, stops at first match):
    1. Exact key match  — the model passes the exact stored key.
    2. Substring match on value — e.g. "finish report" matches a task whose
       value is "Finish the Q1 report".  Case-insensitive.

    The agent should use this immediately when the user says they completed
    something ("I finished the report", "Done with the call").
    """
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

        # --- Pass 1: exact key lookup (O(1) via index) ---
        for kind in ("task", "commitment"):
            entity = store.get_by_key(kind, key)
            if entity:
                store.mark_done(entity.id)
                return ToolResult(tool_name=self.name, content=f"Marked '{entity.value}' as done.")

        # --- Pass 2: substring match on value (O(n) table scan) ---
        # This handles natural-language references like "I finished the report"
        # when the stored key is something opaque like "task_3a8f".
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
    """Store or update a structured fact about the user in the entity store.

    This is an upsert: if a record with the same (kind, key) already exists
    its value is overwritten; otherwise a new record is created.

    The agent uses this to persist things the user mentions in conversation
    that are not automatically extracted by LLMEntityExtractor — e.g. an
    explicit preference stated mid-task ("remember I prefer dark mode").
    """
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
    """Return a formatted summary of all active tasks and commitments.

    The agent uses this at the start of a planning or review session to
    see what the user has on their plate before deciding what to surface
    or act on.  The output is the same format as what's injected into
    the normal chat system prompt via EntityStore.recall_for_prompt().
    """
    name = "list_open_tasks"
    description = "List all active tasks and commitments in memory"
    # No required arguments — always returns everything active.
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, **kwargs) -> ToolResult:
        # char_budget=800 gives more detail than the 400-char system-prompt
        # injection; the observation is only seen by the model, not spoken.
        content = _store(self._db_path).recall_for_prompt(char_budget=800)
        if not content:
            return ToolResult(tool_name=self.name, content="No active tasks or commitments.")
        return ToolResult(tool_name=self.name, content=content)
