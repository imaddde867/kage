from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.platform.storage.knowledge_store import KnowledgeStore


@dataclass
class Entity:
    id: str
    kind: str
    key: str
    value: str
    status: str
    due_date: Optional[str]
    source_id: Optional[str]
    created_at: str
    updated_at: str


class EntityStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._store = KnowledgeStore(self.db_path)

    def _init_schema(self) -> None:
        self._store = KnowledgeStore(self.db_path)

    def upsert(
        self,
        kind: str,
        key: str,
        value: str,
        *,
        status: str = "active",
        due_date: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> str:
        return self._store.upsert(
            kind,
            key,
            value,
            status=status,
            due_date=due_date,
            source_id=source_id,
        )

    def get_by_kind(self, kind: str, *, status: str = "active") -> list[Entity]:
        rows = self._store.get_by_kind(kind, status=status)
        return [
            Entity(
                id=row.id,
                kind=row.kind,
                key=row.key,
                value=row.value,
                status=row.status,
                due_date=row.due_date,
                source_id=row.source_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    def get_by_key(self, kind: str, key: str) -> Optional[Entity]:
        entity = self._store.get_by_key(kind, key)
        if entity is None:
            return None
        return Entity(
            id=entity.id,
            kind=entity.kind,
            key=entity.key,
            value=entity.value,
            status=entity.status,
            due_date=entity.due_date,
            source_id=entity.source_id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    def mark_done(self, entity_id: str) -> None:
        self._store.mark_done(entity_id)

    def recall_personal_context(self, *, char_budget: int = 150) -> str:
        """Profile + preferences only — always safe to inject regardless of intent."""
        sections: list[str] = []

        profiles = self.get_by_kind("profile", status="active")
        if profiles:
            items = [f"{e.key}={e.value}" for e in profiles]
            sections.append(f"Profile: {', '.join(items)}")

        prefs = self.get_by_kind("preference", status="active")
        if prefs:
            items = [e.value for e in prefs]
            sections.append(f"Preferences: {', '.join(items)}")

        result = "\n".join(sections)
        if len(result) > char_budget:
            result = result[: char_budget - 3].rstrip() + "..."
        return result

    def recall_for_prompt(self, *, char_budget: int = 400) -> str:
        sections: list[str] = []

        tasks = self.get_by_kind("task", status="active")
        if tasks:
            items = [
                f"{e.value} (due {e.due_date})" if e.due_date else e.value
                for e in tasks
            ]
            sections.append(f"Tasks: {', '.join(items)}")

        commitments = self.get_by_kind("commitment", status="active")
        if commitments:
            items = [
                f"{e.value} ({e.due_date})" if e.due_date else e.value
                for e in commitments
            ]
            sections.append(f"Commitments: {', '.join(items)}")

        profiles = self.get_by_kind("profile", status="active")
        if profiles:
            items = [f"{e.key}={e.value}" for e in profiles]
            sections.append(f"Profile: {', '.join(items)}")

        prefs = self.get_by_kind("preference", status="active")
        if prefs:
            items = [e.value for e in prefs]
            sections.append(f"Preferences: {', '.join(items)}")

        result = "\n".join(sections)
        if len(result) > char_budget:
            result = result[: char_budget - 3].rstrip() + "..."
        return result
