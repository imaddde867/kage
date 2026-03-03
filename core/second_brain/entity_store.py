from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


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
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id         TEXT PRIMARY KEY,
                    kind       TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    status     TEXT DEFAULT 'active',
                    due_date   TEXT,
                    source_id  TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(kind, key)")

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
        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM entities WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
            if row:
                entity_id = row[0]
                conn.execute(
                    "UPDATE entities SET value=?, status=?, due_date=?, source_id=?, updated_at=? WHERE id=?",
                    (value, status, due_date, source_id, now, entity_id),
                )
                return entity_id
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO entities
                        (id, kind, key, value, status, due_date, source_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (entity_id, kind, key, value, status, due_date, source_id, now, now),
                )
                return entity_id

    def get_by_kind(self, kind: str, *, status: str = "active") -> list[Entity]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, key, value, status, due_date, source_id, created_at, updated_at
                FROM entities
                WHERE kind=? AND status=?
                ORDER BY updated_at DESC
                """,
                (kind, status),
            ).fetchall()
        return [Entity(*row) for row in rows]

    def get_by_key(self, kind: str, key: str) -> Optional[Entity]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, kind, key, value, status, due_date, source_id, created_at, updated_at
                FROM entities
                WHERE kind=? AND key=?
                """,
                (kind, key),
            ).fetchone()
        return Entity(*row) if row else None

    def mark_done(self, entity_id: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE entities SET status='done', updated_at=? WHERE id=?",
                (now, entity_id),
            )

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
