from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.platform.storage.schema import connect_db, ensure_schema


@dataclass
class KnowledgeEntity:
    id: str
    kind: str
    key: str
    value: str
    status: str
    due_date: Optional[str]
    source_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class KnowledgeStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

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
        with connect_db(self.db_path) as conn:
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

    def get_by_kind(self, kind: str, *, status: str = "active") -> list[KnowledgeEntity]:
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, kind, key, value, status, due_date, source_id, created_at, updated_at
                FROM entities
                WHERE kind=? AND status=?
                ORDER BY updated_at DESC
                """,
                (kind, status),
            ).fetchall()
        return [KnowledgeEntity(*row) for row in rows]

    def get_by_key(self, kind: str, key: str) -> KnowledgeEntity | None:
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, kind, key, value, status, due_date, source_id, created_at, updated_at
                FROM entities
                WHERE kind=? AND key=?
                """,
                (kind, key),
            ).fetchone()
        return KnowledgeEntity(*row) if row else None

    def mark_done(self, entity_id: str) -> None:
        now = datetime.now().isoformat()
        with connect_db(self.db_path) as conn:
            conn.execute(
                "UPDATE entities SET status='done', updated_at=? WHERE id=?",
                (now, entity_id),
            )

