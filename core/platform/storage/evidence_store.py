from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.platform.storage.schema import connect_db, ensure_schema


@dataclass
class EvidenceStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

    def count(self) -> int:
        """Return total number of stored evidence records."""
        with connect_db(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()
        return int(row[0]) if row else 0

    def prune(self, keep_last: int) -> int:
        """Delete all but the most recent *keep_last* evidence records.

        Returns the number of rows deleted.
        """
        keep = max(0, keep_last)
        with connect_db(self.db_path) as conn:
            before = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            conn.execute(
                """
                DELETE FROM evidence
                WHERE rowid NOT IN (
                    SELECT rowid FROM evidence
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (keep,),
            )
            after = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        return int(before) - int(after)

    def record(
        self,
        *,
        tool_name: str,
        status: str,
        query_text: str | None,
        content: str,
        structured: dict[str, Any] | None,
        sources: list[str],
        latency_ms: float | None,
    ) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evidence
                    (tool_name, status, query_text, content, structured_json, sources_json, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    status,
                    query_text,
                    content,
                    json.dumps(structured, ensure_ascii=False) if structured is not None else None,
                    json.dumps(sources, ensure_ascii=False),
                    latency_ms,
                    datetime.now().isoformat(),
                ),
            )

