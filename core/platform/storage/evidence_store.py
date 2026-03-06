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

