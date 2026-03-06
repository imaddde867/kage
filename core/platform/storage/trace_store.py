from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.platform.storage.schema import connect_db, ensure_schema


@dataclass
class TraceStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

    def record(
        self,
        *,
        event_kind: str,
        event_name: str,
        status: str,
        latency_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO traces (event_kind, event_name, status, latency_ms, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_kind,
                    event_name,
                    status,
                    latency_ms,
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                    datetime.now().isoformat(),
                ),
            )

    def record_tool_result(
        self,
        *,
        tool_name: str,
        is_error: bool,
        latency_ms: float | None,
        content_preview: str,
    ) -> None:
        self.record(
            event_kind="tool_call",
            event_name=tool_name,
            status="error" if is_error else "ok",
            latency_ms=latency_ms,
            payload={"preview": content_preview[:220]},
        )

    def tool_health(self, tool_name: str, *, window: int = 50) -> float:
        limit = max(5, int(window))
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT status
                FROM traces
                WHERE event_kind='tool_call' AND event_name=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (tool_name, limit),
            ).fetchall()
        if not rows:
            return 1.0
        ok = sum(1 for (status,) in rows if status == "ok")
        return ok / len(rows)

