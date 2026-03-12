from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.platform.models import ExecutionIntent, TaskStatus
from core.platform.storage.schema import connect_db, ensure_schema

_VALID_STATUSES = {status.value for status in TaskStatus}


@dataclass(frozen=True)
class AutonomyTaskRecord:
    task_id: str
    task_text: str
    status: str
    intent_action: str
    risk_tier: str
    reason_codes: tuple[str, ...]
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass
class AutonomyTaskStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

    def create_task(
        self,
        *,
        task_text: str,
        intent: ExecutionIntent,
        status: TaskStatus | str = TaskStatus.PLANNED,
    ) -> str:
        task_id = f"autonomy_{uuid.uuid4().hex[:16]}"
        now = datetime.now().isoformat()
        status_value = self._status_value(status)
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO autonomy_tasks
                    (id, task_text, status, intent_action, risk_tier, reason_codes_json, last_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_text,
                    status_value,
                    intent.action,
                    intent.risk_tier.value,
                    json.dumps(list(intent.reason_codes), ensure_ascii=False),
                    None,
                    now,
                    now,
                ),
            )
        return task_id

    def transition(
        self,
        task_id: str,
        *,
        status: TaskStatus | str,
        last_error: str | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        status_value = self._status_value(status)
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE autonomy_tasks
                SET status = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status_value, last_error, now, task_id),
            )

    def get(self, task_id: str) -> AutonomyTaskRecord | None:
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_text, status, intent_action, risk_tier, reason_codes_json, last_error, created_at, updated_at
                FROM autonomy_tasks
                WHERE id = ?
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return AutonomyTaskRecord(
            task_id=row[0],
            task_text=row[1],
            status=self._normalize_stored_status(row[2]),
            intent_action=row[3],
            risk_tier=row[4],
            reason_codes=tuple(self._decode_reason_codes(row[5])),
            last_error=row[6],
            created_at=row[7],
            updated_at=row[8],
        )

    def list_recent(self, *, limit: int = 30) -> list[AutonomyTaskRecord]:
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_text, status, intent_action, risk_tier, reason_codes_json, last_error, created_at, updated_at
                FROM autonomy_tasks
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            AutonomyTaskRecord(
                task_id=row[0],
                task_text=row[1],
                status=self._normalize_stored_status(row[2]),
                intent_action=row[3],
                risk_tier=row[4],
                reason_codes=tuple(self._decode_reason_codes(row[5])),
                last_error=row[6],
                created_at=row[7],
                updated_at=row[8],
            )
            for row in rows
        ]

    def _decode_reason_codes(self, payload: str | None) -> list[str]:
        if not payload:
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()]

    def _status_value(self, status: TaskStatus | str) -> str:
        if isinstance(status, TaskStatus):
            candidate = status.value
        else:
            candidate = str(status).strip().lower()
        if candidate not in _VALID_STATUSES:
            raise ValueError(f"Unsupported autonomy task status: {status!r}")
        return candidate

    def _normalize_stored_status(self, value: str | None) -> str:
        candidate = str(value or "").strip().lower()
        if candidate in _VALID_STATUSES:
            return candidate
        return TaskStatus.BLOCKED.value
