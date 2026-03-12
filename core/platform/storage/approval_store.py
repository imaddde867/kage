from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.platform.storage.schema import connect_db, ensure_schema

_VALID_SCOPE_KINDS = {"tool", "tier"}


@dataclass(frozen=True)
class ApprovalEntry:
    approval_key: str
    scope_kind: str
    scope_name: str
    note: str
    granted_by: str
    created_at: str
    updated_at: str


def approval_key(scope_kind: str, scope_name: str) -> str:
    kind = (scope_kind or "").strip().lower()
    name = (scope_name or "").strip().lower()
    if kind not in _VALID_SCOPE_KINDS:
        raise ValueError(f"Unsupported approval scope kind: {scope_kind!r}")
    if not name:
        raise ValueError("Approval scope name cannot be empty.")
    return f"{kind}:{name}"


@dataclass
class ApprovalStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

    def grant(
        self,
        *,
        scope_kind: str,
        scope_name: str,
        granted_by: str = "user",
        note: str = "",
    ) -> str:
        key = approval_key(scope_kind, scope_name)
        now = datetime.now().isoformat()
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_approvals
                    (approval_key, scope_kind, scope_name, note, granted_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(approval_key) DO UPDATE SET
                    note=excluded.note,
                    granted_by=excluded.granted_by,
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    scope_kind.strip().lower(),
                    scope_name.strip().lower(),
                    note,
                    granted_by,
                    now,
                    now,
                ),
            )
        return key

    def revoke(self, *, scope_kind: str, scope_name: str) -> bool:
        key = approval_key(scope_kind, scope_name)
        with connect_db(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM agent_approvals WHERE approval_key = ?",
                (key,),
            )
            return int(result.rowcount or 0) > 0

    def is_granted(self, *, scope_kind: str, scope_name: str) -> bool:
        key = approval_key(scope_kind, scope_name)
        return self.is_granted_key(key)

    def is_granted_key(self, approval: str) -> bool:
        key = (approval or "").strip().lower()
        if not key:
            return False
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM agent_approvals WHERE approval_key = ? LIMIT 1",
                (key,),
            ).fetchone()
        return row is not None

    def list_entries(self, *, limit: int = 200) -> list[ApprovalEntry]:
        rows: list[tuple[str, str, str, str, str, str, str]]
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT approval_key, scope_kind, scope_name, note, granted_by, created_at, updated_at
                FROM agent_approvals
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            ApprovalEntry(
                approval_key=row[0],
                scope_kind=row[1],
                scope_name=row[2],
                note=row[3] or "",
                granted_by=row[4],
                created_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]

    def count_entries(self) -> int:
        with connect_db(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM agent_approvals").fetchone()
        return int(row[0]) if row else 0
