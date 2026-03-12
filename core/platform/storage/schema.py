from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")


@contextmanager
def connect_db(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    try:
        _configure_connection(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(db_path: Path | str) -> None:
    with connect_db(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_input TEXT,
                kage_response TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)"
        )

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

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                query_text TEXT,
                content TEXT,
                structured_json TEXT,
                sources_json TEXT,
                latency_ms REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_tool_name ON evidence(tool_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_created_at ON evidence(created_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_kind TEXT NOT NULL,
                event_name TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL,
                payload_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_kind_name ON traces(event_kind, event_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_approvals (
                approval_key TEXT PRIMARY KEY,
                scope_kind TEXT NOT NULL,
                scope_name TEXT NOT NULL,
                note TEXT,
                granted_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_approvals_scope ON agent_approvals(scope_kind, scope_name)"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS autonomy_tasks (
                id TEXT PRIMARY KEY,
                task_text TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'planned',
                    'in_progress',
                    'blocked',
                    'awaiting_approval',
                    'done',
                    'failed'
                )),
                intent_action TEXT NOT NULL,
                risk_tier TEXT NOT NULL,
                reason_codes_json TEXT NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_autonomy_tasks_status ON autonomy_tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_autonomy_tasks_updated_at ON autonomy_tasks(updated_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proactive_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                reason TEXT NOT NULL,
                due_date TEXT,
                entity_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proactive_created_at ON proactive_opportunities(created_at)"
        )

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(user_input, kage_response, content='conversations', content_rowid='rowid')
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, user_input, kage_response)
                    VALUES (new.rowid, new.user_input, new.kage_response);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, user_input, kage_response)
                    VALUES('delete', old.rowid, old.user_input, old.kage_response);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, user_input, kage_response)
                    VALUES('delete', old.rowid, old.user_input, old.kage_response);
                    INSERT INTO conversations_fts(rowid, user_input, kage_response)
                    VALUES (new.rowid, new.user_input, new.kage_response);
                END
                """
            )
        except sqlite3.OperationalError:
            # Some SQLite builds may not include FTS5 support.
            pass
