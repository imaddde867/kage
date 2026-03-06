from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_db(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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

