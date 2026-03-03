from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import config

_DB_FILENAME = "kage_memory.db"
_SCAN_LIMIT = 100


@dataclass
class MemoryStore:
    db_path: Path | None = None

    def __post_init__(self) -> None:
        default_path = Path(config.get().memory_dir).expanduser() / _DB_FILENAME
        self.db_path = Path(self.db_path).expanduser() if self.db_path is not None else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
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

    def store_exchange(self, user_input: str, assistant_response: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_input, kage_response, timestamp) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), user_input, assistant_response, datetime.now().isoformat()),
            )

    def recall(self, query: str, n_results: int = 5) -> str:
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        if not keywords:
            return ""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_input, kage_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (_SCAN_LIMIT,),
            ).fetchall()

        matches: list[tuple[int, str, str]] = []
        for user_text, reply_text in rows:
            user_text = (user_text or "").strip()
            reply_text = (reply_text or "").strip()
            haystack = f"{user_text} {reply_text}".lower()
            score = sum(1 for kw in keywords if kw in haystack)
            if score > 0:
                matches.append((score, user_text, reply_text))

        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return ""

        parts = ["--- Relevant past exchanges ---"]
        for _, user_text, reply_text in matches[:n_results]:
            parts.append(f"User: {user_text}\nKage: {reply_text}")
        return "\n".join(parts)
