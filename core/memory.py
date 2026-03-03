from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import config

_DEFAULT_DB_FILENAME = "kage_memory.db"
_CONVERSATION_SCAN_LIMIT = 100
_FACT_SCAN_LIMIT = 200
_FACT_OUTPUT_LIMIT = 3


@dataclass
class MemoryStore:
    db_path: Path | None = None

    def __post_init__(self) -> None:
        default_path = Path(config.get_settings().memory_dir).expanduser() / _DEFAULT_DB_FILENAME
        base_path = Path(self.db_path).expanduser() if self.db_path is not None else default_path
        self.db_path = base_path
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
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    fact TEXT,
                    category TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts(timestamp)")

    def store_exchange(self, user_input: str, assistant_response: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_input, kage_response, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), user_input, assistant_response, datetime.now().isoformat()),
            )

    def store_fact(self, fact: str, category: str = "general") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO facts (id, fact, category, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), fact, category, datetime.now().isoformat()),
            )

    def recall(self, query: str, n_results: int = 5) -> str:
        """
        Retrieve relevant memory for a given query using keyword matching.
        Returns a formatted string ready to inject into the LLM prompt.
        """
        keywords = self._tokenize_query(query)
        if not keywords:
            return ""

        parts: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_input, kage_response
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (_CONVERSATION_SCAN_LIMIT,),
            ).fetchall()

            matches: list[tuple[int, str, str]] = []
            for user_input, response in rows:
                user_text = (user_input or "").strip()
                reply_text = (response or "").strip()
                if not user_text and not reply_text:
                    continue
                score = self._score_text(f"{user_text} {reply_text}", keywords)
                if score > 0:
                    matches.append((score, user_text, reply_text))

            matches.sort(key=lambda item: item[0], reverse=True)
            top_matches = matches[: max(n_results, 0)]
            if top_matches:
                parts.append("--- Relevant past exchanges ---")
                for _, user_text, reply_text in top_matches:
                    parts.append(f"User: {user_text}\nKage: {reply_text}")

            fact_rows = conn.execute(
                """
                SELECT fact, category
                FROM facts
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (_FACT_SCAN_LIMIT,),
            ).fetchall()
            fact_matches: list[tuple[int, str]] = []
            for fact, _category in fact_rows:
                fact_text = (fact or "").strip()
                if not fact_text:
                    continue
                score = self._score_text(fact_text, keywords)
                if score > 0:
                    fact_matches.append((score, fact_text))

            fact_matches.sort(key=lambda item: item[0], reverse=True)
            if fact_matches:
                parts.append("--- Facts about you ---")
                for _, fact_text in fact_matches[:_FACT_OUTPUT_LIMIT]:
                    parts.append(fact_text)

        return "\n".join(parts) if parts else ""

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        return [word.lower() for word in query.split() if len(word) > 3]

    @staticmethod
    def _score_text(text: str, keywords: Iterable[str]) -> int:
        haystack = " ".join(text.lower().split())
        return sum(1 for keyword in keywords if keyword in haystack)


_DEFAULT_STORE: MemoryStore | None = None


def get_default_store() -> MemoryStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = MemoryStore()
    return _DEFAULT_STORE


def store_exchange(user_input: str, assistant_response: str) -> None:
    get_default_store().store_exchange(user_input, assistant_response)


def store_fact(fact: str, category: str = "general") -> None:
    get_default_store().store_fact(fact, category)


def recall(query: str, n_results: int = 5) -> str:
    return get_default_store().recall(query, n_results=n_results)


__all__ = [
    "MemoryStore",
    "get_default_store",
    "store_exchange",
    "store_fact",
    "recall",
]
