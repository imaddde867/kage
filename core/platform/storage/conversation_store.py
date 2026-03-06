from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.platform.storage.schema import connect_db, ensure_schema

_SCAN_LIMIT = 100
_ENTRY_CHAR_LIMIT = 220
_RECALL_CHAR_BUDGET = 900
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "about",
    "also",
    "always",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "could",
    "from",
    "have",
    "just",
    "maybe",
    "more",
    "need",
    "please",
    "really",
    "should",
    "some",
    "than",
    "that",
    "them",
    "there",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    clean = _normalize(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


@dataclass
class ConversationStore:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_schema(self.db_path)

    def store_exchange(self, user_input: str, assistant_response: str) -> str:
        exchange_id = str(uuid.uuid4())
        with connect_db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_input, kage_response, timestamp) VALUES (?, ?, ?, ?)",
                (exchange_id, user_input, assistant_response, datetime.now().isoformat()),
            )
        return exchange_id

    def recent_turns(self, limit: int = 4, *, max_chars: int = _ENTRY_CHAR_LIMIT) -> list[tuple[str, str]]:
        if limit <= 0:
            return []
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT user_input, kage_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        turns: list[tuple[str, str]] = []
        for user_text, reply_text in reversed(rows):
            user = _truncate(user_text or "", max_chars)
            reply = _truncate(reply_text or "", max_chars)
            if not user and not reply:
                continue
            turns.append((user, reply))
        return turns

    def recall(self, query: str, n_results: int = 5, *, char_budget: int = _RECALL_CHAR_BUDGET) -> str:
        if n_results <= 0 or char_budget <= 0:
            return ""

        query_tokens = {
            token
            for token in _tokens(query)
            if len(token) >= 3 and token not in _STOPWORDS
        }
        if not query_tokens:
            return ""

        with connect_db(self.db_path) as conn:
            fts_rows = []
            try:
                fts_query = " ".join(sorted(query_tokens))
                fts_rows = conn.execute(
                    """
                    SELECT c.user_input, c.kage_response, c.timestamp
                    FROM conversations_fts f
                    JOIN conversations c ON c.rowid = f.rowid
                    WHERE conversations_fts MATCH ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                    """,
                    (fts_query, _SCAN_LIMIT),
                ).fetchall()
            except Exception:
                fts_rows = []

            rows = fts_rows
            if not rows:
                rows = conn.execute(
                    "SELECT user_input, kage_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
                    (_SCAN_LIMIT,),
                ).fetchall()

        query_phrase = _normalize(query).lower()
        matches: list[tuple[float, str, str]] = []
        for idx, row in enumerate(rows):
            user_text, reply_text, _timestamp = row
            user_text = _normalize(user_text or "")
            reply_text = _normalize(reply_text or "")
            haystack = f"{user_text}\n{reply_text}".lower()
            tokens = _tokens(haystack)
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            phrase_bonus = 2 if query_phrase and query_phrase in haystack else 0
            recency_bonus = max(0.0, 1.0 - (idx / max(len(rows), 1)))
            score = (overlap * 3) + phrase_bonus + (recency_bonus * 3)
            matches.append((score, user_text, reply_text))

        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return ""

        parts = ["--- Relevant past exchanges ---"]
        seen: set[str] = set()
        seen_users: set[str] = set()
        remaining = char_budget - len(parts[0])
        if remaining <= 0:
            return ""

        added = 0
        for _, user_text, reply_text in matches:
            if added >= n_results:
                break
            key = _normalize(f"{user_text}\n{reply_text}").lower()
            user_key = _normalize(user_text).lower()
            if key in seen:
                continue
            if user_key and user_key in seen_users:
                continue
            seen.add(key)
            if user_key:
                seen_users.add(user_key)
            entry = f"User: {_truncate(user_text, _ENTRY_CHAR_LIMIT)}\nKage: {_truncate(reply_text, _ENTRY_CHAR_LIMIT)}"
            cost = len(entry) + 1
            if cost > remaining:
                continue
            parts.append(entry)
            remaining -= cost
            added += 1
        if len(parts) == 1:
            return ""
        return "\n".join(parts)

