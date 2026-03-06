from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import config
from core.platform.storage import ConversationStore, connect_db

_DB_FILENAME = "kage_memory.db"
_RECALL_CHAR_BUDGET = 900
_ENTRY_CHAR_LIMIT = 220
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
class MemoryStore:
    db_path: Path | None = None

    def __post_init__(self) -> None:
        default_path = Path(config.get().memory_dir).expanduser() / _DB_FILENAME
        self.db_path = Path(self.db_path).expanduser() if self.db_path is not None else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conversation = ConversationStore(self.db_path)

    def _connect(self):
        return connect_db(self.db_path)

    def _init_schema(self) -> None:
        # Backward-compat shim: schema initialization is now centralized in
        # core.platform.storage.schema via ConversationStore.
        self._conversation = ConversationStore(self.db_path)

    def _init_schema_entities(self) -> None:
        # Backward-compat shim retained for older callers.
        self._conversation = ConversationStore(self.db_path)

    def store_exchange(self, user_input: str, assistant_response: str) -> str:
        return self._conversation.store_exchange(user_input, assistant_response)

    def recent_turns(self, limit: int = 4, *, max_chars: int = _ENTRY_CHAR_LIMIT) -> list[tuple[str, str]]:
        return self._conversation.recent_turns(limit=limit, max_chars=max_chars)

    def recall(self, query: str, n_results: int = 5, *, char_budget: int = _RECALL_CHAR_BUDGET) -> str:
        return self._conversation.recall(query, n_results=n_results, char_budget=char_budget)
