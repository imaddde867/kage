"""NeuroCache Connector — Kage's long-term memory via second-brain.

Responsibilities
----------------
1. Auto-inject relevant vault notes into the system prompt before each response.
2. Expose VaultSearchTool so the agent can explicitly search the vault.
3. Write Kage-generated content back to the vault inbox (optional).
4. Report vault stats via get_stats().

second-brain API (api/server.py) endpoints used:
    POST /api/search    — vector search, returns [{title, path, tags, snippet}]
    GET  /api/stats     — note/tag/entity counts

Enabled with NEUROCACHE_ENABLED=true in .env.
The second-brain server must be running on NEUROCACHE_API_URL (default :8765).
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from core.agent.tool_base import Tool, ToolOutcome, ToolResult

logger = logging.getLogger(__name__)

_SNIPPET_CHAR_LIMIT = 500
_DEFAULT_K = 5
_MAX_K = 10
_MAX_VAULT_CHARS = 3000  # hard cap for auto-inject context


class NeuroCacheConnector:
    """HTTP client for second-brain's FastAPI server.

    Designed to be cheap to construct and resilient to the server being
    offline — every method returns a safe empty value on failure.
    """

    def __init__(self, api_url: str, vault_inbox: str = "") -> None:
        self.api_url = api_url.rstrip("/")
        self._vault_inbox = Path(vault_inbox).expanduser() if vault_inbox else None
        self._client: Any | None = None

    def _http(self) -> Any:
        """Lazy-init httpx.Client (avoids import cost at startup)."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=5.0)
        return self._client

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the second-brain server is reachable."""
        try:
            r = self._http().get(f"{self.api_url}/api/stats")
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        *,
        k: int = _DEFAULT_K,
        max_chars: int = _MAX_VAULT_CHARS,
    ) -> str:
        """Search the vault and return a formatted block for system-prompt injection.

        Returns "" if the server is offline, the query is empty, or no results.
        Never raises — safe to call on the hot path.
        """
        query = (query or "").strip()
        if not query:
            return ""
        k = max(1, min(k, _MAX_K))
        try:
            r = self._http().post(
                f"{self.api_url}/api/search",
                json={"query": query, "top_k": k},
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as exc:
            logger.debug("[neurocache] search failed: %s", exc)
            return ""

        if not results:
            return ""

        parts: list[str] = []
        char_count = 0
        for res in results:
            title = res.get("title", "untitled")
            snippet = (res.get("snippet") or "")[:_SNIPPET_CHAR_LIMIT]
            tags = res.get("tags", "")
            tag_line = f"  tags: {tags}\n" if tags else ""
            block = f"**{title}**\n{tag_line}{snippet}\n\n"
            if char_count + len(block) > max_chars:
                break
            parts.append(block)
            char_count += len(block)

        if not parts:
            return ""
        return "Relevant notes from your vault:\n\n" + "".join(parts).rstrip()

    def raw_search(self, query: str, *, k: int = _DEFAULT_K) -> list[dict[str, str]]:
        """Return raw search results for use in VaultSearchTool."""
        query = (query or "").strip()
        if not query:
            return []
        k = max(1, min(k, _MAX_K))
        try:
            r = self._http().post(
                f"{self.api_url}/api/search",
                json={"query": query, "top_k": k},
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as exc:
            logger.debug("[neurocache] raw_search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Vault write-back
    # ------------------------------------------------------------------

    def save_to_vault(
        self,
        title: str,
        content: str,
        *,
        tags: list[str] | None = None,
    ) -> str:
        """Write a Kage-generated note to the vault inbox.

        Returns the path to the written file, or "" if vault_inbox is not
        configured or the write fails.
        """
        if self._vault_inbox is None:
            logger.debug("[neurocache] save_to_vault called but NEUROCACHE_VAULT_INBOX is not set")
            return ""
        try:
            self._vault_inbox.mkdir(parents=True, exist_ok=True)
            safe = "".join(c for c in title if c.isalnum() or c in " -_").strip()
            if not safe:
                safe = "kage-note"
            note_path = self._vault_inbox / f"{safe}.md"
            tag_line = ""
            if tags:
                tag_line = "tags: [" + ", ".join(tags) + "]\n"
            now = datetime.datetime.now().isoformat(timespec="seconds")
            text = (
                f"---\ncreated: {now}\nsource: kage\n{tag_line}---\n\n"
                f"# {title}\n\n{content}\n"
            )
            note_path.write_text(text, encoding="utf-8")
            return str(note_path)
        except Exception as exc:
            logger.warning("[neurocache] save_to_vault failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        try:
            r = self._http().get(f"{self.api_url}/api/stats")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "offline"}


# ------------------------------------------------------------------
# Module-level singleton — shared across BrainService lifetime
# ------------------------------------------------------------------

_connector: NeuroCacheConnector | None = None


def get_connector(api_url: str, vault_inbox: str = "") -> NeuroCacheConnector:
    """Return (and lazily create) the process-wide NeuroCacheConnector."""
    global _connector
    if _connector is None:
        _connector = NeuroCacheConnector(api_url, vault_inbox)
    return _connector


# ------------------------------------------------------------------
# Agent-facing Tool
# ------------------------------------------------------------------

class VaultSearchTool(Tool):
    """Search your Obsidian vault for notes relevant to a topic or question.

    Use this when the user asks about their own notes, research, past work,
    projects, or anything they might have written down.  Returns note titles
    and excerpts with file paths.
    """

    name = "vault_search"
    description = "Search your Obsidian vault for relevant notes and research"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the vault",
            },
            "k": {
                "type": "integer",
                "description": "Number of results (1-10, default 5)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_url: str, vault_inbox: str = "") -> None:
        self._connector = get_connector(api_url, vault_inbox)

    def execute(self, *, query: str, k: int = _DEFAULT_K, **kwargs) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(
                tool_name=self.name,
                content="Provide a non-empty search query.",
                is_error=True,
            )

        results = self._connector.raw_search(query, k=max(1, min(k, _MAX_K)))

        if not results:
            return ToolResult(
                tool_name=self.name,
                content=json.dumps({"query": query, "results": []}),
                outcome=ToolOutcome(
                    status="ok",
                    structured={"query": query, "results": []},
                    sources=[],
                    retryable=False,
                    side_effects=False,
                ),
            )

        rows = [
            {
                "title": r.get("title", ""),
                "path": r.get("path", ""),
                "tags": r.get("tags", ""),
                "snippet": (r.get("snippet") or "")[:300],
            }
            for r in results
        ]
        payload = json.dumps({"query": query, "results": rows}, ensure_ascii=False)
        sources = [r["path"] for r in rows if r.get("path")]
        return ToolResult(
            tool_name=self.name,
            content=payload,
            outcome=ToolOutcome(
                status="ok",
                structured={"query": query, "results": rows},
                sources=sources,
                retryable=False,
                side_effects=False,
            ),
        )
