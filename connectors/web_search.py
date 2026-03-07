"""Web search connector using DuckDuckGo — no API key, no account required.

Returns compact JSON to keep AgentLoop observations small and machine-readable.
This makes downstream URL selection and de-duplication more reliable.
"""
from __future__ import annotations

import json
import warnings

from core.agent.tool_base import Tool, ToolOutcome, ToolResult

_DDGS = None  # type: ignore[assignment]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    try:
        from ddgs import DDGS as _DDGS  # type: ignore[import]
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore[import]
        except ImportError:
            pass

_DEFAULT_RESULTS = 5
_MAX_RESULTS = 10
_MAX_QUERY_CHARS = 300
_SNIPPET_MAX_CHARS = 200
_RESULT_TITLE_MAX_CHARS = 180
_MAX_CONTENT_CHARS = 2500
_SEARCH_ATTEMPTS = 2
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "tempor",
    "connection",
    "network",
    "try again",
    "rate limit",
    "too many requests",
    "unavailable",
    "429",
    "502",
    "503",
    "504",
)


def _is_transient_search_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


class WebSearchTool(Tool):
    """Search the web with DuckDuckGo and return result title/URL/snippet lines.

    The agent uses this tool when asked for current events, facts it might
    not know, or anything that benefits from up-to-date web results.
    """
    name = "web_search"
    description = "Search the web for recent or factual information with URLs"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5)",
            },
        },
        "required": ["query"],
    }

    def _error_result(self, message: str, *, retryable: bool) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            content=message,
            is_error=True,
            outcome=ToolOutcome(
                status="error",
                structured=None,
                sources=[],
                retryable=retryable,
                side_effects=False,
            ),
        )

    def _normalize_query(self, query: object) -> str:
        if not isinstance(query, str):
            return ""
        text = query.strip()
        if not text:
            return ""
        condensed = " ".join(text.split())
        return condensed[:_MAX_QUERY_CHARS]

    def _clamp_limit(self, max_results: object) -> int:
        try:
            return max(1, min(int(max_results), _MAX_RESULTS))
        except Exception:
            return _DEFAULT_RESULTS

    def _fetch_results(self, query: str, *, limit: int) -> list[object]:
        last_exc: Exception | None = None
        for attempt in range(_SEARCH_ATTEMPTS):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    return list(_DDGS().text(query, max_results=limit))
            except Exception as exc:
                last_exc = exc
                should_retry = _is_transient_search_error(exc) and attempt + 1 < _SEARCH_ATTEMPTS
                if not should_retry:
                    raise
        if last_exc is not None:
            raise last_exc
        return []

    def _compact_payload(self, query: str, rows: list[dict[str, str]]) -> str:
        payload: dict[str, object] = {"query": query, "results": []}
        for row in rows:
            candidate = dict(payload)
            candidate_results = list(payload["results"])  # type: ignore[index]
            candidate_results.append(row)
            candidate["results"] = candidate_results
            rendered = json.dumps(candidate, ensure_ascii=False)
            if len(rendered) > _MAX_CONTENT_CHARS:
                break
            payload = candidate

        rendered = json.dumps(payload, ensure_ascii=False)
        if len(rendered) <= _MAX_CONTENT_CHARS:
            return rendered
        return json.dumps({"query": query[:120], "results": []}, ensure_ascii=False)

    def execute(self, *, query: str, max_results: int = _DEFAULT_RESULTS, **kwargs) -> ToolResult:
        """Run a DuckDuckGo text search and return compact structured JSON."""
        del kwargs
        if _DDGS is None:
            return self._error_result(
                "DuckDuckGo search is not installed. Run: pip install ddgs",
                retryable=True,
            )

        query_text = self._normalize_query(query)
        if not query_text:
            return self._error_result(
                "Invalid search query: provide a non-empty text query.",
                retryable=False,
            )

        limit = self._clamp_limit(max_results)

        try:
            results = self._fetch_results(query_text, limit=limit)
        except Exception as exc:
            return self._error_result(
                f"Search failed: {exc}",
                retryable=_is_transient_search_error(exc),
            )

        if not results:
            return ToolResult(
                tool_name=self.name,
                content=json.dumps({"query": query_text, "results": []}, ensure_ascii=False),
                outcome=ToolOutcome(
                    status="ok",
                    structured={"query": query_text, "results": []},
                    sources=[],
                    retryable=False,
                    side_effects=False,
                ),
            )

        rows: list[dict[str, str]] = []
        rank = 1
        for raw in results:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "Untitled result").strip()
            snippet = str(raw.get("body") or "").strip()
            url = str(raw.get("href") or raw.get("url") or "").strip()
            if not url:
                continue
            rows.append(
                {
                    "rank": rank,
                    "title": title[:_RESULT_TITLE_MAX_CHARS],
                    "url": url,
                    "snippet": snippet[:_SNIPPET_MAX_CHARS],
                }
            )
            rank += 1

        payload = self._compact_payload(query=query_text, rows=rows)
        sources = [row["url"] for row in rows if isinstance(row.get("url"), str)]
        return ToolResult(
            tool_name=self.name,
            content=payload,
            outcome=ToolOutcome(
                status="ok",
                structured={"query": query_text, "results": rows},
                sources=sources,
                retryable=False,
                side_effects=False,
            ),
        )
