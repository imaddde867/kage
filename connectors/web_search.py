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
_SNIPPET_MAX_CHARS = 200
_RESULT_TITLE_MAX_CHARS = 180
_MAX_CONTENT_CHARS = 2500


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
            return ToolResult(
                tool_name=self.name,
                content="DuckDuckGo search is not installed. Run: pip install ddgs",
                is_error=True,
                outcome=ToolOutcome(
                    status="error",
                    structured=None,
                    sources=[],
                    retryable=True,
                    side_effects=False,
                ),
            )
        try:
            limit = max(1, min(int(max_results), _MAX_RESULTS))
        except Exception:
            limit = _DEFAULT_RESULTS

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results = list(_DDGS().text(query, max_results=limit))
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                content=f"Search failed: {exc}",
                is_error=True,
                outcome=ToolOutcome(
                    status="error",
                    structured=None,
                    sources=[],
                    retryable=True,
                    side_effects=False,
                ),
            )

        if not results:
            return ToolResult(
                tool_name=self.name,
                content=json.dumps({"query": query, "results": []}, ensure_ascii=False),
                outcome=ToolOutcome(
                    status="ok",
                    structured={"query": query, "results": []},
                    sources=[],
                    retryable=False,
                    side_effects=False,
                ),
            )

        rows: list[dict[str, str]] = []
        for idx, raw in enumerate(results, start=1):
            title = (raw.get("title") or "Untitled result").strip()
            snippet = (raw.get("body") or "").strip()
            url = (raw.get("href") or raw.get("url") or "").strip()
            if not url:
                continue
            rows.append(
                {
                    "rank": idx,
                    "title": title[:_RESULT_TITLE_MAX_CHARS],
                    "url": url,
                    "snippet": snippet[:_SNIPPET_MAX_CHARS],
                }
            )

        payload = self._compact_payload(query=query, rows=rows)
        sources = [row["url"] for row in rows if isinstance(row.get("url"), str)]
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
