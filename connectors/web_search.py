"""Web search connector using DuckDuckGo — no API key, no account required.

The duckduckgo-search library is imported once at module load time so _DDGS
can be patched in tests without importing the package itself.

Result format includes title + URL + snippet so the agent can chain:
web_search -> web_fetch for deeper reading.

Install:
    pip install duckduckgo-search
"""
from __future__ import annotations

import warnings

from core.agent.tool_base import Tool, ToolResult

# Optional import — None if duckduckgo-search is not installed.
# Keeping this at module level (rather than inside execute) makes it
# patchable in tests via @patch("connectors.web_search._DDGS", ...).
# DeprecationWarnings from the duckduckgo_search package (rename notices) are
# suppressed here so they don't appear in user-facing terminal output.
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        from duckduckgo_search import DDGS as _DDGS  # type: ignore[import]
except ImportError:
    _DDGS = None  # type: ignore[assignment]

_DEFAULT_RESULTS = 5
_MAX_RESULTS = 10


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

    def execute(self, *, query: str, max_results: int = _DEFAULT_RESULTS, **kwargs) -> ToolResult:
        """Run a DuckDuckGo text search and return title/URL/snippet lines.

        Args:
            query: The search string sent to DuckDuckGo.
            max_results: Requested result count (clamped to 1..10).

        Returns:
            ToolResult with newline-separated bullet blocks,
            or an error result if the library is missing or the search fails.
        """
        if _DDGS is None:
            return ToolResult(
                tool_name=self.name,
                content="duckduckgo-search is not installed. Run: pip install duckduckgo-search",
                is_error=True,
            )
        try:
            limit = max(1, min(int(max_results), _MAX_RESULTS))
        except Exception:
            limit = _DEFAULT_RESULTS

        try:
            results = list(_DDGS().text(query, max_results=limit))
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Search failed: {exc}", is_error=True)

        if not results:
            return ToolResult(tool_name=self.name, content="No results found.")

        lines: list[str] = []
        for raw in results:
            title = (raw.get("title") or "Untitled result").strip()
            snippet = (raw.get("body") or "").strip()
            url = (raw.get("href") or raw.get("url") or "").strip()

            parts = [f"- {title}"]
            if url:
                parts.append(f"  URL: {url}")
            if snippet:
                parts.append(f"  Snippet: {snippet}")
            lines.append("\n".join(parts))

        text = "\n".join(lines)
        return ToolResult(tool_name=self.name, content=text)
