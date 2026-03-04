"""Web search connector using DuckDuckGo — no API key, no account required.

The duckduckgo-search library is an optional dependency.  It is imported once
at module load time so the result (_DDGS) can be patched in tests without
needing to import the library at all.  If the library is not installed,
_DDGS is None and execute() returns an informative error result.

Install:
    pip install duckduckgo-search
"""
from __future__ import annotations

from core.agent.tool_base import Tool, ToolResult

# Optional import — None if duckduckgo-search is not installed.
# Keeping this at module level (rather than inside execute) makes it
# patchable in tests via @patch("connectors.web_search._DDGS", ...).
try:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore[import]
except ImportError:
    _DDGS = None  # type: ignore[assignment]


class WebSearchTool(Tool):
    """Search the web with DuckDuckGo and return the top 3 result snippets.

    The agent uses this tool when asked for current events, facts it might
    not know, or anything that benefits from up-to-date web results.

    Result format (injected as an observation into the next agent step):
        - Title: Snippet body
        - Title: Snippet body
        ...
    """
    name = "web_search"
    description = "Search the web for recent or factual information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }

    def execute(self, *, query: str, **kwargs) -> ToolResult:
        """Run a DuckDuckGo text search and return up to 3 result snippets.

        Args:
            query: The search string sent to DuckDuckGo.

        Returns:
            ToolResult with a newline-separated list of "- title: body" lines,
            or an error result if the library is missing or the search fails.
        """
        if _DDGS is None:
            return ToolResult(
                tool_name=self.name,
                content="duckduckgo-search is not installed. Run: pip install duckduckgo-search",
                is_error=True,
            )
        try:
            results = list(_DDGS().text(query, max_results=3))
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Search failed: {exc}", is_error=True)

        if not results:
            return ToolResult(tool_name=self.name, content="No results found.")

        text = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
        return ToolResult(tool_name=self.name, content=text)
