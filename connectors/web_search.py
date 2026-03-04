"""Web search connector using DuckDuckGo (no API key required).

Install: pip install duckduckgo-search
"""
from __future__ import annotations

from core.agent.tool_base import Tool, ToolResult

try:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore[import]
except ImportError:
    _DDGS = None  # type: ignore[assignment]


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for recent or factual information"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }

    def execute(self, *, query: str, **kwargs) -> ToolResult:
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
