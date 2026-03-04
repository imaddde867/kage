"""Web fetch connector — fetch a URL and return readable text.

Requires: pip install httpx trafilatura
"""
from __future__ import annotations

from core.agent.tool_base import Tool, ToolResult

_MAX_CHARS = 2000


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return the readable text content"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL to fetch"}},
        "required": ["url"],
    }

    def execute(self, *, url: str, **kwargs) -> ToolResult:
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return ToolResult(tool_name=self.name, content="httpx is not installed. Run: pip install httpx", is_error=True)

        try:
            import trafilatura  # type: ignore[import]
        except ImportError:
            return ToolResult(tool_name=self.name, content="trafilatura is not installed. Run: pip install trafilatura", is_error=True)

        try:
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Fetch failed: {exc}", is_error=True)

        text = trafilatura.extract(response.text) or response.text
        if not text or not text.strip():
            return ToolResult(tool_name=self.name, content="No readable content found at that URL.", is_error=True)

        return ToolResult(tool_name=self.name, content=text.strip()[:_MAX_CHARS])
