"""Web fetch connector — retrieve a URL and return its readable text content.

Uses httpx for the HTTP request and trafilatura for main-content extraction.
trafilatura strips navigation, ads, and boilerplate, returning just the
article or page body — much more useful to the LLM than raw HTML.

The result is truncated to _MAX_CHARS characters to keep it within the
model's context budget.  If the page contains more content than this, the
agent may need to ask the user to narrow the request or call web_search
instead to get a summary first.

Both dependencies are optional.  If either is missing, execute() returns a
clear error message telling the user what to install.

Install:
    pip install httpx trafilatura
"""
from __future__ import annotations

from core.agent.tool_base import Tool, ToolResult

# Maximum characters returned to the agent.  Keeps the observation from
# consuming the entire context window for pages with lots of text.
_MAX_CHARS = 2000


class WebFetchTool(Tool):
    """Fetch a URL and return the main readable text content (up to 2000 chars).

    Useful when the agent needs to read a specific page — e.g. documentation,
    an article, or a product page — rather than search across many pages.
    """
    name = "web_fetch"
    description = "Fetch a URL and return the readable text content"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL to fetch"}},
        "required": ["url"],
    }

    def execute(self, *, url: str, **kwargs) -> ToolResult:
        """Fetch the URL and extract readable text via trafilatura.

        Imports are deferred to execute() (rather than module level) because
        these are optional heavyweight dependencies; the connector itself
        should load without them installed.

        Args:
            url: The full URL to fetch (http or https).

        Returns:
            ToolResult containing up to _MAX_CHARS of extracted text, or an
            error result describing what went wrong (missing library, network
            error, or no readable content found).
        """
        # Deferred imports — return friendly errors if either lib is missing.
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                content="httpx is not installed. Run: pip install httpx",
                is_error=True,
            )

        try:
            import trafilatura  # type: ignore[import]
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                content="trafilatura is not installed. Run: pip install trafilatura",
                is_error=True,
            )

        # Fetch the page.  follow_redirects handles common redirects (http→https, etc.).
        try:
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(tool_name=self.name, content=f"Fetch failed: {exc}", is_error=True)

        # trafilatura.extract strips boilerplate and returns the main content.
        # Falls back to the raw HTML text if trafilatura returns nothing
        # (e.g. for pages with no detectable main content block).
        text = trafilatura.extract(response.text) or response.text
        if not text or not text.strip():
            return ToolResult(
                tool_name=self.name,
                content="No readable content found at that URL.",
                is_error=True,
            )

        return ToolResult(tool_name=self.name, content=text.strip()[:_MAX_CHARS])
