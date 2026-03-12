"""Browser fetch connector — JavaScript-rendering page fetcher via Playwright.

Use this when web_fetch returns incomplete content because the page requires
JavaScript to render (SPAs, React/Vue/Angular frontends, etc.).

Install:
    pip install playwright
    playwright install chromium
"""
from __future__ import annotations

import re

from core.agent.tool_base import Tool, ToolOutcome, ToolResult

try:
    from playwright.sync_api import sync_playwright as _sync_playwright  # type: ignore[import]
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _sync_playwright = None  # type: ignore[assignment]
    _PLAYWRIGHT_AVAILABLE = False

_DEFAULT_MAX_CHARS = 4000
_MAX_ALLOWED_CHARS = 12000
_TIMEOUT_MS = 20_000  # 20s page.goto timeout

_WS_RE = re.compile(r"\s+")


def _condense(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _clamp_max_chars(max_chars: int) -> int:
    try:
        requested = int(max_chars)
    except Exception:
        return _DEFAULT_MAX_CHARS
    return max(500, min(requested, _MAX_ALLOWED_CHARS))


class BrowserFetchTool(Tool):
    """Fetch a JavaScript-rendered page using a headless Chromium browser.

    Use this instead of web_fetch when the target page is a SPA or requires
    JavaScript execution to display its content.
    """

    name = "browser_fetch"
    description = (
        "Fetch a JS-rendered page using headless Chromium (Playwright). "
        "Use when web_fetch returns empty or incomplete content due to JavaScript."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "wait_for_selector": {
                "type": "string",
                "description": (
                    "Optional CSS selector to wait for before extracting text "
                    "(useful for SPAs that render content after initial load)"
                ),
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters returned (500-12000, default 4000)",
            },
        },
        "required": ["url"],
    }

    def execute(
        self,
        *,
        url: str,
        wait_for_selector: str | None = None,
        max_chars: int = _DEFAULT_MAX_CHARS,
        **kwargs,
    ) -> ToolResult:
        del kwargs

        if not isinstance(url, str) or not url.strip():
            return ToolResult(
                tool_name=self.name,
                content="Invalid URL: provide a non-empty http/https URL string.",
                is_error=True,
                outcome=ToolOutcome(
                    status="error", structured=None, sources=[], retryable=False, side_effects=False
                ),
            )

        if not _PLAYWRIGHT_AVAILABLE or _sync_playwright is None:
            return ToolResult(
                tool_name=self.name,
                content=(
                    "Playwright is not installed. "
                    "Run: pip install playwright && playwright install chromium"
                ),
                is_error=True,
                outcome=ToolOutcome(
                    status="error", structured=None, sources=[], retryable=False, side_effects=False
                ),
            )

        limit = _clamp_max_chars(max_chars)

        try:
            with _sync_playwright() as pw:
                try:
                    browser = pw.chromium.launch(headless=True)
                except Exception as exc:
                    return ToolResult(
                        tool_name=self.name,
                        content=(
                            f"Failed to launch Chromium: {exc}. "
                            "Run: playwright install chromium"
                        ),
                        is_error=True,
                        outcome=ToolOutcome(
                            status="error",
                            structured=None,
                            sources=[],
                            retryable=False,
                            side_effects=False,
                        ),
                    )
                try:
                    page = browser.new_page()
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=_TIMEOUT_MS)
                    except Exception as exc:
                        return ToolResult(
                            tool_name=self.name,
                            content=f"Page navigation timed out or failed for {url}: {exc}",
                            is_error=True,
                            outcome=ToolOutcome(
                                status="error",
                                structured=None,
                                sources=[url],
                                retryable=True,
                                side_effects=False,
                            ),
                        )

                    if wait_for_selector:
                        try:
                            page.wait_for_selector(wait_for_selector, timeout=_TIMEOUT_MS)
                        except Exception:
                            # Selector not found — proceed with what we have
                            pass

                    body_text = page.inner_text("body") if page.query_selector("body") else ""
                    text = _condense(body_text)
                finally:
                    page.close() if "page" in dir() else None  # type: ignore[possibly-undefined]
            # browser.close() is called by the context manager exit above
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                content=f"Browser fetch failed for {url}: {exc}",
                is_error=True,
                outcome=ToolOutcome(
                    status="error",
                    structured=None,
                    sources=[url],
                    retryable=True,
                    side_effects=False,
                ),
            )

        if not text:
            return ToolResult(
                tool_name=self.name,
                content=f"No readable text found at {url} after JavaScript rendering.",
                is_error=True,
                outcome=ToolOutcome(
                    status="error",
                    structured=None,
                    sources=[url],
                    retryable=False,
                    side_effects=False,
                ),
            )

        truncated = text[:limit]
        return ToolResult(
            tool_name=self.name,
            content=f"URL: {url}\n{truncated}",
            outcome=ToolOutcome(
                status="ok",
                structured={"url": url, "text": truncated},
                sources=[url],
                retryable=False,
                side_effects=False,
            ),
        )
