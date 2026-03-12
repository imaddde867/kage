"""Unit tests for connectors.browser_fetch — all playwright calls are mocked."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestBrowserFetchToolImportGuard(unittest.TestCase):
    """Tool handles missing playwright gracefully."""

    def test_playwright_not_installed_returns_error(self):
        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", False), \
             patch("connectors.browser_fetch._sync_playwright", None):
            from connectors.browser_fetch import BrowserFetchTool
            tool = BrowserFetchTool()
            result = tool.execute(url="https://example.com")
            self.assertTrue(result.is_error)
            self.assertIn("playwright", result.content.lower())
            self.assertFalse(result.outcome.retryable)

    def test_invalid_url_returns_error(self):
        from connectors.browser_fetch import BrowserFetchTool
        tool = BrowserFetchTool()
        result = tool.execute(url="")
        self.assertTrue(result.is_error)
        self.assertFalse(result.outcome.retryable)


def _make_pw_mock(body_text: str = "Hello world"):
    """Build a mock playwright context manager that returns body_text."""
    page = MagicMock()
    page.inner_text.return_value = body_text
    page.query_selector.return_value = MagicMock()  # truthy — body exists
    page.goto.return_value = None
    page.wait_for_selector.return_value = None
    page.close.return_value = None

    browser = MagicMock()
    browser.new_page.return_value = page

    pw = MagicMock()
    pw.chromium.launch.return_value = browser

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pw)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, page, browser


class TestBrowserFetchToolSuccess(unittest.TestCase):

    def test_success_returns_content(self):
        ctx, page, _ = _make_pw_mock("Page content here")
        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            result = BrowserFetchTool().execute(url="https://example.com")
        self.assertFalse(result.is_error)
        self.assertIn("Page content here", result.content)
        self.assertIn("https://example.com", result.content)

    def test_max_chars_truncation(self):
        long_text = "A" * 10000
        ctx, _, _ = _make_pw_mock(long_text)
        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            result = BrowserFetchTool().execute(url="https://example.com", max_chars=500)
        self.assertFalse(result.is_error)
        # content = "URL: ...\n" + text[:500]
        text_part = result.content.split("\n", 1)[1]
        self.assertLessEqual(len(text_part), 500)

    def test_wait_for_selector_called_when_provided(self):
        ctx, page, _ = _make_pw_mock("SPA content")
        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            BrowserFetchTool().execute(url="https://example.com", wait_for_selector=".main")
        page.wait_for_selector.assert_called_once_with(".main", timeout=unittest.mock.ANY)


class TestBrowserFetchToolErrors(unittest.TestCase):

    def test_goto_timeout_is_retryable(self):
        page = MagicMock()
        page.goto.side_effect = Exception("Timeout exceeded")
        page.close.return_value = None
        page.query_selector.return_value = MagicMock()

        browser = MagicMock()
        browser.new_page.return_value = page

        pw = MagicMock()
        pw.chromium.launch.return_value = browser

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=pw)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            result = BrowserFetchTool().execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertTrue(result.outcome.retryable)

    def test_empty_body_text_not_retryable(self):
        ctx, _, _ = _make_pw_mock("")
        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            result = BrowserFetchTool().execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertFalse(result.outcome.retryable)

    def test_chromium_launch_failure_not_retryable(self):
        pw = MagicMock()
        pw.chromium.launch.side_effect = Exception("Chromium not found")

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=pw)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("connectors.browser_fetch._PLAYWRIGHT_AVAILABLE", True), \
             patch("connectors.browser_fetch._sync_playwright", return_value=ctx):
            from connectors.browser_fetch import BrowserFetchTool
            result = BrowserFetchTool().execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertFalse(result.outcome.retryable)


if __name__ == "__main__":
    unittest.main()
