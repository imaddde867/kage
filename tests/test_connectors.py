"""Unit tests for connectors — mock external calls, no network/subprocess required.

All external side-effects are patched:
    subprocess.run        → mocked for NotifyTool / osascript calls
    core.speaker.speak    → mocked for SpeakTool
    connectors.web_search._DDGS → mocked for WebSearchTool
    connectors.web_fetch._ScraplingFetcher / _HTTPX → mocked for WebFetchTool
    _run_osascript        → mocked for ReminderAddTool / CalendarReadTool

Real I/O is only used by ShellTool (allowlisted commands like date/pwd/echo)
and memory-op tools (which write to a temp SQLite file per test).

Test classes:
    TestEscapeAs           — AppleScript injection-prevention helper
    TestNotifyTool         — macOS notification banner via osascript
    TestSpeakTool          — TTS via core.speaker.speak()
    TestShellTool          — allowlist + metachar security layers (real subprocess)
    TestWebSearchTool      — DuckDuckGo search with mocked _DDGS
    TestWebFetchTool       — Scrapling-first fetch with mocked fallback paths
    TestMemoryOpTools      — mark_task_done, update_fact, list_open_tasks
    TestReminderAddTool    — AppleScript safety: quote escaping + date handling
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from connectors.memory_ops import ListOpenTasksTool, MarkTaskDoneTool, UpdateFactTool
from connectors.notify import NotifyTool, SpeakTool, _escape_as
from connectors.shell import ShellTool
from connectors.web_search import WebSearchTool
from connectors.web_fetch import (
    WebFetchTool,
    _try_parse_json,
    _is_json_content_type,
    _is_ssl_error,
    _is_domain_allowlisted,
)
from connectors.apple_calendar import (
    ReminderAddTool, CalendarReadTool, _escape_as as _cal_escape_as,
    _parse_due_datetime, _due_date_applescript,
)


# ---------------------------------------------------------------------------
# _escape_as (AppleScript escaping)
# ---------------------------------------------------------------------------

class TestEscapeAs(unittest.TestCase):
    """Verify the AppleScript string escaping helper used by notify and calendar connectors.

    _escape_as() must escape backslashes before double-quotes (order matters) so
    user-supplied text can be safely embedded inside AppleScript double-quoted strings.
    Both connectors (notify.py and apple_calendar.py) have their own copy of the
    helper; this class tests both are consistent.
    """

    def test_plain_string_unchanged(self) -> None:
        """Strings without special chars pass through unmodified."""
        self.assertEqual(_escape_as("hello world"), "hello world")

    def test_double_quote_escaped(self) -> None:
        """Double quotes are replaced with backslash-double-quote."""
        self.assertEqual(_escape_as('say "hi"'), 'say \\"hi\\"')

    def test_backslash_escaped(self) -> None:
        """Backslashes are doubled so they are not consumed by AppleScript."""
        self.assertEqual(_escape_as("path\\to"), "path\\\\to")

    def test_both_escaped(self) -> None:
        """When both special chars are present, backslashes are handled first."""
        self.assertEqual(_escape_as('"\\n"'), '\\"\\\\n\\"')

    def test_calendar_escape_same_behaviour(self) -> None:
        """Both copies of _escape_as (notify and apple_calendar) behave identically."""
        self.assertEqual(_escape_as("test"), _cal_escape_as("test"))
        self.assertEqual(_escape_as('"quoted"'), _cal_escape_as('"quoted"'))


# ---------------------------------------------------------------------------
# NotifyTool
# ---------------------------------------------------------------------------

class TestNotifyTool(unittest.TestCase):
    """Verify NotifyTool calls osascript correctly and handles all failure modes."""

    def setUp(self) -> None:
        self.tool = NotifyTool()

    @patch("subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        """A successful osascript call returns a non-error result containing the message."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.tool.execute(message="Hello", title="Test")
        self.assertFalse(result.is_error)
        self.assertIn("Hello", result.content)
        # Verify osascript was called
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "osascript")

    @patch("subprocess.run")
    def test_quotes_in_message_are_escaped(self, mock_run: MagicMock) -> None:
        """Double quotes in the message are escaped in the AppleScript string argument."""
        mock_run.return_value = MagicMock(returncode=0)
        self.tool.execute(message='Say "hi" now', title="Test")
        script_arg = mock_run.call_args[0][0][2]
        self.assertIn('\\"hi\\"', script_arg)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_osascript_not_found(self, _mock: MagicMock) -> None:
        """Missing osascript binary (non-macOS) returns an error ToolResult."""
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)
        self.assertIn("osascript", result.content)

    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "osascript", stderr=b"AppleScript error"),
    )
    def test_osascript_error(self, _mock: MagicMock) -> None:
        """A non-zero osascript exit code returns an error ToolResult."""
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)


# ---------------------------------------------------------------------------
# SpeakTool
# ---------------------------------------------------------------------------

class TestSpeakTool(unittest.TestCase):
    """Verify SpeakTool delegates to core.speaker.speak() and surfaces errors."""

    def setUp(self) -> None:
        self.tool = SpeakTool()

    @patch("core.speaker.speak")
    def test_speak_success(self, mock_speak: MagicMock) -> None:
        """Successful speak() call returns a non-error result."""
        result = self.tool.execute(message="Hello there")
        mock_speak.assert_called_once_with("Hello there")
        self.assertFalse(result.is_error)

    @patch("core.speaker.speak", side_effect=RuntimeError("audio error"))
    def test_speak_failure(self, _mock: MagicMock) -> None:
        """An exception from speak() is caught and returned as an error ToolResult."""
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)
        self.assertIn("audio error", result.content)


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------

class TestShellTool(unittest.TestCase):
    """Verify ShellTool's three security layers: allowlist, metachar block, shell=False.

    Allowed-command tests use real subprocess calls (date, pwd, echo) since
    those are safe and deterministic.  Blocked-command tests only touch the
    security-check code paths — no shell is spawned for rejected commands.
    """

    def setUp(self) -> None:
        self.tool = ShellTool()

    def test_allowed_command_date(self) -> None:
        """'date' is on the allowlist and returns non-empty output."""
        result = self.tool.execute(command="date")
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.strip())

    def test_allowed_command_pwd(self) -> None:
        """'pwd' returns an absolute path starting with '/'."""
        result = self.tool.execute(command="pwd")
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.startswith("/"))

    def test_allowed_command_uname(self) -> None:
        """'uname' is allowlisted for safe local system introspection."""
        result = self.tool.execute(command="uname -s")
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.strip())

    def test_blocked_command(self) -> None:
        """'rm' is not on the allowlist — layer 1 rejects it with an error."""
        result = self.tool.execute(command="rm -rf /")
        self.assertTrue(result.is_error)
        self.assertIn("rm", result.content)

    def test_pipe_blocked(self) -> None:
        """The '|' metacharacter triggers layer 2 rejection regardless of command."""
        result = self.tool.execute(command="ls | grep foo")
        self.assertTrue(result.is_error)

    def test_redirect_blocked(self) -> None:
        """The '>' metacharacter triggers layer 2 rejection."""
        result = self.tool.execute(command="echo hello > file.txt")
        self.assertTrue(result.is_error)

    def test_semicolon_blocked(self) -> None:
        """The ';' metacharacter triggers layer 2 rejection."""
        result = self.tool.execute(command="pwd; ls")
        self.assertTrue(result.is_error)

    def test_empty_command(self) -> None:
        """An empty command string is rejected before reaching the allowlist."""
        result = self.tool.execute(command="")
        self.assertTrue(result.is_error)

    def test_invalid_syntax(self) -> None:
        """An unterminated quote raises ValueError from shlex.split — returned as error."""
        result = self.tool.execute(command="echo 'unterminated")
        self.assertTrue(result.is_error)

    def test_echo(self) -> None:
        """'echo' is allowlisted; its output appears in the result content."""
        result = self.tool.execute(command="echo kage")
        self.assertFalse(result.is_error)
        self.assertIn("kage", result.content)


# ---------------------------------------------------------------------------
# WebSearchTool (mocked DDGS)
# ---------------------------------------------------------------------------

class TestWebSearchTool(unittest.TestCase):
    """Verify WebSearchTool formats results correctly and handles all failure modes.

    _DDGS is a module-level alias for the DuckDuckGo DDGS class (or None when the
    package is not installed).  It is patched here so tests never hit the network.
    """

    def setUp(self) -> None:
        self.tool = WebSearchTool()

    @patch("connectors.web_search._DDGS")
    def test_returns_results(self, mock_ddgs_cls: MagicMock) -> None:
        """Search results are returned as compact structured JSON."""
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": "Python 3.13", "body": "Released in October 2024.", "href": "https://python.org"},
            {"title": "Release notes", "body": "Various improvements."},
        ]
        result = self.tool.execute(query="Python 3.13 release")
        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertEqual(payload["query"], "Python 3.13 release")
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["url"], "https://python.org")
        self.assertIn("Python 3.13", payload["results"][0]["title"])
        mock_ddgs_cls.return_value.text.assert_called_once_with("Python 3.13 release", max_results=5)

    @patch("connectors.web_search._DDGS")
    def test_empty_query_returns_input_error(self, mock_ddgs_cls: MagicMock) -> None:
        result = self.tool.execute(query="   ")
        self.assertTrue(result.is_error)
        self.assertIn("invalid search query", result.content.lower())
        self.assertFalse(result.outcome.retryable)
        mock_ddgs_cls.return_value.text.assert_not_called()

    @patch("connectors.web_search._DDGS")
    def test_no_results(self, mock_ddgs_cls: MagicMock) -> None:
        """An empty result list returns an empty structured result payload."""
        mock_ddgs_cls.return_value.text.return_value = []
        result = self.tool.execute(query="xyzzy nothing here")
        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertEqual(payload["query"], "xyzzy nothing here")
        self.assertEqual(payload["results"], [])

    @patch("connectors.web_search._DDGS")
    def test_search_error(self, mock_ddgs_cls: MagicMock) -> None:
        """A network error from DDGS.text() is caught and returned as an error ToolResult."""
        mock_ddgs_cls.return_value.text.side_effect = ConnectionError("network down")
        result = self.tool.execute(query="anything")
        self.assertTrue(result.is_error)
        self.assertIn("Search failed", result.content)
        self.assertTrue(result.outcome.retryable)

    @patch("connectors.web_search._DDGS")
    def test_retry_once_on_transient_error_then_success(self, mock_ddgs_cls: MagicMock) -> None:
        mock_ddgs_cls.return_value.text.side_effect = [
            ConnectionError("temporary network outage"),
            [{"title": "Recovered", "body": "ok", "href": "https://example.com"}],
        ]
        result = self.tool.execute(query="resilient search")
        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(mock_ddgs_cls.return_value.text.call_count, 2)

    @patch("connectors.web_search._DDGS")
    def test_max_results_clamped(self, mock_ddgs_cls: MagicMock) -> None:
        """max_results is clamped so huge values don't explode request cost."""
        mock_ddgs_cls.return_value.text.return_value = []
        self.tool.execute(query="anything", max_results=999)
        mock_ddgs_cls.return_value.text.assert_called_once_with("anything", max_results=10)

    @patch("connectors.web_search._DDGS")
    def test_malformed_result_rows_are_ignored(self, mock_ddgs_cls: MagicMock) -> None:
        mock_ddgs_cls.return_value.text.return_value = [
            "bad row",
            {"title": "No URL"},
            {"title": "Valid", "body": "snippet", "href": "https://example.com/ok"},
        ]
        result = self.tool.execute(query="mixed rows")
        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["url"], "https://example.com/ok")

    @patch("connectors.web_search._DDGS")
    def test_output_is_size_capped(self, mock_ddgs_cls: MagicMock) -> None:
        long_snippet = "A" * 2000
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": f"Result {idx}", "body": long_snippet, "href": f"https://example.com/{idx}"}
            for idx in range(1, 12)
        ]
        result = self.tool.execute(query="long query", max_results=10)
        self.assertFalse(result.is_error)
        self.assertLessEqual(len(result.content), 2500)

    @patch("connectors.web_search._DDGS", None)
    def test_missing_ddgs_import(self) -> None:
        """When duckduckgo-search is not installed, _DDGS is None → error with install hint."""
        result = self.tool.execute(query="test")
        self.assertTrue(result.is_error)
        self.assertIn("ddgs", result.content.lower())


# ---------------------------------------------------------------------------
# WebFetchTool (Scrapling-first with fallback)
# ---------------------------------------------------------------------------

class TestWebFetchTool(unittest.TestCase):
    """Verify WebFetchTool uses Scrapling first and falls back safely."""

    def setUp(self) -> None:
        self.tool = WebFetchTool()

    @patch("connectors.web_fetch._ScraplingFetcher")
    def test_scrapling_success(self, mock_fetcher: MagicMock) -> None:
        """Readable text is returned when Scrapling succeeds."""
        fake_response = MagicMock()
        fake_response.body = b"<html><body><h1>Hello</h1><p>world</p></body></html>"
        mock_fetcher.get.return_value = fake_response

        result = self.tool.execute(url="https://example.com")
        self.assertFalse(result.is_error)
        self.assertIn("URL: https://example.com", result.content)
        self.assertIn("Hello", result.content)
        self.assertIn("world", result.content)
        mock_fetcher.get.assert_called_once()

    @patch("connectors.web_fetch._ScraplingFetcher")
    def test_scrapling_status_blocked_returns_error(self, mock_fetcher: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 403
        fake_response.url = "https://example.com/challenge"
        fake_response.body = b"<html><body>Forbidden</body></html>"
        mock_fetcher.get.return_value = fake_response

        result = self.tool.execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertIn("anti-bot", result.content.lower())
        self.assertIn("403", result.content)

    @patch("connectors.web_fetch._ScraplingFetcher")
    def test_scrapling_js_challenge_returns_error(self, mock_fetcher: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.url = "https://example.com/challenge"
        fake_response.body = b"<html><body>Please enable JavaScript to continue.</body></html>"
        mock_fetcher.get.return_value = fake_response

        result = self.tool.execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertIn("anti-bot", result.content.lower())

    @patch("connectors.web_fetch._ScraplingFetcher")
    def test_unsupported_region_url_returns_error(self, mock_fetcher: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.url = "https://eu.usatoday.com/unsupported-eu/"
        fake_response.body = b"<html><body>Some generic content</body></html>"
        mock_fetcher.get.return_value = fake_response

        result = self.tool.execute(url="https://eu.usatoday.com/unsupported-eu/")
        self.assertTrue(result.is_error)
        self.assertIn("blocked", result.content.lower())

    @patch("connectors.web_fetch._HTTPX")
    @patch("connectors.web_fetch._ScraplingFetcher")
    def test_fallback_to_httpx(self, mock_fetcher: MagicMock, mock_httpx: MagicMock) -> None:
        """When Scrapling fails, HTTP fallback still returns content."""
        mock_fetcher.get.side_effect = RuntimeError("blocked")
        http_response = MagicMock()
        http_response.text = "<html><body><p>Fallback content</p></body></html>"
        http_response.url = "https://example.com/final"
        http_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = http_response

        result = self.tool.execute(url="https://example.com")
        self.assertFalse(result.is_error)
        self.assertIn("Fallback content", result.content)
        self.assertIn("https://example.com/final", result.content)

    def test_invalid_url(self) -> None:
        """Invalid URLs are rejected early with a clear error."""
        result = self.tool.execute(url="not-a-url")
        self.assertTrue(result.is_error)
        self.assertIn("Invalid URL", result.content)

    def test_invalid_url_type(self) -> None:
        result = self.tool.execute(url=123)  # type: ignore[arg-type]
        self.assertTrue(result.is_error)
        self.assertIn("Invalid URL type", result.content)

    @patch("connectors.web_fetch._HTTPX", None)
    @patch("connectors.web_fetch._ScraplingFetcher", None)
    def test_missing_fetchers(self) -> None:
        """Missing Scrapling and HTTP fallback returns install guidance."""
        result = self.tool.execute(url="https://example.com")
        self.assertTrue(result.is_error)
        self.assertIn("scrapling[fetchers]", result.content)


# ---------------------------------------------------------------------------
# MarkTaskDoneTool / UpdateFactTool / ListOpenTasksTool
# ---------------------------------------------------------------------------

class TestMemoryOpTools(unittest.TestCase):
    """Verify memory-op tools read and write the EntityStore SQLite database correctly.

    A fresh temp database is created for each test to ensure isolation.
    These tests exercise real SQLite I/O (no mocking) because the tools'
    correctness depends on actual database state.
    """

    def setUp(self) -> None:
        """Create a temp SQLite database and instantiate all three tools against it."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)
        self.mark = MarkTaskDoneTool(self._db)
        self.update = UpdateFactTool(self._db)
        self.list_tasks = ListOpenTasksTool(self._db)

    def tearDown(self) -> None:
        """Delete the temp database after each test."""
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_list_empty(self) -> None:
        """An empty store returns a 'No active' message (not an error)."""
        result = self.list_tasks.execute()
        self.assertFalse(result.is_error)
        self.assertIn("No active", result.content)

    def test_update_fact_then_list(self) -> None:
        """A stored task appears in ListOpenTasksTool output after UpdateFactTool writes it."""
        result = self.update.execute(kind="task", key="report", value="Finish Q1 report")
        self.assertFalse(result.is_error)
        self.assertIn("task/report", result.content)

        listed = self.list_tasks.execute()
        self.assertIn("Finish Q1 report", listed.content)

    def test_mark_done_exact_key(self) -> None:
        """Marking a task done by exact key removes it from the active list."""
        self.update.execute(kind="task", key="report", value="Finish Q1 report")
        result = self.mark.execute(key="report")
        self.assertFalse(result.is_error)
        self.assertIn("Finish Q1 report", result.content)

        # Task should no longer appear in list
        listed = self.list_tasks.execute()
        self.assertNotIn("Finish Q1 report", listed.content)

    def test_mark_done_fuzzy_value_match(self) -> None:
        """A substring of the task's value can be used to find and mark it done."""
        self.update.execute(kind="task", key="t1", value="Submit the weekly report")
        result = self.mark.execute(key="weekly report")
        self.assertFalse(result.is_error)
        self.assertIn("Submit the weekly report", result.content)

    def test_mark_done_not_found(self) -> None:
        """Attempting to mark a non-existent task done returns an error ToolResult."""
        result = self.mark.execute(key="nonexistent task")
        self.assertTrue(result.is_error)
        self.assertIn("nonexistent task", result.content)

    def test_update_multiple_kinds(self) -> None:
        """profile and preference entities also appear in the list output."""
        self.update.execute(kind="profile", key="name", value="Imad")
        self.update.execute(kind="preference", key="tone", value="concise")
        listed = self.list_tasks.execute()
        self.assertIn("Imad", listed.content)
        self.assertIn("concise", listed.content)


# ---------------------------------------------------------------------------
# ReminderAddTool (AppleScript injection safety)
# ---------------------------------------------------------------------------

class TestReminderAddTool(unittest.TestCase):
    """Verify ReminderAddTool builds correct AppleScript and handles all edge cases.

    _run_osascript is patched so no actual Reminders app is touched.
    Injection safety is verified by checking the AppleScript string passed
    to the mock.
    """

    def setUp(self) -> None:
        self.tool = ReminderAddTool()

    @patch("connectors.apple_calendar._run_osascript")
    def test_plain_title(self, mock_run: MagicMock) -> None:
        """A plain title without special chars appears verbatim in the AppleScript."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Buy groceries")
        self.assertFalse(result.is_error)
        script = mock_run.call_args[0][0]
        self.assertIn("Buy groceries", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_title_with_quotes_escaped(self, mock_run: MagicMock) -> None:
        """Double quotes in the title are escaped so they cannot break the AppleScript string."""
        mock_run.return_value = ("", False)
        self.tool.execute(title='Finish "report"')
        script = mock_run.call_args[0][0]
        self.assertIn('\\"report\\"', script)
        # The raw unescaped quote should not appear in the script
        self.assertNotIn('name:"Finish "report""', script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_due_date_valid(self, mock_run: MagicMock) -> None:
        """A valid ISO date triggers AppleScript component setters and result includes timestamp."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="2026-03-15")
        self.assertFalse(result.is_error)
        # Result must include a normalized timestamp (date-only defaults to 09:00)
        self.assertIn("2026-03-15T09:00", result.content)
        script = mock_run.call_args[0][0]
        # Component setters used (not locale-dependent date string)
        self.assertIn("set year of d to 2026", script)
        self.assertIn("set month of d to 3", script)
        self.assertIn("set day of d to 15", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_due_date_invalid_silently_skipped(self, mock_run: MagicMock) -> None:
        """An invalid due date is silently ignored; the reminder is still created without one."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="not-a-date")
        # Should still succeed, just without a due clause
        self.assertFalse(result.is_error)
        script = mock_run.call_args[0][0]
        self.assertNotIn("set due date", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_osascript_error_propagated(self, mock_run: MagicMock) -> None:
        """An osascript failure (e.g. permission denied) is returned as an error ToolResult."""
        mock_run.return_value = ("Permission denied", True)
        result = self.tool.execute(title="Task")
        self.assertTrue(result.is_error)


# ---------------------------------------------------------------------------
# CalendarReadTool (retry-on-timeout behavior)
# ---------------------------------------------------------------------------

class TestCalendarReadTool(unittest.TestCase):
    """Verify CalendarReadTool retries timeout failures once and reports cleanly."""

    def setUp(self) -> None:
        self.tool = CalendarReadTool()

    @patch("connectors.apple_calendar._run_osascript")
    @patch("connectors.apple_calendar._config")
    def test_success_first_attempt(self, mock_cfg: MagicMock, mock_run: MagicMock) -> None:
        mock_cfg.get.return_value = MagicMock(
            calendar_read_timeout_seconds=7,
            calendar_read_retry_count=1,
            calendar_read_retry_delay_seconds=0.0,
        )
        mock_run.return_value = ("Standup at Thursday 10:00", False)

        result = self.tool.execute(days=2)
        self.assertFalse(result.is_error)
        self.assertIn("Standup", result.content)
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs.get("timeout"), 7)

    @patch("connectors.apple_calendar._run_osascript")
    @patch("connectors.apple_calendar._config")
    def test_timeout_then_success_retries_once(
        self, mock_cfg: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_cfg.get.return_value = MagicMock(
            calendar_read_timeout_seconds=10,
            calendar_read_retry_count=1,
            calendar_read_retry_delay_seconds=0.0,
        )
        mock_run.side_effect = [
            ("osascript timed out.", True),
            ("Planning review at Friday 09:00", False),
        ]

        result = self.tool.execute(days=3)
        self.assertFalse(result.is_error)
        self.assertIn("Planning review", result.content)
        self.assertEqual(mock_run.call_count, 2)

    @patch("connectors.apple_calendar._run_osascript")
    @patch("connectors.apple_calendar._config")
    def test_timeout_retries_exhausted_returns_deterministic_error(
        self, mock_cfg: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_cfg.get.return_value = MagicMock(
            calendar_read_timeout_seconds=10,
            calendar_read_retry_count=1,
            calendar_read_retry_delay_seconds=0.0,
        )
        mock_run.side_effect = [
            ("osascript timed out.", True),
            ("osascript timed out.", True),
        ]

        result = self.tool.execute(days=1)
        self.assertTrue(result.is_error)
        self.assertIn("timed out after 2 attempts", result.content.lower())
        self.assertEqual(mock_run.call_count, 2)

    @patch("connectors.apple_calendar._run_osascript")
    @patch("connectors.apple_calendar._config")
    def test_non_timeout_error_does_not_retry(
        self, mock_cfg: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_cfg.get.return_value = MagicMock(
            calendar_read_timeout_seconds=10,
            calendar_read_retry_count=3,
            calendar_read_retry_delay_seconds=0.0,
        )
        mock_run.return_value = ("Permission denied", True)

        result = self.tool.execute(days=1)
        self.assertTrue(result.is_error)
        self.assertIn("Permission denied", result.content)
        self.assertEqual(mock_run.call_count, 1)


# ---------------------------------------------------------------------------
# Reminder datetime parsing (_parse_due_datetime, _due_date_applescript)
# ---------------------------------------------------------------------------

class TestReminderDatetimeParsing(unittest.TestCase):
    """Verify the due date/time parsing helpers handle all supported ISO formats."""

    def test_date_only_defaults_to_0900(self) -> None:
        """YYYY-MM-DD is parsed and time defaults to 09:00:00."""
        dt = _parse_due_datetime("2026-03-15")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 3)
        self.assertEqual(dt.day, 15)
        self.assertEqual(dt.hour, 9)
        self.assertEqual(dt.minute, 0)
        self.assertEqual(dt.second, 0)

    def test_datetime_hhmm_parsed(self) -> None:
        """YYYY-MM-DDTHH:MM is parsed with seconds defaulting to 0."""
        dt = _parse_due_datetime("2026-03-15T14:30")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.second, 0)

    def test_datetime_full_parsed(self) -> None:
        """YYYY-MM-DDTHH:MM:SS is parsed in full."""
        dt = _parse_due_datetime("2026-03-15T14:30:45")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.second, 45)

    def test_invalid_string_returns_none(self) -> None:
        """Unparseable strings return None without raising."""
        self.assertIsNone(_parse_due_datetime("not-a-date"))
        self.assertIsNone(_parse_due_datetime("tomorrow"))
        self.assertIsNone(_parse_due_datetime(""))

    def test_applescript_uses_component_setters(self) -> None:
        """_due_date_applescript uses year/month/day/time setters, not date string literals."""
        from datetime import datetime
        dt = datetime(2026, 3, 15, 9, 0, 0)
        snippet = _due_date_applescript(dt)
        self.assertIn("set year of d to 2026", snippet)
        self.assertIn("set month of d to 3", snippet)
        self.assertIn("set day of d to 15", snippet)
        self.assertIn("set time of d to 32400", snippet)  # 9*3600
        self.assertIn("set due date of newReminder to d", snippet)
        # Must NOT use locale-dependent date string literal
        self.assertNotIn('date "', snippet)

    @patch("connectors.apple_calendar._run_osascript")
    def test_datetime_result_contains_normalized_timestamp(self, mock_run: MagicMock) -> None:
        """Result message contains the normalized ISO timestamp when due_date is valid."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Meeting", due_date="2026-03-15T14:30")
        self.assertFalse(result.is_error)
        self.assertIn("2026-03-15T14:30", result.content)

    @patch("connectors.apple_calendar._run_osascript")
    def test_date_only_in_result_uses_0900(self, mock_run: MagicMock) -> None:
        """Date-only input appears in result as YYYY-MM-DDTHH:MM with 09:00."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="2026-04-01")
        self.assertIn("2026-04-01T09:00", result.content)

    def setUp(self) -> None:
        self.tool = ReminderAddTool()


# ---------------------------------------------------------------------------
# WebFetchTool — JSON detection helpers
# ---------------------------------------------------------------------------

class TestWebFetchJsonDetection(unittest.TestCase):
    """Verify the JSON detection helpers used by WebFetchTool."""

    def test_json_object_parsed(self) -> None:
        """A valid JSON object body returns pretty-printed JSON."""
        raw = '{"price": 50000, "currency": "USD"}'
        parsed = _try_parse_json(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIn('"price"', parsed)

    def test_json_array_parsed(self) -> None:
        """A JSON array body is detected and returned."""
        raw = '[{"id": 1}, {"id": 2}]'
        parsed = _try_parse_json(raw)
        self.assertIsNotNone(parsed)

    def test_html_not_detected_as_json(self) -> None:
        """HTML content is not mistaken for JSON."""
        self.assertIsNone(_try_parse_json("<html><body>Hello</body></html>"))

    def test_plain_text_not_detected_as_json(self) -> None:
        """Plain text is not mistaken for JSON."""
        self.assertIsNone(_try_parse_json("Just some text."))

    def test_broken_json_returns_none(self) -> None:
        """A body starting with '{' but invalid JSON returns None without raising."""
        self.assertIsNone(_try_parse_json('{"broken": '))

    def test_json_content_type_detected(self) -> None:
        """application/json content-type header triggers JSON detection."""
        headers = {"content-type": "application/json; charset=utf-8"}
        self.assertTrue(_is_json_content_type(headers))

    def test_html_content_type_not_json(self) -> None:
        """text/html content-type is not detected as JSON."""
        headers = {"content-type": "text/html"}
        self.assertFalse(_is_json_content_type(headers))

    @patch("connectors.web_fetch._ScraplingFetcher", None)
    @patch("connectors.web_fetch._HTTPX")
    def test_json_content_type_response_returns_json(self, mock_httpx: MagicMock) -> None:
        """When server responds with application/json, raw JSON is returned."""
        http_response = MagicMock()
        http_response.headers = {"content-type": "application/json"}
        http_response.text = '{"price": 50000}'
        http_response.url = "https://api.example.com/price"
        http_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = http_response

        tool = WebFetchTool()
        result = tool.execute(url="https://api.example.com/price")
        self.assertFalse(result.is_error)
        self.assertIn('"price"', result.content)


# ---------------------------------------------------------------------------
# WebFetchTool — TLS fallback toggle
# ---------------------------------------------------------------------------

class TestWebFetchTLSFallback(unittest.TestCase):
    """Verify TLS fallback behaviour controlled by WEB_FETCH_TLS_MODE config."""

    @patch("connectors.web_fetch._ScraplingFetcher", None)
    @patch("connectors.web_fetch._HTTPX")
    @patch("connectors.web_fetch._config")
    def test_ssl_error_retried_when_mode_allow(
        self, mock_cfg: MagicMock, mock_httpx: MagicMock
    ) -> None:
        """SSL failure triggers insecure retry when TLS mode is allow_insecure_fallback."""
        mock_cfg.get.return_value.web_fetch_tls_mode = "allow_insecure_fallback"
        mock_cfg.get.return_value.web_fetch_insecure_fallback_domains = (
            "self-signed.example.com",
        )
        mock_cfg.get.return_value.web_fetch_tls_retry_with_certifi = False

        ssl_exc = Exception("ssl certificate verify failed")
        success_response = MagicMock()
        success_response.headers = {"content-type": "text/html"}
        success_response.text = "<html><body><p>Insecure page</p></body></html>"
        success_response.url = "https://self-signed.example.com"
        success_response.raise_for_status.return_value = None

        # First call raises SSL error; second (verify=False) succeeds
        mock_httpx.get.side_effect = [ssl_exc, success_response]

        tool = WebFetchTool()
        result = tool.execute(url="https://self-signed.example.com")
        self.assertFalse(result.is_error)
        self.assertIn("TLS verification disabled", result.content)
        self.assertEqual(mock_httpx.get.call_count, 2)
        # Second call must have verify=False
        _, second_kwargs = mock_httpx.get.call_args_list[1]
        self.assertFalse(second_kwargs.get("verify", True))

    @patch("connectors.web_fetch._ScraplingFetcher", None)
    @patch("connectors.web_fetch._HTTPX")
    @patch("connectors.web_fetch._CERTIFI")
    @patch("connectors.web_fetch._config")
    def test_ssl_error_retried_with_certifi_in_strict_mode(
        self,
        mock_cfg: MagicMock,
        mock_certifi: MagicMock,
        mock_httpx: MagicMock,
    ) -> None:
        mock_cfg.get.return_value.web_fetch_tls_mode = "strict"
        mock_cfg.get.return_value.web_fetch_insecure_fallback_domains = ()
        mock_cfg.get.return_value.web_fetch_tls_retry_with_certifi = True
        mock_certifi.where.return_value = "/tmp/certifi.pem"

        ssl_exc = Exception("ssl certificate verify failed")
        success_response = MagicMock()
        success_response.headers = {"content-type": "text/html"}
        success_response.text = "<html><body><p>Secure retry page</p></body></html>"
        success_response.url = "https://secure.example.com"
        success_response.raise_for_status.return_value = None
        mock_httpx.get.side_effect = [ssl_exc, success_response]

        tool = WebFetchTool()
        result = tool.execute(url="https://secure.example.com")
        self.assertFalse(result.is_error)
        self.assertIn("CA bundle: certifi", result.content)
        self.assertEqual(mock_httpx.get.call_count, 2)
        _, second_kwargs = mock_httpx.get.call_args_list[1]
        self.assertEqual(second_kwargs.get("verify"), "/tmp/certifi.pem")

    @patch("connectors.web_fetch._ScraplingFetcher", None)
    @patch("connectors.web_fetch._HTTPX")
    @patch("connectors.web_fetch._config")
    def test_ssl_error_allow_mode_not_allowlisted_blocks_insecure_fallback(
        self, mock_cfg: MagicMock, mock_httpx: MagicMock
    ) -> None:
        mock_cfg.get.return_value.web_fetch_tls_mode = "allow_insecure_fallback"
        mock_cfg.get.return_value.web_fetch_insecure_fallback_domains = ()
        mock_cfg.get.return_value.web_fetch_tls_retry_with_certifi = False
        mock_httpx.get.side_effect = Exception("ssl certificate verify failed")

        tool = WebFetchTool()
        result = tool.execute(url="https://not-allowlisted.example.com")
        self.assertTrue(result.is_error)
        self.assertEqual(mock_httpx.get.call_count, 1)
        self.assertIn("WEB_FETCH_INSECURE_FALLBACK_DOMAINS", result.content)

    @patch("connectors.web_fetch._ScraplingFetcher", None)
    @patch("connectors.web_fetch._HTTPX")
    @patch("connectors.web_fetch._config")
    def test_ssl_error_not_retried_in_strict_mode(
        self, mock_cfg: MagicMock, mock_httpx: MagicMock
    ) -> None:
        """SSL failure is not retried (only one attempt) when TLS mode is strict."""
        mock_cfg.get.return_value.web_fetch_tls_mode = "strict"
        mock_cfg.get.return_value.web_fetch_insecure_fallback_domains = ()
        mock_cfg.get.return_value.web_fetch_tls_retry_with_certifi = False

        ssl_exc = Exception("ssl certificate verify failed")
        mock_httpx.get.side_effect = ssl_exc

        tool = WebFetchTool()
        result = tool.execute(url="https://self-signed.example.com")
        self.assertTrue(result.is_error)
        self.assertEqual(mock_httpx.get.call_count, 1)

    def test_ssl_error_helper_detects_ssl_messages(self) -> None:
        """_is_ssl_error returns True for SSL-related exception messages."""
        self.assertTrue(_is_ssl_error(Exception("ssl certificate verify failed")))
        self.assertTrue(_is_ssl_error(Exception("TLS handshake error")))
        self.assertTrue(_is_ssl_error(Exception("certificate has expired")))
        self.assertFalse(_is_ssl_error(Exception("connection refused")))
        self.assertFalse(_is_ssl_error(Exception("timeout after 12s")))

    def test_domain_allowlist_supports_subdomains(self) -> None:
        self.assertTrue(
            _is_domain_allowlisted("https://docs.example.com/page", ("example.com",))
        )
        self.assertTrue(_is_domain_allowlisted("https://example.com", ("example.com",)))
        self.assertFalse(_is_domain_allowlisted("https://evil-example.com", ("example.com",)))


if __name__ == "__main__":
    unittest.main()
