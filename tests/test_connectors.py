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
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from connectors.memory_ops import ListOpenTasksTool, MarkTaskDoneTool, UpdateFactTool
from connectors.notify import NotifyTool, SpeakTool, _escape_as
from connectors.shell import ShellTool
from connectors.web_search import WebSearchTool
from connectors.web_fetch import WebFetchTool
from connectors.apple_calendar import ReminderAddTool, CalendarReadTool, _escape_as as _cal_escape_as


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
        """Search results include title, URL, and snippet blocks."""
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": "Python 3.13", "body": "Released in October 2024.", "href": "https://python.org"},
            {"title": "Release notes", "body": "Various improvements."},
        ]
        result = self.tool.execute(query="Python 3.13 release")
        self.assertFalse(result.is_error)
        self.assertIn("Python 3.13", result.content)
        self.assertIn("https://python.org", result.content)
        self.assertIn("Released in October", result.content)
        mock_ddgs_cls.return_value.text.assert_called_once_with("Python 3.13 release", max_results=5)

    @patch("connectors.web_search._DDGS")
    def test_no_results(self, mock_ddgs_cls: MagicMock) -> None:
        """An empty result list returns a 'No results' message (not an error)."""
        mock_ddgs_cls.return_value.text.return_value = []
        result = self.tool.execute(query="xyzzy nothing here")
        self.assertFalse(result.is_error)
        self.assertIn("No results", result.content)

    @patch("connectors.web_search._DDGS")
    def test_search_error(self, mock_ddgs_cls: MagicMock) -> None:
        """A network error from DDGS.text() is caught and returned as an error ToolResult."""
        mock_ddgs_cls.return_value.text.side_effect = ConnectionError("network down")
        result = self.tool.execute(query="anything")
        self.assertTrue(result.is_error)
        self.assertIn("Search failed", result.content)

    @patch("connectors.web_search._DDGS")
    def test_max_results_clamped(self, mock_ddgs_cls: MagicMock) -> None:
        """max_results is clamped so huge values don't explode request cost."""
        mock_ddgs_cls.return_value.text.return_value = []
        self.tool.execute(query="anything", max_results=999)
        mock_ddgs_cls.return_value.text.assert_called_once_with("anything", max_results=10)

    @patch("connectors.web_search._DDGS", None)
    def test_missing_ddgs_import(self) -> None:
        """When duckduckgo-search is not installed, _DDGS is None → error with install hint."""
        result = self.tool.execute(query="test")
        self.assertTrue(result.is_error)
        self.assertIn("duckduckgo-search", result.content)


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
        """A valid ISO date is converted to 'Month DD, YYYY' format in the AppleScript."""
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="2026-03-15")
        self.assertFalse(result.is_error)
        self.assertIn("due 2026-03-15", result.content)
        script = mock_run.call_args[0][0]
        self.assertIn("March 15, 2026", script)

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


if __name__ == "__main__":
    unittest.main()
