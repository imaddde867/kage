"""Unit tests for connectors — mock external calls, no network/subprocess required."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from connectors.memory_ops import ListOpenTasksTool, MarkTaskDoneTool, UpdateFactTool
from connectors.notify import NotifyTool, SpeakTool, _escape_as
from connectors.shell import ShellTool
from connectors.web_search import WebSearchTool
from connectors.apple_calendar import ReminderAddTool, CalendarReadTool, _escape_as as _cal_escape_as


# ---------------------------------------------------------------------------
# _escape_as (AppleScript escaping)
# ---------------------------------------------------------------------------

class TestEscapeAs(unittest.TestCase):
    def test_plain_string_unchanged(self) -> None:
        self.assertEqual(_escape_as("hello world"), "hello world")

    def test_double_quote_escaped(self) -> None:
        self.assertEqual(_escape_as('say "hi"'), 'say \\"hi\\"')

    def test_backslash_escaped(self) -> None:
        self.assertEqual(_escape_as("path\\to"), "path\\\\to")

    def test_both_escaped(self) -> None:
        self.assertEqual(_escape_as('"\\n"'), '\\"\\\\n\\"')

    def test_calendar_escape_same_behaviour(self) -> None:
        self.assertEqual(_escape_as("test"), _cal_escape_as("test"))
        self.assertEqual(_escape_as('"quoted"'), _cal_escape_as('"quoted"'))


# ---------------------------------------------------------------------------
# NotifyTool
# ---------------------------------------------------------------------------

class TestNotifyTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotifyTool()

    @patch("subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        result = self.tool.execute(message="Hello", title="Test")
        self.assertFalse(result.is_error)
        self.assertIn("Hello", result.content)
        # Verify osascript was called
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "osascript")

    @patch("subprocess.run")
    def test_quotes_in_message_are_escaped(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        self.tool.execute(message='Say "hi" now', title="Test")
        script_arg = mock_run.call_args[0][0][2]
        self.assertIn('\\"hi\\"', script_arg)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_osascript_not_found(self, _mock: MagicMock) -> None:
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)
        self.assertIn("osascript", result.content)

    @patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "osascript", stderr=b"AppleScript error"),
    )
    def test_osascript_error(self, _mock: MagicMock) -> None:
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)


# ---------------------------------------------------------------------------
# SpeakTool
# ---------------------------------------------------------------------------

class TestSpeakTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = SpeakTool()

    @patch("core.speaker.speak")
    def test_speak_success(self, mock_speak: MagicMock) -> None:
        result = self.tool.execute(message="Hello there")
        mock_speak.assert_called_once_with("Hello there")
        self.assertFalse(result.is_error)

    @patch("core.speaker.speak", side_effect=RuntimeError("audio error"))
    def test_speak_failure(self, _mock: MagicMock) -> None:
        result = self.tool.execute(message="hi")
        self.assertTrue(result.is_error)
        self.assertIn("audio error", result.content)


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------

class TestShellTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ShellTool()

    def test_allowed_command_date(self) -> None:
        result = self.tool.execute(command="date")
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.strip())

    def test_allowed_command_pwd(self) -> None:
        result = self.tool.execute(command="pwd")
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.startswith("/"))

    def test_blocked_command(self) -> None:
        result = self.tool.execute(command="rm -rf /")
        self.assertTrue(result.is_error)
        self.assertIn("rm", result.content)

    def test_pipe_blocked(self) -> None:
        result = self.tool.execute(command="ls | grep foo")
        self.assertTrue(result.is_error)

    def test_redirect_blocked(self) -> None:
        result = self.tool.execute(command="echo hello > file.txt")
        self.assertTrue(result.is_error)

    def test_semicolon_blocked(self) -> None:
        result = self.tool.execute(command="pwd; ls")
        self.assertTrue(result.is_error)

    def test_empty_command(self) -> None:
        result = self.tool.execute(command="")
        self.assertTrue(result.is_error)

    def test_invalid_syntax(self) -> None:
        result = self.tool.execute(command="echo 'unterminated")
        self.assertTrue(result.is_error)

    def test_echo(self) -> None:
        result = self.tool.execute(command="echo kage")
        self.assertFalse(result.is_error)
        self.assertIn("kage", result.content)


# ---------------------------------------------------------------------------
# WebSearchTool (mocked DDGS)
# ---------------------------------------------------------------------------

class TestWebSearchTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = WebSearchTool()

    @patch("connectors.web_search._DDGS")
    def test_returns_results(self, mock_ddgs_cls: MagicMock) -> None:
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": "Python 3.13", "body": "Released in October 2024."},
            {"title": "Release notes", "body": "Various improvements."},
        ]
        result = self.tool.execute(query="Python 3.13 release")
        self.assertFalse(result.is_error)
        self.assertIn("Python 3.13", result.content)
        self.assertIn("Released in October", result.content)

    @patch("connectors.web_search._DDGS")
    def test_no_results(self, mock_ddgs_cls: MagicMock) -> None:
        mock_ddgs_cls.return_value.text.return_value = []
        result = self.tool.execute(query="xyzzy nothing here")
        self.assertFalse(result.is_error)
        self.assertIn("No results", result.content)

    @patch("connectors.web_search._DDGS")
    def test_search_error(self, mock_ddgs_cls: MagicMock) -> None:
        mock_ddgs_cls.return_value.text.side_effect = ConnectionError("network down")
        result = self.tool.execute(query="anything")
        self.assertTrue(result.is_error)
        self.assertIn("Search failed", result.content)

    @patch("connectors.web_search._DDGS", None)
    def test_missing_ddgs_import(self) -> None:
        result = self.tool.execute(query="test")
        self.assertTrue(result.is_error)
        self.assertIn("duckduckgo-search", result.content)


# ---------------------------------------------------------------------------
# MarkTaskDoneTool / UpdateFactTool / ListOpenTasksTool
# ---------------------------------------------------------------------------

class TestMemoryOpTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)
        self.mark = MarkTaskDoneTool(self._db)
        self.update = UpdateFactTool(self._db)
        self.list_tasks = ListOpenTasksTool(self._db)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_list_empty(self) -> None:
        result = self.list_tasks.execute()
        self.assertFalse(result.is_error)
        self.assertIn("No active", result.content)

    def test_update_fact_then_list(self) -> None:
        result = self.update.execute(kind="task", key="report", value="Finish Q1 report")
        self.assertFalse(result.is_error)
        self.assertIn("task/report", result.content)

        listed = self.list_tasks.execute()
        self.assertIn("Finish Q1 report", listed.content)

    def test_mark_done_exact_key(self) -> None:
        self.update.execute(kind="task", key="report", value="Finish Q1 report")
        result = self.mark.execute(key="report")
        self.assertFalse(result.is_error)
        self.assertIn("Finish Q1 report", result.content)

        # Task should no longer appear in list
        listed = self.list_tasks.execute()
        self.assertNotIn("Finish Q1 report", listed.content)

    def test_mark_done_fuzzy_value_match(self) -> None:
        self.update.execute(kind="task", key="t1", value="Submit the weekly report")
        result = self.mark.execute(key="weekly report")
        self.assertFalse(result.is_error)
        self.assertIn("Submit the weekly report", result.content)

    def test_mark_done_not_found(self) -> None:
        result = self.mark.execute(key="nonexistent task")
        self.assertTrue(result.is_error)
        self.assertIn("nonexistent task", result.content)

    def test_update_multiple_kinds(self) -> None:
        self.update.execute(kind="profile", key="name", value="Imad")
        self.update.execute(kind="preference", key="tone", value="concise")
        listed = self.list_tasks.execute()
        self.assertIn("Imad", listed.content)
        self.assertIn("concise", listed.content)


# ---------------------------------------------------------------------------
# ReminderAddTool (AppleScript injection safety)
# ---------------------------------------------------------------------------

class TestReminderAddTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReminderAddTool()

    @patch("connectors.apple_calendar._run_osascript")
    def test_plain_title(self, mock_run: MagicMock) -> None:
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Buy groceries")
        self.assertFalse(result.is_error)
        script = mock_run.call_args[0][0]
        self.assertIn("Buy groceries", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_title_with_quotes_escaped(self, mock_run: MagicMock) -> None:
        mock_run.return_value = ("", False)
        self.tool.execute(title='Finish "report"')
        script = mock_run.call_args[0][0]
        self.assertIn('\\"report\\"', script)
        # The raw unescaped quote should not appear in the script
        self.assertNotIn('name:"Finish "report""', script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_due_date_valid(self, mock_run: MagicMock) -> None:
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="2026-03-15")
        self.assertFalse(result.is_error)
        self.assertIn("due 2026-03-15", result.content)
        script = mock_run.call_args[0][0]
        self.assertIn("March 15, 2026", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_due_date_invalid_silently_skipped(self, mock_run: MagicMock) -> None:
        mock_run.return_value = ("", False)
        result = self.tool.execute(title="Task", due_date="not-a-date")
        # Should still succeed, just without a due clause
        self.assertFalse(result.is_error)
        script = mock_run.call_args[0][0]
        self.assertNotIn("set due date", script)

    @patch("connectors.apple_calendar._run_osascript")
    def test_osascript_error_propagated(self, mock_run: MagicMock) -> None:
        mock_run.return_value = ("Permission denied", True)
        result = self.tool.execute(title="Task")
        self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
