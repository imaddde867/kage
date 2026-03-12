"""Unit tests for connectors.scheduled_reminders — uses real temp SQLite."""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


def _make_db() -> tuple[Path, str]:
    """Create a temp SQLite file and return (path, filename)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name), f.name


class TestParseWhen(unittest.TestCase):

    def test_full_iso_datetime(self):
        from connectors.scheduled_reminders import _parse_when
        result = _parse_when("2030-06-15T14:30:00")
        self.assertEqual(result, "2030-06-15T14:30:00")

    def test_iso_without_seconds(self):
        from connectors.scheduled_reminders import _parse_when
        result = _parse_when("2030-06-15T14:30")
        self.assertEqual(result, "2030-06-15T14:30:00")

    def test_date_only(self):
        from connectors.scheduled_reminders import _parse_when
        result = _parse_when("2030-06-15")
        self.assertEqual(result, "2030-06-15T00:00:00")

    def test_with_timezone_stripped(self):
        from connectors.scheduled_reminders import _parse_when
        result = _parse_when("2030-06-15T14:30:00+05:00")
        # tzinfo stripped, naive local time
        self.assertIsNotNone(result)
        self.assertNotIn("+", result)

    def test_invalid_returns_none(self):
        from connectors.scheduled_reminders import _parse_when
        self.assertIsNone(_parse_when("not-a-date"))
        self.assertIsNone(_parse_when(""))
        self.assertIsNone(_parse_when("tomorrow at noon"))


class TestScheduleReminderTool(unittest.TestCase):

    def setUp(self):
        self.db_path, self.db_file = _make_db()

    def tearDown(self):
        os.unlink(self.db_file)

    def test_schedule_success(self):
        from connectors.scheduled_reminders import ScheduleReminderTool
        result = ScheduleReminderTool(self.db_path).execute(
            message="Call mom", when="2030-01-01T09:00:00"
        )
        self.assertFalse(result.is_error)
        self.assertIn("2030-01-01", result.content)
        self.assertIn("Call mom", result.content)

    def test_schedule_bad_date_returns_error(self):
        from connectors.scheduled_reminders import ScheduleReminderTool
        result = ScheduleReminderTool(self.db_path).execute(
            message="Call mom", when="not-a-date"
        )
        self.assertTrue(result.is_error)

    def test_scheduled_reminder_stored_in_entity_store(self):
        from connectors.scheduled_reminders import ListScheduledTool, ScheduleReminderTool
        ScheduleReminderTool(self.db_path).execute(
            message="Water plants", when="2030-03-01T08:00:00"
        )
        result = ListScheduledTool(self.db_path).execute()
        self.assertIn("Water plants", result.content)


class TestListScheduledTool(unittest.TestCase):

    def setUp(self):
        self.db_path, self.db_file = _make_db()

    def tearDown(self):
        os.unlink(self.db_file)

    def test_empty_list(self):
        from connectors.scheduled_reminders import ListScheduledTool
        result = ListScheduledTool(self.db_path).execute()
        self.assertFalse(result.is_error)
        self.assertIn("No pending", result.content)

    def test_list_sorted_by_due_date(self):
        from connectors.scheduled_reminders import ListScheduledTool, ScheduleReminderTool
        tool = ScheduleReminderTool(self.db_path)
        tool.execute(message="Later", when="2030-06-01T10:00:00")
        tool.execute(message="Earlier", when="2030-01-01T09:00:00")
        result = ListScheduledTool(self.db_path).execute()
        idx_earlier = result.content.index("Earlier")
        idx_later = result.content.index("Later")
        self.assertLess(idx_earlier, idx_later)


class TestCancelReminderTool(unittest.TestCase):

    def setUp(self):
        self.db_path, self.db_file = _make_db()

    def tearDown(self):
        os.unlink(self.db_file)

    def _schedule(self, message: str) -> None:
        from connectors.scheduled_reminders import ScheduleReminderTool
        ScheduleReminderTool(self.db_path).execute(
            message=message, when="2030-01-01T09:00:00"
        )

    def test_cancel_by_value_substring(self):
        self._schedule("Pick up dry cleaning")
        from connectors.scheduled_reminders import CancelReminderTool, ListScheduledTool
        result = CancelReminderTool(self.db_path).execute(query="dry cleaning")
        self.assertFalse(result.is_error)
        self.assertIn("dry cleaning", result.content.lower())
        # Should no longer appear in list
        list_result = ListScheduledTool(self.db_path).execute()
        self.assertNotIn("dry cleaning", list_result.content.lower())

    def test_cancel_nonexistent_returns_error(self):
        from connectors.scheduled_reminders import CancelReminderTool
        result = CancelReminderTool(self.db_path).execute(query="nonexistent")
        self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
