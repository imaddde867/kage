"""Unit tests for HeartbeatAgent — no real timer, no audio, no LLM.

Strategy
--------
All time-dependent behaviour is tested by:
  - Patching `core.agent.heartbeat.datetime` to control the reported hour for DND tests.
  - Setting `_last_proactive` directly on the agent instance for debounce tests.
  - Using a real temp SQLite database for EntityStore so message composition tests
    exercise the actual SQL queries.
  - Patching `core.speaker.speak` to assert that speech is (or is not) triggered.

Test classes:
    TestHeartbeatDND            — _in_dnd() logic for overnight and same-day windows
    TestHeartbeatComposeMessage — _compose_message() with various entity states
    TestHeartbeatDebounce       — _debounce_ok() timing logic
    TestHeartbeatAudioGuard     — _tick() is a no-op when audio is not IDLE
"""
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from core.agent.heartbeat import HeartbeatAgent
from core.second_brain.entity_store import EntityStore


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> config.Settings:
    """Return a new frozen Settings with selected fields overridden.

    Uses the live config as a base so all defaults are filled in, then
    substitutes the requested fields.  Useful for testing with custom
    dnd_start_hour / dnd_end_hour / proactive_debounce_seconds values.
    """
    base = config.get()
    # Build a new frozen Settings with overrides applied
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields.update(overrides)
    return config.Settings(**fields)


def _brain_with_db(db_path: Path) -> MagicMock:
    """Return a mock BrainService whose _entity_store points to a real EntityStore."""
    brain = MagicMock()
    brain._entity_store = EntityStore(db_path)
    return brain


def _coordinator(idle: bool = True) -> MagicMock:
    """Return a mock AudioCoordinator whose state is IDLE or SPEAKING."""
    from core.audio_coordinator import AudioState
    coord = MagicMock()
    coord.state = AudioState.IDLE if idle else AudioState.SPEAKING
    return coord


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeartbeatDND(unittest.TestCase):
    """Verify _in_dnd() correctly classifies hours for both overnight and same-day windows."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def _agent(self, **settings_overrides) -> HeartbeatAgent:
        """Build a HeartbeatAgent with custom DND settings."""
        settings = _make_settings(**settings_overrides)
        return HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=settings,
        )

    def test_in_dnd_overnight_window(self) -> None:
        """An overnight window (23–7) correctly identifies hours inside and outside DND."""
        agent = self._agent(dnd_start_hour=23, dnd_end_hour=7)
        # 23:00 → in DND
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=23)
            self.assertTrue(agent._in_dnd())
        # 03:00 → in DND
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=3)
            self.assertTrue(agent._in_dnd())
        # 08:00 → not in DND
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=8)
            self.assertFalse(agent._in_dnd())

    def test_in_dnd_same_day_window(self) -> None:
        """A same-day window (9–17) correctly identifies noon as DND and 18:00 as active."""
        agent = self._agent(dnd_start_hour=9, dnd_end_hour=17)
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            self.assertTrue(agent._in_dnd())
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=18)
            self.assertFalse(agent._in_dnd())


class TestHeartbeatComposeMessage(unittest.TestCase):
    """Verify _compose_message() produces the right output based on entity state.

    A fresh EntityStore is used for each test so entity state is isolated.
    Tests cover: no tasks, future task (suppressed), due-today, overdue,
    multiple items, and brain without an entity store attribute.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)
        self.store = EntityStore(self._db)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def _agent(self) -> HeartbeatAgent:
        """Build a HeartbeatAgent backed by self.store."""
        brain = MagicMock()
        brain._entity_store = self.store
        return HeartbeatAgent(
            brain=brain,
            coordinator=_coordinator(),
            settings=config.get(),
        )

    def test_no_tasks_returns_none(self) -> None:
        """An empty entity store produces no message (None)."""
        agent = self._agent()
        self.assertIsNone(agent._compose_message())

    def test_future_task_returns_none(self) -> None:
        """A task due 5 days from now is not surfaced (not yet urgent)."""
        future = (date.today() + timedelta(days=5)).isoformat()
        self.store.upsert("task", "future_task", "Do something later", due_date=future)
        agent = self._agent()
        self.assertIsNone(agent._compose_message())

    def test_due_today_task_triggers_message(self) -> None:
        """A task due today appears in the message with a 'due today' label."""
        today = date.today().isoformat()
        self.store.upsert("task", "report", "Finish report", due_date=today)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("Finish report", msg)
        self.assertIn("due today", msg)

    def test_overdue_task_shows_overdue(self) -> None:
        """A task due yesterday is labelled 'overdue' in the message."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        self.store.upsert("task", "old", "Old task", due_date=yesterday)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("overdue", msg)

    def test_multiple_tasks_message(self) -> None:
        """Three due items (task, task, commitment) produce a '3 things due' summary."""
        today = date.today().isoformat()
        self.store.upsert("task", "t1", "Task one", due_date=today)
        self.store.upsert("task", "t2", "Task two", due_date=today)
        self.store.upsert("commitment", "c1", "Call doctor", due_date=today)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("3 things due", msg)

    def test_brain_without_entity_store_returns_none(self) -> None:
        """If brain has no _entity_store attribute, compose returns None gracefully."""
        brain = MagicMock(spec=[])  # no _entity_store attribute
        agent = HeartbeatAgent(
            brain=brain,
            coordinator=_coordinator(),
            settings=config.get(),
        )
        self.assertIsNone(agent._compose_message())


class TestHeartbeatDebounce(unittest.TestCase):
    """Verify _debounce_ok() prevents rapid-fire proactive messages."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_debounce_ok_when_never_spoken(self) -> None:
        """A freshly created agent (no prior speak) passes the debounce check."""
        agent = HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=config.get(),
        )
        self.assertTrue(agent._debounce_ok())

    def test_debounce_blocks_immediately_after_speak(self) -> None:
        """Setting _last_proactive to now blocks a second speak with a long debounce window."""
        settings = _make_settings(proactive_debounce_seconds=999)
        agent = HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=settings,
        )
        agent._last_proactive = time.monotonic()
        self.assertFalse(agent._debounce_ok())


class TestHeartbeatAudioGuard(unittest.TestCase):
    """Verify _tick() skips speaking when the audio system is busy."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_tick_skips_when_not_idle(self) -> None:
        """When coordinator.state is SPEAKING, _tick() must not call speak() even if a task is due."""
        today = date.today().isoformat()
        store = EntityStore(self._db)
        store.upsert("task", "t", "Due task", due_date=today)

        brain = MagicMock()
        brain._entity_store = store

        # Coordinator reports SPEAKING (not idle)
        coord = _coordinator(idle=False)

        agent = HeartbeatAgent(brain=brain, coordinator=coord, settings=config.get())

        with patch("core.speaker.speak") as mock_speak:
            agent._tick()
            mock_speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
