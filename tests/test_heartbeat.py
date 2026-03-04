"""Unit tests for HeartbeatAgent — no real timer, no audio, no LLM."""
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from core.agent.heartbeat import HeartbeatAgent
from core.second_brain.entity_store import EntityStore


def _make_settings(**overrides) -> config.Settings:
    base = config.get()
    # Build a new frozen Settings with overrides applied
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields.update(overrides)
    return config.Settings(**fields)


def _brain_with_db(db_path: Path) -> MagicMock:
    brain = MagicMock()
    brain._entity_store = EntityStore(db_path)
    return brain


def _coordinator(idle: bool = True) -> MagicMock:
    from core.audio_coordinator import AudioState
    coord = MagicMock()
    coord.state = AudioState.IDLE if idle else AudioState.SPEAKING
    return coord


class TestHeartbeatDND(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def _agent(self, **settings_overrides) -> HeartbeatAgent:
        settings = _make_settings(**settings_overrides)
        return HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=settings,
        )

    def test_in_dnd_overnight_window(self) -> None:
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
        agent = self._agent(dnd_start_hour=9, dnd_end_hour=17)
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            self.assertTrue(agent._in_dnd())
        with patch("core.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=18)
            self.assertFalse(agent._in_dnd())


class TestHeartbeatComposeMessage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)
        self.store = EntityStore(self._db)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def _agent(self) -> HeartbeatAgent:
        brain = MagicMock()
        brain._entity_store = self.store
        return HeartbeatAgent(
            brain=brain,
            coordinator=_coordinator(),
            settings=config.get(),
        )

    def test_no_tasks_returns_none(self) -> None:
        agent = self._agent()
        self.assertIsNone(agent._compose_message())

    def test_future_task_returns_none(self) -> None:
        future = (date.today() + timedelta(days=5)).isoformat()
        self.store.upsert("task", "future_task", "Do something later", due_date=future)
        agent = self._agent()
        self.assertIsNone(agent._compose_message())

    def test_due_today_task_triggers_message(self) -> None:
        today = date.today().isoformat()
        self.store.upsert("task", "report", "Finish report", due_date=today)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("Finish report", msg)
        self.assertIn("due today", msg)

    def test_overdue_task_shows_overdue(self) -> None:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        self.store.upsert("task", "old", "Old task", due_date=yesterday)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("overdue", msg)

    def test_multiple_tasks_message(self) -> None:
        today = date.today().isoformat()
        self.store.upsert("task", "t1", "Task one", due_date=today)
        self.store.upsert("task", "t2", "Task two", due_date=today)
        self.store.upsert("commitment", "c1", "Call doctor", due_date=today)
        agent = self._agent()
        msg = agent._compose_message()
        self.assertIsNotNone(msg)
        self.assertIn("3 things due", msg)

    def test_brain_without_entity_store_returns_none(self) -> None:
        brain = MagicMock(spec=[])  # no _entity_store attribute
        agent = HeartbeatAgent(
            brain=brain,
            coordinator=_coordinator(),
            settings=config.get(),
        )
        self.assertIsNone(agent._compose_message())


class TestHeartbeatDebounce(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_debounce_ok_when_never_spoken(self) -> None:
        agent = HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=config.get(),
        )
        self.assertTrue(agent._debounce_ok())

    def test_debounce_blocks_immediately_after_speak(self) -> None:
        settings = _make_settings(proactive_debounce_seconds=999)
        agent = HeartbeatAgent(
            brain=_brain_with_db(self._db),
            coordinator=_coordinator(),
            settings=settings,
        )
        agent._last_proactive = time.monotonic()
        self.assertFalse(agent._debounce_ok())


class TestHeartbeatAudioGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.close()
        self._db.unlink(missing_ok=True)

    def test_tick_skips_when_not_idle(self) -> None:
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
