from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from core.audio_coordinator import AudioCoordinator, AudioState


def _settings(*, debounce_ms: int = 0, guard_ms: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        allow_barge_in=True,
        interrupt_debounce_ms=debounce_ms,
        post_tts_guard_ms=guard_ms,
    )


class AudioCoordinatorTests(unittest.TestCase):
    def test_request_interrupt_marks_state(self) -> None:
        coordinator = AudioCoordinator(settings=_settings())
        coordinator.begin_speaking()

        self.assertTrue(coordinator.request_interrupt())
        self.assertEqual(coordinator.state, AudioState.INTERRUPTED)

    def test_interrupt_debounce_blocks_rapid_retrigger(self) -> None:
        coordinator = AudioCoordinator(settings=_settings(debounce_ms=150))
        coordinator.begin_speaking()
        self.assertTrue(coordinator.request_interrupt())

        coordinator.begin_speaking()
        self.assertFalse(coordinator.request_interrupt())

    def test_post_tts_guard_expires(self) -> None:
        coordinator = AudioCoordinator(settings=_settings(guard_ms=15))
        coordinator.begin_speaking()
        coordinator.end_speaking()

        self.assertTrue(coordinator.in_post_tts_guard())
        time.sleep(0.03)
        self.assertFalse(coordinator.in_post_tts_guard())


if __name__ == "__main__":
    unittest.main()
