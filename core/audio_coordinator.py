from __future__ import annotations

import threading
import time
from enum import Enum

import config


class AudioState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class AudioCoordinator:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get()
        self._state = AudioState.IDLE
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._last_tts_stop = 0.0
        self._last_interrupt = 0.0

    @property
    def state(self) -> AudioState:
        with self._lock:
            return self._state

    @property
    def allow_barge_in(self) -> bool:
        return self.settings.allow_barge_in

    @property
    def cancel_token(self) -> threading.Event:
        with self._lock:
            return self._cancel_event

    def transition(self, state: AudioState) -> None:
        with self._lock:
            self._state = state

    def begin_speaking(self) -> threading.Event:
        with self._lock:
            self._cancel_event = threading.Event()
            self._state = AudioState.SPEAKING
            return self._cancel_event

    def request_interrupt(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._state != AudioState.SPEAKING:
                return False

            debounce_s = self.settings.interrupt_debounce_ms / 1000.0
            if debounce_s > 0 and now - self._last_interrupt < debounce_s:
                return False

            self._cancel_event.set()
            self._state = AudioState.INTERRUPTED
            self._last_interrupt = now
            return True

    def end_speaking(self, *, interrupted: bool = False) -> None:
        with self._lock:
            self._last_tts_stop = time.monotonic()
            if interrupted:
                self._state = AudioState.INTERRUPTED
            elif self._state == AudioState.SPEAKING:
                self._state = AudioState.THINKING

    def post_tts_guard_remaining(self) -> float:
        with self._lock:
            elapsed = time.monotonic() - self._last_tts_stop
        guard_s = self.settings.post_tts_guard_ms / 1000.0
        return max(0.0, guard_s - elapsed)

    def in_post_tts_guard(self) -> bool:
        return self.post_tts_guard_remaining() > 0.0

    def wait_for_listen_window(self) -> None:
        remaining = self.post_tts_guard_remaining()
        if remaining > 0:
            time.sleep(remaining)
