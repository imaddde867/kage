from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

import config
from core.listener import ListenerService


class _FakeWakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores[:]

    def predict(self, _audio: np.ndarray) -> dict[str, float]:
        if self._scores:
            return {"hey_jarvis": self._scores.pop(0)}
        return {"hey_jarvis": 0.0}


def _interrupt_settings() -> config.Settings:
    return replace(
        config.get(),
        interrupt_min_score=0.5,
        interrupt_hold_ms=180,
        silence_threshold=50,
        sample_rate=16000,
        wake_word_chunk_size=1600,
    )


class ListenerInterruptTests(unittest.TestCase):
    def test_interrupt_requires_wake_then_speech_hold(self) -> None:
        settings = _interrupt_settings()
        listener = ListenerService(settings=settings)
        listener._wake_model = _FakeWakeModel([0.9, 0.0])  # type: ignore[attr-defined]

        speech = np.full(settings.wake_word_chunk_size, 600, dtype=np.int16)
        self.assertFalse(listener.detect_interrupt(speech))
        self.assertTrue(listener.detect_interrupt(speech))

    def test_interrupt_not_triggered_when_followup_is_silence(self) -> None:
        settings = _interrupt_settings()
        listener = ListenerService(settings=settings)
        listener._wake_model = _FakeWakeModel([0.9, 0.0, 0.0])  # type: ignore[attr-defined]

        silence = np.zeros(settings.wake_word_chunk_size, dtype=np.int16)
        self.assertFalse(listener.detect_interrupt(silence))
        self.assertFalse(listener.detect_interrupt(silence))
        self.assertFalse(listener.detect_interrupt(silence))

    def test_transcript_name_normalization_rewrites_variants(self) -> None:
        settings = replace(
            _interrupt_settings(),
            assistant_name="Kage",
            stt_name_normalization_enabled=True,
            stt_name_variants=("cage", "kaj"),
        )
        listener = ListenerService(settings=settings)

        text = "hey cage, are you there kaj?"
        self.assertEqual(
            listener._normalize_transcript_name(text),
            "hey Kage, are you there Kage?",
        )


if __name__ == "__main__":
    unittest.main()
