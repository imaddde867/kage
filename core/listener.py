from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


def _normalize_stt_backend(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"apple", "macos", "siri", "native"}:
        return "apple"
    if raw in {"whisper", "faster_whisper", "faster-whisper"}:
        return "whisper"
    return "apple"


class ListenerService:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get_settings()
        self._whisper: Any | None = None
        self._wake_model: Any | None = None
        self._models_loaded = False
        self._stt_backend = _normalize_stt_backend(self.settings.stt_backend)

    def load_models(self) -> None:
        if self._models_loaded:
            return

        if self._stt_backend == "apple":
            print("[Listener] STT backend: macOS native speech recognition (no model load needed)")
        else:
            print("[Listener] Loading Whisper model...")
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu", compute_type="int8")
            print("[Listener] Whisper ready.")

        print("[Listener] Loading wake word model...")
        import openwakeword
        from openwakeword.model import Model as WakeWordModel

        openwakeword.utils.download_models([self.settings.wake_word_model])
        self._wake_model = WakeWordModel(
            wakeword_models=[self.settings.wake_word_model],
            inference_framework="onnx",
        )
        self._models_loaded = True
        print(f"[Listener] Wake word ready. Say '{self.settings.wake_word.title()}' to activate.\n")

    def _ensure_models_loaded(self) -> None:
        if not self._models_loaded:
            self.load_models()

    def wait_for_wake_word(self) -> bool:
        """Block until the wake word is detected. Returns True when heard."""
        self._ensure_models_loaded()
        chunk = self.settings.wake_word_chunk_size

        with sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk,
        ) as stream:
            while True:
                audio, overflowed = stream.read(chunk)
                if overflowed:
                    logger.warning("Audio input overflow while waiting for wake word")

                scores = self._wake_model.predict(np.asarray(audio).flatten().astype(np.int16))
                for name, score in scores.items():
                    if float(score) > self.settings.wake_word_threshold:
                        print(f"[Wake] '{name}' detected (score: {float(score):.2f})")
                        return True

    def record_until_silence(self) -> np.ndarray:
        """Record until the user stops talking. Returns int16 audio array."""
        print("[Listener] Listening...")
        chunk = self.settings.record_chunk_size
        chunks: list[np.ndarray] = []
        silence = 0
        silence_needed = int(self.settings.silence_duration * self.settings.sample_rate / chunk)
        max_chunks = int(self.settings.max_record_seconds * self.settings.sample_rate / chunk)

        with sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk,
        ) as stream:
            for _ in range(max_chunks):
                audio, overflowed = stream.read(chunk)
                if overflowed:
                    logger.warning("Audio input overflow while recording")

                flat = np.asarray(audio, dtype=np.int16).reshape(-1)
                chunks.append(flat.copy())

                if np.abs(flat).mean() < self.settings.silence_threshold:
                    silence += 1
                    if silence >= silence_needed:
                        break
                else:
                    silence = 0

        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16, copy=False)

    def _transcribe_apple(self, audio: np.ndarray) -> str:
        """Transcribe using macOS native speech recognition (fast, hardware-accelerated)."""
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("SpeechRecognition not installed. Falling back to Whisper.")
            self._stt_backend = "whisper"
            return self._transcribe_whisper(audio)

        try:
            audio_bytes = audio.tobytes()
            audio_data = sr.AudioData(audio_bytes, self.settings.sample_rate, 2)
            recognizer = sr.Recognizer()
            return recognizer.recognize_apple(audio_data)
        except sr.UnknownValueError:
            return ""
        except Exception as exc:
            logger.warning("Apple STT failed (%s). Falling back to Whisper.", exc)
            self._stt_backend = "whisper"
            if self._whisper is None:
                print("[Listener] Loading Whisper model as fallback...")
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel(self.settings.whisper_model, device="cpu", compute_type="int8")
            return self._transcribe_whisper(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        """Transcribe using faster-whisper."""
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu", compute_type="int8")
        audio_f32 = audio.astype(np.float32) / 32768.0
        segments, _ = self._whisper.transcribe(audio_f32, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Convert int16 audio array to text using the configured STT backend."""
        if audio.size == 0:
            return ""
        self._ensure_models_loaded()

        if self._stt_backend == "apple":
            return self._transcribe_apple(audio)
        return self._transcribe_whisper(audio)

    def listen_and_transcribe(self) -> str:
        """Record then transcribe. Returns text string."""
        audio = self.record_until_silence()
        if audio.size == 0:
            return ""
        text = self.transcribe(audio)
        print(f"[You]: {text}")
        return text


_DEFAULT_LISTENER: ListenerService | None = None


def get_default_listener() -> ListenerService:
    global _DEFAULT_LISTENER
    if _DEFAULT_LISTENER is None:
        _DEFAULT_LISTENER = ListenerService()
    return _DEFAULT_LISTENER


def listen_for_wake_word() -> bool:
    return get_default_listener().wait_for_wake_word()


def record_until_silence() -> np.ndarray:
    return get_default_listener().record_until_silence()


def transcribe(audio: np.ndarray) -> str:
    return get_default_listener().transcribe(audio)


def listen_and_transcribe() -> str:
    return get_default_listener().listen_and_transcribe()


__all__ = [
    "ListenerService",
    "get_default_listener",
    "listen_for_wake_word",
    "record_until_silence",
    "transcribe",
    "listen_and_transcribe",
]
