from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


class ListenerService:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get()
        self._whisper: Any | None = None
        self._wake_model: Any | None = None
        self._models_loaded = False
        self._stt_backend = self._normalize_backend(self.settings.stt_backend)

    @staticmethod
    def _normalize_backend(value: str) -> str:
        raw = (value or "").strip().lower()
        if raw in {"whisper", "faster_whisper", "faster-whisper"}:
            return "whisper"
        return "apple"

    def load_models(self) -> None:
        if self._models_loaded:
            return

        if self._stt_backend == "apple":
            print("[Listener] STT: macOS native speech recognition")
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
        print(f"[Listener] Ready. Say '{self.settings.wake_word.title()}' to activate.\n")

    def wait_for_wake_word(self) -> None:
        if not self._models_loaded:
            self.load_models()
        chunk = self.settings.wake_word_chunk_size
        with sd.InputStream(samplerate=self.settings.sample_rate, channels=1, dtype="int16", blocksize=chunk) as stream:
            while True:
                audio, _ = stream.read(chunk)
                scores = self._wake_model.predict(np.asarray(audio).flatten().astype(np.int16))
                for name, score in scores.items():
                    if float(score) > self.settings.wake_word_threshold:
                        print(f"[Wake] '{name}' detected ({float(score):.2f})")
                        return

    def record_until_silence(self) -> np.ndarray:
        print("[Listener] Listening...")
        chunk = self.settings.record_chunk_size
        chunks: list[np.ndarray] = []
        silence = 0
        silence_needed = int(self.settings.silence_duration * self.settings.sample_rate / chunk)
        max_chunks = int(self.settings.max_record_seconds * self.settings.sample_rate / chunk)

        with sd.InputStream(samplerate=self.settings.sample_rate, channels=1, dtype="int16", blocksize=chunk) as stream:
            for _ in range(max_chunks):
                audio, _ = stream.read(chunk)
                flat = np.asarray(audio, dtype=np.int16).reshape(-1)
                chunks.append(flat.copy())
                if np.abs(flat).mean() < self.settings.silence_threshold:
                    silence += 1
                    if silence >= silence_needed:
                        break
                else:
                    silence = 0

        return np.concatenate(chunks).astype(np.int16, copy=False) if chunks else np.array([], dtype=np.int16)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if self._stt_backend == "apple":
            return self._transcribe_apple(audio)
        return self._transcribe_whisper(audio)

    def _transcribe_apple(self, audio: np.ndarray) -> str:
        try:
            import speech_recognition as sr
            audio_data = sr.AudioData(audio.tobytes(), self.settings.sample_rate, 2)
            return sr.Recognizer().recognize_apple(audio_data)
        except ImportError:
            logger.warning("SpeechRecognition not installed; falling back to Whisper")
            self._stt_backend = "whisper"
            return self._transcribe_whisper(audio)
        except Exception as exc:
            logger.warning("Apple STT failed (%s); falling back to Whisper", exc)
            self._stt_backend = "whisper"
            return self._transcribe_whisper(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu", compute_type="int8")
        audio_f32 = audio.astype(np.float32) / 32768.0
        segments, _ = self._whisper.transcribe(audio_f32, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
