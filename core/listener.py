from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional in test environments
    sd = None  # type: ignore[assignment]

import config

logger = logging.getLogger(__name__)


class ListenerService:
    def __init__(self, *, settings: config.Settings | None = None) -> None:
        self.settings = settings or config.get()
        self._whisper: Any | None = None
        self._wake_model: Any | None = None
        self._models_loaded = False
        self._stt_backend = self._normalize_backend(self.settings.stt_backend)
        self._interrupt_wake_armed = False
        self._interrupt_speech_frames = 0
        self._interrupt_wake_deadline = 0.0
        # Rolling pre-buffer: last ~500ms of audio captured during wake-word detection.
        # Prepended to the recording so speech said right after "hey kage" isn't lost.
        self._pre_buffer: list[np.ndarray] = []

    @staticmethod
    def _normalize_backend(value: str) -> str:
        raw = (value or "").strip().lower()
        if raw in {"whisper", "faster_whisper", "faster-whisper"}:
            return "whisper"
        return "apple"

    @staticmethod
    def _require_sounddevice() -> Any:
        if sd is None:
            raise RuntimeError("sounddevice is not installed. Install it with: pip install sounddevice")
        return sd

    def load_models(self) -> None:
        if self._models_loaded:
            return

        if self._stt_backend == "apple":
            logger.info("Listener STT backend: apple")
        else:
            logger.info("Listener STT backend: mlx-whisper (model will load on first transcription)")

        logger.info("Loading wake word model")
        import openwakeword
        from openwakeword.model import Model as WakeWordModel
        ww_model = self.settings.wake_word_model
        if not os.path.isfile(ww_model):
            openwakeword.utils.download_models([ww_model])
        self._wake_model = WakeWordModel(
            wakeword_models=[ww_model],
            inference_framework="onnx",
        )
        self._models_loaded = True
        logger.info("Listener ready for wake word '%s'", self.settings.wake_word)

    def wait_for_wake_word(self) -> None:
        if not self._models_loaded:
            self.load_models()
        chunk = self.settings.wake_word_chunk_size
        # Keep ~500ms rolling buffer so speech immediately after wake word isn't lost
        pre_buffer_chunks = max(1, int(0.5 * self.settings.sample_rate / chunk))
        rolling: list[np.ndarray] = []
        sd_lib = self._require_sounddevice()
        with sd_lib.InputStream(samplerate=self.settings.sample_rate, channels=1, dtype="int16", blocksize=chunk) as stream:
            while True:
                audio, _ = stream.read(chunk)
                flat = np.asarray(audio).flatten().astype(np.int16)
                rolling.append(flat)
                if len(rolling) > pre_buffer_chunks:
                    rolling.pop(0)
                scores = self._wake_model.predict(flat)
                for name, score in scores.items():
                    if float(score) > self.settings.wake_word_threshold:
                        logger.info("Wake word '%s' detected (%.2f)", name, float(score))
                        self._pre_buffer = list(rolling)
                        return

    def record_until_silence(self) -> np.ndarray:
        logger.info("Listening for user speech")
        chunk = self.settings.record_chunk_size
        chunks: list[np.ndarray] = list(self._pre_buffer)
        self._pre_buffer = []
        silence = 0
        silence_needed = int(self.settings.silence_duration * self.settings.sample_rate / chunk)
        max_chunks = int(self.settings.max_record_seconds * self.settings.sample_rate / chunk)

        sd_lib = self._require_sounddevice()
        with sd_lib.InputStream(samplerate=self.settings.sample_rate, channels=1, dtype="int16", blocksize=chunk) as stream:
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

    def record_with_partial_transcripts(self) -> Iterator[tuple[str, bool]]:
        """Yield partial transcripts while recording, then yield the final transcript.

        Tuples are `(text, is_final)`. The final item is always emitted with
        `is_final=True` (possibly with empty text when nothing was captured).
        """
        logger.info("Listening for user speech (streaming STT)")
        chunk = self.settings.record_chunk_size
        chunks: list[np.ndarray] = list(self._pre_buffer)
        self._pre_buffer = []
        silence = 0
        silence_needed = int(self.settings.silence_duration * self.settings.sample_rate / chunk)
        max_chunks = int(self.settings.max_record_seconds * self.settings.sample_rate / chunk)
        partial_interval = float(getattr(self.settings, "stt_stream_interval_seconds", 0.8))
        partial_every = max(1, int(partial_interval * self.settings.sample_rate / chunk))
        min_chars = max(0, int(getattr(self.settings, "stt_stream_min_chars", 0)))
        last_partial = ""

        sd_lib = self._require_sounddevice()
        with sd_lib.InputStream(samplerate=self.settings.sample_rate, channels=1, dtype="int16", blocksize=chunk) as stream:
            for idx in range(max_chunks):
                audio, _ = stream.read(chunk)
                flat = np.asarray(audio, dtype=np.int16).reshape(-1)
                chunks.append(flat.copy())

                if (idx + 1) % partial_every == 0 and chunks:
                    preview_audio = np.concatenate(chunks).astype(np.int16, copy=False)
                    partial = self.transcribe(preview_audio).strip()
                    if partial and len(partial) >= min_chars and partial != last_partial:
                        last_partial = partial
                        yield partial, False

                if np.abs(flat).mean() < self.settings.silence_threshold:
                    silence += 1
                    if silence >= silence_needed:
                        break
                else:
                    silence = 0

        final_audio = np.concatenate(chunks).astype(np.int16, copy=False) if chunks else np.array([], dtype=np.int16)
        final_text = self.transcribe(final_audio).strip()
        if final_text:
            yield final_text, True
            return
        if last_partial:
            yield last_partial, True
            return
        yield "", True

    def reset_interrupt_detector(self) -> None:
        self._interrupt_wake_armed = False
        self._interrupt_speech_frames = 0
        self._interrupt_wake_deadline = 0.0

    def detect_interrupt(self, audio_chunk: np.ndarray) -> bool:
        if self._wake_model is None:
            return False

        flat = np.asarray(audio_chunk, dtype=np.int16).reshape(-1)
        if flat.size == 0:
            return False

        now = time.monotonic()
        wake_scores = self._wake_model.predict(flat)
        wake_hit = any(float(score) >= self.settings.interrupt_min_score for score in wake_scores.values())
        if wake_hit:
            self._interrupt_wake_armed = True
            self._interrupt_speech_frames = 0
            self._interrupt_wake_deadline = now + 2.0

        if not self._interrupt_wake_armed:
            return False

        if now > self._interrupt_wake_deadline:
            self.reset_interrupt_detector()
            return False

        if np.abs(flat).mean() >= self.settings.silence_threshold:
            self._interrupt_speech_frames += 1
        else:
            self._interrupt_speech_frames = max(0, self._interrupt_speech_frames - 1)

        chunk_ms = (flat.size / self.settings.sample_rate) * 1000.0
        needed_frames = max(1, math.ceil(self.settings.interrupt_hold_ms / max(chunk_ms, 1.0)))
        if self._interrupt_speech_frames >= needed_frames:
            self.reset_interrupt_detector()
            return True

        return False

    def _normalize_transcript_name(self, text: str) -> str:
        if not text or not self.settings.stt_name_normalization_enabled:
            return text

        canonical = self.settings.assistant_name.strip()
        if not canonical:
            return text

        variants = tuple(v.strip() for v in self.settings.stt_name_variants if v.strip())
        if not variants:
            return text

        options = "|".join(
            sorted((re.escape(v) for v in variants), key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_])(?:{options})(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        return pattern.sub(canonical, text)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if self._stt_backend == "apple":
            text = self._transcribe_apple(audio)
        else:
            text = self._transcribe_whisper(audio)
        return self._normalize_transcript_name(text)

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
        import mlx_whisper
        model = self.settings.whisper_model
        if "/" not in model:
            model = f"mlx-community/whisper-{model}-mlx"
        audio_f32 = audio.astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=model,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )
        return result.get("text", "").strip()
