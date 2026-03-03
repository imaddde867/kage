from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional in test environments
    sd = None  # type: ignore[assignment]

import config

_MODEL_LOCK = threading.Lock()
_SPEAK_LOCK = threading.Lock()
_SPEAK_STOP_EVENT = threading.Event()
_KOKORO_MODEL: Any | None = None


@dataclass(frozen=True)
class SpeakResult:
    completed: bool

    @property
    def interrupted(self) -> bool:
        return not self.completed


def _sanitize(text: str) -> str:
    text = re.sub(r"[*#`_]", "", text)
    text = text.replace("...", ".").replace("—", ", ").replace("–", ", ")
    return re.sub(r"\s+", " ", text).strip()


def _apply_name_pronunciation(text: str, settings: config.Settings) -> str:
    if not settings.tts_name_override_enabled:
        return text

    name = settings.assistant_name.strip()
    spoken = settings.tts_name_pronunciation.strip()
    if not name or not spoken:
        return text

    pattern = re.compile(
        rf"\b{re.escape(name)}(?P<possessive>'s)?\b",
        flags=re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        suffix = match.group("possessive") or ""
        return f"{spoken}{suffix}"

    return pattern.sub(_replace, text)


def _require_sounddevice() -> Any:
    if sd is None:
        raise RuntimeError("sounddevice is not installed. Install it with: pip install sounddevice")
    return sd


def _load_model(settings: config.Settings) -> Any:
    global _KOKORO_MODEL
    if _KOKORO_MODEL is not None:
        return _KOKORO_MODEL

    with _MODEL_LOCK:
        if _KOKORO_MODEL is not None:
            return _KOKORO_MODEL

        try:
            # Support both mlx-audio layouts:
            # - newer: mlx_audio.tts.load_model
            # - some builds: mlx_audio.tts.utils.load_model
            try:
                from mlx_audio.tts import load_model as mlx_load_model  # type: ignore[import]
            except ImportError:
                from mlx_audio.tts.utils import load_model as mlx_load_model  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio TTS is not installed. Install it with: pip install 'mlx-audio[tts]'"
            ) from exc

        print(f"[Speaker] Loading Kokoro model {settings.kokoro_model}…", flush=True)
        _KOKORO_MODEL = mlx_load_model(settings.kokoro_model, lazy=True)
        print("[Speaker] Kokoro ready.", flush=True)

    return _KOKORO_MODEL


def _should_stop(cancel_token: threading.Event | None) -> bool:
    return _SPEAK_STOP_EVENT.is_set() or (cancel_token is not None and cancel_token.is_set())


def _speak_kokoro(
    model: Any,
    text: str,
    settings: config.Settings,
    *,
    cancel_token: threading.Event | None = None,
    on_chunk: Callable[[np.ndarray, int], None] | None = None,
) -> bool:
    stream: Any | None = None
    stream_opened = False
    completed = True
    sd_lib = _require_sounddevice()
    try:
        for result in model.generate(
            text=text,
            voice=settings.kokoro_voice,
            speed=settings.kokoro_speed,
            lang_code=settings.kokoro_lang_code,
        ):
            if _should_stop(cancel_token):
                completed = False
                break

            chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1, 1)
            if chunk.size == 0:
                continue

            if on_chunk is not None:
                on_chunk(chunk, int(result.sample_rate))

            if _should_stop(cancel_token):
                completed = False
                break

            if stream is None:
                stream = sd_lib.OutputStream(
                    samplerate=int(result.sample_rate),
                    channels=1,
                    dtype="float32",
                )
                stream.start()
                stream_opened = True

            stream.write(chunk)
    finally:
        if stream is not None and stream_opened:
            try:
                stream.stop()
            finally:
                stream.close()
    return completed


def stop_speaking() -> None:
    _SPEAK_STOP_EVENT.set()


def speak(
    text: str,
    *,
    cancel_token: threading.Event | None = None,
    on_chunk: Callable[[np.ndarray, int], None] | None = None,
) -> SpeakResult:
    clean = _sanitize(text)
    settings = config.get()
    clean = _apply_name_pronunciation(clean, settings)
    if not clean:
        return SpeakResult(completed=True)

    with _SPEAK_LOCK:
        _SPEAK_STOP_EVENT.clear()
        model = _load_model(settings)
        completed = _speak_kokoro(
            model,
            clean,
            settings,
            cancel_token=cancel_token,
            on_chunk=on_chunk,
        )
        _SPEAK_STOP_EVENT.clear()
        return SpeakResult(completed=completed)
