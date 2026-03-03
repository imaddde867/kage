from __future__ import annotations

import re
import threading
from typing import Any

import numpy as np
import sounddevice as sd

import config

_MODEL_LOCK = threading.Lock()
_KOKORO_MODEL: Any | None = None


def _sanitize(text: str) -> str:
    text = re.sub(r"[*#`_]", "", text)
    text = text.replace("...", ".").replace("—", ", ").replace("–", ", ")
    return re.sub(r"\s+", " ", text).strip()


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


def _speak_kokoro(model: Any, text: str, settings: config.Settings) -> None:
    stream: sd.OutputStream | None = None
    stream_opened = False
    try:
        for result in model.generate(
            text=text,
            voice=settings.kokoro_voice,
            speed=settings.kokoro_speed,
            lang_code=settings.kokoro_lang_code,
        ):
            chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1, 1)
            if chunk.size == 0:
                continue

            if stream is None:
                stream = sd.OutputStream(
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


def speak(text: str) -> None:
    clean = _sanitize(text)
    if not clean:
        return

    settings = config.get()
    model = _load_model(settings)
    _speak_kokoro(model, clean, settings)
